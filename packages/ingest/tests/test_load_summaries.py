"""ingest load with SAC summaries: what the index holds of them, and the checks on them.

Postgres is real, the index is the stand-in directory. Nothing here goes through
the service's retrieval tool. The leak test reads the index the way a persona's
clearance filter would, so it holds whatever ends up doing the searching.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from spatial_core.records import PERSONAS, Chunk, ChunkRole, Persona
from spatial_core.search import LocalIndex
from spatial_ingest.search import AzureSearchIndex
from stubs import AzureSearch, OpenAI
from test_load import C, ingest, read_jsonl, rewrite
from test_load_documents import INDEX_C, Bundle, azure, bundle, chunks, edit_manifest, failures

__all__ = ["azure", "bundle"]


def summaries(bundle: Path) -> list[dict[str, Any]]:
    return read_jsonl(bundle / "documents" / "summaries.jsonl")


def edit_summary(bundle: Path, summary_id: str, **changes: Any) -> None:
    records = summaries(bundle)
    for record in records:
        if record["id"] == summary_id:
            record.update(changes)
    rewrite(bundle, "documents/summaries.jsonl", records)


def admitted(index: list[Chunk], persona: Persona) -> list[Chunk]:
    """What a persona's clearance filter lets through, summaries and passages alike."""
    return [chunk for chunk in index if persona.rank >= chunk.clearance_class.rank]


def words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def test_load_indexes_each_summary_as_a_summary_chunk_of_its_document(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    held = {chunk.id: chunk for chunk in LocalIndex(search_dir).chunks(INDEX_C)}
    by_role = {role: [c for c in held.values() if c.role is role] for role in ChunkRole}
    assert len(by_role[ChunkRole.PASSAGE]) == 8
    assert sorted(c.id for c in by_role[ChunkRole.SUMMARY]) == sorted(
        s["id"] for s in summaries(built)
    )
    room, passage = held["room:107#summary"], held["room:107#0"]
    assert room.text == next(s["text"] for s in summaries(built) if s["id"] == room.id)
    assert room.ordinal == -1
    # Its document's in everything a filter looks at.
    for field in ("document_id", "kind", "title", "url", "clearance_class", "space_ids"):
        assert getattr(room, field) == getattr(passage, field), field
    assert room.ancestor_ids == passage.ancestor_ids
    assert room.clearance_class == "staff"


def test_a_visitors_retrieval_never_returns_text_derived_from_a_higher_clearance_class(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    index = LocalIndex(search_dir).chunks(INDEX_C)
    made = {s["id"]: s for s in summaries(built)}
    passages = {c.id: c for c in index if c.role is ChunkRole.PASSAGE}

    for position, persona in enumerate(PERSONAS[:-1]):
        offered = admitted(index, persona)
        above = [c for c in passages.values() if c.clearance_class.rank > position]
        assert offered and above
        # Everything the persona's own documents say, plus the stand-in summariser's
        # one word of its own. A summary is made of the words its model read.
        own = words("covers")
        for chunk in offered:
            if chunk.role is ChunkRole.PASSAGE:
                own |= words(chunk.title) | words(chunk.text)
        for chunk in offered:
            if chunk.role is not ChunkRole.SUMMARY:
                continue
            # Where it came from, by the bundle's own record of what the model read.
            sources = [passages[source] for source in made[chunk.id]["source_chunk_ids"]]
            assert all(source.clearance_class.rank <= position for source in sources), chunk.id
            # And what it says: nothing a higher class's document says and theirs do not.
            assert words(chunk.text) <= own, chunk.id
        told = set().union(*(words(chunk.text) for chunk in offered))
        for hidden in above:
            assert words(hidden.text) - own, hidden.id
            assert not (words(hidden.text) - own) & told, hidden.id

    # Said plainly for the visitor: the staff page and the maintenance manual.
    visible = " ".join(chunk.text for chunk in admitted(index, Persona.VISITOR))
    assert "bell" not in visible and "staff corridor" not in visible
    assert "valve" not in visible.lower()
    # A maintenance summary does exist. It is just not the visitor's to find.
    assert "valve" in next(c for c in index if c.id == "manual:valve#summary").text


def test_load_fails_when_a_summary_was_built_across_clearance_classes(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    # The visitor summary of the Gaede page now claims it also read the staff page.
    edit_summary(built, "hall:gaede#summary", source_chunk_ids=["hall:gaede#0", "room:107#0"])

    assert ingest("load", built, database_url) == 1

    assert failures(database_url, C) == [
        "a summary is built from passages of its own document and clearance class: "
        "fails for hall:gaede#summary"
    ]


def test_load_fails_when_a_summary_is_filed_under_another_class_than_its_document(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    edit_summary(built, "manual:valve#summary", clearance_class="visitor")

    def recount(manifest: dict[str, Any]) -> None:
        manifest["documents"]["summaries"]["by_clearance_class"] = {
            "visitor": 6,
            "staff": 1,
            "maintenance": 0,
        }

    edit_manifest(built, recount)

    assert ingest("load", built, database_url) == 1

    found = failures(database_url, C)
    assert (
        "a chunk's clearance class is its strictest anchor's: fails for manual:valve#summary"
        in found
    )
    assert (
        "a summary is built from passages of its own document and clearance class: "
        "fails for manual:valve#summary"
    ) in found


def test_load_fails_when_a_document_in_the_index_has_no_summary(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    rewrite(
        built,
        "documents/summaries.jsonl",
        [s for s in summaries(built) if s["id"] != "manual:ext#summary"],
    )

    assert ingest("load", built, database_url) == 1

    found = failures(database_url, C)
    assert "every document in the index has one summary: fails for manual:ext" in found
    assert "the index holds 4 visitor summaries, the manifest says 5" in found


def test_reloading_a_bundle_reuses_its_summaries_and_makes_none(
    bundle: Bundle, database_url: str, search_dir: Path, openai: OpenAI
) -> None:
    built = bundle(C)
    asked = len(openai.calls)
    # A model that would answer differently now, if anyone asked it.
    openai.status = 500

    assert ingest("load", built, database_url) == 0
    first = (search_dir / f"{INDEX_C}.json").read_bytes()
    assert ingest("load", built, database_url) == 0

    assert (search_dir / f"{INDEX_C}.json").read_bytes() == first
    assert len(openai.calls) == asked
    held = {chunk.id: chunk.text for chunk in LocalIndex(search_dir).chunks(INDEX_C)}
    for summary in summaries(built):
        assert held[summary["id"]] == summary["text"]


def test_the_azure_index_can_tell_a_summary_from_a_passage(
    bundle: Bundle, database_url: str, azure: AzureSearch
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    fields = {field["name"]: field for field in azure.definitions[INDEX_C]["fields"]}
    assert fields["role"]["filterable"] is True
    roles = {document["id"]: document["role"] for document in azure.documents[INDEX_C].values()}
    assert roles["manual:b1#0"] == "passage" and roles["manual:b1#summary"] == "summary"
    assert sorted(roles) == sorted(
        [*(c["id"] for c in chunks(built)), *(s["id"] for s in summaries(built))]
    )
    client = AzureSearchIndex(endpoint=azure.url, api_key="search-key")
    assert sum(1 for c in client.chunks(INDEX_C) if c.role is ChunkRole.SUMMARY) == 7
