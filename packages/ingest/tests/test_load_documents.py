"""ingest load and promote with a document corpus: the search index and its checks.

Postgres is real. The search index is the stand-in directory, except where a test
points load at a stand-in Azure AI Search service on localhost.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import prose
from spatial_core.records import Chunk
from spatial_core.search import LocalIndex
from spatial_ingest import search
from spatial_ingest.build import MANIFEST_FILE
from spatial_ingest.cli import main
from spatial_ingest.promote import promote
from spatial_ingest.search import AzureSearchIndex, SearchError
from stubs import AzureSearch, serve_azure_search
from test_load import A, B, C, D, ingest, pointer, query, read_jsonl, rewrite, snapshots

Bundle = Callable[[str], Path]
INDEX_C = "spatial-mini-campus-20260915"


@pytest.fixture
def bundle(tmp_path: Path, venue_file: Path, corpus_file: Path) -> Bundle:
    """Builds the mini campus with its corpus as the snapshot of a date, once."""
    out = tmp_path / "bundles"

    def built(date: str) -> Path:
        if not (out / date / MANIFEST_FILE).exists():
            args = ["build", "--date", date, "--venue", str(venue_file), "--out", str(out)]
            assert main([*args, "--documents", str(corpus_file)]) == 0
        return out / date

    return built


def chunks(bundle: Path) -> list[dict[str, Any]]:
    return read_jsonl(bundle / "documents" / "chunks.jsonl")


def edit_chunk(bundle: Path, chunk_id: str, **changes: Any) -> None:
    records = chunks(bundle)
    for record in records:
        if record["id"] == chunk_id:
            record.update(changes)
    rewrite(bundle, "documents/chunks.jsonl", records)


def edit_manifest(bundle: Path, change: Callable[[dict[str, Any]], None]) -> None:
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    change(manifest)
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))


def failures(database_url: str, date: str) -> list[str]:
    ((found,),) = query(
        database_url, "SELECT failures FROM control.snapshots WHERE snapshot = %s", date
    )
    return list(found)


def indexes(database_url: str) -> tuple[Any, ...]:
    (row,) = query(
        database_url,
        "SELECT search_index, previous_search_index FROM control.current_snapshot",
    )
    return row


def test_load_writes_one_index_for_the_snapshot_holding_every_chunk(
    bundle: Bundle, database_url: str, search_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    assert f"with index {INDEX_C}" in capsys.readouterr().out
    assert [path.name for path in search_dir.iterdir()] == [f"{INDEX_C}.json"]
    stored = LocalIndex(search_dir).entries(INDEX_C)
    assert len(stored) == 8
    held = {entry.chunk.id: entry for entry in stored}
    vectors = {v["chunk_id"]: v["vector"] for v in read_jsonl(built / "documents/vectors.jsonl")}
    for record in chunks(built):
        assert held[record["id"]].chunk == Chunk.model_validate(record)
        assert list(held[record["id"]].vector) == vectors[record["id"]]
        assert len(held[record["id"]].vector) == prose.DIMENSIONS


def test_an_indexed_chunk_carries_its_anchors_its_paths_and_its_clearance_class(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    assert ingest("load", bundle(C), database_url) == 0

    held = {chunk.id: chunk for chunk in LocalIndex(search_dir).chunks(INDEX_C)}
    room = held["room:107#0"]
    assert room.space_ids == ("osm:way/100", "osm:way/1001")
    assert room.ancestor_ids == (
        "floor:way/100:1",
        "osm:way/100",
        "osm:way/1001",
        "site:mini-campus",
    )
    assert room.clearance_class == "staff"
    air = held["manual:air#0"]
    assert air.asset_type_ids == (prose.DIFFUSER, prose.VAV)
    # The shared manual is visitor although one of its types is maintenance, and
    # load's clearance check agrees, or this load would have failed.
    assert air.clearance_class == "visitor"
    assert held["manual:ext#0"].asset_ids == (prose.EXTINGUISHER_ASSET,)


def test_the_snapshot_row_names_its_index_and_load_does_not_move_the_pointer(
    bundle: Bundle, database_url: str
) -> None:
    assert ingest("load", bundle(C), database_url) == 0

    assert query(database_url, "SELECT search_index FROM control.snapshots") == [(INDEX_C,)]
    assert indexes(database_url) == (None, None)


def test_the_pointer_names_the_index_as_well_as_the_schema_pair(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    for date in (A, B):
        assert ingest("load", bundle(date), database_url) == 0

    assert ingest("promote", bundle(A), database_url) == 0
    assert indexes(database_url) == ("spatial-mini-campus-20260901", None)

    assert ingest("promote", bundle(B), database_url) == 0
    assert pointer(database_url) == (B, A)
    assert indexes(database_url) == (
        "spatial-mini-campus-20260908",
        "spatial-mini-campus-20260901",
    )
    # One row says it all: the snapshot, its schema pair and its index.
    assert query(
        database_url,
        """
        SELECT now.snapshot::text, held.osm_schema, held.seed_schema, held.views_schema,
            now.search_index
        FROM control.current_snapshot now JOIN control.snapshots held USING (snapshot)
        """,
    ) == [
        (
            B,
            "snap_20260908_osm",
            "snap_20260908_seed",
            "snap_20260908",
            "spatial-mini-campus-20260908",
        )
    ]

    # Rollback is the same command, and the indexes swap with the snapshots.
    capsys.readouterr()
    assert ingest("promote", bundle(A), database_url) == 0
    assert indexes(database_url) == (
        "spatial-mini-campus-20260901",
        "spatial-mini-campus-20260908",
    )
    assert "current index spatial-mini-campus-20260901" in capsys.readouterr().out
    # Promoting the current snapshot again changes nothing.
    assert ingest("promote", bundle(A), database_url) == 0
    assert indexes(database_url) == (
        "spatial-mini-campus-20260901",
        "spatial-mini-campus-20260908",
    )


def test_promote_returns_the_indexes_it_now_points_at(bundle: Bundle, database_url: str) -> None:
    assert ingest("load", bundle(C), database_url) == 0

    moved = promote(snapshot=C, database_url=database_url)

    assert (moved.current, moved.index, moved.previous_index) == (C, INDEX_C, None)


def test_load_fails_when_a_chunks_clearance_class_is_not_its_strictest_anchors(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    built = bundle(C)
    # Room 1001 is access=private, so its page is staff. Filed as visitor, it leaks.
    edit_chunk(built, "room:107#0", clearance_class="visitor")

    assert ingest("load", built, database_url) == 1

    assert snapshots(database_url) == {C: "failed"}
    assert "a chunk's clearance class is its strictest anchor's: fails for room:107#0" in failures(
        database_url, C
    )
    assert "cannot be promoted" in capsys.readouterr().out
    assert ingest("promote", built, database_url) == 1
    assert pointer(database_url) == (None, None)


def test_load_fails_when_a_shared_manual_is_filed_stricter_than_its_least_restrictive_type(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    edit_chunk(built, "manual:air#0", clearance_class="maintenance")

    assert ingest("load", built, database_url) == 1
    assert (
        "a chunk's clearance class is its strictest anchor's: fails for manual:air#0"
        in failures(database_url, C)
    )


def test_load_fails_when_the_index_does_not_hold_the_chunks_the_manifest_counts(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    rewrite(built, "documents/chunks.jsonl", chunks(built)[:-1])

    def recount(manifest: dict[str, Any]) -> None:
        manifest["parts"]["documents/chunks.jsonl"]["records"] = 8

    edit_manifest(built, recount)

    assert ingest("load", built, database_url) == 1
    found = failures(database_url, C)
    assert "the index holds 7 chunks, the manifest says 8" in found
    # room:107 is the last chunk and the only staff one.
    assert "the index holds 0 staff chunks, the manifest says 1" in found


def test_load_fails_when_the_counts_by_clearance_class_disagree(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)

    def recount(manifest: dict[str, Any]) -> None:
        manifest["documents"]["chunks_by_clearance_class"] = {
            "visitor": 5,
            "staff": 2,
            "maintenance": 1,
        }

    edit_manifest(built, recount)

    assert ingest("load", built, database_url) == 1
    assert failures(database_url, C) == [
        "the index holds 6 visitor chunks, the manifest says 5",
        "the index holds 1 staff chunks, the manifest says 2",
    ]


def test_load_fails_when_a_chunk_is_anchored_to_nothing_in_the_snapshot(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    edit_chunk(built, "hall:gaede#0", space_ids=["osm:way/424242"])
    edit_chunk(built, "manual:b1#0", asset_type_ids=[])

    assert ingest("load", built, database_url) == 1
    assert (
        "a chunk's anchors are spaces, types and assets of the snapshot: "
        "fails for hall:gaede#0, manual:b1#0"
    ) in failures(database_url, C)


def test_load_fails_when_a_chunks_ancestor_paths_disagree_with_the_tree(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    edit_chunk(built, "manual:ext#0", ancestor_ids=["site:mini-campus"])

    assert ingest("load", built, database_url) == 1
    assert failures(database_url, C) == [
        "a chunk's ancestor paths agree with the tree: fails for manual:ext#0"
    ]


def test_a_chunk_without_a_vector_cannot_be_loaded(bundle: Bundle, database_url: str) -> None:
    built = bundle(C)
    vectors = read_jsonl(built / "documents" / "vectors.jsonl")
    rewrite(built, "documents/vectors.jsonl", [v for v in vectors if v["chunk_id"] != "room:107#0"])

    assert ingest("load", built, database_url) == 1
    (failure,) = failures(database_url, C)
    assert "chunk room:107#0 has no vector" in failure


def test_a_document_part_that_does_not_match_its_hash_cannot_be_loaded(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    with (built / "documents" / "translations.jsonl").open("a") as part:
        part.write("\n")

    assert ingest("load", built, database_url) == 1
    (failure,) = failures(database_url, C)
    assert "documents/translations.jsonl does not match the sha256" in failure


def test_loading_the_same_bundle_twice_yields_the_same_index(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    first = (search_dir / f"{INDEX_C}.json").read_bytes()

    assert ingest("load", built, database_url) == 0

    assert (search_dir / f"{INDEX_C}.json").read_bytes() == first
    assert [path.name for path in search_dir.iterdir()] == [f"{INDEX_C}.json"]


def test_a_reload_replaces_what_the_index_held(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    kept = [c for c in chunks(built) if c["document_id"] != "manual:b1"]
    rewrite(built, "documents/chunks.jsonl", kept)

    def recount(manifest: dict[str, Any]) -> None:
        manifest["documents"]["chunks_by_clearance_class"]["visitor"] = 4

    edit_manifest(built, recount)

    assert ingest("load", built, database_url) == 0
    assert len(LocalIndex(search_dir).chunks(INDEX_C)) == 6


def test_load_makes_no_model_call(
    bundle: Bundle, database_url: str, openai: Any, site: Any
) -> None:
    built = bundle(C)
    asked = (len(openai.calls), len(site.requests))
    openai.status = 500

    assert ingest("load", built, database_url) == 0
    assert (len(openai.calls), len(site.requests)) == asked


def test_an_index_goes_when_its_snapshot_is_dropped(
    bundle: Bundle, database_url: str, search_dir: Path
) -> None:
    for date in (A, B, C):
        assert ingest("load", bundle(date), database_url) == 0
        assert ingest("promote", bundle(date), database_url) == 0
    assert len(list(search_dir.iterdir())) == 3

    # C is current and B previous, so loading D drops A, index and all.
    assert ingest("load", bundle(D), database_url) == 0

    assert sorted(path.stem for path in search_dir.iterdir()) == [
        "spatial-mini-campus-20260908",
        "spatial-mini-campus-20260915",
        "spatial-mini-campus-20260922",
    ]
    assert sorted(snapshots(database_url)) == [B, C, D]


def test_load_reports_the_link_check_and_an_offline_source_does_not_block(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    (log,) = (built / "runs").glob("*-load.log")
    assert "link check: 1 of 8 sources were offline at build" in log.read_text()
    assert "/halls/gone.html (hall:gone)" in log.read_text()


def test_load_refuses_to_run_with_no_search_index_configured(
    bundle: Bundle,
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    built = bundle(C)
    monkeypatch.delenv("SPATIAL_SEARCH_DIR")

    assert ingest("load", built, database_url) == 1

    assert "AZURE_SEARCH_ENDPOINT" in capsys.readouterr().err
    assert query(database_url, "SELECT count(*) FROM pg_namespace WHERE nspname = 'control'") == [
        (0,)
    ]


def test_the_reader_role_can_read_which_index_is_current(bundle: Bundle, database_url: str) -> None:
    assert ingest("load", bundle(C), database_url) == 0
    assert ingest("promote", bundle(C), database_url) == 0

    assert query(
        database_url,
        "SELECT has_column_privilege('spatial_reader', 'control.current_snapshot', "
        "'search_index', 'SELECT')",
    ) == [(True,)]


@pytest.fixture
def azure(monkeypatch: pytest.MonkeyPatch) -> Any:
    """A stand-in Azure AI Search service, with the environment pointing load at it."""
    with serve_azure_search() as state:
        monkeypatch.setenv(search.ENDPOINT_ENV, state.url)
        monkeypatch.setenv(search.API_KEY_ENV, "search-key")
        yield state


def test_load_writes_to_azure_ai_search_when_the_environment_names_one(
    bundle: Bundle, database_url: str, azure: AzureSearch, search_dir: Path
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    assert not search_dir.exists()
    assert list(azure.definitions) == [INDEX_C]
    held = sorted(azure.documents[INDEX_C].values(), key=lambda document: document["id"])
    assert [document["id"] for document in held] == [record["id"] for record in chunks(built)]
    room = next(document for document in held if document["id"] == "room:107#0")
    assert room["clearance_class"] == "staff"
    assert room["space_ids"] == ["osm:way/100", "osm:way/1001"]
    assert len(room["vector"]) == prose.DIMENSIONS
    # A key Azure takes: letters and digits only.
    assert all(document["key"].isalnum() for document in held)


def test_the_azure_index_is_hybrid_and_filters_on_anchors_paths_and_clearance(
    azure: AzureSearch,
) -> None:
    AzureSearchIndex(endpoint=azure.url, api_key="search-key").create("spatial-x-20260915", 4)

    fields = {field["name"]: field for field in azure.definitions["spatial-x-20260915"]["fields"]}
    assert fields["key"]["key"] is True
    # BM25 over the text and the title, a vector beside them.
    assert fields["text"]["searchable"] is True and fields["title"]["searchable"] is True
    assert fields["vector"]["type"] == "Collection(Edm.Single)"
    assert fields["vector"]["dimensions"] == 4
    profile = fields["vector"]["vectorSearchProfile"]
    search_config = azure.definitions["spatial-x-20260915"]["vectorSearch"]
    assert [p["name"] for p in search_config["profiles"]] == [profile]
    for name in ("space_ids", "asset_type_ids", "asset_ids", "ancestor_ids"):
        assert fields[name]["type"] == "Collection(Edm.String)"
        assert fields[name]["filterable"] is True
    assert fields["clearance_class"]["filterable"] is True
    assert fields["url"]["searchable"] is False


def test_the_azure_client_reads_back_what_it_wrote_and_drops_it(
    bundle: Bundle, database_url: str, azure: AzureSearch
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    client = AzureSearchIndex(endpoint=azure.url, api_key="search-key")

    assert client.exists(INDEX_C)
    assert client.chunks(INDEX_C) == [Chunk.model_validate(record) for record in chunks(built)]

    client.delete(INDEX_C)
    client.delete(INDEX_C)
    assert not client.exists(INDEX_C)
    assert azure.definitions == {}


def test_the_azure_client_waits_for_the_index_to_catch_up(
    bundle: Bundle, database_url: str, azure: AzureSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    azure.lagging = 3

    assert ingest("load", bundle(C), database_url) == 0
    counted = [path for _, path in azure.requests if path.endswith("$count")]
    assert len(counted) == 2


def test_an_index_that_never_catches_up_fails_the_upload(azure: AzureSearch) -> None:
    client = AzureSearchIndex(endpoint=azure.url, api_key="search-key", settle_seconds=0)
    client.create("spatial-x-20260915", prose.DIMENSIONS)
    azure.lagging = 1
    entry = search.IndexEntry(
        chunk=Chunk(
            id="d#0",
            document_id="d",
            ordinal=0,
            kind="manual",  # type: ignore[arg-type]
            title="A manual",
            text="Text.",
            language="en",
            url="https://example.invalid/d",
            original_text_sha256="0" * 64,
            translated=False,
            space_ids=(),
            asset_type_ids=("type",),
            asset_ids=(),
            ancestor_ids=(),
            clearance_class="visitor",  # type: ignore[arg-type]
            link_status="online",  # type: ignore[arg-type]
            link_checked_on="2026-09-15",
        ),
        vector=(0.1, 0.2, 0.3, 0.4),
    )

    with pytest.raises(SearchError, match="holds 0 chunks after upload, expected 1"):
        client.upload("spatial-x-20260915", [entry])


def test_a_search_service_that_refuses_fails_the_load_and_says_so(
    bundle: Bundle, database_url: str, azure: AzureSearch, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(search.API_KEY_ENV, "the-wrong-key")

    assert ingest("load", bundle(C), database_url) == 1

    assert snapshots(database_url) == {C: "failed"}
    (failure,) = failures(database_url, C)
    assert "answered 403" in failure
