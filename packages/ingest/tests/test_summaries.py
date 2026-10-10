"""SAC summaries: built by ingest build one clearance class at a time, indexed by load.

Same stand-ins as the document tests. The stand-in summariser answers with the
first words of what it was shown, so a summary can only say what its model read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import prose
from spatial_ingest import summaries
from spatial_ingest.cli import main
from stubs import OpenAI
from test_documents import DATE, build, chunks, files, manifest, read_jsonl


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "bundles"


@pytest.fixture
def bundle(out: Path, venue_file: Path, corpus_file: Path) -> Path:
    assert build(venue_file, corpus_file, out) == 0
    return out / DATE


def stored(bundle: Path) -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in read_jsonl(bundle / "documents" / "summaries.jsonl")}


def test_build_stores_one_summary_per_document_with_what_made_it(bundle: Path) -> None:
    made = stored(bundle)

    assert sorted(made) == sorted({f"{c['document_id']}#summary" for c in chunks(bundle)})
    valve = made["manual:valve#summary"]
    assert valve["text"] == f"Covers: Ball valve manual {prose.VALVE}"
    assert valve["document_id"] == "manual:valve"
    assert valve["source_chunk_ids"] == ["manual:valve#0"]
    assert valve["clearance_class"] == "maintenance"
    assert valve["deployment"] == prose.TRANSLATION_DEPLOYMENT
    assert valve["model"] == prose.TRANSLATION_MODEL
    assert valve["prompt_version"] == summaries.PROMPT_VERSION == "summarise-document@1"
    assert valve["input_tokens"] > 0 and valve["output_tokens"] > 0


def test_no_summary_is_built_across_clearance_classes(bundle: Path, openai: OpenAI) -> None:
    passages = {c["id"]: c for c in chunks(bundle)}
    assert {c["clearance_class"] for c in passages.values()} == {"visitor", "staff", "maintenance"}

    for made in stored(bundle).values():
        read = {passages[source]["clearance_class"] for source in made["source_chunk_ids"]}
        assert read == {made["clearance_class"]}, made["id"]
        assert {passages[s]["document_id"] for s in made["source_chunk_ids"]} == {
            made["document_id"]
        }

    # And what the model was actually shown, call by call: text of one class only.
    shown = openai.summarised()
    assert len(shown) == 7
    for text in shown:
        inside = {c["clearance_class"] for c in passages.values() if c["text"] in text}
        assert len(inside) == 1, text


def test_classes_are_summarised_one_at_a_time_lowest_first(bundle: Path, openai: OpenAI) -> None:
    passages = chunks(bundle)

    def cleared(text: str) -> str:
        return next(c["clearance_class"] for c in passages if c["text"] in text)

    order = [cleared(text) for text in openai.summarised()]
    assert order == ["visitor"] * 5 + ["staff"] + ["maintenance"]


def test_a_passage_is_embedded_with_its_documents_summary_in_front(bundle: Path) -> None:
    vectors = {v["chunk_id"]: v for v in read_jsonl(bundle / "documents" / "vectors.jsonl")}
    summary = stored(bundle)["manual:b1#summary"]["text"]

    for passage in chunks(bundle, "manual:b1"):
        augmented = f"{passage['title']}\n\n{summary}\n\n{passage['text']}"
        assert vectors[passage["id"]]["vector"] == prose.embedding(augmented)
        # Only the vector knows the summary. What gets quoted is still the passage.
        assert "Covers:" not in passage["text"]
    assert vectors["manual:b1#summary"]["vector"] == prose.embedding(
        f"B1 light fixture manual\n\n{summary}"
    )


def test_summaries_are_not_in_the_chunks_part(bundle: Path) -> None:
    assert all("#summary" not in c["id"] for c in chunks(bundle))
    assert {c["role"] for c in chunks(bundle)} == {"passage"}


def test_the_manifest_records_what_made_the_summaries_and_what_they_cost(bundle: Path) -> None:
    built = manifest(bundle)

    assert built["documents"]["summaries"] == {
        "documents": 7,
        "parts": 0,
        "by_clearance_class": {"visitor": 5, "staff": 1, "maintenance": 1},
        "made_by": [
            {
                "deployment": prose.TRANSLATION_DEPLOYMENT,
                "model": prose.TRANSLATION_MODEL,
                "prompt_version": summaries.PROMPT_VERSION,
                "summaries": 7,
            }
        ],
    }
    part = built["parts"]["documents/summaries.jsonl"]
    assert part["records"] == 7
    assert summaries.PROMPT_VERSION in part["produced_by"]
    spent = {call["purpose"]: call for call in built["documents"]["spend"]["calls"]}["summary"]
    made = stored(bundle).values()
    assert spent["calls"] == 7
    assert spent["input_tokens"] == sum(s["input_tokens"] for s in made)
    assert spent["output_tokens"] == sum(len(s["text"].split()) for s in made)


def test_building_again_reuses_the_summaries_and_asks_nobody(
    bundle: Path, out: Path, venue_file: Path, corpus_file: Path, openai: OpenAI
) -> None:
    before = files(bundle)
    asked = len(openai.calls)
    openai.status = 500

    assert build(venue_file, corpus_file, out) == 0

    assert files(bundle) == before
    assert len(openai.calls) == asked


def test_a_summary_made_under_another_prompt_version_is_made_again(
    bundle: Path,
    out: Path,
    venue_file: Path,
    corpus_file: Path,
    openai: OpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(summaries, "PROMPT_VERSION", "summarise-document@2")

    assert build(venue_file, corpus_file, out) == 0

    assert len(openai.summarised()) == 14
    assert {s["prompt_version"] for s in stored(bundle).values()} == {"summarise-document@2"}


def test_a_long_document_is_summarised_in_parts_and_then_as_a_whole(
    out: Path,
    venue_file: Path,
    corpus_file: Path,
    openai: OpenAI,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The B1 manual is two chunks, each longer than this.
    monkeypatch.setattr(summaries, "WINDOW", 400)

    assert build(venue_file, corpus_file, out) == 0

    made = stored(out / DATE)
    parts = [made["manual:b1#summary-part-0"], made["manual:b1#summary-part-1"]]
    assert [p["source_chunk_ids"] for p in parts] == [["manual:b1#0"], ["manual:b1#1"]]
    assert [p["prompt"] for p in parts] == ["part", "part"]
    whole = made["manual:b1#summary"]
    assert whole["prompt"] == "reduce"
    assert whole["source_chunk_ids"] == ["manual:b1#0", "manual:b1#1"]
    listed = f"B1 light fixture manual\n\n1. {parts[0]['text']}\n2. {parts[1]['text']}"
    assert listed in openai.summarised()
    assert manifest(out / DATE)["documents"]["summaries"]["parts"] == 2
    assert manifest(out / DATE)["documents"]["summaries"]["documents"] == 7


def test_a_venue_with_no_corpus_has_no_summaries(out: Path, venue_file: Path) -> None:
    assert main(["build", "--date", DATE, "--venue", str(venue_file), "--out", str(out)]) == 0

    assert (out / DATE / "documents" / "summaries.jsonl").read_bytes() == b""
    assert manifest(out / DATE)["documents"]["summaries"]["documents"] == 0
