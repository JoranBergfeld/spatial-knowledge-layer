"""ingest build over a document corpus, end to end.

The venue's pages and the manuals are a stand-in site on localhost, and so is the
Azure OpenAI resource that translates and embeds. Nothing here leaves the machine.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import prose
from spatial_ingest import models
from spatial_ingest.cli import main
from spatial_ingest.corpus import default_corpus, read_corpus
from spatial_ingest.rules.passages import CHUNK_LIMIT
from stubs import OpenAI, Site

DATE = "2026-09-15"
BUILDING_A, BUILDING_E = "osm:way/100", "osm:way/500"
SITE = "site:mini-campus"


def build(venue_file: Path, corpus_file: Path, out: Path, *extra: str) -> int:
    args = ["build", "--date", DATE, "--venue", str(venue_file), "--out", str(out)]
    return main([*args, "--documents", str(corpus_file), *extra])


@pytest.fixture
def out(tmp_path: Path) -> Path:
    return tmp_path / "bundles"


@pytest.fixture
def bundle(out: Path, venue_file: Path, corpus_file: Path) -> Path:
    assert build(venue_file, corpus_file, out) == 0
    return out / DATE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def documents(bundle: Path) -> dict[str, dict[str, Any]]:
    return {d["id"]: d for d in read_jsonl(bundle / "documents" / "documents.jsonl")}


def chunks(bundle: Path, document: str | None = None) -> list[dict[str, Any]]:
    every = read_jsonl(bundle / "documents" / "chunks.jsonl")
    return [c for c in every if document is None or c["document_id"] == document]


def manifest(bundle: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
    return loaded


def files(bundle: Path) -> dict[str, str]:
    return {
        str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
    }


def today() -> str:
    return datetime.datetime.now(datetime.UTC).date().isoformat()


def test_build_ingests_the_pages_and_the_manuals_of_placed_types(bundle: Path) -> None:
    assert sorted(documents(bundle)) == [
        "hall:bau",
        "hall:gaede",
        "manual:air",
        "manual:b1",
        "manual:ext",
        "manual:valve",
        "room:107",
    ]
    kinds = {d["id"]: d["kind"] for d in documents(bundle).values()}
    assert kinds["hall:gaede"] == "accessibility-page"
    assert kinds["manual:b1"] == "manual"
    assert all(d["record_provenance"] == "sourced" for d in documents(bundle).values())


def test_a_manual_for_a_type_nobody_placed_is_not_ingested_or_even_fetched(
    bundle: Path, site: Site
) -> None:
    summary = manifest(bundle)["documents"]

    assert "manual:chiller" not in documents(bundle)
    assert "/manuals/chiller.txt" not in site.requests
    assert {"id": "manual:chiller", "reason": "none of its anchors is in the snapshot"} in summary[
        "skipped"
    ]
    # The valve manual stays, anchored to the one of its two types that is placed.
    assert [a["id"] for a in documents(bundle)["manual:valve"]["anchors"]] == [prose.VALVE_TYPE]
    assert {
        "document": "manual:valve",
        "reason": f"no asset of type {prose.CHILLER} is placed",
    } in summary["dropped_anchors"]


def test_a_page_for_a_building_the_venue_lacks_is_skipped_and_said_so(bundle: Path) -> None:
    summary = manifest(bundle)["documents"]

    assert "hall:nowhere" not in documents(bundle)
    assert {"id": "hall:nowhere", "reason": "none of its anchors is in the snapshot"} in summary[
        "skipped"
    ]
    assert {"document": "hall:nowhere", "reason": "no building numbered 77.77"} in summary[
        "dropped_anchors"
    ]


def test_originals_stay_in_the_bundle_and_only_the_link_goes_out(bundle: Path, site: Site) -> None:
    kept = {p.name for p in (bundle / "documents" / "originals").iterdir()}
    fetched = json.loads((bundle / "documents" / "fetched.json").read_text())

    assert len(kept) == 7
    assert (bundle / "documents" / "originals" / fetched["manual:air"]["file"]).read_bytes() == (
        prose.PAGES["/manuals/air.pdf"][0]
    )
    assert fetched["manual:air"]["file"].endswith(".pdf")
    # A chunk carries where the original lives and never the file.
    for chunk in chunks(bundle, "manual:air"):
        assert chunk["url"] == site.url + "/manuals/air.pdf"
        assert not any("originals" in str(value) for value in chunk.values())
    # The manifest pins each original, and no part of the bundle is one.
    air = prose.PAGES["/manuals/air.pdf"][0]
    assert manifest(bundle)["documents"]["originals"]["manual:air"] == (
        hashlib.sha256(air).hexdigest()
    )
    assert documents(bundle)["manual:air"]["original_sha256"] == hashlib.sha256(air).hexdigest()
    assert not any("originals" in part for part in manifest(bundle)["parts"])


def test_a_document_has_one_or_more_anchors_each_a_space_a_type_or_an_asset(bundle: Path) -> None:
    anchors = {d["id"]: d["anchors"] for d in documents(bundle).values()}

    assert anchors["hall:gaede"] == [
        {
            "kind": "space",
            "id": BUILDING_A,
            "provenance": "derived",
            "rule": "anchor-building-by-ref@1",
        }
    ]
    assert anchors["room:107"] == [
        {
            "kind": "space",
            "id": BUILDING_A,
            "provenance": "derived",
            "rule": "anchor-building-by-ref@1",
        },
        {"kind": "space", "id": "osm:way/1001", "provenance": "sourced", "rule": None},
    ]
    assert [(a["kind"], a["id"]) for a in anchors["manual:air"]] == [
        ("asset_type", prose.DIFFUSER),
        ("asset_type", prose.VAV),
    ]
    assert [(a["kind"], a["id"]) for a in anchors["manual:ext"]] == [
        ("asset", prose.EXTINGUISHER_ASSET)
    ]


def test_a_document_takes_its_strictest_anchors_clearance(bundle: Path) -> None:
    classes = {d["id"]: d["clearance_class"] for d in documents(bundle).values()}

    # Buildings carry no clearance of their own, so a page on one is for anyone.
    assert classes["hall:gaede"] == "visitor"
    # Room 1001 is access=private. Its building limits nothing, the room does.
    assert classes["room:107"] == "staff"
    assert classes["manual:b1"] == "visitor"
    assert classes["manual:valve"] == "maintenance"
    # An asset anchor takes its type's: an extinguisher is for anyone.
    assert classes["manual:ext"] == "visitor"


def test_a_manual_shared_by_several_types_takes_the_least_restrictive(bundle: Path) -> None:
    # The VAV box is maintenance and the diffuser is visitor.
    assert documents(bundle)["manual:air"]["clearance_class"] == "visitor"


def test_every_chunk_carries_its_documents_clearance_class(bundle: Path) -> None:
    classes = {d["id"]: d["clearance_class"] for d in documents(bundle).values()}

    assert chunks(bundle)
    for chunk in chunks(bundle):
        assert chunk["clearance_class"] == classes[chunk["document_id"]]
    assert manifest(bundle)["documents"]["chunks_by_clearance_class"] == {
        "visitor": 6,
        "staff": 1,
        "maintenance": 1,
    }


def test_german_text_is_translated_by_the_composition_deployment(
    bundle: Path, openai: OpenAI
) -> None:
    translated = [body["messages"][1]["content"] for body in openai.chats(summaries=False)]

    assert sorted(translated) == [prose.BAU_GERMAN, prose.GAEDE_GERMAN]
    assert {deployment for deployment, op, _ in openai.calls if op == "chat/completions"} == {
        prose.TRANSLATION_DEPLOYMENT
    }
    (bau,) = chunks(bundle, "hall:bau")
    assert bau["text"] == prose.BAU_ENGLISH
    assert bau["translated"] is True


def test_english_text_is_not_translated(bundle: Path) -> None:
    (room,) = chunks(bundle, "room:107")

    assert room["text"] == prose.ROOM_107
    assert room["translated"] is False
    assert room["language"] == "en"
    assert documents(bundle)["room:107"]["translation"] is None


def test_each_chunk_keeps_the_originals_language_url_and_text_hash(
    bundle: Path, site: Site
) -> None:
    (gaede,) = chunks(bundle, "hall:gaede")

    assert gaede["language"] == "de"
    assert gaede["url"] == site.url + "/halls/gaede.html"
    assert gaede["original_text_sha256"] == hashlib.sha256(prose.GAEDE_GERMAN.encode()).hexdigest()
    # The German itself is in the private bundle and not on the chunk.
    assert "Stockwerk" not in json.dumps(gaede, ensure_ascii=False)
    (kept,) = [
        t
        for t in read_jsonl(bundle / "documents" / "translations.jsonl")
        if t["chunk_id"] == gaede["id"]
    ]
    assert kept["original"] == prose.GAEDE_GERMAN
    assert kept["english"] == prose.GAEDE_TRANSLATED


def test_small_counts_times_and_floors_are_in_words_and_measures_stay_digits(
    bundle: Path,
) -> None:
    (gaede,) = chunks(bundle, "hall:gaede")

    assert gaede["text"] == prose.GAEDE_ENGLISH
    assert "second upper floor" in gaede["text"]
    assert "two steps" in gaede["text"]
    assert "seven thirty" in gaede["text"]
    assert "90 cm" in gaede["text"]
    assert "37.1 MHz" in gaede["text"]
    # English originals get the rule too: "every 12 months" is a measure and stays.
    (air,) = chunks(bundle, "manual:air")
    assert "every 12 months" in air["text"]


def test_a_document_date_is_recorded_only_when_the_rule_reads_it(bundle: Path) -> None:
    dated = documents(bundle)

    assert dated["hall:gaede"]["date"] == {
        "value": "2012-05-31",
        "provenance": "derived",
        "rule": "date-from-last-change@1",
        "basis": None,
    }
    assert dated["hall:bau"]["date"] is None
    assert all(d["date"] is None for d in dated.values() if d["kind"] == "manual")
    assert chunks(bundle, "hall:gaede")[0]["document_date"] == "2012-05-31"
    assert chunks(bundle, "hall:bau")[0]["document_date"] is None
    assert manifest(bundle)["documents"]["dated"] == 1
    assert manifest(bundle)["documents"]["undated"] == 6


def test_each_source_link_is_recorded_online_or_offline_as_of_the_check(
    bundle: Path, site: Site
) -> None:
    summary = manifest(bundle)["documents"]
    links = {link["document"]: link for link in summary["links"]}

    assert links["hall:gaede"] == {
        "document": "hall:gaede",
        "url": site.url + "/halls/gaede.html",
        "status": "online",
        "checked_on": today(),
    }
    assert links["hall:gone"]["status"] == "offline"
    assert links["hall:gone"]["checked_on"] == today()
    assert summary["offline"] == 1
    assert documents(bundle)["hall:gaede"]["link"] == {"status": "online", "checked_on": today()}
    (gaede,) = chunks(bundle, "hall:gaede")
    assert (gaede["link_status"], gaede["link_checked_on"]) == ("online", today())


def test_a_source_that_never_answered_has_nothing_to_ingest(bundle: Path) -> None:
    assert "hall:gone" not in documents(bundle)
    assert {
        "id": "hall:gone",
        "reason": "its source is offline and the bundle has no original",
    } in manifest(bundle)["documents"]["skipped"]


def test_a_source_that_went_offline_is_still_quoted_from_the_bundle(
    bundle: Path, out: Path, venue_file: Path, corpus_file: Path, site: Site
) -> None:
    site.offline.add("/halls/gaede.html")

    assert build(venue_file, corpus_file, out, "--refetch-documents") == 0

    assert documents(bundle)["hall:gaede"]["link"]["status"] == "offline"
    (gaede,) = chunks(bundle, "hall:gaede")
    assert gaede["text"] == prose.GAEDE_ENGLISH
    assert gaede["link_status"] == "offline"
    assert documents(bundle)["hall:bau"]["link"]["status"] == "online"


def test_chunks_carry_anchor_ids_by_kind_and_flattened_ancestor_paths(bundle: Path) -> None:
    (room,) = chunks(bundle, "room:107")
    (air,) = chunks(bundle, "manual:air")
    (extinguisher,) = chunks(bundle, "manual:ext")

    assert room["space_ids"] == [BUILDING_A, "osm:way/1001"]
    assert room["asset_type_ids"] == [] and room["asset_ids"] == []
    assert room["ancestor_ids"] == ["floor:way/100:1", BUILDING_A, "osm:way/1001", SITE]

    assert air["asset_type_ids"] == [prose.DIFFUSER, prose.VAV]
    # A type is in no one place, so it has no path.
    assert air["space_ids"] == [] and air["ancestor_ids"] == []

    assert extinguisher["asset_ids"] == [prose.EXTINGUISHER_ASSET]
    # The extinguisher hangs in corridor 1006: its path, corridor included.
    assert extinguisher["ancestor_ids"] == ["floor:way/100:0", BUILDING_A, "osm:way/1006", SITE]


def test_a_long_manual_is_cut_into_chunks_in_order(bundle: Path) -> None:
    cut = chunks(bundle, "manual:b1")

    assert [c["id"] for c in cut] == ["manual:b1#0", "manual:b1#1"]
    assert [c["ordinal"] for c in cut] == [0, 1]
    assert all(len(c["text"]) <= CHUNK_LIMIT for c in cut)
    assert cut[0]["text"].startswith("B1 light fixture.")
    assert cut[1]["text"].startswith("Cleaning.")
    assert documents(bundle)["manual:b1"]["chunks"] == 2


def test_every_chunk_gets_a_vector_from_the_embedding_deployment(
    bundle: Path, openai: OpenAI
) -> None:
    vectors = {v["chunk_id"]: v for v in read_jsonl(bundle / "documents" / "vectors.jsonl")}

    # A vector per passage and one per document summary. test_summaries has what is in them.
    documents_ = {c["document_id"] for c in chunks(bundle)}
    assert sorted(vectors) == sorted(
        [*(c["id"] for c in chunks(bundle)), *(f"{d}#summary" for d in documents_)]
    )
    assert all(len(v["vector"]) == prose.DIMENSIONS for v in vectors.values())
    assert {deployment for deployment, op, _ in openai.calls if op == "embeddings"} == {
        prose.EMBEDDING_DEPLOYMENT
    }
    assert manifest(bundle)["documents"]["embedding"] == {
        "deployment": prose.EMBEDDING_DEPLOYMENT,
        "model": prose.EMBEDDING_MODEL,
        "dimensions": prose.DIMENSIONS,
    }


def test_the_manifest_records_what_made_each_translation(bundle: Path) -> None:
    made = {
        "deployment": prose.TRANSLATION_DEPLOYMENT,
        "model": prose.TRANSLATION_MODEL,
        "prompt_version": models.PROMPT_VERSION,
    }

    assert manifest(bundle)["documents"]["translations"] == [
        {"document": "hall:bau", "chunks": 1, **made},
        {"document": "hall:gaede", "chunks": 1, **made},
    ]
    assert documents(bundle)["hall:gaede"]["translation"] == made
    assert (
        models.PROMPT_VERSION in manifest(bundle)["parts"]["documents/chunks.jsonl"]["produced_by"]
    )
    assert (
        "numbers-in-words@2" in manifest(bundle)["parts"]["documents/chunks.jsonl"]["produced_by"]
    )


def test_the_manifest_records_ingestion_spend(bundle: Path, openai: OpenAI) -> None:
    spend = manifest(bundle)["documents"]["spend"]
    by_purpose = {call["purpose"]: call for call in spend["calls"]}
    translations = read_jsonl(bundle / "documents" / "translations.jsonl")
    embedded = sum(len(body["input"][0].split()) for body in openai.made("embeddings"))

    assert by_purpose["translation"]["deployment"] == prose.TRANSLATION_DEPLOYMENT
    assert by_purpose["translation"]["calls"] == 2
    assert by_purpose["translation"]["input_tokens"] == sum(t["input_tokens"] for t in translations)
    assert by_purpose["translation"]["output_tokens"] == (
        len(prose.GAEDE_TRANSLATED.split()) + len(prose.BAU_ENGLISH.split())
    )
    # 0.75 in and 4.50 out a million, as the rate table has it.
    translation = by_purpose["translation"]
    assert translation["cost_usd"] == round(
        (translation["input_tokens"] * 0.75 + translation["output_tokens"] * 4.5) / 1_000_000, 6
    )
    assert spend["unpriced"] == []

    # Eight passages and seven summaries.
    assert by_purpose["embedding"]["calls"] == 15
    assert by_purpose["embedding"]["input_tokens"] == embedded
    # 0.02 dollars a million tokens, as the rate table has it.
    assert by_purpose["embedding"]["rate"]["usd_per_million_input"] == 0.02
    assert by_purpose["embedding"]["cost_usd"] == round(embedded * 0.02 / 1_000_000, 6)
    # The total is rounded once, the calls each, so they can differ in the last place.
    assert spend["total_usd"] == pytest.approx(
        sum(call["cost_usd"] for call in spend["calls"]), abs=1e-5
    )


def test_the_manifest_pins_the_corpus_and_counts_the_parts(bundle: Path, corpus_file: Path) -> None:
    built = manifest(bundle)

    assert built["inputs"]["documents"] == {
        "file": corpus_file.name,
        "sha256": hashlib.sha256(corpus_file.read_bytes()).hexdigest(),
        "sources": 10,
    }
    assert built["parts"]["documents/chunks.jsonl"]["records"] == 8
    assert built["parts"]["documents/documents.jsonl"]["records"] == 7
    assert built["documents"]["documents"] == 7
    assert built["documents"]["chunks"] == 8
    assert built["documents"]["by_kind"] == {"accessibility-page": 3, "manual": 4}
    assert built["documents"]["index"] == "spatial-mini-campus-20260915"
    for part in ("documents", "chunks", "translations", "vectors"):
        path = bundle / "documents" / f"{part}.jsonl"
        assert built["parts"][f"documents/{part}.jsonl"]["sha256"] == (
            hashlib.sha256(path.read_bytes()).hexdigest()
        )


def test_building_again_asks_nobody_and_writes_the_same_bytes(
    bundle: Path, out: Path, venue_file: Path, corpus_file: Path, site: Site, openai: OpenAI
) -> None:
    before = files(bundle)
    asked = (len(site.requests), len(openai.calls))
    # Whatever the sites say now, the bundle already has its answers.
    site.offline.update(prose.PAGES)
    openai.status = 500

    assert build(venue_file, corpus_file, out) == 0

    assert files(bundle) == before
    assert (len(site.requests), len(openai.calls)) == asked


def test_a_translation_is_never_paid_for_twice(
    bundle: Path, out: Path, venue_file: Path, corpus_file: Path, openai: OpenAI
) -> None:
    spend = manifest(bundle)["documents"]["spend"]

    assert build(venue_file, corpus_file, out, "--refetch-documents") == 0

    assert len(openai.chats(summaries=False)) == 2
    assert manifest(bundle)["documents"]["spend"] == spend


def test_a_venue_with_no_corpus_builds_with_no_documents_and_no_model(
    out: Path, venue_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (models.ENDPOINT_ENV, models.API_KEY_ENV):
        monkeypatch.delenv(name, raising=False)

    assert main(["build", "--date", DATE, "--venue", str(venue_file), "--out", str(out)]) == 0

    built = manifest(out / DATE)
    assert built["inputs"]["documents"] is None
    assert built["documents"]["documents"] == 0
    assert built["documents"]["chunks"] == 0
    assert built["documents"]["embedding"] is None
    assert built["parts"]["documents/chunks.jsonl"]["records"] == 0
    assert (out / DATE / "documents" / "chunks.jsonl").read_bytes() == b""


def test_build_stops_when_the_corpus_needs_a_model_and_none_is_configured(
    out: Path,
    venue_file: Path,
    corpus_file: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(models.ENDPOINT_ENV)

    assert build(venue_file, corpus_file, out) == 1

    assert models.ENDPOINT_ENV in capsys.readouterr().err
    assert not (out / DATE / "manifest.json").exists()


def test_build_stops_when_the_model_cannot_be_reached(
    out: Path, venue_file: Path, corpus_file: Path, openai: OpenAI
) -> None:
    openai.status = 503

    assert build(venue_file, corpus_file, out) == 1
    assert not (out / DATE / "manifest.json").exists()


def test_a_corpus_that_does_not_parse_stops_the_build(
    out: Path, venue_file: Path, corpus_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = json.loads(corpus_file.read_text())
    broken["documents"][0]["anchors"] = [{"building": "50.34", "space": "osm:way/1001"}]
    corpus_file.write_text(json.dumps(broken))

    assert build(venue_file, corpus_file, out) == 1
    assert "exactly one of" in capsys.readouterr().err


# A drawing as tesseract reads it: what readable-passage@1 is there to keep out.
DRAWING = "l' ~-~~~~~~~I | r-----' I I _ ~ | C) ) ET-2/ . | _ Iw 100¢ HWP—2 | = ~~ \\ | "


def test_ocr_junk_never_becomes_a_chunk(
    out: Path, venue_file: Path, corpus_file: Path, site: Site
) -> None:
    # One manual with a page of prose and a page of drawing, one with drawings only.
    prose_page = "Lamp replacement. " + "Switch off the breaker before opening the housing. " * 13
    site.pages["/manuals/scanned.txt"] = (
        f"{prose_page}\n\n{DRAWING * 10}".encode(),
        "text/plain",
    )
    site.pages["/manuals/plan.txt"] = (f"{DRAWING * 10}\n\n{DRAWING * 10}".encode(), "text/plain")
    corpus = json.loads(corpus_file.read_text())
    for source_id, path, title in (
        ("manual:scanned", "/manuals/scanned.txt", "B1 scanned manual"),
        ("manual:plan", "/manuals/plan.txt", "B1 mounting plan"),
    ):
        corpus["documents"].append(
            {
                "id": source_id,
                "kind": "manual",
                "language": "en",
                "title": title,
                "url": site.url + path,
                "anchors": [{"asset_type": prose.B1}],
            }
        )
    corpus_file.write_text(json.dumps(corpus))

    assert build(venue_file, corpus_file, out) == 0
    bundle = out / DATE

    # The drawing page is dropped, and the prose keeps its place and its id.
    (kept,) = chunks(bundle, "manual:scanned")
    assert kept["id"] == "manual:scanned#0"
    assert kept["text"].startswith("Lamp replacement.")
    assert documents(bundle)["manual:scanned"]["chunks"] == 1
    # Nothing readable at all: found by its title, and nothing quoted from it.
    (title,) = chunks(bundle, "manual:plan")
    assert title["text"] == "B1 mounting plan"
    assert not any("~~~" in chunk["text"] for chunk in chunks(bundle))

    gated = manifest(bundle)["documents"]["unreadable_passages"]
    assert gated == {
        "dropped": 3,
        "by_document": {"manual:plan": 2, "manual:scanned": 1},
        "indexed_by_title": ["manual:plan"],
    }
    assert (
        "readable-passage@1" in (manifest(bundle)["parts"]["documents/chunks.jsonl"]["produced_by"])
    )


def test_the_published_kit_corpus_is_its_lecture_hall_pages_keyed_by_building() -> None:
    path = default_corpus("kit-campus-sued")
    assert path is not None
    corpus = read_corpus(path)

    assert corpus.venue == "kit-campus-sued"
    assert len(corpus.sources) == 36
    assert all(source.kind == "accessibility-page" for source in corpus.sources)
    assert all(source.language == "de" for source in corpus.sources)
    for source in corpus.sources:
        (anchor,) = source.anchors
        assert anchor.building is not None and source.title.startswith(anchor.building)
    audimax = next(s for s in corpus.sources if s.id == "kit:hoersaal/406")
    assert audimax.url == "https://www.studiumundbehinderung.kit.edu/76_406.php"
    assert audimax.anchors[0].building == "30.95"


def test_a_venue_without_a_published_corpus_has_none() -> None:
    assert default_corpus("mini-campus") is None
