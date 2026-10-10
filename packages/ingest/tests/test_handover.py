"""The test bench through the ingestion CLI: build and load over the hand-made handover.

The real handover is never fetched here. Everything below runs on
handover_fixture/, and the expected values are the ones in mini_handover.py.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import psycopg
import pytest

import mini_handover
from spatial_core.search import LocalIndex
from spatial_ingest.cli import main
from spatial_ingest.handover import ifc
from stubs import OpenAI

DATE = "2026-09-15"
INDEX = "spatial-mini-bench-20260915"


def build(out: Path, *extra: str, handover: Path = mini_handover.DIR) -> int:
    args = ["build", "--date", DATE, "--venue", str(handover / "definition.json")]
    return main([*args, "--out", str(out), "--handover", str(handover), *extra])


@pytest.fixture
def bundle(tmp_path: Path, openai: OpenAI) -> Path:
    assert build(tmp_path / "bundles") == 0
    return tmp_path / "bundles" / DATE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def records(bundle: Path, part: str) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in read_jsonl(bundle / part)}


def manifest(bundle: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
    return loaded


def seen(bundle: Path, part: str, key: str) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for row in read_jsonl(bundle / part):
        found.setdefault(row["persona"], set()).add(row[key])
    return found


# --- the tree ------------------------------------------------------------------------------


def test_the_facility_is_the_building_and_floors_and_spaces_hang_from_it(bundle: Path) -> None:
    spaces = records(bundle, "osm/spaces.jsonl")

    assert Counter(space["tier"]["value"] for space in spaces.values()) == {
        "building": 1,
        "floor": 3,
        "room": 13,
    }
    building = spaces["building:b-12"]
    assert building["parent_id"] is None
    assert building["label"]["value"] == "Harbour Clinic"
    assert building["ref"]["value"] == "B 12"
    assert spaces["floor:first-floor"]["parent_id"] == "building:b-12"
    assert spaces["space:2f02"]["parent_id"] == "floor:first-floor"
    assert spaces["space:2f02"]["ancestor_path"] == [
        {"tier": "building", "id": "building:b-12"},
        {"tier": "floor", "id": "floor:first-floor"},
    ]
    assert spaces["space:2f02"]["source"] == "cobie:Space/2F02"
    assert all(space["record_provenance"] == "sourced" for space in spaces.values())


def test_a_space_is_labelled_by_its_description_or_else_building_and_number(bundle: Path) -> None:
    spaces = records(bundle, "osm/spaces.jsonl")

    assert spaces["space:1g01"]["label"] == {
        "value": "RECEPTION",
        "provenance": "sourced",
        "rule": None,
        "basis": None,
    }
    # "n/a" is COBie for nothing there, so 2F05 has no name and takes the rule's label.
    assert spaces["space:2f05"]["name"] is None
    assert spaces["space:2f05"]["label"]["value"] == "B 12 room 2F05"
    assert spaces["space:2f05"]["label"]["rule"] == "room-label@1"


# --- clearance -----------------------------------------------------------------------------


def test_spaces_take_clearance_from_their_table_13_category_as_an_authored_default(
    bundle: Path,
) -> None:
    spaces = records(bundle, "osm/spaces.jsonl")
    rooms = {i: s for i, s in spaces.items() if s["tier"]["value"] == "room"}

    assert {i.removeprefix("space:"): s["clearance"]["persona"] for i, s in rooms.items()} == {
        "1g01": "visitor",
        "1g02": "visitor",
        "1g03": "visitor",
        "1g04": "staff",
        "1g05": "maintenance",
        "1g06": "visitor",
        "2f01": "visitor",
        "2f02": "staff",
        "2f03": "staff",
        "2f04": "maintenance",
        "2f05": "visitor",
        "2f06": "staff",
        "3r01": "maintenance",
    }
    for room in rooms.values():
        assert room["clearance"]["rule"] == "clearance-from-space-category@1"
        assert room["clearance"]["authored_default"] is True
        assert room["clearance"]["access"] is None
    said = manifest(bundle)["clearance"]
    assert said["sourced_share"] == 0.0
    assert said["authored_defaults"] == 13
    assert said["rooms_by_persona"] == {"visitor": 6, "staff": 4, "maintenance": 3}


def test_asset_types_use_the_shared_table_23_rule(bundle: Path) -> None:
    types = records(bundle, "osm/asset_types.jsonl")

    assert {i.removeprefix("type:"): t["clearance"]["persona"] for i, t in types.items()} == {
        "light-fixture-a": "visitor",
        "door-type-a": "visitor",
        "lavatory-l1": "visitor",
        "sprinkler-head": "visitor",
        "exam-table": "staff",
        "work-station": "staff",
        "panelboard-p1": "maintenance",
        "ball-valve-50": "maintenance",
        "ahu": "maintenance",
        "smoke-detector": "maintenance",
    }
    assert {t["clearance"]["rule"] for t in types.values()} == {"clearance-from-omniclass@1"}
    assert types["type:smoke-detector"]["clearance"]["unrecognised_category"] is True
    assert manifest(bundle)["inventory"]["unrecognised_categories"] == 1


def test_each_persona_sees_what_the_two_rules_leave_it(bundle: Path) -> None:
    spaces = seen(bundle, "osm/visible_spaces.jsonl", "space_id")
    assets = seen(bundle, "osm/visible_assets.jsonl", "asset_id")
    types = seen(bundle, "osm/visible_asset_types.jsonl", "type_id")

    # Six rooms, their two floors and the building.
    assert len(spaces["visitor"]) == 9
    assert "floor:roof" not in spaces["staff"]
    assert len(spaces["maintenance"]) == 17
    assert {p: len(found) for p, found in assets.items()} == {
        "visitor": 14,
        "staff": 23,
        "maintenance": 35,
    }
    # A panelboard in a corridor anyone walks through is still not a visitor's to see.
    assert "asset:panel-p1-03" not in assets["staff"]
    assert "asset:panel-p1-03" in assets["maintenance"]
    assert types["visitor"] == {
        "type:light-fixture-a",
        "type:door-type-a",
        "type:lavatory-l1",
        "type:sprinkler-head",
    }


# --- assets --------------------------------------------------------------------------------


def test_a_component_is_an_asset_of_its_type_in_its_space(bundle: Path) -> None:
    assets = records(bundle, "osm/assets.jsonl")
    light = assets["asset:light-a-01"]

    assert light["type_id"] == "type:light-fixture-a"
    assert light["space_id"] == "space:1g01"
    assert [step["id"] for step in light["ancestor_path"]] == [
        "building:b-12",
        "floor:ground-floor",
        "space:1g01",
    ]
    assert light["record_provenance"] == "sourced"
    assert light["source"] == "cobie:Component/Light A-01"
    assert light["attributes"]["serial_number"]["value"] == "LA-0001"
    assert light["attributes"]["installation_date"]["value"] == "2018-03-01T09:00:00"
    # A blank stays a gap. It is not an attribute with "n/a" in it.
    assert "serial_number" not in assets["asset:light-a-02"]["attributes"]
    assert assets["asset:light-a-13"]["attributes"] == {}
    assert manifest(bundle)["inventory"]["attribute_catalogue"]["type:light-fixture-a"] == [
        "installation_date",
        "serial_number",
        "warranty_start_date",
    ]


def test_a_component_listed_in_two_spaces_sits_in_the_first(bundle: Path) -> None:
    assets = records(bundle, "osm/assets.jsonl")

    assert assets["asset:door-d-g04"]["space_id"] == "space:1g04"
    assert manifest(bundle)["counts"]["assets_listed_in_several_spaces"] == 3


def test_a_padded_type_name_still_joins_its_components(bundle: Path) -> None:
    types = records(bundle, "osm/asset_types.jsonl")
    assets = records(bundle, "osm/assets.jsonl")

    assert types["type:work-station"]["name"]["value"] == "Work Station"
    assert assets["asset:work-station-ws-01"]["type_id"] == "type:work-station"


def test_a_type_carries_what_the_handover_records_and_no_more(bundle: Path) -> None:
    types = records(bundle, "osm/asset_types.jsonl")
    light, sprinkler = types["type:light-fixture-a"], types["type:sprinkler-head"]

    assert light["omniclass"]["value"] == "23-80 70 11 14 11: Downlights"
    assert light["manufacturer"] == {
        "value": "Lithonia",
        "provenance": "derived",
        "rule": "manufacturer-from-email@1",
        "basis": None,
    }
    assert light["model"]["value"] == "GT2 MV"
    assert light["warranty_parts_years"]["value"] == "2"
    assert light["expected_life_years"] is None
    # A placeholder address names nobody.
    assert sprinkler["manufacturer"] is None
    assert sprinkler["model"] is None


def test_placement_comes_from_the_coordinate_sheet_and_else_from_the_ifc_model(
    bundle: Path,
) -> None:
    assets = records(bundle, "osm/assets.jsonl")

    assert assets["asset:ahu-1"]["position"]["rule"] == "position-from-cobie-coordinate@1"
    assert assets["asset:door-d-g03"]["position"]["rule"] == "position-from-cobie-coordinate@1"
    assert assets["asset:light-a-01"]["position"]["rule"] == "position-from-ifc-placement@1"
    assert assets["asset:panel-p1-01"]["position"]["provenance"] == "derived"
    assert assets["asset:light-a-02"]["position"] is None
    assert manifest(bundle)["counts"]["placed"] == {
        "by_coordinate_sheet": 2,
        "by_ifc_placement": 2,
        "share": round(4 / 35, 4),
    }


def test_an_ifc_placement_is_followed_through_the_placements_it_sits_in() -> None:
    placed = ifc.placements(mini_handover.DIR / mini_handover.IFC)

    # Millimetres in the file. The light's storey is turned a quarter, so its own
    # (1000, 500) lands at (10000 - 500, 2000 + 1000).
    assert placed["0aLightA01GlobalIdxxx1"] == pytest.approx((9.5, 3.0, 2.7))
    assert placed["0aPanelP101GlobalIdxx2"] == pytest.approx((7.0, 1.5, 0.0))
    assert "0aRelationGlobalIdxxx3" not in placed


def test_jobs_become_maintenance_tasks_on_their_type(bundle: Path) -> None:
    tasks = records(bundle, "osm/maintenance_tasks.jsonl")

    annual = tasks["task:ahu-ahu-annual-maintenance-0"]
    assert annual["type_id"] == "type:ahu"
    assert (annual["frequency"]["value"], annual["frequency_unit"]["value"]) == ("1", "year")
    assert annual["record_provenance"] == "sourced"
    assert len(tasks) == 2


# --- documents -----------------------------------------------------------------------------


def test_a_document_keyed_to_a_system_is_anchored_to_that_systems_assets_as_derived(
    bundle: Path,
) -> None:
    documents = records(bundle, "documents/documents.jsonl")

    assert documents["doc:hvac-om-pdf"]["anchors"] == [
        {"kind": "asset", "id": asset_id, "provenance": "derived", "rule": "anchors-from-system@1"}
        for asset_id in ("asset:ahu-1", "asset:ahu-2", "asset:ball-valve-bv-02")
    ]
    # The sheet name has a trailing space in the workbook, as eleven real rows do.
    assert [a["id"] for a in documents["doc:sprinkler-test-pdf"]["anchors"]] == [
        "asset:sprinkler-s-01",
        "asset:sprinkler-s-02",
    ]
    assert documents["doc:sprinkler-test-pdf"]["clearance_class"] == "visitor"
    # The smoke detectors' code is not in the table, so their report fails closed with them.
    assert documents["doc:firealarm-test-pdf"]["clearance_class"] == "maintenance"
    [chunk] = [
        c
        for c in read_jsonl(bundle / "documents" / "chunks.jsonl")
        if c["document_id"] == "doc:hvac-om-pdf"
    ]
    assert chunk["asset_ids"] == ["asset:ahu-1", "asset:ahu-2", "asset:ball-valve-bv-02"]
    assert "floor:roof" in chunk["ancestor_ids"]


def test_a_document_keyed_to_a_row_is_anchored_to_it_as_sourced(bundle: Path) -> None:
    documents = records(bundle, "documents/documents.jsonl")

    def anchors(document: str) -> list[tuple[str, str, str]]:
        found = documents[document]["anchors"]
        return [(a["kind"], a["id"], a["provenance"]) for a in found]

    assert anchors("doc:closeout-pdf") == [("space", "building:b-12", "sourced")]
    assert anchors("doc:firstfloor-notes-pdf") == [("space", "floor:first-floor", "sourced")]
    assert anchors("doc:elecroom-schedule-pdf") == [("space", "space:1g05", "sourced")]
    assert anchors("doc:ahu-startup-pdf") == [("asset", "asset:ahu-1", "sourced")]
    assert anchors("doc:lighta-productdata-pdf") == [
        ("asset_type", "type:light-fixture-a", "sourced")
    ]
    assert documents["doc:elecroom-schedule-pdf"]["clearance_class"] == "maintenance"
    assert documents["doc:closeout-pdf"]["clearance_class"] == "visitor"
    assert documents["doc:examtable-pdf"]["clearance_class"] == "staff"


def test_one_file_listed_for_several_rows_is_one_document(bundle: Path) -> None:
    warranty = records(bundle, "documents/documents.jsonl")["doc:warranty-general-pdf"]

    assert [a["id"] for a in warranty["anchors"]] == [
        "type:door-type-a",
        "type:lavatory-l1",
        "type:panelboard-p1",
    ]
    # One of its rows files it as product data, so it is a manual, and a manual
    # shared by several types takes the least restrictive of them.
    assert warranty["kind"] == "manual"
    assert warranty["clearance_class"] == "visitor"
    assert records(bundle, "documents/documents.jsonl")["doc:closeout-pdf"]["kind"] == (
        "handover-record"
    )


def test_the_text_is_the_ocr_and_a_document_without_any_is_skipped(
    bundle: Path, openai: OpenAI
) -> None:
    said = manifest(bundle)
    chunks = read_jsonl(bundle / "documents" / "chunks.jsonl")

    [panel] = [c for c in chunks if c["document_id"] == "doc:panelboard-om-pdf"]
    assert "isolate the supply and remove the cover" in panel["text"]
    assert panel["translated"] is False
    assert panel["title"] == "Panelboard Manual"
    assert said["documents"]["documents"] == 12
    assert [skipped["id"] for skipped in said["documents"]["skipped"]] == ["doc:lost-pdf"]
    assert said["counts"]["documents_without_text"] == ["doc:lost-pdf"]
    assert said["counts"]["documents_keyed_to_nothing"] == ["Ghost.pdf"]
    # English already, so nothing was translated. Summaries and vectors were made.
    assert openai.chats(summaries=False) == []
    assert len(openai.chats(summaries=True)) == 12


# --- pinned inputs -------------------------------------------------------------------------


def test_the_manifest_pins_the_handover_files_and_the_ocr_text(bundle: Path) -> None:
    said = manifest(bundle)
    workbook = mini_handover.DIR / mini_handover.WORKBOOK

    assert said["venue"] == "mini-bench"
    assert said["source"] == "handover"
    assert said["inputs"]["handover"]["files"] == {
        "handover.xlsx": hashlib.sha256(workbook.read_bytes()).hexdigest(),
        "model.ifc": hashlib.sha256((mini_handover.DIR / "model.ifc").read_bytes()).hexdigest(),
    }
    text = (mini_handover.DIR / "ocr" / "Closeout.txt").read_bytes()
    assert said["documents"]["originals"]["doc:closeout-pdf"] == hashlib.sha256(text).hexdigest()
    assert said["documents"]["index"] == INDEX


def test_a_file_that_does_not_match_its_pin_stops_the_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = tmp_path / "handover"
    shutil.copytree(mini_handover.DIR, copy)
    (copy / "model.ifc").write_text("not the model")

    assert build(tmp_path / "bundles", handover=copy) == 1
    assert "does not match the hash it is pinned to" in capsys.readouterr().err
    assert not (tmp_path / "bundles" / DATE).exists()


def test_a_handover_venue_is_not_built_without_its_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["build", "--date", DATE, "--venue", str(mini_handover.DEFINITION)]

    assert main([*args, "--out", str(tmp_path)]) == 2
    assert "--handover" in capsys.readouterr().err


def test_building_twice_writes_the_same_bytes_and_asks_no_model_again(
    bundle: Path, openai: OpenAI
) -> None:
    def files() -> dict[str, str]:
        return {
            str(path.relative_to(bundle)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(bundle.rglob("*"))
            if path.is_file()
        }

    first, calls = files(), len(openai.calls)

    assert build(bundle.parent) == 0
    assert files() == first
    assert len(openai.calls) == calls


@contextmanager
def serve(files: dict[str, bytes]) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = files.get(self.path.lstrip("/"))
            self.send_response(200 if body is not None else 404)
            self.send_header("Content-Length", str(len(body or b"")))
            self.end_headers()
            self.wfile.write(body or b"")

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


def test_fetching_a_handover_keeps_only_files_that_match_their_pins(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    workbook = (mini_handover.DIR / mini_handover.WORKBOOK).read_bytes()
    with serve({"handover.xlsx": workbook, "model.ifc": b"tampered"}) as url:
        definition = json.loads(mini_handover.DEFINITION.read_text())
        definition["cobie"]["url"] = f"{url}/handover.xlsx"
        definition["ifc"][0]["url"] = f"{url}/model.ifc"
        path = tmp_path / "definition.json"
        path.write_text(json.dumps(definition))
        fetched = tmp_path / "fetched"

        assert main(["fetch-handover", "--venue", str(path), "--to", str(fetched)]) == 1

    assert "is not the file that was pinned" in capsys.readouterr().err
    assert (fetched / "handover.xlsx").read_bytes() == workbook
    assert sorted(p.name for p in fetched.iterdir()) == ["handover.xlsx"]


def test_the_built_in_test_bench_pins_the_real_handover() -> None:
    from spatial_ingest.handover.definition import load_definition

    bench = load_definition("test-bench")

    assert bench is not None
    assert bench.id == "test-bench"
    # The same workbook the seed inventory was drawn from.
    assert bench.cobie.sha256 == (
        "cc539fe27e3a8df5c80bc6710c17f96f329c3d5d1b023b4246536333bcd77b7a"
    )
    assert all(pinned.url and len(pinned.sha256) == 64 for pinned in bench.pinned())
    assert load_definition("kit-campus-sued") is None


# --- load ----------------------------------------------------------------------------------


def test_the_bench_loads_through_the_same_load_and_passes_every_invariant(
    bundle: Path, database_url: str, search_dir: Path
) -> None:
    assert main(["load", "--date", DATE, "--out", str(bundle.parent)]) == 2
    code = main(
        ["load", "--date", DATE, "--out", str(bundle.parent), "--database-url", database_url]
    )

    assert code == 0
    with psycopg.connect(database_url) as conn:
        status, failures, venue, index = conn.execute(
            "SELECT status, failures, venue, search_index FROM control.snapshots"
        ).fetchone() or (None, None, None, None)
        assets = conn.execute("SELECT count(*) FROM snap_20260915.assets").fetchone()
        seeded = conn.execute("SELECT count(*) FROM snap_20260915_seed.assets").fetchone()
    assert (status, failures, venue, index) == ("loaded", [], "mini-bench", INDEX)
    assert assets == (35,)
    assert seeded == (0,)
    # Twelve documents: a passage and a summary each.
    assert len(LocalIndex(search_dir).chunks(INDEX)) == 24


def test_the_bench_is_never_loaded_into_a_database_that_holds_another_venue(
    bundle: Path,
    tmp_path: Path,
    venue_file: Path,
    database_url: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    campus = tmp_path / "campus"
    built = ["build", "--date", "2026-09-08", "--venue", str(venue_file), "--out", str(campus)]
    assert main(built) == 0
    loaded = ["--out", str(campus), "--database-url", database_url]
    assert main(["load", "--date", "2026-09-08", *loaded]) == 0

    code = main(
        ["load", "--date", DATE, "--out", str(bundle.parent), "--database-url", database_url]
    )

    assert code == 1
    with psycopg.connect(database_url) as conn:
        venues = conn.execute("SELECT venue FROM control.snapshots").fetchall()
        schemas = conn.execute(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = 'snap_20260915'"
        ).fetchone()
    assert venues == [("mini-campus",)]
    assert schemas == (0,)


def test_the_demo_venue_is_never_loaded_into_the_benchs_database(
    bundle: Path, tmp_path: Path, venue_file: Path, database_url: str
) -> None:
    out = ["--out", str(bundle.parent), "--database-url", database_url]
    assert main(["load", "--date", DATE, *out]) == 0
    campus = tmp_path / "campus"
    built = ["build", "--date", "2026-09-22", "--venue", str(venue_file), "--out", str(campus)]
    assert main(built) == 0

    code = main(
        ["load", "--date", "2026-09-22", "--out", str(campus), "--database-url", database_url]
    )

    assert code == 1
