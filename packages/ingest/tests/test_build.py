"""The ingestion CLI's build command, end to end over the mini campus."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs

import pytest

from mini_campus import PERSON, VENUE, overpass_response, write_pbf
from spatial_ingest.cli import main
from spatial_ingest.osm import from_osm_file

DATE = "2026-09-15"


@dataclass
class Overpass:
    """A stand-in Overpass endpoint on localhost."""

    url: str = ""
    status: int = 200
    queries: list[str] = field(default_factory=list)


@pytest.fixture
def overpass() -> Iterator[Overpass]:
    state = Overpass()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            form = self.rfile.read(int(self.headers["Content-Length"])).decode()
            state.queries.append(parse_qs(form)["data"][0])
            body = overpass_response() if state.status == 200 else b"rate limited"
            self.send_response(state.status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, args=(0.01,), daemon=True).start()
    state.url = f"http://127.0.0.1:{server.server_port}/api/interpreter"
    yield state
    server.shutdown()


@pytest.fixture
def venue_file(tmp_path: Path, overpass: Overpass) -> Path:
    path = tmp_path / "mini-campus.json"
    path.write_text(json.dumps({**VENUE, "overpass_url": overpass.url}))
    return path


def build(venue_file: Path, out: Path, *extra: str) -> int:
    return main(["build", "--date", DATE, "--venue", str(venue_file), "--out", str(out), *extra])


@pytest.fixture
def bundle(tmp_path: Path, venue_file: Path) -> Path:
    assert build(venue_file, tmp_path / "bundles") == 0
    return tmp_path / "bundles" / DATE


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def spaces(bundle: Path) -> dict[str, dict[str, Any]]:
    return {space["id"]: space for space in read_jsonl(bundle / "osm" / "spaces.jsonl")}


def manifest(bundle: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
    return loaded


def files(bundle: Path) -> dict[str, str]:
    return {str(p.relative_to(bundle)): sha256(p) for p in sorted(bundle.rglob("*")) if p.is_file()}


def test_build_pulls_a_dated_query_over_the_bounding_box(bundle: Path, overpass: Overpass) -> None:
    (query,) = overpass.queries

    assert '[date:"2026-09-15T00:00:00Z"]' in query
    assert "[bbox:49.009,8.409,49.016,8.418]" in query
    assert "area" not in query
    assert "way(id:9001,9002,9003)" in query


def test_the_response_is_stored_with_its_query_text_and_sha256(
    bundle: Path, overpass: Overpass
) -> None:
    pull = manifest(bundle)["inputs"]["pull"]

    assert (bundle / "pull" / "overpass.json").read_bytes() == overpass_response()
    assert (bundle / "pull" / "query.overpassql").read_text() == overpass.queries[0]
    assert pull["kind"] == "overpass"
    assert pull["query"] == overpass.queries[0]
    assert pull["sha256"] == hashlib.sha256(overpass_response()).hexdigest()
    assert pull["endpoint"] == overpass.url


def test_only_the_three_parts_of_the_site_are_kept(bundle: Path) -> None:
    kept = spaces(bundle)

    assert "osm:way/700" not in kept
    assert "osm:way/7001" not in kept
    buildings = sorted(s["id"] for s in kept.values() if s["tier"]["value"] == "building")
    assert buildings == [
        "osm:relation/300",
        "osm:way/100",
        "osm:way/200",
        "osm:way/400",
        "osm:way/500",
        "osm:way/600",
    ]


def test_the_pull_becomes_one_strict_tree(bundle: Path) -> None:
    tree = spaces(bundle)
    roots = [s for s in tree.values() if s["parent_id"] is None]

    assert [root["id"] for root in roots] == ["site:mini-campus"]
    assert len(tree) == 29
    for space in tree.values():
        assert space["tier"]["value"] in {"site", "building", "floor", "room"}
        walked = []
        parent_id = space["parent_id"]
        while parent_id is not None:
            walked.append({"tier": tree[parent_id]["tier"]["value"], "id": parent_id})
            parent_id = tree[parent_id]["parent_id"]
        assert space["ancestor_path"] == walked[::-1]


def test_a_room_sits_under_site_building_and_floor(bundle: Path) -> None:
    room = spaces(bundle)["osm:way/1001"]

    assert room["parent_id"] == "floor:way/100:1"
    assert room["ancestor_path"] == [
        {"tier": "site", "id": "site:mini-campus"},
        {"tier": "building", "id": "osm:way/100"},
        {"tier": "floor", "id": "floor:way/100:1"},
    ]


def test_floors_are_derived_from_level_tags_and_say_so(bundle: Path) -> None:
    tree = spaces(bundle)
    floors = sorted(s["id"] for s in tree.values() if s["tier"]["value"] == "floor")

    assert floors == [
        "floor:relation/300:0",
        "floor:way/100:0",
        "floor:way/100:1",
        "floor:way/100:2",
        "floor:way/200:1",
        "floor:way/400:0",
        "floor:way/500:1",
    ]
    floor = tree["floor:way/100:1"]
    assert floor["tier"] == {
        "value": "floor",
        "provenance": "derived",
        "rule": "floor-from-level-tags@1",
        "basis": None,
    }
    assert floor["source"] is None
    assert floor["parent_id"] == "osm:way/100"
    assert floor["label"]["value"] == "50.34 level 1"


def test_a_space_covering_several_floors_hangs_from_the_lowest_and_spans_the_rest(
    bundle: Path,
) -> None:
    stairs = spaces(bundle)["osm:way/1004"]
    spans = read_jsonl(bundle / "osm" / "spans.jsonl")

    assert stairs["parent_id"] == "floor:way/100:0"
    assert stairs["levels"]["value"] == "0;1;2"
    assert spans == [
        {
            "space_id": "osm:way/1004",
            "floor_id": "floor:way/100:1",
            "provenance": "derived",
            "rule": "level-parse@1",
        },
        {
            "space_id": "osm:way/1004",
            "floor_id": "floor:way/100:2",
            "provenance": "derived",
            "rule": "level-parse@1",
        },
    ]


def test_a_room_without_a_level_hangs_from_its_building(bundle: Path) -> None:
    assert spaces(bundle)["osm:way/1008"]["parent_id"] == "osm:way/100"


def test_room_labels(bundle: Path) -> None:
    tree = spaces(bundle)

    def label(space_id: str) -> str | None:
        value = tree[space_id]["label"]
        return value["value"] if value else None

    assert label("osm:way/1002") == "Gaede-Hörsaal"
    assert label("osm:way/1001") == "50.34 room 107"
    assert label("osm:node/1007") == "50.34 room 109"
    assert label("osm:way/2001") == "Kollegiengebäude am Zirkel room 107"
    assert label("osm:way/2002") == "Kollegiengebäude am Zirkel room 102"
    assert label("osm:relation/300") == "30.10"
    assert label("osm:way/100") == "Informatik"
    # What the name filter leaves of a name is not a sourced name, so the number wins.
    assert label("osm:way/1003") == "50.34 room 108"
    assert tree["osm:way/1001"]["label"]["provenance"] == "derived"
    assert tree["osm:way/1001"]["label"]["rule"] == "room-label@1"
    assert tree["osm:way/1002"]["label"]["provenance"] == "sourced"


def test_rooms_no_rule_can_address_are_kept_and_marked(bundle: Path) -> None:
    tree = spaces(bundle)
    unaddressable = sorted(s["id"] for s in tree.values() if not s["addressable"])

    assert unaddressable == [
        "floor:way/400:0",  # its building has neither number nor name
        "osm:way/1005",  # no name, no number
        "osm:way/1006",  # a corridor with neither
        "osm:way/3002",  # in the courtyard, so in no building
        "osm:way/400",
        "osm:way/4001",  # numbered, but its building is not
    ]
    for space_id in unaddressable:
        assert tree[space_id]["label"] is None
    assert tree["osm:way/3002"]["parent_id"] == "site:mini-campus"


def test_names_of_people_are_stripped_and_flagged_as_derived(bundle: Path) -> None:
    tree = spaces(bundle)
    office = tree["osm:way/1003"]

    assert office["name"] == {
        "value": "Büro",
        "provenance": "derived",
        "rule": "strip-person-names@1",
        "basis": None,
    }
    assert office["ref"]["value"] == "108"
    assert "description" not in tree["osm:way/1009"]["attributes"]
    assert tree["osm:way/100"]["attributes"]["description"]["value"] == "Leitung"
    assert tree["osm:way/1002"]["name"]["provenance"] == "sourced"
    published = (bundle / "osm" / "spaces.jsonl").read_text()
    assert "Mustermann" not in published
    assert PERSON in (bundle / "pull" / "overpass.json").read_text()


def test_every_record_carries_provenance_and_every_derived_value_names_its_rule(
    bundle: Path,
) -> None:
    derived = 0

    def check(node: Any) -> None:
        nonlocal derived
        if isinstance(node, dict):
            if node.get("provenance") == "derived":
                derived += 1
                assert node["rule"], node
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    for space in spaces(bundle).values():
        assert space["record_provenance"] == "sourced"
        assert space["tier"]["provenance"] in {"sourced", "derived"}
        check(space)
    assert derived > 0
    assert spaces(bundle)["osm:way/1001"]["attributes"]["access"] == {
        "value": "private",
        "provenance": "sourced",
        "rule": None,
        "basis": None,
    }
    assert spaces(bundle)["osm:way/1001"]["source"] == "way/1001"


def test_the_manifest_records_input_hashes_and_what_produced_each_part(bundle: Path) -> None:
    recorded = manifest(bundle)

    assert recorded["snapshot"] == DATE
    assert recorded["venue"] == "mini-campus"
    assert recorded["inputs"]["pull"]["sha256"] == sha256(bundle / "pull" / "overpass.json")
    assert recorded["inputs"]["venue"]["definition"]["part_ways"] == [9001, 9002, 9003]
    assert len(recorded["inputs"]["venue"]["sha256"]) == 64
    assert "strip-person-names@1" not in recorded["parts"]["osm/spans.jsonl"]["produced_by"]
    assert sorted(recorded["parts"]) == ["osm/spaces.jsonl", "osm/spans.jsonl"]
    for name, part in recorded["parts"].items():
        assert part["sha256"] == sha256(bundle / name)
        assert part["records"] == len(read_jsonl(bundle / name))
        assert part["produced_by"]
    assert "strip-person-names@1" in recorded["parts"]["osm/spaces.jsonl"]["produced_by"]
    assert recorded["counts"]["spaces"] == {"site": 1, "building": 6, "floor": 7, "room": 15}
    assert recorded["counts"]["rooms_addressable"] == 11
    assert recorded["counts"]["names_filtered"] == 3


def test_building_twice_from_the_same_stored_pull_gives_the_same_bundle(
    bundle: Path, venue_file: Path, overpass: Overpass, tmp_path: Path
) -> None:
    first = files(bundle)

    assert build(venue_file, tmp_path / "bundles") == 0

    assert files(bundle) == first
    assert len(overpass.queries) == 1


def test_a_stored_pull_that_changed_since_it_was_pinned_fails_the_build(
    bundle: Path, venue_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stored = bundle / "pull" / "overpass.json"
    stored.write_bytes(stored.read_bytes() + b"\n")

    assert build(venue_file, tmp_path / "bundles") != 0
    assert "sha256" in capsys.readouterr().err


def test_rate_limiting_fails_the_build_and_points_at_the_extract_fallback(
    venue_file: Path, overpass: Overpass, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    overpass.status = 429

    assert build(venue_file, tmp_path / "bundles") != 0
    assert "--extract" in capsys.readouterr().err
    assert not (tmp_path / "bundles" / DATE / "manifest.json").exists()


def test_a_regional_extract_cut_down_to_the_box_gives_the_same_structure(
    bundle: Path, venue_file: Path, overpass: Overpass, tmp_path: Path
) -> None:
    extract = tmp_path / "region-260915.osm.pbf"
    write_pbf(extract)

    assert build(venue_file, tmp_path / "fallback", "--extract", str(extract)) == 0

    fallback = tmp_path / "fallback" / DATE
    assert len(overpass.queries) == 1  # only the first bundle's pull
    for part in ("osm/spaces.jsonl", "osm/spans.jsonl"):
        assert (fallback / part).read_bytes() == (bundle / part).read_bytes()


def test_the_cut_down_extract_is_pinned_the_same_way(venue_file: Path, tmp_path: Path) -> None:
    extract = tmp_path / "region-260915.osm.pbf"
    write_pbf(extract)
    out = tmp_path / "fallback"

    assert build(venue_file, out, "--extract", str(extract)) == 0
    first = files(out / DATE)
    extract.unlink()
    assert build(venue_file, out) == 0

    pull = manifest(out / DATE)["inputs"]["pull"]
    assert pull["kind"] == "extract"
    assert pull["sha256"] == sha256(out / DATE / "pull" / "extract.osm.pbf")
    assert pull["cut_from"]["name"] == "region-260915.osm.pbf"
    assert len(pull["cut_from"]["sha256"]) == 64
    assert pull["bbox"] == VENUE["bbox"]
    assert files(out / DATE) == first
    cut = from_osm_file(out / DATE / "pull" / "extract.osm.pbf")
    assert 100 in cut.ways
    assert 880 not in cut.ways
