"""The ingestion CLI's build command, end to end over the mini campus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mini_campus import PERSON, VENUE, overpass_response, write_pbf
from spatial_ingest.cli import main
from spatial_ingest.osm import from_osm_file
from stubs import Overpass

DATE = "2026-09-15"


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
    assert len(tree) == 30
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
        "osm:way/1010",  # no name, no number, no door
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
    assert sorted(recorded["parts"]) == [
        "documents/chunks.jsonl",
        "documents/documents.jsonl",
        "documents/summaries.jsonl",
        "documents/translations.jsonl",
        "documents/vectors.jsonl",
        "osm/asset_types.jsonl",
        "osm/assets.jsonl",
        "osm/edges.jsonl",
        "osm/spaces.jsonl",
        "osm/spans.jsonl",
        "osm/visible_asset_types.jsonl",
        "osm/visible_assets.jsonl",
        "osm/visible_spaces.jsonl",
        "seed/asset_types.jsonl",
        "seed/assets.jsonl",
        "seed/maintenance_tasks.jsonl",
        "seed/service_events.jsonl",
        "seed/visible_asset_types.jsonl",
        "seed/visible_assets.jsonl",
    ]
    for name, part in recorded["parts"].items():
        assert part["sha256"] == sha256(bundle / name)
        assert part["records"] == len(read_jsonl(bundle / name))
        assert part["produced_by"]
    assert "strip-person-names@1" in recorded["parts"]["osm/spaces.jsonl"]["produced_by"]
    assert recorded["counts"]["spaces"] == {"site": 1, "building": 6, "floor": 7, "room": 16}
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
    for part in (
        "osm/spaces.jsonl",
        "osm/spans.jsonl",
        "osm/edges.jsonl",
        "osm/visible_spaces.jsonl",
    ):
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


def edges(bundle: Path) -> dict[str, dict[str, Any]]:
    return {edge["id"]: edge for edge in read_jsonl(bundle / "osm" / "edges.jsonl")}


def visible(bundle: Path) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {"visitor": set(), "staff": set(), "maintenance": set()}
    for entry in read_jsonl(bundle / "osm" / "visible_spaces.jsonl"):
        sets[entry["persona"]].add(entry["space_id"])
    return sets


def test_the_circulation_edges_are_built_from_the_doors_in_the_pull(bundle: Path) -> None:
    built = edges(bundle)

    assert sorted(built) == [
        "node/1101:outside>osm:way/1006",
        "node/1102:osm:way/1005>osm:way/1006",
        "node/1102:osm:way/1006>osm:way/1005",
        "node/1103:outside>osm:way/1002",
        "node/1104:outside>osm:way/1001",
        "snap:osm:way/1006>osm:way/1010",
        "snap:osm:way/1010>osm:way/1006",
    ]
    inward = built["node/1102:osm:way/1006>osm:way/1005"]
    assert inward["door"] == "node/1102"
    assert (inward["from_id"], inward["to_id"]) == ("osm:way/1006", "osm:way/1005")
    assert (inward["provenance"], inward["rule"]) == ("derived", "edges-from-doors@1")
    assert built["node/1101:outside>osm:way/1006"]["from_id"] is None
    # Five through doors, and the two of a snap.
    assert manifest(bundle)["counts"]["edges"] == 7
    # A door on no outline joins nothing.
    assert manifest(bundle)["counts"]["doors_on_no_space"] == 1


def test_an_edge_takes_the_stricter_of_its_door_and_the_space_it_enters(bundle: Path) -> None:
    built = edges(bundle)

    def cleared(edge_id: str) -> tuple[str, str | None]:
        clearance = built[edge_id]["clearance"]
        return clearance["persona"], clearance["access"]

    # A yes door never opens a private room, and the raw tag that decided is kept.
    assert cleared("node/1104:outside>osm:way/1001") == ("staff", "private")
    assert built["node/1104:outside>osm:way/1001"]["door_clearance"]["access"] == "yes"
    # A private door closes a room whose own default is visitor.
    assert cleared("node/1103:outside>osm:way/1002") == ("staff", "private")
    # An untagged door constrains nothing: the edge takes the space it enters.
    assert built["node/1102:osm:way/1006>osm:way/1005"]["door_clearance"] is None
    assert cleared("node/1102:osm:way/1006>osm:way/1005") == ("staff", None)
    assert cleared("node/1102:osm:way/1005>osm:way/1006") == ("visitor", None)
    assert built["node/1102:osm:way/1005>osm:way/1006"]["clearance"]["authored_default"]


def test_space_clearance_is_read_from_access_tags_or_flagged_as_an_authored_default(
    bundle: Path,
) -> None:
    tree = spaces(bundle)

    assert tree["osm:way/1001"]["clearance"] == {
        "persona": "staff",
        "access": "private",
        "rule": "clearance-from-access@1",
        "authored_default": False,
        "unrecognised_category": False,
    }
    assert tree["osm:way/2001"]["clearance"]["persona"] == "visitor"  # customers
    # An unrecognised value fails closed, and the raw tag is kept beside it.
    assert tree["osm:way/1009"]["clearance"]["persona"] == "maintenance"
    assert tree["osm:way/1009"]["clearance"]["access"] == "staff_only"
    assert tree["osm:way/1006"]["clearance"] == {
        "persona": "visitor",
        "access": None,
        "rule": "clearance-authored-default@1",
        "authored_default": True,
        "unrecognised_category": False,
    }
    assert tree["osm:way/1002"]["clearance"]["persona"] == "visitor"  # an auditorium
    assert tree["osm:way/1005"]["clearance"]["persona"] == "staff"
    # Above the room tier an untagged space has no clearance of its own.
    assert tree["osm:way/100"]["clearance"] is None
    assert tree["floor:way/100:0"]["clearance"] is None


def test_each_persona_gets_its_visible_set_and_the_sets_nest(bundle: Path) -> None:
    sets = visible(bundle)

    assert sets["visitor"] == {
        "site:mini-campus",
        "osm:way/100",
        "floor:way/100:0",
        "osm:way/1006",  # entered from outside through an untagged entrance
        "osm:way/1004",  # no edge, so its own clearance: stairs are visitor
        "osm:way/200",
        "floor:way/200:1",
        "osm:way/2001",
    }
    # Behind a door that admits staff, whatever the room's own clearance.
    assert "osm:way/1002" in sets["staff"] - sets["visitor"]
    assert "osm:way/1005" in sets["staff"] - sets["visitor"]
    assert sets["visitor"] <= sets["staff"] <= sets["maintenance"]
    # Only maintenance reaches 1009, and the floor it sits on comes with it.
    assert sets["maintenance"] - sets["staff"] == {"osm:way/1009", "floor:way/100:2"}
    # A building with no rooms and no tag is entered by nothing.
    assert set(spaces(bundle)) - sets["maintenance"] == {"osm:way/600"}


def test_the_manifest_records_the_share_of_rooms_whose_clearance_is_sourced(
    bundle: Path,
) -> None:
    recorded = manifest(bundle)

    assert recorded["clearance"] == {
        "rooms": 15,
        "rooms_sourced": 3,
        "sourced_share": 0.2,
        "unrecognised_access": 1,
        "visible_spaces": {"visitor": 8, "staff": 27, "maintenance": 29},
    }
    produced_by = recorded["parts"]["osm/visible_spaces.jsonl"]["produced_by"]
    assert "visible-by-edges@1" in produced_by
    assert "edge-clearance@1" in recorded["parts"]["osm/edges.jsonl"]["produced_by"]


# --- what routes need -----------------------------------------------------------------------


def test_an_edge_says_where_its_door_is_and_on_which_level(bundle: Path) -> None:
    built = edges(bundle)

    entrance = built["node/1101:outside>osm:way/1006"]
    assert (entrance["lon"], entrance["lat"], entrance["level"]) == (8.4103, 49.01115, "0")
    # Both ways through a door are at the same door.
    there, back = (
        built["node/1102:osm:way/1005>osm:way/1006"],
        built["node/1102:osm:way/1006>osm:way/1005"],
    )
    assert (there["lon"], there["lat"], there["level"]) == (back["lon"], back["lat"], "0")


def test_a_doorless_room_is_snapped_to_circulation_within_two_metres(bundle: Path) -> None:
    built = edges(bundle)
    snaps = {id: edge for id, edge in built.items() if edge["door"] is None}

    assert sorted(snaps) == ["snap:osm:way/1006>osm:way/1010", "snap:osm:way/1010>osm:way/1006"]
    into_room = snaps["snap:osm:way/1006>osm:way/1010"]
    assert (into_room["provenance"], into_room["rule"]) == ("derived", "snap-doorless-room@1")
    assert into_room["level"] == "0"
    # On the corridor's outline, where the room is nearest.
    assert into_room["lat"] == pytest.approx(49.0112)
    # The room it enters decides the clearance, as through a door.
    assert into_room["clearance"]["persona"] == "staff"
    assert snaps["snap:osm:way/1010>osm:way/1006"]["clearance"]["persona"] == "visitor"
    # 3001 has no door either, and no circulation anywhere near it.
    assert not any("osm:way/3001" in id for id in built)
    assert manifest(bundle)["counts"]["snapped_rooms"] == 1
    produced_by = manifest(bundle)["parts"]["osm/edges.jsonl"]["produced_by"]
    assert "snap-doorless-room@1" in produced_by


def test_a_snap_shows_nobody_a_room(bundle: Path) -> None:
    # 1010 is an untagged room, so staff. The snap from the visitor corridor does not
    # open it, and no door asset is hung on a door that is not there.
    assert "osm:way/1010" not in visible(bundle)["visitor"]
    assert "osm:way/1010" in visible(bundle)["staff"]
    assets = read_jsonl(bundle / "seed" / "assets.jsonl")
    assert all(asset["door"] is None or asset["door"].startswith("node/") for asset in assets)


def test_the_default_origin_is_an_entrance_from_the_venue_definition(bundle: Path) -> None:
    assert manifest(bundle)["default_origin"] == {
        "door": "node/1101",
        "edge": "node/1101:outside>osm:way/1006",
    }


def test_a_default_origin_that_is_no_entrance_fails_the_build(
    tmp_path: Path, venue_file: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # 1102 is a door between two rooms: nobody comes in from outside through it.
    definition = json.loads(venue_file.read_text())
    venue_file.write_text(json.dumps({**definition, "default_origin": "node/1102"}))

    assert build(venue_file, tmp_path / "bundles") != 0
    assert "node/1102" in capsys.readouterr().err
