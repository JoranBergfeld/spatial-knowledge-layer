"""The seed inventory and the sourced fire assets, built over the mini campus."""

from __future__ import annotations

import datetime
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from shapely import Point
from shapely.geometry import shape

from spatial_core.clearance import type_clearance
from spatial_ingest.cli import main
from spatial_ingest.rules import fire
from spatial_ingest.seed import generate

DATE = "2026-09-15"
INVENTORY = generate.default_inventory()


def build(venue_file: Path, out: Path, *extra: str) -> int:
    return main(["build", "--date", DATE, "--venue", str(venue_file), "--out", str(out), *extra])


@pytest.fixture
def bundle(tmp_path: Path, venue_file: Path) -> Path:
    assert build(venue_file, tmp_path / "bundles") == 0
    return tmp_path / "bundles" / DATE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def records(bundle: Path, part: str) -> dict[str, dict[str, Any]]:
    return {record["id"]: record for record in read_jsonl(bundle / part)}


def manifest(bundle: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads((bundle / "manifest.json").read_text())
    return loaded


def inputs() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(INVENTORY.read_text())
    return loaded


def write_inputs(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(json.dumps(data))
    return path


def seeded(bundle: Path) -> list[dict[str, Any]]:
    return read_jsonl(bundle / "seed" / "assets.jsonl")


def in_space(bundle: Path, space_id: str) -> Counter[str]:
    """What a room's category drew for it: its assets other than doors."""
    return Counter(
        a["type_id"] for a in seeded(bundle) if a["space_id"] == space_id and a["door"] is None
    )


def test_asset_types_are_records_with_their_type_data(bundle: Path) -> None:
    types = records(bundle, "seed/asset_types.jsonl")

    door = types["seed:type:door-type-a"]
    assert door["record_provenance"] == "seed"
    assert door["name"]["value"] == "Door Type A"
    assert door["manufacturer"]["value"] == "Fibrdor"
    assert "manufacturer-from-email@1" in door["manufacturer"]["basis"]
    assert door["model"]["value"] == "Standard Fib-R-Dor"
    assert door["omniclass"]["value"] == "23-30 10"
    assert door["warranty_parts_years"]["value"] == "2"
    assert door["warranty_labor_years"]["value"] == "0"
    assert door["expected_life_years"]["value"] == "15"
    # The bench records an expected life on few types. A gap stays a gap.
    assert types["seed:type:light-fixture-b1"]["expected_life_years"] is None
    for kind in types.values():
        cleared = type_clearance(kind["omniclass"]["value"]).model_dump(mode="json")
        assert kind["clearance"] == cleared
        assert kind["clearance"]["authored_default"]
    # Only the types some asset uses are written.
    used = {asset["type_id"] for asset in seeded(bundle)}
    assert set(types) == used


def test_every_type_code_is_recognised_and_the_manifest_counts_the_fallbacks(
    bundle: Path, venue_file: Path, tmp_path: Path
) -> None:
    for kind in inputs()["types"].values():
        assert not type_clearance(kind["omniclass"]).unrecognised_category, kind["name"]
    assert manifest(bundle)["inventory"]["unrecognised_categories"] == 0

    data = inputs()
    data["types"]["light-fixture-b1"]["omniclass"] = "23-99 99"
    path = write_inputs(tmp_path / "unlisted.json", data)

    # A code the rule does not list fails closed and is counted. It blocks nothing.
    assert build(venue_file, tmp_path / "out", "--inventory", str(path)) == 0
    built = tmp_path / "out" / DATE
    assert manifest(built)["inventory"]["unrecognised_categories"] == 1
    light = records(built, "seed/asset_types.jsonl")["seed:type:light-fixture-b1"]
    assert light["clearance"]["persona"] == "maintenance"
    assert light["clearance"]["unrecognised_category"]


@pytest.mark.parametrize(
    ("email", "name"),
    [
        ("sales@lithonia.com", "Lithonia"),
        ("sales@krueger-hvac.com", "Krueger-hvac"),
        ("supplycompany@usace.army.mil", None),
        ("n/a", None),
        (None, None),
    ],
)
def test_a_manufacturer_is_read_from_the_benchs_contact_address(
    email: str | None, name: str | None
) -> None:
    placeholders = inputs()["placeholder_manufacturers"]

    assert generate.manufacturer_from_email(email, placeholders) == name


def test_the_same_inputs_give_the_same_inventory(
    bundle: Path, venue_file: Path, tmp_path: Path
) -> None:
    assert build(venue_file, tmp_path / "again") == 0

    again = tmp_path / "again" / DATE
    for part in sorted((bundle / "seed").glob("*.jsonl")):
        assert (again / "seed" / part.name).read_bytes() == part.read_bytes()


def test_the_seed_and_the_inputs_are_pinned_in_the_manifest(bundle: Path) -> None:
    pin = manifest(bundle)["inputs"]["inventory"]

    assert pin == {
        "file": "inventory.json",
        "sha256": hashlib.sha256(INVENTORY.read_bytes()).hexdigest(),
        "seed": inputs()["seed"],
    }
    assert manifest(bundle)["inventory"]["seed"] == inputs()["seed"]


def test_another_seed_gives_another_inventory(
    bundle: Path, venue_file: Path, tmp_path: Path
) -> None:
    other = write_inputs(tmp_path / "other.json", {**inputs(), "seed": inputs()["seed"] + 1})

    assert build(venue_file, tmp_path / "other", "--inventory", str(other)) == 0

    built = tmp_path / "other" / DATE
    assert manifest(built)["inputs"]["inventory"]["seed"] == inputs()["seed"] + 1
    # The same rooms and counts, but other serials, dates and gaps.
    assert [a["id"] for a in seeded(built)] == [a["id"] for a in seeded(bundle)]
    assert seeded(built) != seeded(bundle)


def test_each_mapped_room_draws_its_categorys_mix_and_unmapped_rooms_get_nothing(
    bundle: Path,
) -> None:
    rooms = {a["space_id"] for a in seeded(bundle) if a["door"] is None}
    categories = inputs()["categories"]

    assert rooms == {
        "osm:way/1002",
        "osm:way/1004",
        "osm:way/1005",
        "osm:way/1006",
        "osm:node/1007",
    }
    for space_id, category in (
        ("osm:way/1002", "Classroom"),  # room=auditorium
        ("osm:way/1004", "Stairway"),  # room=stairs
        ("osm:way/1005", "Office"),  # room=office
        ("osm:way/1006", "Corridor"),  # indoor=corridor
    ):
        drawn = {f"seed:type:{item['type']}" for item in categories[category]["mix"]}
        assert set(in_space(bundle, space_id)) <= drawn, space_id
        assert in_space(bundle, space_id), space_id
    counts = manifest(bundle)["inventory"]["generated"]
    assert counts["rooms_mapped"] == 5
    assert counts["rooms_unmapped"] == 11


def test_every_room_mapping_names_a_bench_category() -> None:
    data = inputs()

    mapped = {m["tags"].get("room"): m["category"] for m in data["room_categories"]}
    assert mapped["office"] == "Office"
    assert mapped["toilets"] == "Restroom"
    assert mapped["seminar_room"] == "Conference Room"
    assert mapped["computer_lab"] == "Classroom"
    assert mapped["hall"] == "Waiting Room"
    assert set(mapped.values()) == set(data["categories"])


@pytest.mark.parametrize(
    ("item", "area", "count"),
    [
        ({"m2_per_item": 8, "min": 1}, 40.0, 5),
        ({"m2_per_item": 8, "min": 1}, 2.0, 1),
        ({"m2_per_item": 8, "min": 0}, 2.0, 0),
        ({"m2_per_item": 8, "min": 1}, 1000.0, 125),
        ({"m2_per_item": 8, "min": 1}, 20.0, 3),
        ({"m2_per_item": 5.4, "min": 0}, 10.0, 2),
    ],
)
def test_counts_scale_with_area_and_are_at_least_the_categorys_min(
    item: dict[str, Any], area: float, count: int
) -> None:
    assert generate.count_for(item, area) == count


def test_a_room_without_an_outline_gets_a_count_and_no_placement(bundle: Path) -> None:
    office = inputs()["categories"]["Office"]
    typical = float(office["typical_area_m2"])

    placed = [a for a in seeded(bundle) if a["space_id"] == "osm:node/1007"]

    assert placed
    assert all(a["position"] is None for a in placed)
    expected = {
        f"seed:type:{item['type']}": generate.count_for(item, typical) for item in office["mix"]
    }
    assert in_space(bundle, "osm:node/1007") == {k: v for k, v in expected.items() if v}
    assert manifest(bundle)["inventory"]["generated"]["rooms_without_outline"] == 1


def test_placement_follows_the_real_room_outline(bundle: Path) -> None:
    spaces = records(bundle, "osm/spaces.jsonl")

    for asset in seeded(bundle):
        if asset["door"] is not None or asset["position"] is None:
            continue
        outline = shape(spaces[asset["space_id"]]["outline"])
        point = Point(asset["position"]["lon"], asset["position"]["lat"])
        assert outline.covers(point), asset["id"]
        assert asset["position"]["provenance"] == "inferred"
        assert "layout-in-outline@1" in asset["position"]["basis"]


def test_only_domain_neutral_types_are_generated(bundle: Path) -> None:
    kinds = inputs()["types"].values()

    # The spec's list: lights, AC units, air terminals, doors, valves, sanitary
    # fittings, electrical panels. Nothing else, and every one of them is used.
    assert {kind["family"] for kind in kinds} == set(generate.NEUTRAL_FAMILIES)
    assert not [kind["name"] for kind in kinds if "camera" in kind["name"].lower()]
    published = (bundle / "seed" / "asset_types.jsonl").read_text().lower()
    assert "camera" not in published
    assert "demo " not in published


def test_an_inventory_with_a_camera_type_is_refused(
    venue_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = inputs()
    data["types"]["dome-camera"] = {**data["types"]["light-fixture-b1"], "name": "Dome camera"}
    data["categories"]["Corridor"]["mix"].append(
        {"type": "dome-camera", "m2_per_item": 50, "min": 1}
    )
    path = write_inputs(tmp_path / "cameras.json", data)

    assert build(venue_file, tmp_path / "out", "--inventory", str(path)) == 1
    assert "not domain-neutral" in capsys.readouterr().err


def test_an_inventory_with_a_type_outside_the_neutral_families_is_refused(
    venue_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = inputs()
    data["types"]["office-desk"] = {
        **data["types"]["light-fixture-b1"],
        "name": "Office desk",
        "family": "furniture",
    }
    path = write_inputs(tmp_path / "desks.json", data)

    assert build(venue_file, tmp_path / "out", "--inventory", str(path)) == 1
    assert "not domain-neutral" in capsys.readouterr().err


def test_a_door_asset_binds_to_a_sourced_door_and_is_no_edge(bundle: Path) -> None:
    edges = read_jsonl(bundle / "osm" / "edges.jsonl")
    doors = {a["door"]: a for a in seeded(bundle) if a["door"] is not None}

    # 1104 opens only into 1001, which maps to no category.
    assert set(doors) == {"node/1101", "node/1102", "node/1103"}
    assert all(door in {edge["door"] for edge in edges} for door in doors)
    # A door between a room and a corridor sits in the room.
    assert doors["node/1102"]["space_id"] == "osm:way/1005"
    assert doors["node/1101"]["space_id"] == "osm:way/1006"
    assert doors["node/1102"]["type_id"] == "seed:type:door-type-a"
    assert not (bundle / "seed" / "edges.jsonl").exists()
    # The edges are what they were: the doors' own, derived from OSM.
    assert {edge["provenance"] for edge in edges} == {"derived"}


def test_gaps_mirror_the_bench_fill_rates_per_type(bundle: Path) -> None:
    rates = inputs()["bench"]["fill_rates"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for asset in seeded(bundle):
        by_type.setdefault(asset["type_id"], []).append(asset)

    assert rates == {"serial_number": 0.43, "installation_date": 1.0, "warranty_start_date": 1.0}
    for assets in by_type.values():
        for key, rate in rates.items():
            filled = sum(1 for asset in assets if key in asset["attributes"])
            assert filled == math.floor(rate * len(assets) + 0.5)
    fill = manifest(bundle)["inventory"]["generated"]["fill"]
    for key, rate in rates.items():
        assert fill[key]["bench"] == rate
        assert abs(fill[key]["generated"] - rate) < 0.02


def test_serials_and_install_dates_follow_their_rules(bundle: Path) -> None:
    types = inputs()["types"]
    spread = inputs()["bench"]["install_days_before_handover_deciles"]
    snapshot = datetime.date.fromisoformat(DATE)
    built = datetime.date(2024, 3, 1)  # building A's start_date, 2024-03
    held = 0

    for asset in seeded(bundle):
        kind = types[asset["type_id"].removeprefix("seed:type:")]
        attributes = asset["attributes"]
        if "serial_number" in attributes:
            serial = attributes["serial_number"]["value"]
            assert serial.startswith(kind["model"] + "-")
            assert len(serial) == len(kind["model"]) + 9
        installed = datetime.date.fromisoformat(attributes["installation_date"]["value"])
        # Within the bench's spread before the snapshot, and never before the building.
        assert snapshot - datetime.timedelta(days=spread[-1]) <= installed
        assert installed <= snapshot - datetime.timedelta(days=spread[0])
        assert installed >= built
        if "start_date" in attributes["installation_date"]["basis"]:
            assert installed == built
            held += 1
        # The bench starts every warranty on the day of installation.
        assert attributes["warranty_start_date"]["value"] == installed.isoformat()
        assert "warranty-start@1" in attributes["warranty_start_date"]["basis"]
    assert 0 < held < len(seeded(bundle))


@pytest.mark.parametrize(
    ("value", "date"),
    [
        ("1965", datetime.date(1965, 1, 1)),
        ("1965-03", datetime.date(1965, 3, 1)),
        ("1965-03-17", datetime.date(1965, 3, 17)),
        ("~1960", None),
        ("C19", None),
        ("1965-13", None),
        (None, None),
    ],
)
def test_a_start_date_bounds_only_where_it_is_a_plain_date(
    value: str | None, date: datetime.date | None
) -> None:
    assert generate.start_date(value) == date


def test_an_asset_carries_its_spaces_path_and_then_its_space(bundle: Path) -> None:
    spaces = records(bundle, "osm/spaces.jsonl")
    assets = [*seeded(bundle), *read_jsonl(bundle / "osm" / "assets.jsonl")]

    assert assets
    for asset in assets:
        space = spaces[asset["space_id"]]
        assert asset["ancestor_path"] == [
            *space["ancestor_path"],
            {"tier": space["tier"]["value"], "id": space["id"]},
        ]


def test_maintenance_tasks_follow_cobie_job(bundle: Path) -> None:
    tasks = records(bundle, "seed/maintenance_tasks.jsonl")

    task = tasks["seed:task:seed:type:door-type-a:1"]
    assert task["type_id"] == "seed:type:door-type-a"
    for field in (
        "name",
        "category",
        "status",
        "description",
        "duration",
        "duration_unit",
        "start",
        "task_start_unit",
        "frequency",
        "frequency_unit",
        "task_number",
    ):
        assert task[field]["value"], field
    assert task["category"]["value"] == "PM"
    assert task["frequency"]["value"] == "12"
    assert task["frequency_unit"]["value"] == "month"
    # The bench has no job for any generated type, so only the authored ones are
    # written: the door's and the panelboard's, and the sourced fire types'.
    authored = {f"seed:type:{key}" for key, kind in inputs()["types"].items() if kind["jobs"]}
    used = {t["id"] for t in read_jsonl(bundle / "seed" / "asset_types.jsonl")}
    fire = {t["id"] for t in read_jsonl(bundle / "osm" / "asset_types.jsonl")}
    assert {task["type_id"] for task in tasks.values()} == (authored & used) | fire


def test_each_asset_has_at_most_one_latest_service_event(bundle: Path) -> None:
    events = read_jsonl(bundle / "seed" / "service_events.jsonl")
    tasks = records(bundle, "seed/maintenance_tasks.jsonl")
    assets = {a["id"]: a for a in seeded(bundle)}
    assets |= records(bundle, "osm/assets.jsonl")
    snapshot = datetime.date.fromisoformat(DATE)
    with_task = {task["type_id"] for task in tasks.values()}

    assert len({event["asset_id"] for event in events}) == len(events)
    rate = inputs()["bench"]["service_event_fill_rate"]
    assert rate == 0.7
    assert len(events) == math.floor(rate * len(assets) + 0.5)
    for event in events:
        asset = assets[event["asset_id"]]
        # COBie v3 Event: a start, an optional end, a performer. No outcome.
        assert set(event) == {
            "id",
            "record_provenance",
            "asset_id",
            "task_id",
            "name",
            "start",
            "end",
            "performer",
        }
        assert event["end"] is None
        assert event["performer"]["value"] == inputs()["bench"]["service_performer"]
        start = datetime.date.fromisoformat(event["start"]["value"])
        assert start <= snapshot
        if asset["type_id"] in with_task:
            task = tasks[event["task_id"]]
            assert task["type_id"] == asset["type_id"]
            assert (snapshot - start).days < int(task["frequency"]["value"]) * 31
        else:
            # No task for the type: within the default one-year interval.
            assert event["task_id"] is None
            assert (snapshot - start).days < 365
        installed = asset["attributes"].get("installation_date")
        if installed:
            assert start >= datetime.date.fromisoformat(installed["value"])
    assert {event["task_id"] is None for event in events} == {True, False}
    # Sourced fire assets get events too, in the seed half.
    assert any(event["asset_id"].startswith("osm:asset:") for event in events)


def test_generated_records_are_seed_and_every_value_is_inferred_with_its_basis(
    bundle: Path,
) -> None:
    inferred = 0

    def check(node: Any) -> None:
        nonlocal inferred
        if isinstance(node, dict):
            if "provenance" in node:
                assert node["provenance"] == "inferred", node
                assert node["basis"], node
                inferred += 1
            for key, child in node.items():
                if key != "clearance":
                    check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    for part in ("asset_types", "assets", "maintenance_tasks", "service_events"):
        for record in read_jsonl(bundle / "seed" / f"{part}.jsonl"):
            assert record["record_provenance"] == "seed"
            check(record)
    assert inferred > 0
    # Nothing generated reaches the OSM half.
    for part in ("asset_types", "assets"):
        for record in read_jsonl(bundle / "osm" / f"{part}.jsonl"):
            assert record["record_provenance"] == "sourced"
    assert not (bundle / "osm" / "maintenance_tasks.jsonl").exists()
    assert not (bundle / "osm" / "service_events.jsonl").exists()


def test_the_fire_assets_in_the_pull_are_sourced(bundle: Path) -> None:
    assets = records(bundle, "osm/assets.jsonl")
    types = records(bundle, "osm/asset_types.jsonl")

    assert sorted(assets) == [
        "osm:asset:node/1201",
        "osm:asset:node/1202",
        "osm:asset:node/1203",
    ]
    extinguisher = assets["osm:asset:node/1201"]
    assert extinguisher["source"] == "node/1201"
    assert extinguisher["type_id"] == "osm:type:fire_extinguisher"
    assert extinguisher["position"]["provenance"] == "sourced"
    # In the corridor on its level; over 1002 but on another level, so in the building;
    # a hydrant in no building, so on the site.
    assert extinguisher["space_id"] == "osm:way/1006"
    assert assets["osm:asset:node/1202"]["space_id"] == "osm:way/100"
    assert assets["osm:asset:node/1202"]["attributes"]["fire_extinguisher:type"] == {
        "value": "powder",
        "provenance": "sourced",
        "rule": None,
        "basis": None,
    }
    assert assets["osm:asset:node/1203"]["space_id"] == "site:mini-campus"
    assert sorted(types) == ["osm:type:fire_extinguisher", "osm:type:fire_hydrant"]
    kind = types["osm:type:fire_extinguisher"]
    assert kind["record_provenance"] == "sourced"
    assert kind["omniclass"]["provenance"] == "derived"
    assert kind["manufacturer"] is None
    assert manifest(bundle)["inventory"]["sourced"]["dropped_outside_site"] == 1


def test_every_fire_type_is_filed_where_a_visitor_sees_it() -> None:
    for value, kind in fire.TYPES.items():
        cleared = type_clearance(kind.omniclass)
        assert kind.omniclass.startswith("23-65 70"), value
        assert cleared.persona.value == "visitor", value
        assert not cleared.unrecognised_category, value


def test_the_attribute_catalogue_is_derived_per_snapshot(bundle: Path) -> None:
    catalogue = manifest(bundle)["inventory"]["attribute_catalogue"]

    assert catalogue["osm:type:fire_extinguisher"] == ["fire_extinguisher:type"]
    assert catalogue["osm:type:fire_hydrant"] == ["fire_hydrant:type"]
    assert catalogue["seed:type:light-fixture-b1"] == [
        "installation_date",
        "serial_number",
        "warranty_start_date",
    ]
    assert set(catalogue) == {
        asset["type_id"]
        for asset in [*seeded(bundle), *read_jsonl(bundle / "osm" / "assets.jsonl")]
    }


def visible(bundle: Path, half: str, part: str, key: str) -> dict[str, set[str]]:
    sets: dict[str, set[str]] = {"visitor": set(), "staff": set(), "maintenance": set()}
    for entry in read_jsonl(bundle / half / f"{part}.jsonl"):
        sets[entry["persona"]].add(entry[key])
    return sets


def test_an_asset_is_visible_when_its_type_is_cleared_and_its_space_is_visible(
    bundle: Path,
) -> None:
    spaces = visible(bundle, "osm", "visible_spaces", "space_id")
    seen = visible(bundle, "seed", "visible_assets", "asset_id")
    sourced = visible(bundle, "osm", "visible_assets", "asset_id")
    types = {
        **records(bundle, "seed/asset_types.jsonl"),
        **records(bundle, "osm/asset_types.jsonl"),
    }
    order = ["visitor", "staff", "maintenance"]

    every = [*seeded(bundle), *read_jsonl(bundle / "osm" / "assets.jsonl")]
    for persona in order:
        for asset in every:
            cleared = order.index(types[asset["type_id"]]["clearance"]["persona"])
            expected = cleared <= order.index(persona) and asset["space_id"] in spaces[persona]
            assert (asset["id"] in seen[persona] | sourced[persona]) == expected, (persona, asset)

    # The extinguisher in the corridor is there for a visitor. A light in the
    # auditorium is hidden by its room, behind a staff door. An air terminal box in
    # the office is hidden by its type, in a room staff can see.
    assert "osm:asset:node/1201" in sourced["visitor"]
    assert "seed:asset:osm:way/1002:light-fixture-b1:1" in seen["staff"]
    assert "seed:asset:osm:way/1002:light-fixture-b1:1" not in seen["visitor"]
    assert "osm:way/1005" in spaces["staff"]
    assert "seed:asset:osm:way/1005:vav-200-mm:1" in seen["maintenance"]
    assert "seed:asset:osm:way/1005:vav-200-mm:1" not in seen["staff"]
    # The sets nest.
    assert seen["visitor"] <= seen["staff"] <= seen["maintenance"]
    assert sourced["visitor"] <= sourced["staff"] <= sourced["maintenance"]


def test_a_type_is_visible_only_through_a_visible_asset(bundle: Path) -> None:
    seen = visible(bundle, "seed", "visible_assets", "asset_id")
    types = visible(bundle, "seed", "visible_asset_types", "type_id")
    assets = {a["id"]: a for a in seeded(bundle)}

    for persona, ids in types.items():
        assert ids == {assets[asset_id]["type_id"] for asset_id in seen[persona]}
    assert "seed:type:vav-200-mm" not in types["staff"]
    assert "seed:type:vav-200-mm" in types["maintenance"]
