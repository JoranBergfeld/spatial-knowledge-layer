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
    """What a room's category drew for it: its assets other than door leaves."""
    return Counter(
        a["type_id"] for a in seeded(bundle) if a["space_id"] == space_id and a["door"] is None
    )


def test_asset_types_are_records_with_their_type_data(bundle: Path) -> None:
    types = records(bundle, "seed/asset_types.jsonl")

    desk = types["seed:type:office-desk"]
    assert desk["record_provenance"] == "seed"
    assert desk["manufacturer"]["value"] == "Demo Office Furniture"
    assert desk["model"]["value"] == "DOF-D160"
    assert desk["omniclass"]["value"] == "23-21 19 13"
    assert desk["warranty_parts_years"]["value"] == "5"
    assert desk["warranty_labor_years"]["value"] == "2"
    assert desk["expected_life_years"]["value"] == "15"
    for kind in types.values():
        cleared = type_clearance(kind["omniclass"]["value"]).model_dump(mode="json")
        assert kind["clearance"] == cleared
    # Only the types some asset uses are written.
    used = {asset["type_id"] for asset in seeded(bundle)}
    assert set(types) == used


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

    assert rooms == {"osm:way/1002", "osm:way/1005", "osm:way/1006", "osm:node/1007"}
    assert set(in_space(bundle, "osm:way/1005")) == {
        "seed:type:office-desk",
        "seed:type:office-chair",
        "seed:type:bookshelf",
        "seed:type:desktop-computer",
    }
    assert set(in_space(bundle, "osm:way/1002")) == {
        "seed:type:auditorium-seat",
        "seed:type:projector",
        "seed:type:whiteboard",
    }
    counts = manifest(bundle)["inventory"]["generated"]
    assert counts["rooms_mapped"] == 4
    assert counts["rooms_unmapped"] == 11


@pytest.mark.parametrize(
    ("item", "area", "count"),
    [
        ({"m2_per_item": 8, "min": 1, "max": 12}, 40.0, 5),
        ({"m2_per_item": 8, "min": 1, "max": 12}, 2.0, 1),
        ({"m2_per_item": 8, "min": 0, "max": 12}, 2.0, 0),
        ({"m2_per_item": 8, "min": 1, "max": 12}, 1000.0, 12),
        ({"m2_per_item": 8, "min": 1, "max": 12}, 20.0, 3),
        ({"m2_per_item": None, "min": 1, "max": 1}, 1000.0, 1),
    ],
)
def test_counts_scale_with_area_within_the_categorys_bounds(
    item: dict[str, Any], area: float, count: int
) -> None:
    assert generate.count_for(item, area) == count


def test_a_room_without_an_outline_gets_a_count_and_no_placement(bundle: Path) -> None:
    office = inputs()["categories"]["office"]
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
    names = [kind["name"].lower() for kind in inputs()["types"].values()]
    assert not [name for name in names if "camera" in name]
    published = (bundle / "seed" / "asset_types.jsonl").read_text().lower()
    assert "camera" not in published


def test_an_inventory_with_a_camera_type_is_refused(
    venue_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data = inputs()
    data["types"]["dome-camera"] = {**data["types"]["projector"], "name": "Dome camera"}
    data["categories"]["circulation"]["mix"].append(
        {"type": "dome-camera", "m2_per_item": 50, "min": 1, "max": 4}
    )
    path = write_inputs(tmp_path / "cameras.json", data)

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
    assert doors["node/1102"]["type_id"] == "seed:type:door-leaf"
    assert not (bundle / "seed" / "edges.jsonl").exists()
    # The edges are what they were: the doors' own, derived from OSM.
    assert {edge["provenance"] for edge in edges} == {"derived"}


def test_gaps_mirror_the_bench_fill_rates_per_type(bundle: Path) -> None:
    rates = inputs()["bench"]["fill_rates"]
    by_type: dict[str, list[dict[str, Any]]] = {}
    for asset in seeded(bundle):
        by_type.setdefault(asset["type_id"], []).append(asset)

    for assets in by_type.values():
        for key in ("serial_number", "installation_date"):
            filled = sum(1 for asset in assets if key in asset["attributes"])
            assert filled == math.floor(rates[key] * len(assets) + 0.5)
    fill = manifest(bundle)["inventory"]["generated"]["fill"]
    assert fill["serial_number"]["bench"] == rates["serial_number"]
    assert abs(fill["serial_number"]["generated"] - rates["serial_number"]) < 0.02
    assert abs(fill["installation_date"]["generated"] - rates["installation_date"]) < 0.02


def test_serials_and_install_dates_follow_their_rules(bundle: Path) -> None:
    types = inputs()["types"]
    snapshot = datetime.date.fromisoformat(DATE)

    for asset in seeded(bundle):
        kind = types[asset["type_id"].removeprefix("seed:type:")]
        attributes = asset["attributes"]
        if "serial_number" in attributes:
            serial = attributes["serial_number"]["value"]
            assert serial.startswith(kind["model"] + "-")
            assert len(serial) == len(kind["model"]) + 9
        if "installation_date" in attributes:
            installed = datetime.date.fromisoformat(attributes["installation_date"]["value"])
            life = datetime.timedelta(days=kind["expected_life_years"] * 365.25)
            assert snapshot - life < installed <= snapshot - datetime.timedelta(days=30)


def test_maintenance_tasks_follow_cobie_job(bundle: Path) -> None:
    tasks = records(bundle, "seed/maintenance_tasks.jsonl")

    task = tasks["seed:task:seed:type:office-chair:1"]
    assert task["type_id"] == "seed:type:office-chair"
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
    # Every type in use has its jobs, the sourced fire types too.
    types = {t["id"] for t in read_jsonl(bundle / "seed" / "asset_types.jsonl")}
    types |= {t["id"] for t in read_jsonl(bundle / "osm" / "asset_types.jsonl")}
    assert {task["type_id"] for task in tasks.values()} == types


def test_each_asset_has_at_most_one_latest_service_event(bundle: Path) -> None:
    events = read_jsonl(bundle / "seed" / "service_events.jsonl")
    tasks = records(bundle, "seed/maintenance_tasks.jsonl")
    assets = {a["id"]: a for a in seeded(bundle)}
    assets |= records(bundle, "osm/assets.jsonl")
    snapshot = datetime.date.fromisoformat(DATE)

    assert len({event["asset_id"] for event in events}) == len(events)
    rate = inputs()["bench"]["service_event_fill_rate"]
    assert len(events) == math.floor(rate * len(assets) + 0.5)
    for event in events:
        asset = assets[event["asset_id"]]
        task = tasks[event["task_id"]]
        assert task["type_id"] == asset["type_id"]
        assert event["category"]["value"] == "Service"
        date = datetime.date.fromisoformat(event["date"]["value"])
        assert date <= snapshot
        assert (snapshot - date).days < int(task["frequency"]["value"]) * 31
        installed = asset["attributes"].get("installation_date")
        if installed:
            assert date >= datetime.date.fromisoformat(installed["value"])
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


def test_the_attribute_catalogue_is_derived_per_snapshot(bundle: Path) -> None:
    catalogue = manifest(bundle)["inventory"]["attribute_catalogue"]

    assert catalogue["osm:type:fire_extinguisher"] == ["fire_extinguisher:type"]
    assert catalogue["osm:type:fire_hydrant"] == ["fire_hydrant:type"]
    assert catalogue["seed:type:office-desk"] == ["installation_date", "serial_number"]
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

    # The extinguisher in the corridor is there for a visitor; the projector in the
    # auditorium is staff equipment behind a staff door.
    assert "osm:asset:node/1201" in sourced["visitor"]
    assert "seed:asset:osm:way/1002:projector:1" in seen["staff"]
    assert "seed:asset:osm:way/1002:projector:1" not in seen["visitor"]


def test_a_type_is_visible_only_through_a_visible_asset(bundle: Path) -> None:
    seen = visible(bundle, "seed", "visible_assets", "asset_id")
    types = visible(bundle, "seed", "visible_asset_types", "type_id")
    assets = {a["id"]: a for a in seeded(bundle)}

    for persona, ids in types.items():
        assert ids == {assets[asset_id]["type_id"] for asset_id in seen[persona]}
    assert "seed:type:projector" not in types["visitor"]
    assert "seed:type:projector" in types["staff"]
