"""The seed inventory: equipment for the demo venue, generated, and saying so.

Pinned inputs in, the same inventory out. The inputs are the snapshot's structure
(itself pinned by the pull), the snapshot date, and inventory.json beside this
file, which holds the fixed seed, the asset types, the test bench's space
categories with their equipment mixes and fill rates, and which room types map to
which category. The manifest records the file's sha256 and the seed.

Every record made here is seed, and every value on it is inferred with its basis.
It is written to the seed half of the bundle and never to the OSM half. The rules,
in order:

    room-category@1      a room-tier space takes the category of the first mapping
                         whose tags it carries. An unmapped space gets nothing.
    count-by-area@1      per item of the category's mix: the room's area over the
                         item's square metres, rounded, between its min and max. A
                         fixed item (no square metres) is its min. A room without
                         an outline counts with the category's typical area.
    layout-in-outline@1  where each one stands (rules/layout.py). A room without an
                         outline gets its count and no placement.
    door-asset@1         a door leaf for each sourced door an edge goes through,
                         in the mapped space it opens into (a room before a
                         corridor, then by id). The asset is bound to the door; the
                         edge stays the door's, and no asset is ever an edge.
    fill-by-rate@1       serial and installation date, each on the share of a
                         type's assets the bench measured, picked by a hash of the
                         seed and the asset id. The rest are gaps.
    install-date@1       between 30 days and nine tenths of the type's expected
                         life before the snapshot.
    serial@1             the type's model and eight hex digits of that hash.
    latest-service@1     on the chosen share of all assets, generated and
                         sourced: one event for the type's most frequent task,
                         within one interval of it before the snapshot and never
                         before installation.

The service-event share is chosen here, not measured: 0.7. A bench that records
events at all records them on most of its assets, but not all; seven in ten keeps
the gap visible without making "when was this last serviced" mostly unanswerable.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from spatial_core import clearance
from spatial_core.records import (
    Asset,
    AssetType,
    Edge,
    MaintenanceTask,
    Position,
    RecordProvenance,
    ServiceEvent,
    Space,
    Value,
    ValueProvenance,
)
from spatial_ingest.osm import OsmData
from spatial_ingest.rules import fire, layout, tiers

GENERATOR = "seed-inventory@1"
CATEGORY_RULE = "room-category@1"
COUNT_RULE = "count-by-area@1"
DOOR_RULE = "door-asset@1"
FILL_RULE = "fill-by-rate@1"
INSTALL_RULE = "install-date@1"
SERIAL_RULE = "serial@1"
SERVICE_RULE = "latest-service@1"
RULES = (
    GENERATOR,
    CATEGORY_RULE,
    COUNT_RULE,
    layout.RULE,
    DOOR_RULE,
    FILL_RULE,
    INSTALL_RULE,
    SERIAL_RULE,
    SERVICE_RULE,
    clearance.TYPE_RULE,
)

INVENTORY_FILE = "inventory.json"
SERIAL = "serial_number"
INSTALLED = "installation_date"
# Types whose names could make the demo about surveillance or a specific trade.
# None of these may be generated.
EXCLUDED_WORDS = ("camera", "cctv", "surveillance")

_MONTH_DAYS = 30.4375
_UNIT_DAYS = {"day": 1.0, "week": 7.0, "month": _MONTH_DAYS, "year": 365.25}


class InventoryError(Exception):
    pass


def default_inventory() -> Path:
    return Path(str(resources.files("spatial_ingest.seed") / INVENTORY_FILE))


@dataclass(frozen=True)
class Inputs:
    path: Path
    sha256: str
    data: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.data["seed"])

    def pin(self) -> dict[str, Any]:
        return {"file": self.path.name, "sha256": self.sha256, "seed": self.seed}


def read_inputs(path: Path) -> Inputs:
    raw = path.read_bytes()
    data = json.loads(raw)
    _check(data)
    return Inputs(path, hashlib.sha256(raw).hexdigest(), data)


def _check(data: Mapping[str, Any]) -> None:
    types = data["types"]
    for key, kind in types.items():
        if any(word in f"{key} {kind['name']}".lower() for word in EXCLUDED_WORDS):
            raise InventoryError(f"type {key!r} is not domain-neutral and cannot be generated")
        if kind["layout"] not in (*layout.LAYOUTS, None):
            raise InventoryError(f"type {key!r} has an unknown layout {kind['layout']!r}")
    if data["door_type"] not in types:
        raise InventoryError(f"the door type {data['door_type']!r} is not a type")
    for name, category in data["categories"].items():
        for item in category["mix"]:
            if item["type"] not in types:
                raise InventoryError(f"category {name!r} draws unknown type {item['type']!r}")
            if types[item["type"]]["layout"] is None:
                raise InventoryError(
                    f"category {name!r} draws {item['type']!r}, which has no layout"
                )
    for mapping in data["room_categories"]:
        if mapping["category"] not in data["categories"]:
            raise InventoryError(f"a room mapping names unknown category {mapping['category']!r}")


@dataclass(frozen=True)
class Inventory:
    types: tuple[AssetType, ...]
    assets: tuple[Asset, ...]
    tasks: tuple[MaintenanceTask, ...]
    events: tuple[ServiceEvent, ...]
    counts: dict[str, Any]


def _inferred(value: object, basis: str) -> Value:
    return Value(value=str(value), provenance=ValueProvenance.INFERRED, basis=basis)


def _hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def _rng(*parts: object) -> random.Random:
    return random.Random(_hash(*parts))


def category_of(space: Space, mappings: Sequence[Mapping[str, Any]]) -> str | None:
    """room-category: the first mapping whose tags the space carries."""
    if space.tier.value != tiers.ROOM:
        return None
    tags = {key: value.value for key, value in space.attributes.items()}
    for mapping in mappings:
        if all(tags.get(key) == value for key, value in mapping["tags"].items()):
            return str(mapping["category"])
    return None


def count_for(item: Mapping[str, Any], area: float) -> int:
    """count-by-area: the area over the item's square metres, rounded, within min and max."""
    low, high = int(item["min"]), int(item["max"])
    per = item["m2_per_item"]
    if per is None:
        return low
    return max(low, min(high, math.floor(area / float(per) + 0.5)))


class _Generator:
    def __init__(
        self,
        inputs: Inputs,
        snapshot: str,
        spaces: Sequence[Space],
        edges: Sequence[Edge],
        osm: OsmData,
        sourced: fire.FireAssets,
    ) -> None:
        self.data = inputs.data
        self.seed = inputs.seed
        self.date = datetime.date.fromisoformat(snapshot)
        self.spaces = {space.id: space for space in spaces}
        self.edges = edges
        self.osm = osm
        self.sourced = sourced
        self.counts: Counter[str] = Counter()
        self._by_category: Counter[str] = Counter()
        self.categories: dict[str, str] = {}
        for space in spaces:
            category = category_of(space, self.data["room_categories"])
            if category is not None:
                self.categories[space.id] = category
            elif space.tier.value == tiers.ROOM:
                self.counts["rooms_unmapped"] += 1

    def _basis(self, detail: str) -> str:
        return f"{GENERATOR} seed {self.seed}: {detail}"

    def generate(self) -> Inventory:
        assets = [*self._room_assets(), *self._door_assets()]
        assets = self._fill(assets)
        used = sorted({asset.type_id for asset in assets})
        types = [self._type(type_id) for type_id in used]
        tasks = self._tasks(used)
        tasks_by_type: dict[str, list[MaintenanceTask]] = defaultdict(list)
        for task in tasks:
            tasks_by_type[task.type_id].append(task)
        events = self._events([*self.sourced.assets, *assets], tasks_by_type)

        self.counts["rooms_mapped"] = len(self.categories)
        counts: dict[str, Any] = {
            "asset_types": len(types),
            "assets": len(assets),
            "assets_by_category": dict(sorted(self._by_category.items())),
            "door_assets": sum(1 for asset in assets if asset.door is not None),
            "maintenance_tasks": len(tasks),
            "service_events": len(events),
            "rooms_mapped": self.counts["rooms_mapped"],
            "rooms_unmapped": self.counts["rooms_unmapped"],
            "rooms_without_outline": self.counts["rooms_without_outline"],
            "assets_without_position": sum(1 for asset in assets if asset.position is None),
            "fill": self._fill_counts(assets, len(events), len(self.sourced.assets)),
        }
        return Inventory(
            tuple(types),
            tuple(sorted(assets, key=lambda asset: asset.id)),
            tuple(tasks),
            tuple(sorted(events, key=lambda event: event.id)),
            counts,
        )

    # The assets.

    def _room_assets(self) -> list[Asset]:
        assets = []
        for space_id in sorted(self.categories):
            space = self.spaces[space_id]
            name = self.categories[space_id]
            category = self.data["categories"][name]
            area = layout.area_m2(space.outline)
            if area is None:
                self.counts["rooms_without_outline"] += 1
                area_basis = f"typical area {category['typical_area_m2']} m2 of {name}"
                area = float(category["typical_area_m2"])
            else:
                area_basis = f"outline area {area:.0f} m2"
            for item in category["mix"]:
                key = item["type"]
                count = count_for(item, area)
                kind = self.data["types"][key]
                points = layout.place(space.outline, count, kind["layout"])
                for index in range(count):
                    number = index + 1
                    basis = self._basis(
                        f"{CATEGORY_RULE} {name}, {COUNT_RULE} {count} over {area_basis}"
                    )
                    position = None
                    if points is not None:
                        lon, lat = points[index]
                        position = Position(
                            lon=lon,
                            lat=lat,
                            provenance=ValueProvenance.INFERRED,
                            basis=self._basis(f"{layout.RULE} {kind['layout']} in {space_id}"),
                        )
                    assets.append(
                        Asset(
                            id=f"seed:asset:{space_id}:{key}:{number}",
                            record_provenance=RecordProvenance.SEED,
                            source=None,
                            type_id=_type_id(key),
                            space_id=space_id,
                            name=_inferred(f"{kind['name']} {number}", basis),
                            position=position,
                        )
                    )
                self._by_category[name] += count
        return assets

    def _door_assets(self) -> list[Asset]:
        joined: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            joined[edge.door].add(edge.to_id)
            if edge.from_id is not None:
                joined[edge.door].add(edge.from_id)
        key = self.data["door_type"]
        name = self.data["types"][key]["name"]
        assets = []
        for door in sorted(joined):
            mapped = [space_id for space_id in joined[door] if space_id in self.categories]
            if not mapped:
                continue
            space_id = min(mapped, key=lambda s: (not self._is_room(s), s))
            node = int(door.split("/")[1])
            lon, lat = self.osm.locations[node]
            basis = self._basis(f"{DOOR_RULE} bound to {door}, opening into {space_id}")
            assets.append(
                Asset(
                    id=f"seed:asset:{door}:{key}",
                    record_provenance=RecordProvenance.SEED,
                    source=None,
                    type_id=_type_id(key),
                    space_id=space_id,
                    name=_inferred(name, basis),
                    door=door,
                    position=Position(
                        lon=lon,
                        lat=lat,
                        provenance=ValueProvenance.INFERRED,
                        basis=self._basis(f"{DOOR_RULE} at the position of {door}"),
                    ),
                )
            )
        return assets

    def _is_room(self, space_id: str) -> bool:
        indoor = self.spaces[space_id].attributes.get("indoor")
        return indoor is not None and indoor.value == "room"

    # The attributes, with the bench's gaps.

    def _chosen(self, ids: Iterable[str], rate: float, purpose: str) -> set[str]:
        """fill-by-rate: round(rate * n) of the ids, the same ones for the same seed."""
        ranked = sorted(ids, key=lambda i: _hash(self.seed, purpose, i))
        return set(ranked[: math.floor(rate * len(ranked) + 0.5)])

    def _fill(self, assets: list[Asset]) -> list[Asset]:
        rates = self.data["bench"]["fill_rates"]
        by_type: dict[str, list[str]] = defaultdict(list)
        for asset in assets:
            by_type[asset.type_id].append(asset.id)
        with_serial: set[str] = set()
        with_date: set[str] = set()
        for ids in by_type.values():
            with_serial |= self._chosen(ids, float(rates[SERIAL]), SERIAL)
            with_date |= self._chosen(ids, float(rates[INSTALLED]), INSTALLED)

        filled = []
        for asset in assets:
            kind = self.data["types"][_type_key(asset.type_id)]
            attributes: dict[str, Value] = {}
            if asset.id in with_serial:
                digits = _hash(self.seed, SERIAL_RULE, asset.id)[:8].upper()
                attributes[SERIAL] = _inferred(
                    f"{kind['model']}-{digits}", self._basis(f"{SERIAL_RULE} after {FILL_RULE}")
                )
            if asset.id in with_date:
                attributes[INSTALLED] = _inferred(
                    self._installed(asset.id, kind).isoformat(),
                    self._basis(f"{INSTALL_RULE} within {kind['expected_life_years']} years"),
                )
            filled.append(asset.model_copy(update={"attributes": attributes}))
        return filled

    def _installed(self, asset_id: str, kind: Mapping[str, Any]) -> datetime.date:
        longest = max(31, int(float(kind["expected_life_years"]) * 365.25 * 0.9))
        return self.date - datetime.timedelta(
            days=_rng(self.seed, INSTALL_RULE, asset_id).randrange(30, longest)
        )

    def _fill_counts(self, assets: Sequence[Asset], events: int, sourced: int) -> dict[str, Any]:
        def share(count: int, total: int) -> float:
            return round(count / total, 4) if total else 0.0

        rates = self.data["bench"]["fill_rates"]
        result: dict[str, Any] = {}
        for key in (SERIAL, INSTALLED):
            count = sum(1 for asset in assets if key in asset.attributes)
            result[key] = {"bench": rates[key], "generated": share(count, len(assets))}
        result["service_event"] = {
            "chosen": self.data["bench"]["service_event_fill_rate"],
            "generated": share(events, len(assets) + sourced),
        }
        return result

    # The types, their tasks and the latest service.

    def _type(self, type_id: str) -> AssetType:
        key = _type_key(type_id)
        kind = self.data["types"][key]
        basis = self._basis(f"types.{key} in {INVENTORY_FILE}")
        code = _inferred(kind["omniclass"], basis)
        return AssetType(
            id=type_id,
            record_provenance=RecordProvenance.SEED,
            name=_inferred(kind["name"], basis),
            omniclass=code,
            manufacturer=_inferred(kind["manufacturer"], basis),
            model=_inferred(kind["model"], basis),
            warranty_parts_years=_inferred(kind["warranty_parts_years"], basis),
            warranty_labor_years=_inferred(kind["warranty_labor_years"], basis),
            expected_life_years=_inferred(kind["expected_life_years"], basis),
            clearance=clearance.type_clearance(code.value),
        )

    def _tasks(self, used: Sequence[str]) -> list[MaintenanceTask]:
        jobs: list[tuple[str, str, Mapping[str, Any]]] = []
        for type_id in used:
            key = _type_key(type_id)
            jobs += [(type_id, f"types.{key}.jobs", job) for job in self.data["types"][key]["jobs"]]
        for kind in self.sourced.types:
            value = kind.id.removeprefix("osm:type:")
            for job in self.data["sourced_type_jobs"].get(value, []):
                jobs.append((kind.id, f"sourced_type_jobs.{value}", job))

        tasks = []
        numbers: Counter[str] = Counter()
        for type_id, where, job in jobs:
            numbers[type_id] += 1
            number = numbers[type_id]
            basis = self._basis(f"{where}[{number - 1}] in {INVENTORY_FILE}")
            resources = job.get("resource_names")
            tasks.append(
                MaintenanceTask(
                    id=f"seed:task:{type_id}:{number}",
                    record_provenance=RecordProvenance.SEED,
                    type_id=type_id,
                    name=_inferred(job["name"], basis),
                    category=_inferred(job["category"], basis),
                    status=_inferred("Scheduled", basis),
                    description=_inferred(job["description"], basis),
                    duration=_inferred(job["duration"], basis),
                    duration_unit=_inferred(job["duration_unit"], basis),
                    start=_inferred(0, basis),
                    task_start_unit=_inferred(job["frequency_unit"], basis),
                    frequency=_inferred(job["frequency"], basis),
                    frequency_unit=_inferred(job["frequency_unit"], basis),
                    task_number=_inferred(number, basis),
                    resource_names=_inferred(resources, basis) if resources else None,
                )
            )
        return tasks

    def _events(
        self, assets: Sequence[Asset], tasks: Mapping[str, Sequence[MaintenanceTask]]
    ) -> list[ServiceEvent]:
        rate = float(self.data["bench"]["service_event_fill_rate"])
        serviceable = [asset for asset in assets if tasks.get(asset.type_id)]
        chosen = self._chosen((asset.id for asset in serviceable), rate, SERVICE_RULE)
        events = []
        for asset in serviceable:
            if asset.id not in chosen:
                continue
            task = min(tasks[asset.type_id], key=lambda t: (_interval_days(t), t.id))
            interval = max(1, int(_interval_days(task)))
            date = self.date - datetime.timedelta(
                days=_rng(self.seed, SERVICE_RULE, asset.id).randrange(interval)
            )
            installed = asset.attributes.get(INSTALLED)
            if installed is not None:
                date = max(date, datetime.date.fromisoformat(installed.value))
            basis = self._basis(f"{SERVICE_RULE} for {task.id}, within one interval")
            events.append(
                ServiceEvent(
                    id=f"seed:event:{asset.id}",
                    record_provenance=RecordProvenance.SEED,
                    asset_id=asset.id,
                    task_id=task.id,
                    name=_inferred(task.name.value, basis),
                    category=_inferred("Service", basis),
                    date=_inferred(date.isoformat(), basis),
                    description=_inferred(f"Carried out: {task.description.value}", basis),
                )
            )
        return events


def _type_id(key: str) -> str:
    return f"seed:type:{key}"


def _type_key(type_id: str) -> str:
    return type_id.removeprefix("seed:type:")


def _interval_days(task: MaintenanceTask) -> float:
    return float(task.frequency.value) * _UNIT_DAYS[task.frequency_unit.value]


def generate(
    *,
    inputs: Inputs,
    snapshot: str,
    spaces: Sequence[Space],
    edges: Sequence[Edge],
    osm: OsmData,
    sourced: fire.FireAssets,
) -> Inventory:
    return _Generator(inputs, snapshot, spaces, edges, osm, sourced).generate()
