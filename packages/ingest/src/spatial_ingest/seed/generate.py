"""The seed inventory: equipment for the demo venue, generated, and saying so.

Pinned inputs in, the same inventory out. The inputs are the snapshot's structure
(itself pinned by the pull), the snapshot date, and inventory.json beside this
file. That file is adapted from the test bench's COBie handover (the buildingSMART
Medical-Dental Clinic sample, CC BY 4.0) and names it: the type library, the
bench's space categories with their equipment mixes, its measured fill rates and
install-date spread, and which room types map to which category. It also holds
the fixed seed. The manifest records the file's sha256 and the seed.

Only domain-neutral types are generated: lights, AC units, air terminals, doors,
valves, sanitary fittings and electrical panels. Each is the bench's most
installed type with its Table 23 code. What the bench leaves blank stays blank.

Every record made here is seed, and every value on it is inferred with its basis.
It is written to the seed half of the bundle and never to the OSM half. The rules,
in order:

    room-category@1      a room-tier space takes the category of the first mapping
                         whose tags it carries. An unmapped space gets nothing.
    count-by-area@2      per item of the category's mix: the room's area over the
                         item's square metres on the bench, rounded, at least its
                         min. A room without an outline counts with the category's
                         typical area.
    layout-in-outline@1  where each one stands (rules/layout.py). A room without an
                         outline gets its count and no placement.
    door-asset@1         a door for each sourced door an edge goes through, in the
                         mapped space it opens into (a room before a corridor, then
                         by id). The asset is bound to the door; the edge stays the
                         door's, and no asset is ever an edge.
    manufacturer-from-email@1
                         the bench names a manufacturer by a contact address. The
                         name is the label before the top-level domain, capitalised
                         (sales@lithonia.com is Lithonia). A placeholder address
                         reads not recorded.
    fill-by-rate@1       serial, installation date and warranty start, each on the
                         share of a type's assets the bench measured, picked by a
                         hash of the seed and the asset id. The rest are gaps.
    install-date@2       as long before the snapshot as a draw from the bench's
                         spread of installation dates before its handover, and
                         never before the building's sourced `start_date` where
                         OSM has one.
    warranty-start@1     the installation date: on the bench the two agree on
                         every component.
    serial@1             the type's model and eight hex digits of that hash.
    latest-service@2     on the chosen share of all assets, generated and
                         sourced: one event, within one interval of the type's
                         most frequent task before the snapshot, or within a year
                         for a type with no task, and never before installation.

The service-event share and the performer are chosen here, not measured, since the
bench has no service events. The share is 0.7. A bench that records
events at all records them on most of its assets, but not all; seven in ten keeps
the gap visible without making "when was this last serviced" mostly unanswerable.
The bench has no task for any generated type, so the two that carry one (the door
and the panelboard) have it authored in inventory.json, and the file says so.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import random
import re
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
COUNT_RULE = "count-by-area@2"
DOOR_RULE = "door-asset@1"
MANUFACTURER_RULE = "manufacturer-from-email@1"
FILL_RULE = "fill-by-rate@1"
INSTALL_RULE = "install-date@2"
SERIAL_RULE = "serial@1"
WARRANTY_RULE = "warranty-start@1"
SERVICE_RULE = "latest-service@2"
RULES = (
    GENERATOR,
    CATEGORY_RULE,
    COUNT_RULE,
    layout.RULE,
    DOOR_RULE,
    MANUFACTURER_RULE,
    FILL_RULE,
    INSTALL_RULE,
    WARRANTY_RULE,
    SERIAL_RULE,
    SERVICE_RULE,
    clearance.TYPE_RULE,
)

INVENTORY_FILE = "inventory.json"
SERIAL = "serial_number"
INSTALLED = "installation_date"
WARRANTY = "warranty_start_date"
# What fill-by-rate fills, each at the bench's rate.
FILLED = (SERIAL, INSTALLED, WARRANTY)
# The only families a generated type may belong to.
NEUTRAL_FAMILIES = (
    "lights",
    "AC units",
    "air terminals",
    "doors",
    "valves",
    "sanitary fittings",
    "electrical panels",
)
# Types whose names could make the demo about surveillance. None may be generated.
EXCLUDED_WORDS = ("camera", "cctv", "surveillance")
# The default service-due interval, for a type with no task.
_YEAR_DAYS = 365

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
        if kind["family"] not in NEUTRAL_FAMILIES:
            raise InventoryError(
                f"type {key!r} is of family {kind['family']!r}, which is not domain-neutral"
            )
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
    """count-by-area: the area over the item's square metres, rounded, at least its min."""
    return max(int(item["min"]), math.floor(area / float(item["m2_per_item"]) + 0.5))


def manufacturer_from_email(email: str | None, placeholders: Sequence[str]) -> str | None:
    """manufacturer-from-email: the label before the top-level domain, capitalised.

    None, for not recorded, where the bench has no address or a placeholder one.
    """
    if not email or "@" not in email or email in placeholders:
        return None
    labels = email.rsplit("@", 1)[1].split(".")
    return labels[-2].capitalize() if len(labels) >= 2 else None


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
                            ancestor_path=fire.asset_path(space),
                            name=_inferred(f"{kind['name']} {number}", basis),
                            position=position,
                        )
                    )
                self._by_category[name] += count
        return assets

    def _door_assets(self) -> list[Asset]:
        joined: dict[str, set[str]] = defaultdict(set)
        for edge in self.edges:
            # A snap is no door, so nothing hangs on it.
            if edge.door is None:
                continue
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
                    ancestor_path=fire.asset_path(self.spaces[space_id]),
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
        chosen: dict[str, set[str]] = {key: set() for key in FILLED}
        for ids in by_type.values():
            for key in FILLED:
                chosen[key] |= self._chosen(ids, float(rates[key]), key)

        filled = []
        for asset in assets:
            kind = self.data["types"][_type_key(asset.type_id)]
            attributes: dict[str, Value] = {}
            if asset.id in chosen[SERIAL]:
                digits = _hash(self.seed, SERIAL_RULE, asset.id)[:8].upper()
                attributes[SERIAL] = _inferred(
                    f"{kind['model']}-{digits}", self._basis(f"{SERIAL_RULE} after {FILL_RULE}")
                )
            if asset.id in chosen[INSTALLED]:
                installed, how = self._installed(asset)
                attributes[INSTALLED] = _inferred(installed.isoformat(), self._basis(how))
                # The bench starts every warranty on the day of installation.
                if asset.id in chosen[WARRANTY]:
                    attributes[WARRANTY] = _inferred(
                        installed.isoformat(),
                        self._basis(f"{WARRANTY_RULE} on the installation date, after {FILL_RULE}"),
                    )
            filled.append(asset.model_copy(update={"attributes": attributes}))
        return filled

    def _installed(self, asset: Asset) -> tuple[datetime.date, str]:
        """install-date: a draw from the bench's spread, never before the building."""
        spread = self.data["bench"]["install_days_before_handover_deciles"]
        at = _rng(self.seed, INSTALL_RULE, asset.id).random() * (len(spread) - 1)
        low = min(int(at), len(spread) - 2)
        days = spread[low] + (spread[low + 1] - spread[low]) * (at - low)
        installed = self.date - datetime.timedelta(days=math.floor(days + 0.5))
        how = f"{INSTALL_RULE} from the bench's install-date spread"
        built = self._built(asset.space_id)
        if built is not None and installed < built:
            return built, f"{how}, held to the building's start_date {built.isoformat()}"
        return installed, how

    def _built(self, space_id: str) -> datetime.date | None:
        """The sourced `start_date` of the building a space is in, where it has one."""
        space = self.spaces[space_id]
        within = [space.id, *(ancestor.id for ancestor in reversed(space.ancestor_path))]
        for candidate in within:
            enclosing = self.spaces.get(candidate)
            if enclosing is None or enclosing.tier.value != tiers.BUILDING:
                continue
            return start_date(_tags(self.osm, enclosing.source).get("start_date"))
        return None

    def _fill_counts(self, assets: Sequence[Asset], events: int, sourced: int) -> dict[str, Any]:
        def share(count: int, total: int) -> float:
            return round(count / total, 4) if total else 0.0

        rates = self.data["bench"]["fill_rates"]
        result: dict[str, Any] = {}
        for key in FILLED:
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
        basis = self._basis(f"types.{key} in {INVENTORY_FILE}, after {self.data['source']['file']}")
        code = _inferred(kind["omniclass"], basis)
        maker = manufacturer_from_email(
            kind["manufacturer_email"], self.data["placeholder_manufacturers"]
        )

        def given(name: str) -> Value | None:
            return None if kind[name] is None else _inferred(kind[name], basis)

        return AssetType(
            id=type_id,
            record_provenance=RecordProvenance.SEED,
            name=_inferred(kind["name"], basis),
            omniclass=code,
            manufacturer=None
            if maker is None
            else _inferred(
                maker, f"{basis}, {MANUFACTURER_RULE} over {kind['manufacturer_email']}"
            ),
            model=given("model"),
            warranty_parts_years=given("warranty_parts_years"),
            warranty_labor_years=given("warranty_labor_years"),
            expected_life_years=given("expected_life_years"),
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
        bench = self.data["bench"]
        chosen = self._chosen(
            (asset.id for asset in assets), float(bench["service_event_fill_rate"]), SERVICE_RULE
        )
        events = []
        for asset in assets:
            if asset.id not in chosen:
                continue
            planned = tasks.get(asset.type_id)
            if planned:
                task = min(planned, key=lambda t: (_interval_days(t), t.id))
                interval = max(1, int(_interval_days(task)))
                name, task_id = task.name.value, task.id
                basis = self._basis(f"{SERVICE_RULE} for {task.id}, within one interval")
            else:
                interval, name, task_id = _YEAR_DAYS, "Service", None
                basis = self._basis(f"{SERVICE_RULE}, no task for the type, within one year")
            start = self.date - datetime.timedelta(
                days=_rng(self.seed, SERVICE_RULE, asset.id).randrange(interval)
            )
            installed = asset.attributes.get(INSTALLED)
            if installed is not None:
                start = max(start, datetime.date.fromisoformat(installed.value))
            events.append(
                ServiceEvent(
                    id=f"seed:event:{asset.id}",
                    record_provenance=RecordProvenance.SEED,
                    asset_id=asset.id,
                    task_id=task_id,
                    name=_inferred(name, basis),
                    start=_inferred(start.isoformat(), basis),
                    performer=_inferred(
                        bench["service_performer"],
                        self._basis(f"bench.service_performer in {INVENTORY_FILE}"),
                    ),
                )
            )
        return events


_START_DATE = re.compile(r"(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?")


def start_date(value: str | None) -> datetime.date | None:
    """An OSM `start_date` as its first day: a year, a year and month, or a full date.

    Anything else ("~1960", "C19", "before 1900") bounds nothing.
    """
    found = _START_DATE.fullmatch(value.strip()) if value else None
    if not found:
        return None
    year, month, day = (int(part) if part else 1 for part in found.groups())
    try:
        return datetime.date(year, month, day)
    except ValueError:
        return None


def _tags(osm: OsmData, source: str | None) -> Mapping[str, str]:
    kind, _, ref = (source or "").partition("/")
    if not ref.isdigit():
        return {}
    if kind == "way" and int(ref) in osm.ways:
        return osm.ways[int(ref)].tags
    if kind == "relation" and int(ref) in osm.relations:
        return osm.relations[int(ref)].tags
    return {}


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
