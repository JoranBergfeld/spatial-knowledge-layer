"""fire-assets-from-emergency: the fire equipment OSM maps, as sourced assets.

A node tagged `emergency` with one of the values below is an asset of the type
that value names. Its type is derived from the tag, with the OmniClass Table 23
code listed here, and has no manufacturer, model, warranties or expected life,
since OSM says nothing of those. Other `emergency` values (defibrillators,
assembly points, phones) are not fire equipment and are left alone, and so are
fire equipment mapped as ways.

asset-in-space-by-point: an asset read from a point sits in the smallest room-tier
space whose outline covers it, on the asset's level where both have one; failing
that, in the smallest building covering it; failing that, in the site. A point
outside the site is dropped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from shapely import Point, STRtree
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from spatial_core import clearance
from spatial_core.records import (
    Asset,
    AssetType,
    Position,
    RecordProvenance,
    Space,
    Value,
    ValueProvenance,
)
from spatial_ingest.osm import OsmData
from spatial_ingest.rules import levels, tiers

RULE = "fire-assets-from-emergency@1"
SPACE_RULE = "asset-in-space-by-point@1"


@dataclass(frozen=True)
class FireType:
    name: str
    omniclass: str


# `emergency` value -> the type it names.
TYPES: Mapping[str, FireType] = {
    "fire_extinguisher": FireType("Fire extinguisher", "23-29 21 11"),
    "fire_blanket": FireType("Fire blanket", "23-29 21 13"),
    "fire_hose": FireType("Fire hose reel", "23-29 21 15"),
    "fire_alarm_box": FireType("Manual fire alarm call point", "23-29 23 11"),
    "fire_hydrant": FireType("Fire hydrant", "23-29 25 11"),
}

# Sourced tags copied onto a fire asset as attributes.
ATTRIBUTE_KEYS = (
    "fire_extinguisher:type",
    "fire_hydrant:diameter",
    "fire_hydrant:position",
    "fire_hydrant:type",
    "location",
    "manufacturer",
    "model",
    "ref",
)


def type_id(value: str) -> str:
    return f"osm:type:{value}"


def asset_type(value: str) -> AssetType:
    kind = TYPES[value]
    code = Value.derived(kind.omniclass, RULE)
    return AssetType(
        id=type_id(value),
        record_provenance=RecordProvenance.SOURCED,
        name=Value.derived(kind.name, RULE),
        omniclass=code,
        clearance=clearance.type_clearance(code.value),
    )


@dataclass
class _Place:
    space_id: str
    geometry: BaseGeometry
    area: float
    levels: frozenset[str]


class Placer:
    """asset-in-space-by-point over a snapshot's spaces."""

    def __init__(self, spaces: Iterable[Space], site_id: str) -> None:
        self.site_id = site_id
        self._rooms: list[_Place] = []
        self._buildings: list[_Place] = []
        self._site: BaseGeometry | None = None
        for space in spaces:
            geometry = _geometry(space.outline)
            if geometry is None:
                continue
            on = frozenset(space.levels.value.split(";")) if space.levels else frozenset()
            place = _Place(space.id, geometry, geometry.area, on)
            if space.id == site_id:
                self._site = geometry
            elif space.tier.value == tiers.ROOM:
                self._rooms.append(place)
            elif space.tier.value == tiers.BUILDING:
                self._buildings.append(place)
        self._room_index = STRtree([p.geometry for p in self._rooms])
        self._building_index = STRtree([p.geometry for p in self._buildings])

    def space_for(self, point: Point, level: str | None) -> str | None:
        on = frozenset(levels.parse_levels(level)) if level else frozenset()
        rooms = [self._rooms[int(i)] for i in self._room_index.query(point, predicate="intersects")]
        rooms = [room for room in rooms if not on or not room.levels or on & room.levels]
        if rooms:
            return min(rooms, key=lambda p: (p.area, p.space_id)).space_id
        buildings = [
            self._buildings[int(i)]
            for i in self._building_index.query(point, predicate="intersects")
        ]
        if buildings:
            return min(buildings, key=lambda p: (p.area, p.space_id)).space_id
        if self._site is not None and self._site.covers(point):
            return self.site_id
        return None


def _geometry(outline: dict[str, Any] | None) -> BaseGeometry | None:
    if outline is None:
        return None
    geometry = shape(outline)
    return geometry if geometry.is_valid and not geometry.is_empty else None


@dataclass(frozen=True)
class FireAssets:
    types: tuple[AssetType, ...]
    assets: tuple[Asset, ...]
    dropped_outside_site: int


def fire_assets(osm: OsmData, placer: Placer) -> FireAssets:
    assets = []
    used: set[str] = set()
    dropped = 0
    for node_id in sorted(osm.node_tags):
        tags = osm.node_tags[node_id]
        value = tags.get("emergency", "")
        if value not in TYPES:
            continue
        lon, lat = osm.locations[node_id]
        space_id = placer.space_for(Point(lon, lat), tags.get("level"))
        if space_id is None:
            dropped += 1
            continue
        used.add(value)
        assets.append(
            Asset(
                id=f"osm:asset:node/{node_id}",
                record_provenance=RecordProvenance.SOURCED,
                source=f"node/{node_id}",
                type_id=type_id(value),
                space_id=space_id,
                name=Value.derived(TYPES[value].name, RULE),
                position=Position(lon=lon, lat=lat, provenance=ValueProvenance.SOURCED),
                attributes={key: Value.sourced(tags[key]) for key in ATTRIBUTE_KEYS if key in tags},
            )
        )
    types = tuple(asset_type(value) for value in sorted(used))
    return FireAssets(types, tuple(assets), dropped)
