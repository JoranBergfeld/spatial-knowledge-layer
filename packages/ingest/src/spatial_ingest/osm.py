"""OSM data in memory, read from either kind of stored pull."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import osmium
from osmium.osm import Node as OsmNode
from osmium.osm import Relation as OsmRelation
from osmium.osm import Way as OsmWay
from shapely import Point, Polygon, unary_union
from shapely.geometry import LineString
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize

Tags = dict[str, str]


@dataclass(frozen=True)
class Way:
    nodes: tuple[int, ...]
    tags: Tags


@dataclass(frozen=True)
class Member:
    type: str
    ref: int
    role: str


@dataclass(frozen=True)
class Relation:
    members: tuple[Member, ...]
    tags: Tags


@dataclass(frozen=True)
class Element:
    """A tagged element with whatever geometry it has."""

    type: str
    id: int
    tags: Tags
    geometry: BaseGeometry | None

    @property
    def ref(self) -> str:
        return f"{self.type}/{self.id}"


@dataclass
class OsmData:
    # node id -> (lon, lat)
    locations: dict[int, tuple[float, float]] = field(default_factory=dict)
    node_tags: dict[int, Tags] = field(default_factory=dict)
    ways: dict[int, Way] = field(default_factory=dict)
    relations: dict[int, Relation] = field(default_factory=dict)

    def way_polygon(self, way_id: int) -> Polygon | None:
        ring = self._ring(way_id)
        if ring is None or len(ring) < 4 or ring[0] != ring[-1]:
            return None
        polygon = Polygon(ring)
        return polygon if polygon.is_valid and not polygon.is_empty else None

    def relation_geometry(self, relation_id: int) -> BaseGeometry | None:
        relation = self.relations[relation_id]
        if relation.tags.get("type") != "multipolygon":
            return None
        outer = self._rings(relation, inner=False)
        if outer.is_empty:
            return None
        geometry = outer.difference(self._rings(relation, inner=True))
        return geometry if geometry.is_valid and not geometry.is_empty else None

    def tagged(self) -> Iterator[Element]:
        """Every element carrying tags, in a fixed order."""
        for node_id in sorted(self.node_tags):
            lon, lat = self.locations[node_id]
            yield Element("node", node_id, self.node_tags[node_id], Point(lon, lat))
        for way_id in sorted(self.ways):
            if self.ways[way_id].tags:
                yield Element("way", way_id, self.ways[way_id].tags, self.way_polygon(way_id))
        for relation_id in sorted(self.relations):
            tags = self.relations[relation_id].tags
            yield Element("relation", relation_id, tags, self.relation_geometry(relation_id))

    def _ring(self, way_id: int) -> list[tuple[float, float]] | None:
        way = self.ways.get(way_id)
        if way is None or any(node not in self.locations for node in way.nodes):
            return None
        return [self.locations[node] for node in way.nodes]

    def _rings(self, relation: Relation, *, inner: bool) -> BaseGeometry:
        # A ring may be split over several ways, so stitch them before making polygons.
        lines = []
        for member in relation.members:
            if member.type != "way" or (member.role == "inner") != inner:
                continue
            ring = self._ring(member.ref)
            if ring is not None and len(ring) >= 2:
                lines.append(LineString(ring))
        return unary_union(list(polygonize(lines)))


def from_overpass_json(raw: bytes) -> OsmData:
    data = OsmData()
    for element in json.loads(raw)["elements"]:
        kind, element_id, tags = element["type"], int(element["id"]), element.get("tags") or {}
        if kind == "node":
            data.locations[element_id] = (float(element["lon"]), float(element["lat"]))
            # `out body; >; out skel` can list a node twice, once with and once without tags.
            if tags:
                data.node_tags[element_id] = dict(tags)
        elif kind == "way":
            data.ways[element_id] = Way(tuple(int(node) for node in element["nodes"]), dict(tags))
        elif kind == "relation":
            members = tuple(
                Member(member["type"], int(member["ref"]), member.get("role", ""))
                for member in element["members"]
            )
            data.relations[element_id] = Relation(members, dict(tags))
    return data


_MEMBER_TYPES = {"n": "node", "w": "way", "r": "relation"}


def from_osm_file(path: Path) -> OsmData:
    """Reads any file libosmium can (.osm.pbf for the extract fallback)."""
    data = OsmData()
    for obj in osmium.FileProcessor(str(path)):
        if not isinstance(obj, OsmNode | OsmWay | OsmRelation):
            continue
        tags = {tag.k: tag.v for tag in obj.tags}
        if isinstance(obj, OsmNode):
            data.locations[obj.id] = (obj.location.lon, obj.location.lat)
            if tags:
                data.node_tags[obj.id] = tags
        elif isinstance(obj, OsmWay):
            data.ways[obj.id] = Way(tuple(node.ref for node in obj.nodes), tags)
        else:
            members = tuple(
                Member(_MEMBER_TYPES[member.type], member.ref, member.role)
                for member in obj.members
            )
            data.relations[obj.id] = Relation(members, tags)
    return data
