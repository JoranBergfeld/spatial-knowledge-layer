"""The OSM join: a pull becomes one strict containment tree of spaces.

site -> building -> floor -> room, with tiers skipped where the data has none:
a room without a readable level hangs from its building, and a room inside the
site but in no building polygon hangs from the site.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from shapely import STRtree, unary_union
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from spatial_core import circulation, clearance
from spatial_core.records import (
    Ancestor,
    Edge,
    RecordProvenance,
    Space,
    Span,
    Value,
    ValueProvenance,
)
from spatial_ingest.osm import Element, OsmData
from spatial_ingest.rules import doors, labels, levels, names, snap, tiers
from spatial_ingest.venue import Venue

# Sourced tags copied onto a space as attributes. Name, ref and level have fields
# of their own. Anything that could hold a person's name stays out unless the
# name filter has been over it.
ATTRIBUTE_KEYS = (
    "access",
    "amenity",
    "building",
    "building:levels",
    "capacity",
    "description",
    "female",
    "highway",
    "indoor",
    "male",
    "office",
    "room",
    "stairs",
    "unisex",
    "university",
    "wheelchair",
)
_FILTERED_KEYS = {"description"}

# Which space encloses which: a space belongs to the site when its representative
# point lies in a part polygon, and to the smallest building polygon holding that point.
CONTAINMENT_RULE = "containment-by-polygon@1"

# The rules behind the tree, and so behind every part of the structure.
TREE_RULES = (
    tiers.SITE_RULE,
    CONTAINMENT_RULE,
    tiers.RULE,
    tiers.FLOOR_RULE,
    levels.RULE,
)
CLEARANCE_RULES = (clearance.ACCESS_RULE, clearance.DEFAULT_RULE)
EDGE_RULES = (
    doors.RULE,
    *CLEARANCE_RULES,
    clearance.EDGE_RULE,
    circulation.TAGS_RULE,
    snap.RULE,
)
RULES = (*TREE_RULES, labels.RULE, names.RULE, *CLEARANCE_RULES)


class StructureError(Exception):
    pass


@dataclass(frozen=True)
class Structure:
    spaces: tuple[Space, ...]
    spans: tuple[Span, ...]
    edges: tuple[Edge, ...]
    counts: dict[str, Any]


@dataclass(frozen=True)
class _Building:
    element: Element
    polygon: BaseGeometry
    space_id: str
    number: str | None
    name: str | None


def _space_id(element: Element) -> str:
    return f"osm:{element.ref}"


def _outline(geometry: BaseGeometry | None) -> dict[str, Any] | None:
    if geometry is None or geometry.geom_type == "Point":
        return None
    shape = mapping(geometry)
    return {"type": shape["type"], "coordinates": _lists(shape["coordinates"])}


def _lists(coordinates: Any) -> Any:
    if isinstance(coordinates, tuple | list):
        return [_lists(item) for item in coordinates]
    return coordinates


def _tags(space: Space) -> dict[str, str]:
    return {key: value.value for key, value in space.attributes.items()}


def _label_value(label: labels.Label | None) -> Value | None:
    if label is None:
        return None
    if label.from_name:
        return Value.sourced(label.text)
    return Value.derived(label.text, labels.RULE)


def _sourced(name: Value | None) -> str | None:
    """Only a name read as is counts as the sourced name. A filtered one does not."""
    return name.value if name and name.provenance is ValueProvenance.SOURCED else None


class _Join:
    def __init__(self, osm: OsmData, venue: Venue) -> None:
        self.osm = osm
        self.venue = venue
        self.spaces: dict[str, Space] = {}
        self.spans: list[Span] = []
        self.edges: list[Edge] = []
        # Room-tier space id -> the nodes of its outline, for finding its doors.
        self.outlines: dict[str, set[int]] = {}
        # Room-tier space id -> its shape and levels, for snapping the doorless ones.
        self.shapes: dict[str, snap.Outline] = {}
        self.counts: Counter[str] = Counter()
        self.site_area = self._site_area(osm)
        self.site_id = f"site:{venue.id}"

        kept: dict[str, list[Element]] = {tiers.BUILDING: [], tiers.ROOM: []}
        for element in osm.tagged():
            tier = tiers.tier_for(element.tags)
            if tier is None:
                continue
            needs_outline = tier == tiers.BUILDING
            if element.geometry is None or (
                needs_outline and element.geometry.geom_type == "Point"
            ):
                self.counts["skipped_without_geometry"] += 1
            elif not self.site_area.covers(element.geometry.representative_point()):
                self.counts["dropped_outside_site"] += 1
            else:
                kept[tier].append(element)

        self._add_site()
        self.buildings = [self._add_building(element) for element in kept[tiers.BUILDING]]
        self._index = STRtree([building.polygon for building in self.buildings])
        for element in kept[tiers.ROOM]:
            self._add_room(element)
        self._add_edges()
        self._add_snaps()

    def _site_area(self, osm: OsmData) -> BaseGeometry:
        parts = []
        for way_id in self.venue.part_ways:
            polygon = osm.way_polygon(way_id)
            if polygon is None:
                raise StructureError(f"the pull does not hold part way {way_id} as a closed way")
            parts.append(polygon)
        return unary_union(parts)

    def _path(self, parent_id: str) -> tuple[Ancestor, ...]:
        parent = self.spaces[parent_id]
        return (*parent.ancestor_path, Ancestor(tier=parent.tier.value, id=parent.id))

    def _add(self, space: Space) -> None:
        self.spaces[space.id] = space

    def _add_site(self) -> None:
        name = Value.derived(self.venue.site_name, tiers.SITE_RULE)
        self._add(
            Space(
                id=self.site_id,
                record_provenance=RecordProvenance.SOURCED,
                source=None,
                tier=Value.derived(tiers.SITE, tiers.SITE_RULE),
                parent_id=None,
                ancestor_path=(),
                name=name,
                label=name,
                addressable=True,
                outline=_outline(self.site_area),
            )
        )

    def _attributes(self, element: Element) -> dict[str, Value]:
        attributes: dict[str, Value] = {}
        for key in ATTRIBUTE_KEYS:
            if key not in element.tags:
                continue
            value = Value.sourced(element.tags[key])
            if key in _FILTERED_KEYS:
                cleaned = self._without_people(element.tags[key])
                if cleaned is None:
                    continue
                value = cleaned
            attributes[key] = value
        return attributes

    def _without_people(self, text: str | None) -> Value | None:
        if text is None or not text.strip():
            return None
        result = names.strip_person_names(text)
        if not result.changed:
            return Value.sourced(text)
        self.counts["names_filtered"] += 1
        return Value.derived(result.text, names.RULE) if result.text else None

    def _add_building(self, element: Element) -> _Building:
        tags = element.tags
        name = self._without_people(tags.get("name"))
        label = labels.building_label(name=_sourced(name), number=tags.get("ref"))
        space = Space(
            id=_space_id(element),
            record_provenance=RecordProvenance.SOURCED,
            source=element.ref,
            tier=Value.derived(tiers.BUILDING, tiers.RULE),
            parent_id=self.site_id,
            ancestor_path=self._path(self.site_id),
            name=name,
            ref=Value.sourced(tags["ref"]) if "ref" in tags else None,
            label=_label_value(label),
            addressable=label is not None,
            attributes=self._attributes(element),
            outline=_outline(element.geometry),
            clearance=clearance.space_clearance(tags),
        )
        self._add(space)
        assert element.geometry is not None
        return _Building(element, element.geometry, space.id, tags.get("ref"), _sourced(name))

    def _building_of(self, element: Element) -> _Building | None:
        assert element.geometry is not None
        point = element.geometry.representative_point()
        hits = [
            self.buildings[int(index)] for index in self._index.query(point, predicate="intersects")
        ]
        # Where polygons overlap, the smallest one is the building proper.
        return min(hits, key=lambda b: (b.polygon.area, b.element.type, b.element.id), default=None)

    def _floor(self, building: _Building, level: str) -> str:
        floor_id = f"floor:{building.element.ref}:{level}"
        if floor_id not in self.spaces:
            label = labels.floor_label(
                building_number=building.number, building_name=building.name, level=level
            )
            self._add(
                Space(
                    id=floor_id,
                    record_provenance=RecordProvenance.SOURCED,
                    source=None,
                    tier=Value.derived(tiers.FLOOR, tiers.FLOOR_RULE),
                    parent_id=building.space_id,
                    ancestor_path=self._path(building.space_id),
                    label=_label_value(label),
                    addressable=label is not None,
                    levels=Value.derived(level, tiers.FLOOR_RULE),
                )
            )
        return floor_id

    def _add_room(self, element: Element) -> None:
        tags = element.tags
        building = self._building_of(element)
        occupied = levels.parse_levels(tags.get("level"))
        space_id = _space_id(element)

        if building is None:
            parent_id = self.site_id
            self.counts["rooms_in_no_building"] += 1
        elif not occupied:
            parent_id = building.space_id
            self.counts["rooms_without_level"] += 1
        else:
            # Hangs from the lowest floor it occupies and spans the rest.
            parent_id = self._floor(building, occupied[0])
            for level in occupied[1:]:
                self.spans.append(
                    Span(
                        space_id=space_id,
                        floor_id=self._floor(building, level),
                        provenance=ValueProvenance.DERIVED,
                        rule=levels.RULE,
                    )
                )

        name = self._without_people(tags.get("name"))
        label = labels.room_label(
            name=_sourced(name),
            ref=tags.get("ref"),
            building_number=building.number if building else None,
            building_name=building.name if building else None,
        )
        # What the name filter leaves behind ("Büro") is the last resort.
        label_value = _label_value(label) or name
        self._add(
            Space(
                id=space_id,
                record_provenance=RecordProvenance.SOURCED,
                source=element.ref,
                tier=Value.derived(tiers.ROOM, tiers.RULE),
                parent_id=parent_id,
                ancestor_path=self._path(parent_id),
                name=name,
                ref=Value.sourced(tags["ref"]) if "ref" in tags else None,
                label=label_value,
                addressable=label_value is not None,
                levels=Value.derived(";".join(occupied), levels.RULE) if occupied else None,
                attributes=self._attributes(element),
                outline=_outline(element.geometry),
                clearance=clearance.space_clearance(tags),
            )
        )
        self.outlines[space_id] = self._outline_nodes(element)
        if element.geometry is not None and element.geometry.geom_type != "Point":
            self.shapes[space_id] = snap.Outline(element.geometry, occupied)

    def _outline_nodes(self, element: Element) -> set[int]:
        if element.type == "way":
            return set(self.osm.ways[element.id].nodes)
        if element.type == "relation":
            return {
                node
                for member in self.osm.relations[element.id].members
                if member.type == "way" and member.ref in self.osm.ways
                for node in self.osm.ways[member.ref].nodes
            }
        return set()

    def _add_edges(self) -> None:
        door_nodes = [node for node, tags in self.osm.node_tags.items() if doors.is_door(tags)]
        joined = doors.joins(door_nodes, self.outlines)
        self.counts["doors_on_no_space"] = len(door_nodes) - len(joined)
        for node, space_ids in joined.items():
            door = f"node/{node}"
            door_clearance = clearance.door_clearance(self.osm.node_tags[node])
            lon, lat = self.osm.locations.get(node, (None, None))
            level = self._door_level(node, space_ids)
            for from_id, to_id in doors.passages(space_ids):
                entered = self.spaces[to_id].clearance
                # Every room-tier space has a clearance, tagged or defaulted.
                assert entered is not None
                self.edges.append(
                    Edge(
                        id=f"{door}:{from_id or 'outside'}>{to_id}",
                        door=door,
                        from_id=from_id,
                        to_id=to_id,
                        door_clearance=door_clearance,
                        clearance=clearance.edge_clearance(door_clearance, entered),
                        provenance=ValueProvenance.DERIVED,
                        rule=doors.RULE,
                        lon=lon,
                        lat=lat,
                        level=level,
                    )
                )

    def _door_level(self, node: int, space_ids: tuple[str, ...]) -> str | None:
        """A door's own level tag if it names one level, else the one level its spaces share."""
        tagged = levels.parse_levels(self.osm.node_tags[node].get("level"))
        if len(tagged) == 1:
            return tagged[0]
        shared: set[str] | None = None
        for space_id in space_ids:
            value = self.spaces[space_id].levels
            occupied = set(value.value.split(";")) if value else set()
            shared = occupied if shared is None else shared & occupied
        return next(iter(shared)) if shared and len(shared) == 1 else None

    def _add_snaps(self) -> None:
        with_door = {edge.to_id for edge in self.edges} | {
            edge.from_id for edge in self.edges if edge.from_id is not None
        }
        walkable = [
            space_id
            for space_id in self.shapes
            if circulation.is_circulation(_tags(self.spaces[space_id]))
        ]
        # Rooms only. A corridor or an area with no door is not a room nobody can get
        # into, it is circulation somebody drew without doors, and stays as mapped.
        doorless = [
            space_id
            for space_id in self.shapes.keys() - with_door
            if circulation.is_room(_tags(self.spaces[space_id]))
        ]
        found = snap.snaps(self.shapes, doorless, walkable)
        self.counts["snapped_rooms"] = len(found)
        for joined in found:
            for from_id, to_id in (
                (joined.room, joined.circulation),
                (joined.circulation, joined.room),
            ):
                entered = self.spaces[to_id].clearance
                assert entered is not None
                self.edges.append(
                    Edge(
                        id=f"snap:{from_id}>{to_id}",
                        door=None,
                        from_id=from_id,
                        to_id=to_id,
                        door_clearance=None,
                        clearance=clearance.edge_clearance(None, entered),
                        provenance=ValueProvenance.DERIVED,
                        rule=snap.RULE,
                        lon=joined.lon,
                        lat=joined.lat,
                        level=joined.level,
                    )
                )

    def result(self) -> Structure:
        spaces = tuple(self.spaces[space_id] for space_id in sorted(self.spaces))
        rooms = [space for space in spaces if space.tier.value == tiers.ROOM]
        by_tier = Counter(space.tier.value for space in spaces)
        counts: dict[str, Any] = {
            "spaces": {tier: by_tier[tier] for tier in tiers.TIERS},
            "spans": len(self.spans),
            "rooms_addressable": sum(room.addressable for room in rooms),
            "rooms_unaddressable": sum(not room.addressable for room in rooms),
        }
        for key in (
            "rooms_in_no_building",
            "rooms_without_level",
            "names_filtered",
            "dropped_outside_site",
            "skipped_without_geometry",
        ):
            counts[key] = self.counts[key]
        counts["edges"] = len(self.edges)
        counts["doors_on_no_space"] = self.counts["doors_on_no_space"]
        counts["snapped_rooms"] = self.counts["snapped_rooms"]
        spans = tuple(sorted(self.spans, key=lambda span: (span.space_id, span.floor_id)))
        edges = tuple(sorted(self.edges, key=lambda edge: edge.id))
        return Structure(spaces, spans, edges, counts)


def build_structure(osm: OsmData, venue: Venue) -> Structure:
    return _Join(osm, venue).result()
