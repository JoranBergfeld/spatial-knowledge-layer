"""Routes: the quickest way between places over one persona's circulation graph.

The graph is built from the venue the host bound, so from the spaces and edges
this persona can see and clear and nothing else. A route that would need a space
beyond clearance is not found, and looks exactly like one a mapping gap broke.

A route minimises modelled duration. Nothing here is measured:

    walking     1.34 m/s everywhere, door to door in a straight line
    stairs      12 s per level, half for a half level
    lift        40 s to take one, plus 4 s per level
    storey      4.0 m, which is where the 12 and the 4 come from

Seconds, metres and those constants go back with every route. Metres are what is
walked on the flat: a level change adds time, never distance.

What a route may pass through (spatial_core.circulation): circulation by its
sourced tags, and an untagged room with three or more doors the persona may take,
flagged derived wherever it is used. The rooms a route starts and ends in are always enterable.
Lifts are in by default. Step-free takes every stair out and leaves the lifts.

A destination is places before anything is routed. A set of places is reached at
whichever is nearest by modelled duration. A point in a mapped space is that space.
A point in open space snaps to the nearest place a route can reach within
OPEN_SPACE_SNAP_M, flagged derived. Points only come from asset positions so far:
a typed coordinate is never read.

Rough edges, all of them: distance inside a space is door to door as the crow
flies, so an L-shaped corridor comes out short. A space with no outline has no
centre to walk to and cannot be routed to. Nothing joins two buildings, because
ingestion makes no spaces of the footways between them.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from itertools import count
from typing import Any

from pydantic import BaseModel, ConfigDict
from shapely.geometry import Point, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

from spatial_core import circulation
from spatial_core.metres import Local
from spatial_core.records import Asset, Edge, Persona, Space
from spatial_service.places import ROOM, Venue, tier

# How far a point in open space may be from a place and still be taken to mean it.
# Chosen, not measured: there is no bundle on this machine to measure on. Five
# metres is about the width of a corridor and a wall, enough for a hydrant on a
# facade or a point dropped just outside a door, and it bounds what the snap hides
# to about four modelled seconds. Past that, which place was meant is a guess.
OPEN_SPACE_SNAP_M = 5.0
OPEN_SPACE_RULE = "snap-open-space-point@1"

WALK = "walk"
STAIRS = "stairs"
LIFT = "lift"


class Constants(BaseModel):
    """What a modelled duration is made of. Returned with every route."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    walking_m_per_s: float = 1.34
    stairs_s_per_level: float = 12.0
    lift_s: float = 40.0
    lift_s_per_level: float = 4.0
    storey_m: float = 4.0


CONSTANTS = Constants()


def subgraph(edges: Iterable[Edge], visible: Collection[str], persona: Persona) -> tuple[Edge, ...]:
    """The edges a persona may take: cleared, and between spaces that exist for them."""
    return tuple(
        edge
        for edge in edges
        if persona.clears(edge.clearance)
        and edge.to_id in visible
        and (edge.from_id is None or edge.from_id in visible)
    )


def door_counts(taken: Iterable[Edge]) -> dict[str, int]:
    """How many doors each space has, counting only the persona's own edges.

    ``taken`` is what `subgraph` kept. A door the persona may not take, or one into a
    room that does not exist for them, is not counted: if it were, a hidden room
    behind a third door would turn a room into a way through, and a route would
    change with something the persona cannot see. So the same room can be crossable
    for staff and not for a visitor.
    """
    doors: dict[str, set[str]] = {}
    for edge in taken:
        if edge.door is None:
            continue
        for space_id in (edge.from_id, edge.to_id):
            if space_id is not None:
                doors.setdefault(space_id, set()).add(edge.door)
    return {space_id: len(found) for space_id, found in doors.items()}


@dataclass(frozen=True)
class Portal:
    """A way between spaces: a door, or the snap of a doorless room."""

    key: str
    door: str | None
    x: float
    y: float
    level: float | None
    rule: str


@dataclass(frozen=True)
class Origin:
    # The space a route starts in the middle of, or
    space_id: str | None = None
    # the edge it comes in by from outside.
    entrance: str | None = None


@dataclass(frozen=True)
class Leg:
    """What a route does in one space."""

    space_id: str
    metres: float
    seconds: float
    mode: str = WALK
    levels: float = 0.0
    # The rule that lets the route pass through. None where it starts or ends.
    crossable: str | None = None
    # How it leaves: the portal, None for the last leg.
    leaves_by: Portal | None = None


@dataclass(frozen=True)
class Found:
    destination: str
    seconds: float
    metres: float
    legs: tuple[Leg, ...]
    # The door a route from outside came in by.
    entered_by: Portal | None = None

    @property
    def origin(self) -> str:
        return self.legs[0].space_id


def _tags(space: Space) -> dict[str, str]:
    return {key: value.value for key, value in space.attributes.items()}


def _level(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:
        return None


def _polygon(space: Space) -> BaseGeometry | None:
    if space.outline is None or space.outline.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    geometry = shape(space.outline)
    return geometry if geometry.is_valid and not geometry.is_empty else None


@dataclass
class Graph:
    """One persona's circulation graph, in metres."""

    local: Local
    shapes: dict[str, BaseGeometry] = field(default_factory=dict)
    centres: dict[str, tuple[float, float]] = field(default_factory=dict)
    portals: dict[str, Portal] = field(default_factory=dict)
    # The portals on each space, and where each leads from there.
    on: dict[str, list[str]] = field(default_factory=dict)
    through: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # Edges that come in from outside: edge id -> (portal, the space entered).
    entrances: dict[str, tuple[str, str]] = field(default_factory=dict)
    crossable: dict[str, str | None] = field(default_factory=dict)
    changes_level: dict[str, str] = field(default_factory=dict)
    edges: int = 0

    @classmethod
    def of(cls, venue: Venue) -> Graph:
        """The venue's graph, built once and kept with it."""
        if "routes" not in venue.memo:
            venue.memo["routes"] = cls._build(venue)
        graph = venue.memo["routes"]
        assert isinstance(graph, Graph)
        return graph

    @classmethod
    def _build(cls, venue: Venue) -> Graph:
        placed = [edge for edge in venue.edges if edge.lon is not None and edge.lat is not None]
        outlined = {i: g for i, s in venue.spaces.items() if (g := _polygon(s)) is not None}
        anchor: tuple[float, float] = (0.0, 0.0)
        if placed:
            anchor = (placed[0].lon or 0.0, placed[0].lat or 0.0)
        elif outlined:
            point = next(iter(outlined.values())).representative_point()
            anchor = (point.x, point.y)
        graph = cls(Local(*anchor))
        for space_id, geometry in outlined.items():
            flat = transform(graph.local.metres, geometry)
            graph.shapes[space_id] = flat
            # The centroid where it is inside, which it is not for an L or a ring.
            centre = flat.centroid if flat.contains(flat.centroid) else flat.representative_point()
            graph.centres[space_id] = (centre.x, centre.y)
        for space_id, space in venue.spaces.items():
            tags = _tags(space)
            graph.crossable[space_id] = circulation.crossable_rule(
                tags, venue.doors.get(space_id, 0)
            )
            if circulation.is_stairs(tags):
                graph.changes_level[space_id] = STAIRS
            elif circulation.is_lift(tags):
                graph.changes_level[space_id] = LIFT
        for edge in placed:
            assert edge.lon is not None and edge.lat is not None
            key = edge.door or "snap:" + "|".join(sorted((edge.from_id or "", edge.to_id)))
            if key not in graph.portals:
                x, y = graph.local.metres(edge.lon, edge.lat)
                graph.portals[key] = Portal(key, edge.door, x, y, _level(edge.level), edge.rule)
            sides = [edge.to_id] if edge.from_id is None else [edge.from_id, edge.to_id]
            for side in sides:
                if key not in graph.on.setdefault(side, []):
                    graph.on[side].append(key)
            if edge.from_id is None:
                graph.entrances[edge.id] = (key, edge.to_id)
            else:
                graph.through.setdefault((key, edge.from_id), []).append(edge.to_id)
            graph.edges += 1
        return graph

    def _walk(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _cross(self, space_id: str, a: Portal, b: Portal) -> tuple[float, float, str, float] | None:
        """Going from one portal of a space to another: metres, seconds, mode, levels."""
        metres = self._walk((a.x, a.y), (b.x, b.y))
        seconds = metres / CONSTANTS.walking_m_per_s
        levels = abs(a.level - b.level) if a.level is not None and b.level is not None else 0.0
        if not levels:
            return metres, seconds, WALK, 0.0
        mode = self.changes_level.get(space_id)
        if mode == STAIRS:
            return metres, seconds + CONSTANTS.stairs_s_per_level * levels, STAIRS, levels
        if mode == LIFT:
            lift = CONSTANTS.lift_s + CONSTANTS.lift_s_per_level * levels
            return metres, seconds + lift, LIFT, levels
        # Two doors on different levels of something that is neither stairs nor a lift.
        return None

    def routes(
        self,
        origin: Origin,
        targets: Collection[str],
        *,
        step_free: bool = False,
        limit: int = 1,
    ) -> list[Found]:
        """The quickest route to each of up to ``limit`` targets, nearest first."""

        def barred(space_id: str) -> bool:
            return step_free and self.changes_level.get(space_id) == STAIRS

        # A state is (portal, the space stood in), or ("", space) for arriving in it.
        best: dict[tuple[str, str], float] = {}
        back: dict[tuple[str, str], tuple[tuple[str, str] | None, Leg | None]] = {}
        heap: list[tuple[float, int, tuple[str, str]]] = []
        order = count()

        def reach(
            state: tuple[str, str], cost: float, via: tuple[str, str] | None, leg: Leg | None
        ) -> None:
            if cost < best.get(state, math.inf):
                best[state] = cost
                back[state] = (via, leg)
                heapq.heappush(heap, (cost, next(order), state))

        entered: Portal | None = None
        start: str | None = None
        if origin.entrance is not None:
            if origin.entrance not in self.entrances:
                return []
            key, start = self.entrances[origin.entrance]
            entered = self.portals[key]
            if barred(start):
                return []
            reach((key, start), 0.0, None, None)
        elif origin.space_id is not None:
            start = origin.space_id
            if barred(start):
                return []
            if start in targets:
                reach(("", start), 0.0, None, Leg(start, 0.0, 0.0))
            # A space with no outline has no middle to set off from.
            portals = self.on.get(start, []) if start in self.centres else []
            for key in portals:
                portal = self.portals[key]
                metres = self._walk(self.centres[start], (portal.x, portal.y))
                seconds = metres / CONSTANTS.walking_m_per_s
                reach((key, start), seconds, None, Leg(start, metres, seconds, leaves_by=portal))
        if start is None:
            return []

        found: list[Found] = []
        done: set[tuple[str, str]] = set()
        while heap and len(found) < limit:
            cost, _, state = heapq.heappop(heap)
            if state in done:
                continue
            done.add(state)
            key, space_id = state
            if not key:
                found.append(self._found(state, cost, back, entered))
                continue
            here = self.portals[key]
            if space_id in targets and space_id in self.centres:
                metres = self._walk((here.x, here.y), self.centres[space_id])
                seconds = metres / CONSTANTS.walking_m_per_s
                reach(("", space_id), cost + seconds, state, Leg(space_id, metres, seconds))
            for beyond in self.through.get(state, []):
                if not barred(beyond):
                    reach((key, beyond), cost, state, None)
            rule = self.crossable.get(space_id)
            if rule is None and space_id != start:
                continue
            for other in self.on.get(space_id, []):
                if other == key:
                    continue
                crossing = self._cross(space_id, here, self.portals[other])
                if crossing is None:
                    continue
                metres, seconds, mode, levels = crossing
                leg = Leg(
                    space_id,
                    metres,
                    seconds,
                    mode,
                    levels,
                    crossable=rule if space_id != start else None,
                    leaves_by=self.portals[other],
                )
                reach((other, space_id), cost + seconds, state, leg)
        return found

    def _found(
        self,
        state: tuple[str, str],
        cost: float,
        back: Mapping[tuple[str, str], tuple[tuple[str, str] | None, Leg | None]],
        entered: Portal | None,
    ) -> Found:
        legs: list[Leg] = []
        at: tuple[str, str] | None = state
        while at is not None:
            at, leg = back[at]
            if leg is not None:
                legs.append(leg)
        legs.reverse()
        return Found(
            destination=state[1],
            seconds=cost,
            metres=sum(leg.metres for leg in legs),
            legs=tuple(legs),
            entered_by=entered,
        )

    def measure(self) -> dict[str, Any]:
        return {"spaces": len(self.on), "edges": self.edges}


@dataclass(frozen=True)
class Target:
    """A place a route may end at, and how it came to be one."""

    space_id: str
    # The assets that brought it in, when the destination was assets.
    assets: tuple[str, ...] = ()
    # How far a point in open space was moved to get here. None when nothing was.
    snapped_m: float | None = None


def places_at(venue: Venue, lon: float, lat: float, under: str | None = None) -> list[Target]:
    """The places a point means: the mapped rooms it lies in, else those within the snap.

    Several rooms can hold one point, one per floor. The route then ends at
    whichever is nearest. ``under`` keeps to rooms inside one space.
    """
    graph = Graph.of(venue)
    x, y = graph.local.metres(lon, lat)
    point = Point(x, y)
    near: list[tuple[float, str]] = []
    for space_id, flat in graph.shapes.items():
        space = venue.spaces[space_id]
        if tier(space) != ROOM:
            continue
        if under is not None and not any(a.id == under for a in space.ancestor_path):
            continue
        distance = float(flat.distance(point))
        if distance <= OPEN_SPACE_SNAP_M:
            near.append((distance, space_id))
    inside = sorted(space_id for distance, space_id in near if distance == 0.0)
    if inside:
        return [Target(space_id) for space_id in inside]
    return [Target(space_id, snapped_m=distance) for distance, space_id in sorted(near)]


def asset_targets(venue: Venue, assets: Iterable[Asset]) -> list[Target]:
    """Where a route to these assets may end: each one's space.

    An asset hung on something bigger than a room (a hydrant on the site) goes by
    its position instead, when it has one.
    """
    by_space: dict[str, list[str]] = {}
    snapped: dict[str, float | None] = {}
    for asset in assets:
        space = venue.spaces.get(asset.space_id)
        if space is None:
            continue
        if tier(space) == ROOM:
            found = [Target(space.id)]
        elif asset.position is not None:
            under = None if space.parent_id is None else space.id
            found = places_at(venue, asset.position.lon, asset.position.lat, under)
        else:
            found = []
        for target in found:
            by_space.setdefault(target.space_id, []).append(asset.id)
            known = snapped.get(target.space_id, target.snapped_m)
            # Standing in the room beats being snapped to it.
            snapped[target.space_id] = (
                None if known is None or target.snapped_m is None else min(known, target.snapped_m)
            )
    return [
        Target(space_id, tuple(sorted(ids)), snapped[space_id])
        for space_id, ids in sorted(by_space.items())
    ]


@dataclass(frozen=True)
class Routed:
    """What a route step came to."""

    origin: Origin
    # True when the question named no origin and the venue's default was used.
    default_origin: bool
    step_free: bool
    targets: tuple[Target, ...]
    # Nearest first. Empty when there is no route. Why not is kept nowhere: every
    # cause of no route is the same abstention, and the trace gives no reason either.
    found: tuple[Found, ...] = ()

    def target(self, space_id: str) -> Target:
        return next(t for t in self.targets if t.space_id == space_id)


def find(
    venue: Venue,
    origin: Origin | None,
    targets: Collection[Target],
    *,
    step_free: bool = False,
    alternatives: int = 1,
) -> Routed:
    """Route from ``origin``, or the default origin, to the nearest of ``targets``."""
    graph = Graph.of(venue)
    default = origin is None
    start = origin
    if start is None and venue.default_origin is not None:
        start = Origin(entrance=venue.default_origin)
    listed = tuple(targets)
    if start is None:
        return Routed(Origin(), default, step_free, listed)
    if not listed:
        return Routed(start, default, step_free, listed)

    ids = {target.space_id for target in listed}
    snapped = [target for target in listed if target.snapped_m is not None]
    if snapped and len(snapped) == len(listed):
        # A point in open space means the nearest place a route reaches, so every
        # candidate is routed to and the least moved one wins.
        reached = graph.routes(start, ids, step_free=step_free, limit=len(ids))
        by_snap = {target.space_id: target.snapped_m or 0.0 for target in listed}
        reached.sort(key=lambda route: (by_snap[route.destination], route.seconds))
        found = reached[:1]
    else:
        found = graph.routes(start, ids, step_free=step_free, limit=alternatives)
    return Routed(start, default, step_free, listed, tuple(found))
