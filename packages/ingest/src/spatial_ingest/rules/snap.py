"""snap-doorless-room: joins a room with no door to the circulation right next to it.

A room nobody mapped a door on has no edge, so no route could start or end there.
Such a room is joined to the nearest circulation space on a level they share, if
their outlines come within SNAP_M of each other. It joins a room to circulation
and nothing else: never two rooms through a wall. And a room only: a corridor or
an area with no door is left as mapped. Which spaces are rooms is the caller's to
say (structure.py hands in the indoor=room ones).

The snap is for routes only. It is not a door, so it shows nobody a room and no
door asset hangs on it.

Rough edges: a room with one mapped door and a second, unmapped opening gets
nothing, and neither does an entrance sitting a little further out than SNAP_M.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from shapely import STRtree
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points, transform

from spatial_core.metres import Local

RULE = "snap-doorless-room@1"
# How far a doorless room may be from circulation and still be joined to it.
SNAP_M = 2.0


@dataclass(frozen=True)
class Snap:
    room: str
    circulation: str
    # The point on the circulation's outline nearest the room.
    lon: float
    lat: float
    # The lowest level the two share.
    level: str


@dataclass(frozen=True)
class Outline:
    geometry: BaseGeometry
    levels: tuple[str, ...]


def snaps(
    outlines: Mapping[str, Outline], doorless: Iterable[str], circulation: Iterable[str]
) -> list[Snap]:
    """Each doorless room with circulation within SNAP_M on a shared level, to the nearest."""
    walkable = sorted(space_id for space_id in circulation if space_id in outlines)
    rooms = sorted(space_id for space_id in doorless if space_id in outlines)
    if not walkable or not rooms:
        return []
    # One projection for the lot: a venue is small enough that it stays true.
    anchor = outlines[walkable[0]].geometry.representative_point()
    local = Local(anchor.x, anchor.y)
    shapes = [transform(local.metres, outlines[space_id].geometry) for space_id in walkable]
    index = STRtree(shapes)

    found = []
    for room in rooms:
        shape = transform(local.metres, outlines[room].geometry)
        levels = set(outlines[room].levels)
        near = []
        for position in index.query(shape.buffer(SNAP_M)):
            other = walkable[int(position)]
            shared = [level for level in outlines[other].levels if level in levels]
            distance = shape.distance(shapes[int(position)])
            if other != room and shared and distance <= SNAP_M:
                near.append((distance, other, shared[0], shapes[int(position)]))
        if not near:
            continue
        _, other, level, target = min(near, key=lambda hit: (hit[0], hit[1]))
        point = nearest_points(target, shape)[0]
        lon, lat = local.degrees(point.x, point.y)
        found.append(Snap(room, other, lon, lat, level))
    return found
