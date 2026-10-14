"""Which spaces a route may cross, and which change level, read from sourced tags.

    crossable-by-tags@1          circulation by its sourced tags: corridors and areas,
                                 stairs and lifts, footways, and rooms tagged hall,
                                 lobby, entrance, foyer or corridor
    crossable-by-door-count@1    an indoor=room with no sourced function and three or
                                 more doors. Derived, and flagged wherever a route
                                 uses one: it may still be an office.

A room with a sourced function that is not circulation is never crossed. Origin and
destination rooms are always enterable, whatever they are.

Ingestion uses this to pick which doorless spaces are rooms and what they snap to,
the service to build the graph a route runs over.
"""

from __future__ import annotations

from collections.abc import Mapping

TAGS_RULE = "crossable-by-tags@1"
DOOR_COUNT_RULE = "crossable-by-door-count@1"
# How many doors make an untagged room crossable.
CROSSABLE_DOORS = 3

_CIRCULATION_INDOOR = {"corridor", "area"}
_CIRCULATION_ROOM = {"hall", "lobby", "entrance", "foyer", "corridor"}
_CIRCULATION_HIGHWAY = {"footway", "corridor", "pedestrian", "path"}
# Tags that give a room a function. One with any of these is never crossed by door count.
_FUNCTION_KEYS = ("room", "amenity", "office")


def is_stairs(tags: Mapping[str, str]) -> bool:
    return (
        tags.get("room") == "stairs"
        or tags.get("stairs", "no") != "no"
        or tags.get("highway") == "steps"
    )


def is_lift(tags: Mapping[str, str]) -> bool:
    return tags.get("room") == "elevator" or tags.get("highway") == "elevator"


def is_room(tags: Mapping[str, str]) -> bool:
    """Mapped as a room: indoor=room, whatever its function. Not a corridor, not an area."""
    return tags.get("indoor") == "room"


def is_circulation(tags: Mapping[str, str]) -> bool:
    """Circulation by sourced tags. Stairs and lifts are circulation too."""
    return (
        is_stairs(tags)
        or is_lift(tags)
        or tags.get("indoor") in _CIRCULATION_INDOOR
        or tags.get("room") in _CIRCULATION_ROOM
        or tags.get("highway") in _CIRCULATION_HIGHWAY
    )


def crossable_rule(tags: Mapping[str, str], doors: int) -> str | None:
    """The rule that makes a space crossable, or None when a route may only start or end in it."""
    if is_circulation(tags):
        return TAGS_RULE
    untagged = is_room(tags) and not any(key in tags for key in _FUNCTION_KEYS)
    if untagged and doors >= CROSSABLE_DOORS:
        return DOOR_COUNT_RULE
    return None
