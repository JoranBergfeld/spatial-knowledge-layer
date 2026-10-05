"""tier-from-osm-tags: which scale level a tagged OSM element sits at, if any.

A tier is a label on a space, not a kind of space. Corridors and areas sit at the
room tier: they are the leaves a floor encloses. Floors never come from this rule.
Campus Süd maps no `indoor=level` elements, so every floor is made from the level
tags of what a building encloses (floor-from-level-tags).
"""

from __future__ import annotations

from collections.abc import Mapping

RULE = "tier-from-osm-tags@1"
# The site has no tags of its own. It comes from the venue definition.
SITE_RULE = "site-from-venue-definition@1"
# A floor nobody mapped as an element, made from the level tags beneath a building.
FLOOR_RULE = "floor-from-level-tags@1"

SITE = "site"
BUILDING = "building"
FLOOR = "floor"
ROOM = "room"

# Root first. Depth in the tree follows this order, tiers may be skipped.
TIERS = (SITE, BUILDING, FLOOR, ROOM)

_ROOM_TIER_INDOOR = {"room", "corridor", "area"}


def tier_for(tags: Mapping[str, str]) -> str | None:
    if tags.get("building", "no") != "no":
        return BUILDING
    if tags.get("indoor") in _ROOM_TIER_INDOOR:
        return ROOM
    return None
