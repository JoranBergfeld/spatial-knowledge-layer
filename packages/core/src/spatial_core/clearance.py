"""Clearance and visibility: who may reach what. Ingestion runs these, the service imports them.

Clearance is read from the venue's sourced `access` tags through a published
mapping. Where an element carries no tag, an authored default fills in, flagged as
one, and it never overrules a tag. There are no per-element overrides: a rule that
misclassifies something is fixed by changing the rule.

    clearance-from-access@1        an `access` value to the lowest persona it admits
    clearance-authored-default@1   untagged corridors, areas and public rooms are
                                   visitor; any other untagged room is staff
    edge-clearance@1               an edge takes the stricter of its door's tag and
                                   the clearance of the space it enters
    visible-by-edges@1             a space is visible when a permitted edge enters
                                   it, or by its own clearance when no edge does,
                                   and its ancestors are visible with it
    clearance-from-omniclass@1     an asset type's OmniClass Table 23 code to the
                                   lowest persona it admits, by its longest listed
                                   prefix; always an authored default, and an
                                   unlisted or missing code fails closed
    visible-assets@1               an asset is visible when its type is cleared and
                                   its space is visible; a type only through a
                                   visible asset

The rules interpret whom the venue admits the public to. They never claim to know
who really holds a key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from spatial_core.records import PERSONAS, Asset, AssetType, Clearance, Edge, Persona, Space

ACCESS_RULE = "clearance-from-access@1"
DEFAULT_RULE = "clearance-authored-default@1"
EDGE_RULE = "edge-clearance@1"
VISIBLE_RULE = "visible-by-edges@1"
TYPE_RULE = "clearance-from-omniclass@1"
ASSET_VISIBLE_RULE = "visible-assets@1"

# The published mapping. Any value not listed fails closed to maintenance.
ACCESS: Mapping[str, Persona] = {
    "yes": Persona.VISITOR,
    "permissive": Persona.VISITOR,
    "customers": Persona.VISITOR,
    "private": Persona.STAFF,
    "no": Persona.STAFF,
    "delivery": Persona.STAFF,
    "permit": Persona.STAFF,
    "emergency": Persona.STAFF,
}
UNRECOGNISED = Persona.MAINTENANCE

# Rooms of a public-by-nature function: toilets, auditorium, classroom, seminar
# room, hall, library, stairs, elevator, entrance. As (key, value) tag pairs.
PUBLIC_FUNCTIONS: frozenset[tuple[str, str]] = frozenset(
    {
        ("room", "toilet"),
        ("room", "toilets"),
        ("amenity", "toilets"),
        ("room", "auditorium"),
        ("room", "lecture"),
        ("room", "class"),
        ("room", "classroom"),
        ("room", "seminar"),
        ("room", "seminar_room"),
        ("room", "hall"),
        ("room", "library"),
        ("amenity", "library"),
        ("room", "stairs"),
        ("stairs", "yes"),
        ("highway", "steps"),
        ("room", "elevator"),
        ("highway", "elevator"),
        ("room", "entrance"),
    }
)
_CIRCULATION = {"corridor", "area"}

# The published mapping over OmniClass Table 23 (Products), by code prefix. A code
# takes its longest listed prefix. A type hidden behind ceilings, in risers or in
# plant rooms is maintenance; working equipment is staff; what anyone in the room
# can see or use is visitor. A row that turns out wrong is fixed here, never by
# overriding a single type.
_TIERS: Mapping[Persona, tuple[str, ...]] = {
    Persona.MAINTENANCE: (
        "23-15",
        "23-60",
        "23-65 35",
        "23-65 50",
        "23-65 55",
        "23-75 10",
        "23-75 35",
        "23-75 50",
        "23-75 70 21 24",
        "23-75 85",
        "23-80 30",
    ),
    Persona.STAFF: (
        "23-20",
        "23-25 30",
        "23-35",
        "23-40 35",
        "23-40 40",
        "23-40 70",
        "23-85",
    ),
    Persona.VISITOR: (
        "23-25 40",
        "23-30",
        "23-40 10",
        "23-40 20",
        "23-40 90",
        "23-40 95",
        "23-45",
        "23-50",
        "23-55",
        "23-65 70",
        "23-75 70 21 27",
        "23-80 50",
        "23-80 70",
    ),
}
OMNICLASS: Mapping[str, Persona] = {
    prefix: persona for persona, prefixes in _TIERS.items() for prefix in prefixes
}
# The numeric code at the front of a category. A title may follow it
# ("23-75 70 21 24: Variable Air Volume Terminals") and is never matched on.
_TABLE_23 = re.compile(r"\s*(23-\d{2}(?: \d{2}){0,5})(?!\d)")


def from_access(value: str) -> Clearance:
    """The clearance a sourced `access` value maps to. Unrecognised values fail closed."""
    return Clearance(
        persona=ACCESS.get(value, UNRECOGNISED),
        access=value,
        rule=ACCESS_RULE,
        authored_default=False,
    )


def _default(persona: Persona) -> Clearance:
    return Clearance(persona=persona, access=None, rule=DEFAULT_RULE, authored_default=True)


def space_clearance(tags: Mapping[str, str]) -> Clearance | None:
    """A space's clearance from its tags.

    A sourced tag always wins. Untagged, a corridor or area is visitor, a room of a
    public-by-nature function is visitor, and any other room is staff. Untagged
    elements above the room tier get no default: they are visible only through
    what they enclose.
    """
    if "access" in tags:
        return from_access(tags["access"])
    # A building tag wins over an indoor one, as it does for the tier.
    if tags.get("building", "no") != "no":
        return None
    indoor = tags.get("indoor")
    if indoor in _CIRCULATION:
        return _default(Persona.VISITOR)
    if indoor == "room":
        public = any(pair in PUBLIC_FUNCTIONS for pair in tags.items())
        return _default(Persona.VISITOR if public else Persona.STAFF)
    return None


def door_clearance(tags: Mapping[str, str]) -> Clearance | None:
    """A door's own clearance. An untagged door gets no default and constrains nothing."""
    return from_access(tags["access"]) if "access" in tags else None


def _code_digits(code: str) -> str:
    return code.replace("-", "").replace(" ", "")


def type_clearance(code: str | None) -> Clearance:
    """An asset type's clearance from its Table 23 code. Always an authored default.

    Only the numeric code is matched, never the title after it. A missing code, one
    not of the Table 23 form, or one with no listed prefix fails closed to
    maintenance and is flagged an unrecognised category.
    """
    found = _TABLE_23.match(code) if code else None
    listed: list[str] = []
    if found:
        digits = _code_digits(found.group(1))
        listed = [p for p in OMNICLASS if digits.startswith(_code_digits(p))]
    return Clearance(
        persona=OMNICLASS[max(listed, key=len)] if listed else UNRECOGNISED,
        access=None,
        rule=TYPE_RULE,
        authored_default=True,
        unrecognised_category=not listed,
    )


def stricter(first: Clearance, second: Clearance) -> Clearance:
    """The more restrictive of two. On a tie, the sourced one, so it is not flagged a default."""
    if first.persona.rank != second.persona.rank:
        return first if first.persona.rank > second.persona.rank else second
    return second if first.authored_default and not second.authored_default else first


def edge_clearance(door: Clearance | None, entered: Clearance) -> Clearance:
    """An edge takes the stricter of its door's tag and the space it enters."""
    return entered if door is None else stricter(door, entered)


def visible_sets(spaces: Iterable[Space], edges: Iterable[Edge]) -> dict[Persona, frozenset[str]]:
    """The spaces that exist for each persona.

    A space is visible when an edge the persona clears enters it. A space no edge
    enters is visible by its own clearance. Either way its ancestors are visible
    with it. The sets nest, visitor within staff within maintenance, because a
    persona clears everything the ones before it clear.
    """
    by_id = {space.id: space for space in spaces}
    entering: dict[str, list[Clearance]] = {}
    for edge in edges:
        entering.setdefault(edge.to_id, []).append(edge.clearance)

    sets = {}
    for persona in PERSONAS:
        visible: set[str] = set()
        for space in by_id.values():
            if space.id in entering:
                reached = any(persona.clears(c) for c in entering[space.id])
            else:
                reached = space.clearance is not None and persona.clears(space.clearance)
            if reached:
                visible.add(space.id)
                visible.update(ancestor.id for ancestor in space.ancestor_path)
        sets[persona] = frozenset(visible)
    return sets


def visible_assets(
    assets: Iterable[Asset],
    types: Iterable[AssetType],
    spaces: Mapping[Persona, frozenset[str]],
) -> tuple[dict[Persona, frozenset[str]], dict[Persona, frozenset[str]]]:
    """The assets, then the asset types, that exist for each persona.

    An asset is visible when the persona clears its type and its space is in the
    persona's visible set. A type is visible only through a visible asset, so a
    type with nothing a persona can see does not exist for them. Like the space
    sets, these nest.
    """
    by_id = {kind.id: kind for kind in types}
    asset_sets: dict[Persona, frozenset[str]] = {}
    type_sets: dict[Persona, frozenset[str]] = {}
    listed = list(assets)
    for persona in PERSONAS:
        seen = [
            asset
            for asset in listed
            if asset.type_id in by_id
            and persona.clears(by_id[asset.type_id].clearance)
            and asset.space_id in spaces[persona]
        ]
        asset_sets[persona] = frozenset(asset.id for asset in seen)
        type_sets[persona] = frozenset(asset.type_id for asset in seen)
    return asset_sets, type_sets
