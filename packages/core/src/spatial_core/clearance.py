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

The rules interpret whom the venue admits the public to. They never claim to know
who really holds a key.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from spatial_core.records import PERSONAS, Clearance, Edge, Persona, Space

ACCESS_RULE = "clearance-from-access@1"
DEFAULT_RULE = "clearance-authored-default@1"
EDGE_RULE = "edge-clearance@1"
VISIBLE_RULE = "visible-by-edges@1"

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
