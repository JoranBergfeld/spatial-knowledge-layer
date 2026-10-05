"""edges-from-doors: the circulation edges, read from the doors in the pull.

A door is a node tagged `door` or `entrance` (anything but `no`). In Simple Indoor
Tagging it is a vertex of the outline of every space it opens into, so the spaces
a door joins are the room-tier spaces whose outline holds its node. A door joining
two spaces makes two edges, one each way. A door on one outline only opens onto no
mapped space (an entrance from outside, or a mapping gap), and makes one edge into
that space.

A room and a corridor that touch with no door between them get no edge. That room
then takes its own clearance, which is what an edge without a door would give it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

RULE = "edges-from-doors@1"


def is_door(tags: Mapping[str, str]) -> bool:
    return any(tags.get(key, "no") != "no" for key in ("door", "entrance"))


def joins(
    doors: Iterable[int], outlines: Mapping[str, Iterable[int]]
) -> dict[int, tuple[str, ...]]:
    """Each door node to the spaces whose outline holds it, in id order. Doors on none drop."""
    wanted = set(doors)
    joined: dict[int, set[str]] = {}
    for space_id, nodes in outlines.items():
        for node in set(nodes) & wanted:
            joined.setdefault(node, set()).add(space_id)
    return {node: tuple(sorted(spaces)) for node, spaces in sorted(joined.items())}


def passages(spaces: tuple[str, ...]) -> list[tuple[str | None, str]]:
    """The (from, to) pairs a door joining these spaces lets through."""
    if len(spaces) == 1:
        return [(None, spaces[0])]
    return [(a, b) for a in spaces for b in spaces if a != b]
