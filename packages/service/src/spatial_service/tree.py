"""The collapsed tree: the venue as the planning model is shown it.

The planner cannot be handed a campus. It gets the top of the containment tree,
cut by code at a depth that keeps it under a node cap, and what lies below the cut
as counts per tier. The model never expands anything: a plan names a scope in
words and place resolution looks under it.

The cut is the deepest level at which every visible space down to it still fits
under the cap. The root is always shown, cap or no cap. Same venue and cap, same
tree: children go by label, then id.

Only what the persona can see is in it, by label and tier. No ids.

The depth is measured, not chosen: every turn's trace records the depth of the cut,
the depth of the whole visible tree, and how many spaces each has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spatial_core.records import Space
from spatial_service.places import Venue, tier

# How many spaces the planner is shown at most, the root aside. Not tuned on the
# real venue yet.
NODE_CAP = 200


@dataclass(frozen=True)
class Collapsed:
    # Nested {label, tier, children, below}, one per root. Usually one.
    tree: list[dict[str, Any]]
    # The tiers of the visible spaces, outermost first.
    tiers: tuple[str, ...]
    # How many levels below the root are shown, and how many there are.
    depth: int
    full_depth: int
    # How many spaces are shown, and how many are visible in all.
    nodes: int
    full_nodes: int
    node_cap: int


def collapse(venue: Venue, node_cap: int = NODE_CAP) -> Collapsed:
    key = ("collapsed", node_cap)
    if key not in venue.memo:
        venue.memo[key] = _collapse(venue, node_cap)
    found = venue.memo[key]
    assert isinstance(found, Collapsed)
    return found


def _collapse(venue: Venue, node_cap: int) -> Collapsed:
    per_level: dict[int, int] = {}
    children: dict[str | None, list[Space]] = {}
    first_seen: dict[str, int] = {}
    for space in venue.spaces.values():
        level = len(space.ancestor_path)
        per_level[level] = per_level.get(level, 0) + 1
        children.setdefault(space.parent_id, []).append(space)
        first_seen[tier(space)] = min(first_seen.get(tier(space), level), level)
    full_depth = max(per_level, default=0)

    depth, shown = 0, per_level.get(0, 0)
    while depth < full_depth and shown + per_level.get(depth + 1, 0) <= node_cap:
        depth += 1
        shown += per_level.get(depth, 0)

    def below(space: Space, counts: dict[str, int]) -> dict[str, int]:
        for child in children.get(space.id, []):
            counts[tier(child)] = counts.get(tier(child), 0) + 1
            below(child, counts)
        return counts

    def node(space: Space) -> dict[str, Any]:
        shown_node: dict[str, Any] = {
            "label": space.label.value if space.label else None,
            "tier": tier(space),
        }
        if len(space.ancestor_path) < depth:
            kids = sorted(children.get(space.id, []), key=_order)
            if kids:
                shown_node["children"] = [node(kid) for kid in kids]
        else:
            hidden = below(space, {})
            if hidden:
                shown_node["below"] = dict(sorted(hidden.items()))
        return shown_node

    return Collapsed(
        tree=[node(root) for root in sorted(children.get(None, []), key=_order)],
        tiers=tuple(sorted(first_seen, key=lambda t: (first_seen[t], t))),
        depth=depth,
        full_depth=full_depth,
        nodes=shown,
        full_nodes=len(venue.spaces),
        node_cap=node_cap,
    )


def _order(space: Space) -> tuple[str, str]:
    return (space.label.value if space.label else "", space.id)
