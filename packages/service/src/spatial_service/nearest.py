"""Nearest: which of some assets is closest to a place, and by what measure.

By modelled duration where a route on the persona's graph reaches one (routes.py).
Where none does, the assets sharing the smallest enclosing space with the place
win, and the answer says that is what it went by.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from spatial_core.records import Asset
from spatial_service.places import Venue
from spatial_service.routes import Found, Graph, Origin, asset_targets


@dataclass(frozen=True)
class Nearest:
    """The assets nearest a place, and what nearest was measured by."""

    # "modelled-duration" where a route reaches one, "shared-containment" otherwise.
    basis: str
    origin: Origin
    anchor: str
    asset_ids: tuple[str, ...]
    # The route to them, by modelled duration.
    route: Found | None = None
    # The smallest space holding both the anchor and the assets, by containment.
    shared: str | None = None
    default_origin: bool = False


BY_DURATION = "modelled-duration"
BY_CONTAINMENT = "shared-containment"


def nearest(venue: Venue, anchor: Origin | None, assets: Collection[Asset]) -> Nearest | None:
    """The nearest of ``assets`` to ``anchor``, or to the default origin. None if none is."""
    graph = Graph.of(venue)
    default = anchor is None
    start = anchor
    if start is None and venue.default_origin is not None:
        start = Origin(entrance=venue.default_origin)
    if start is None or not assets:
        return None
    if start.entrance is not None:
        if start.entrance not in graph.entrances:
            return None
        anchor_space = graph.entrances[start.entrance][1]
    else:
        assert start.space_id is not None
        anchor_space = start.space_id

    targets = asset_targets(venue, assets)
    found = graph.routes(start, {target.space_id for target in targets}) if targets else []
    if found:
        there = next(t for t in targets if t.space_id == found[0].destination)
        return Nearest(
            BY_DURATION, start, anchor_space, there.assets, found[0], default_origin=default
        )

    # No graph between them: the assets sharing the smallest enclosing space win.
    space = venue.spaces[anchor_space]
    path = [*(a.id for a in space.ancestor_path), space.id]

    def shared(asset: Asset) -> int:
        depth = 0
        for mine, theirs in zip(path, (a.id for a in asset.ancestor_path), strict=False):
            if mine != theirs:
                break
            depth += 1
        return depth

    deepest = max(shared(asset) for asset in assets)
    if deepest == 0:
        return None
    ids = tuple(sorted(asset.id for asset in assets if shared(asset) == deepest))
    return Nearest(
        BY_CONTAINMENT, start, anchor_space, ids, shared=path[deepest - 1], default_origin=default
    )
