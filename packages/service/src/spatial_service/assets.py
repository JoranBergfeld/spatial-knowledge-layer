"""The asset tools: the types a mention names, what a scope holds, and counts by scale level.

All three are handed a venue already cut down to one persona's visible set, like
place resolution. An asset beyond clearance is not in it, so it is not counted,
not listed and not noticed missing.

    resolve_asset_type   the visible types a mention names. A set: "lights" is every
                         type of light, and nobody is asked which. No match is
                         nothing found, whether the type is hidden or made up.
    assets_in_scope      every visible asset under one scope node, or under the
                         whole venue, of the given types if any, that passes the
                         given attribute filters if any (attributes.py). One pass
                         over ancestor paths, no call per space.
    aggregate            those assets counted per space of one tier under the scope

Aggregation walks containment only. An asset is counted under the spaces on its
ancestor path, so one in a stairwell spanning three floors counts on the floor the
stairwell hangs from, once.

A count is never one number across provenance. Every tally keeps sourced and
generated apart, and nothing here adds them up for printing.

A space with nothing visible installed in it at all has no recorded installations.
That is a gap in the data, not a zero, so aggregation lists it apart and never
ranks it.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

from spatial_core.records import Asset, AssetType, RecordProvenance, Space
from spatial_service.attributes import FillRate, apply
from spatial_service.places import Outcome, Venue, normalise, tier
from spatial_service.plan import Filter


def _stem(word: str) -> str:
    # Enough to fold "lights" onto "light". Not a stemmer.
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> frozenset[str]:
    return frozenset(_stem(word) for word in re.findall(r"\w+", normalise(text)))


def check_type_arguments(text: str) -> list[str]:
    """Why ``text`` is no argument for resolve_asset_type. Reads no venue, so no persona."""
    if not _words(text):
        return [f"{text!r} has nothing to look up"]
    return []


@dataclass(frozen=True)
class TypeResult:
    outcome: Outcome
    # Every type the mention names, by id. Empty for none.
    ids: tuple[str, ...]


def resolve_asset_type(venue: Venue, text: str) -> TypeResult:
    """The visible asset types ``text`` names: an exact name, else every word of it in a name."""
    query = normalise(text)
    wanted = _words(text)
    types = venue.asset_types.values()
    found = [t for t in types if normalise(t.name.value) == query]
    if not found and wanted:
        found = [t for t in types if wanted <= _type_words(t)]
    if not found:
        return TypeResult(Outcome.NONE, ())
    return TypeResult(Outcome.RESOLVED, tuple(sorted(t.id for t in found)))


def _type_words(kind: AssetType) -> frozenset[str]:
    # The name, and the title where the Table 23 code carries one after a colon.
    _, _, title = kind.omniclass.value.partition(":")
    return _words(kind.name.value) | _words(title)


@dataclass(frozen=True)
class Tally:
    """How many assets, sourced and generated apart."""

    sourced: int = 0
    generated: int = 0

    @property
    def total(self) -> int:
        """For ranking and for telling none from some. Never for printing."""
        return self.sourced + self.generated

    @classmethod
    def of(cls, assets: Iterable[Asset]) -> Tally:
        sourced = generated = 0
        for asset in assets:
            if asset.record_provenance is RecordProvenance.SOURCED:
                sourced += 1
            else:
                generated += 1
        return cls(sourced, generated)


@dataclass(frozen=True)
class ScopeResult:
    # The scope node, or None for the whole venue.
    scope: str | None
    # The types asked for, or None for any.
    types: tuple[str, ...] | None
    # The visible assets under the scope, of those types, that pass every filter.
    ids: tuple[str, ...]
    # How many visible assets the scope holds of any type. Zero means the scope
    # has no recorded installations, which is not the same as none of a type.
    installed: int
    # Each filter, with how many of the assets asked about carry its attribute.
    filters: tuple[FillRate, ...] = ()


def assets_in_scope(
    venue: Venue,
    scope: str | None,
    types: Collection[str] | None = None,
    filters: Sequence[Filter] = (),
) -> ScopeResult:
    """Every visible asset under ``scope`` (or the venue), of ``types``, passing ``filters``."""
    pool = venue.assets.values() if scope is None else venue.assets_under.get(scope, [])
    asked = sorted((a for a in pool if types is None or a.type_id in types), key=lambda a: a.id)
    kept, rates = apply(venue, asked, filters)
    return ScopeResult(
        scope=scope,
        types=None if types is None else tuple(sorted(types)),
        ids=tuple(a.id for a in kept),
        installed=len(pool),
        filters=rates,
    )


@dataclass(frozen=True)
class Group:
    space_id: str
    tally: Tally
    # 1 for the most, shared on a tie.
    rank: int


@dataclass(frozen=True)
class Aggregation:
    outcome: Outcome
    by: str
    # Spaces of the tier with something installed, most first. A space with
    # installations and none of the type asked for is here, at zero.
    ranked: tuple[Group, ...] = ()
    # Spaces of the tier with no recorded installations at all. Never ranked.
    unrecorded: tuple[str, ...] = ()
    # Assets in scope that sit under no space of the tier, such as one hung on a
    # building when the count is per floor.
    ungrouped: int = 0

    @property
    def leaders(self) -> tuple[Group, ...]:
        return tuple(g for g in self.ranked if g.rank == 1 and g.tally.total > 0)


def check_tier(venue: Venue, by: str) -> list[str]:
    """Why ``by`` is no scale level. Reads the snapshot's tier words, not what a persona sees."""
    if normalise(by) not in venue.tiers:
        return [f"{by!r} is not a scale level here; the venue has {', '.join(venue.tiers)}"]
    return []


def aggregate(venue: Venue, found: ScopeResult, by: str) -> Aggregation:
    """``found``'s assets counted per space of tier ``by`` under its scope."""
    by = normalise(by)
    groups = [s for s in venue.spaces.values() if tier(s) == by and _under(s, found.scope)]
    if not groups:
        return Aggregation(Outcome.NONE, by)

    counted: dict[str, list[Asset]] = {group.id: [] for group in groups}
    ungrouped = 0
    for asset_id in found.ids:
        asset = venue.assets[asset_id]
        # Its one enclosing space of the tier. The tree is strict, so there is at most one.
        home = next((a.id for a in asset.ancestor_path if a.id in counted), None)
        if home is None:
            ungrouped += 1
        else:
            counted[home].append(asset)

    recorded = [g for g in groups if venue.assets_under.get(g.id)]
    tallies = {g.id: Tally.of(counted[g.id]) for g in recorded}
    ordered = sorted(recorded, key=lambda g: (-tallies[g.id].total, _sort_label(g), g.id))
    ranked: list[Group] = []
    for position, group in enumerate(ordered, start=1):
        tally = tallies[group.id]
        tied = ranked and ranked[-1].tally.total == tally.total
        ranked.append(Group(group.id, tally, ranked[-1].rank if tied else position))
    unrecorded = sorted(
        (g for g in groups if not venue.assets_under.get(g.id)),
        key=lambda g: (_sort_label(g), g.id),
    )
    return Aggregation(
        Outcome.RESOLVED,
        by,
        ranked=tuple(ranked),
        unrecorded=tuple(g.id for g in unrecorded),
        ungrouped=ungrouped,
    )


def _under(space: Space, scope: str | None) -> bool:
    return scope is None or space.id == scope or any(a.id == scope for a in space.ancestor_path)


def _sort_label(space: Space) -> str:
    return space.label.value if space.label else ""
