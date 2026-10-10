"""The executor: runs a checked plan's steps in order, with no model call between them.

The host binds the venue (one persona, one snapshot) before the first step. A step
that points at an earlier step works with the ids that step produced. The first
step that comes up short ends the run: a place that is not exactly one (a
clarification or an abstention, decided by what it found), a type or a scale level
with nothing under it, or a tool that rejected its arguments, which is a trigger.

An asset step that finds no assets has not come up short. An empty set is a result.

Every step's arguments are checked before the first step runs. Checked as each
step came up, a malformed argument in a later step would be reached only when the
earlier ones resolved, and whether a turn buys a recovery call would then tell
what the persona can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spatial_service.assets import (
    Aggregation,
    ScopeResult,
    aggregate,
    assets_in_scope,
    check_tier,
    check_type_arguments,
    resolve_asset_type,
)
from spatial_service.places import (
    Outcome,
    Venue,
    check_arguments,
    places_matching,
    resolve_place,
)
from spatial_service.plan import (
    Aggregate,
    AssetsInScope,
    Mention,
    Nearest,
    PlacesMatching,
    Plan,
    ResolveAssetType,
    ResolvePlace,
    Resolving,
    Route,
    Step,
)
from spatial_service.recovery import Trigger, TriggerKind
from spatial_service.routes import Nearest as NearestFound
from spatial_service.routes import Origin, Routed, Target, asset_targets, find, nearest


@dataclass(frozen=True)
class StepResult:
    step: str
    tool: str
    args: dict[str, Any]
    outcome: Outcome
    ids: tuple[str, ...]
    # The mention a resolving step read, and its words.
    mention: str | None = None
    text: str | None = None
    how: str | None = None
    matched: int = 0
    # Why the tool rejected its arguments, when it did.
    rejection: tuple[str, ...] = ()
    # What assets_in_scope found, and what aggregate made of it.
    found: ScopeResult | None = None
    aggregation: Aggregation | None = None
    # What a route step came to, and what a nearest step picked.
    routed: Routed | None = None
    near: NearestFound | None = None


@dataclass(frozen=True)
class Execution:
    results: tuple[StepResult, ...]

    @property
    def last(self) -> StepResult:
        return self.results[-1]

    @property
    def complete(self) -> bool:
        """Every step ran and none came up short."""
        return all(r.outcome is Outcome.RESOLVED for r in self.results)

    @property
    def trigger(self) -> Trigger | None:
        """A trigger the run raised, which only a step whose arguments were rejected does."""
        if self.last.outcome is not Outcome.REJECTED:
            return None
        return Trigger(
            kind=TriggerKind.MALFORMED_ARGUMENTS,
            step=self.last.step,
            reasons=list(self.last.rejection),
        )

    def by_tool(self, tool: str) -> list[StepResult]:
        return [r for r in self.results if r.tool == tool]


def execute(plan: Plan, venue: Venue) -> Execution:
    mentions = {mention.id: mention for mention in plan.mentions}
    for step in plan.steps:
        problems = _problems(step, mentions, venue)
        if problems:
            rejected = _result(step, mentions, Outcome.REJECTED, rejection=tuple(problems))
            return Execution((rejected,))

    done: dict[str, StepResult] = {}
    results: list[StepResult] = []
    for step in plan.steps:
        result = _run(step, mentions, venue, done)
        results.append(result)
        if result.outcome is not Outcome.RESOLVED:
            break
        done[step.id] = result
    return Execution(tuple(results))


def _problems(step: Step, mentions: dict[str, Mention], venue: Venue) -> list[str]:
    """Why a step's arguments are malformed. Nothing here reads what a persona can see."""
    if isinstance(step, (ResolvePlace, PlacesMatching)):
        return check_arguments(mentions[step.mention].text)
    if isinstance(step, ResolveAssetType):
        return check_type_arguments(mentions[step.mention].text)
    if isinstance(step, Aggregate):
        return check_tier(venue, step.by)
    return []


def _run(
    step: Step, mentions: dict[str, Mention], venue: Venue, done: dict[str, StepResult]
) -> StepResult:
    if isinstance(step, ResolvePlace):
        within = done[step.within.step].ids if step.within is not None else None
        place = resolve_place(venue, mentions[step.mention].text, within)
        return _result(
            step, mentions, place.outcome, place.ids, how=place.how, matched=place.matched
        )
    if isinstance(step, ResolveAssetType):
        kinds = resolve_asset_type(venue, mentions[step.mention].text)
        return _result(step, mentions, kinds.outcome, kinds.ids, matched=len(kinds.ids))
    if isinstance(step, AssetsInScope):
        # A place step that got this far resolved exactly one place.
        scope = done[step.scope.step].ids[0] if step.scope is not None else None
        types = done[step.types.step].ids if step.types is not None else None
        found = assets_in_scope(venue, scope, types)
        return _result(
            step, mentions, Outcome.RESOLVED, found.ids, matched=len(found.ids), found=found
        )
    if isinstance(step, PlacesMatching):
        fitting = places_matching(venue, mentions[step.mention].text)
        return _result(
            step, mentions, fitting.outcome, fitting.ids, how=fitting.how, matched=fitting.matched
        )
    if isinstance(step, Route):
        origin = None
        if step.origin is not None:
            origin = Origin(space_id=done[step.origin.step].ids[0])
        elif step.after is not None:
            # A route step that got this far found a route. It starts where that ended.
            origin = Origin(space_id=done[step.after.step].ids[0])
        wanted = done[step.destination.step]
        if wanted.found is not None:
            # A route to an asset goes to its space.
            targets = asset_targets(venue, (venue.assets[i] for i in wanted.ids))
        else:
            targets = [Target(space_id) for space_id in wanted.ids]
        routed = find(
            venue, origin, targets, step_free=step.step_free, alternatives=step.alternatives
        )
        ends = tuple(found.destination for found in routed.found)
        outcome = Outcome.RESOLVED if ends else Outcome.NONE
        return _result(step, mentions, outcome, ends, matched=len(ends), routed=routed)
    if isinstance(step, Nearest):
        anchor = Origin(space_id=done[step.to.step].ids[0]) if step.to is not None else None
        among = done[step.assets.step]
        near = nearest(venue, anchor, [venue.assets[i] for i in among.ids])
        ids = near.asset_ids if near is not None else ()
        outcome = Outcome.RESOLVED if near is not None else Outcome.NONE
        return _result(step, mentions, outcome, ids, matched=len(ids), found=among.found, near=near)
    source = done[step.assets.step].found
    assert source is not None
    counted = aggregate(venue, source, step.by)
    ids = (*(g.space_id for g in counted.ranked), *counted.unrecorded)
    return _result(
        step, mentions, counted.outcome, ids, matched=len(ids), found=source, aggregation=counted
    )


def _result(
    step: Step,
    mentions: dict[str, Mention],
    outcome: Outcome,
    ids: tuple[str, ...] = (),
    **rest: Any,
) -> StepResult:
    mention = step.mention if isinstance(step, Resolving) else None
    return StepResult(
        step=step.id,
        tool=step.tool,
        args=step.model_dump(mode="json", exclude={"id", "tool"}),
        outcome=outcome,
        ids=ids,
        mention=mention,
        text=mentions[mention].text if mention is not None else None,
        **rest,
    )
