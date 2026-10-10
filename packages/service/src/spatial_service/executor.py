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
from spatial_service.places import Outcome, Venue, check_arguments, resolve_place
from spatial_service.plan import (
    Aggregate,
    AssetsInScope,
    Mention,
    Plan,
    ResolveAssetType,
    ResolvePlace,
    RetrieveDocuments,
    Step,
)
from spatial_service.recovery import Trigger, TriggerKind
from spatial_service.retrieval import (
    Documents,
    Retrieval,
    anchors_of_assets,
    anchors_of_place,
    check_query,
    document_filter,
)

# The one reason an empty retrieval gives. It never says why nothing came back.
NOTHING_RETRIEVED = "retrieval returned nothing"


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
    # What retrieve_documents searched for and got.
    retrieval: Retrieval | None = None


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
        """A trigger the run raised: a tool rejected its arguments, or retrieval got nothing."""
        if self.last.retrieval is not None and self.last.outcome is Outcome.NONE:
            return Trigger(
                kind=TriggerKind.EMPTY_RETRIEVAL, step=self.last.step, reasons=[NOTHING_RETRIEVED]
            )
        if self.last.outcome is not Outcome.REJECTED:
            return None
        return Trigger(
            kind=TriggerKind.MALFORMED_ARGUMENTS,
            step=self.last.step,
            reasons=list(self.last.rejection),
        )

    def by_tool(self, tool: str) -> list[StepResult]:
        return [r for r in self.results if r.tool == tool]


def execute(plan: Plan, venue: Venue, documents: Documents | None = None) -> Execution:
    """Run the plan. ``documents`` is the retrieval the host bound; with none, nothing is found."""
    mentions = {mention.id: mention for mention in plan.mentions}
    for step in plan.steps:
        problems = _problems(step, mentions, venue)
        if problems:
            rejected = _result(step, mentions, Outcome.REJECTED, rejection=tuple(problems))
            return Execution((rejected,))

    done: dict[str, StepResult] = {}
    results: list[StepResult] = []
    for step in plan.steps:
        if isinstance(step, RetrieveDocuments):
            result = _retrieve(step, mentions, venue, done, documents)
        else:
            result = _run(step, mentions, venue, done)
        results.append(result)
        if result.outcome is not Outcome.RESOLVED:
            break
        done[step.id] = result
    return Execution(tuple(results))


def _problems(step: Step, mentions: dict[str, Mention], venue: Venue) -> list[str]:
    """Why a step's arguments are malformed. Nothing here reads what a persona can see."""
    if isinstance(step, ResolvePlace):
        return check_arguments(mentions[step.mention].text)
    if isinstance(step, ResolveAssetType):
        return check_type_arguments(mentions[step.mention].text)
    if isinstance(step, Aggregate):
        return check_tier(venue, step.by)
    if isinstance(step, RetrieveDocuments):
        return check_query(step.query)
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
    assert isinstance(step, Aggregate)
    source = done[step.assets.step].found
    assert source is not None
    counted = aggregate(venue, source, step.by)
    ids = (*(g.space_id for g in counted.ranked), *counted.unrecorded)
    return _result(
        step, mentions, counted.outcome, ids, matched=len(ids), found=source, aggregation=counted
    )


def _retrieve(
    step: RetrieveDocuments,
    mentions: dict[str, Mention],
    venue: Venue,
    done: dict[str, StepResult],
    documents: Documents | None,
) -> StepResult:
    """The passages anchored to what an earlier step found. None at all is a trigger."""
    source = done[step.anchors.step]
    if source.found is not None:
        anchors = anchors_of_assets(venue, source.found)
    else:
        # A place step that got this far resolved exactly one place.
        anchors = anchors_of_place(venue, source.ids[0])
    if documents is None:
        found = Retrieval(step.query, anchors, document_filter(venue.persona, anchors))
    else:
        found = documents.retrieve(venue, anchors, step.query)
    ids = tuple(passage.chunk.id for passage in found.passages)
    return _result(
        step,
        mentions,
        Outcome.RESOLVED if ids else Outcome.NONE,
        ids,
        matched=len(ids),
        found=source.found,
        retrieval=found,
    )


def _result(
    step: Step,
    mentions: dict[str, Mention],
    outcome: Outcome,
    ids: tuple[str, ...] = (),
    **rest: Any,
) -> StepResult:
    mention = step.mention if isinstance(step, (ResolvePlace, ResolveAssetType)) else None
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
