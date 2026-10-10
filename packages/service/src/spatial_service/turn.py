"""The turn: a persona, a snapshot, prior state and a question in; a turn with its trace out.

    plan        the planning model reads the question and the collapsed tree
                (tree.py) and writes a plan, which is checked in code before any
                tool runs
    execute     the steps run in order with the venue the host bound: what this
                persona can see in this snapshot. No model call between them.
    recover     only on a typed trigger (recovery.py): the planning deployment gets
                its plan back with the reasons and writes a second, which is checked
                and run the same way. A second trigger ends the turn.
    compose     the composing model words the answer over facts by reference; the
                host checks it and fills in every label and number

So a turn makes at most three model calls, however many steps run: CALL_CAP, which
the host enforces rather than trusts.

Every turn is one of four kinds. An answer. A clarification, when a mention names
more than one place the persona can see. An abstention, with its cause. Or
unavailable, when a model could not be reached, which answers nothing at all.

A coverage gap is an abstention with a cause of its own, and says which gap: so
far only that the scope asked about has no recorded installations, which is a gap
in the data and not a count of zero.

A route that cannot be found is one abstention, whatever stopped it: clearance,
the crossing rule or a gap in the mapping. Told apart, they would say where the
restricted corridors are.

Nothing the persona cannot see reaches a model: the planner gets the question,
the collapsed tree of visible spaces and the labels of pending candidates, and the
composer gets references and kinds only. Nothing counts what was left out, so
nothing can say so.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spatial_core.records import Asset, AssetType, Persona, Space
from spatial_service import compose, route_answer
from spatial_service.assets import ScopeResult
from spatial_service.compose import Answer, Citation, CompositionRejected, FactKind, Segment
from spatial_service.executor import Execution, StepResult, execute
from spatial_service.llm import Llm, LlmCall, LlmUnavailable, Purpose
from spatial_service.places import Outcome, Venue
from spatial_service.plan import (
    ALTERNATIVES_CAP,
    DECLINED,
    ROUTE_STEP_CAP,
    STEP_CAP,
    Plan,
    PlanRejected,
    QuestionShape,
    parse_plan,
)
from spatial_service.recovery import PlanChange, RecoveryOutcome, Trigger, diff
from spatial_service.route_answer import RouteOut, RouteTrace
from spatial_service.store import Store
from spatial_service.tree import NODE_CAP, Collapsed, collapse

# Plan, recover, compose.
CALL_CAP = 3
# How much of a plan that is not JSON the trace and the recovery call keep.
UNPARSED_CAP = 2000
# How many assets the trace lists one by one. Past that it still accounts for every
# asset, by provenance class, and the steps still carry every id.
TRACE_ASSET_CAP = 200

POSITION_RECORDED = "recorded"
POSITION_NOT_RECORDED = "position not recorded"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pending(_Model):
    """A clarification asked and not yet answered."""

    mention: str
    candidates: list[str]


class ConversationState(_Model):
    """What a conversation carries from one turn to the next. The client holds it."""

    snapshot: str
    persona: Persona
    turns: int = 0
    # The place the last answer was about: the one located, or the scope looked under.
    place: str | None = None
    # The asset types the last answer was about.
    asset_types: list[str] = []
    pending: Pending | None = None


class TurnRequest(_Model):
    persona: Persona
    # None starts on the current snapshot. A conversation stays on the one it began on.
    snapshot: str | None = None
    state: ConversationState | None = None
    question: str = Field(min_length=1, max_length=500)


class TurnKind(StrEnum):
    ANSWER = "answer"
    CLARIFICATION = "clarification"
    ABSTENTION = "abstention"
    UNAVAILABLE = "unavailable"


class AbstentionCause(StrEnum):
    # Nothing the persona can see matches: either nothing does, or only places
    # beyond their clearance do. The two are one cause on purpose.
    NO_SUCH_PLACE = "no-such-place"
    # The same for an asset type: none by that name, or none the persona can see.
    NO_SUCH_ASSET_TYPE = "no-such-asset-type"
    # The scope holds no visible space of the scale level asked for.
    NOTHING_AT_SCALE_LEVEL = "nothing-at-scale-level"
    # The next step would depend on what an earlier one found.
    STEP_DEPENDENT_QUESTION = "step-dependent-question"
    UNSUPPORTED_QUESTION = "unsupported-question"
    PLAN_REJECTED = "plan-rejected"
    COMPOSITION_REJECTED = "composition-rejected"
    # The venue's data does not reach far enough. The turn's `gap` says where. The
    # metrics count it apart (metrics.py).
    COVERAGE_GAP = "coverage-gap"
    # No route on the mapped graph for this persona. One cause for every reason
    # there might be none, on purpose.
    NO_ROUTE = "no-route"
    # Nothing of the kind asked for can be placed near there, by route or by containment.
    NOTHING_NEAR = "nothing-near"


class CoverageGap(StrEnum):
    """Which gap a coverage-gap abstention found."""

    # Nothing the persona can see is installed in the scope asked about.
    NO_RECORDED_INSTALLATIONS = "no-recorded-installations"


class Candidate(_Model):
    space_id: str
    label: str
    # The label of the building it is in, to tell candidates apart.
    building: str | None


class TraceStep(_Model):
    id: str
    tool: str
    args: dict[str, Any]
    # The words a resolving step looked up.
    text: str | None
    outcome: Outcome
    ids: list[str]
    how: str | None
    # Why the tool rejected its arguments, when it did.
    rejection: list[str] = []


class Verdict(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class PlanVersion(_Model):
    """One plan the turn got: what the model wrote, whether it may run, what it ran."""

    version: int
    # As the model wrote it, when it is a JSON object, checked or not.
    plan: dict[str, Any] | None = None
    # The model's output when it is not, cut to UNPARSED_CAP characters.
    unparsed: str | None = None
    verdict: Verdict
    rejection: list[str] = []
    steps: list[TraceStep] = []
    trigger: Trigger | None = None
    # A second version as changes to the first.
    changes: list[PlanChange] | None = None


class Resolved(_Model):
    mention: str
    text: str
    step: str
    space_id: str
    label: str
    how: str | None


class ResolvedTypes(_Model):
    mention: str
    text: str
    step: str
    type_ids: list[str]
    names: list[str]


class Provenance(_Model):
    space_id: str
    record_provenance: str
    source: str | None
    label: str
    label_provenance: str
    label_rule: str | None


class AuthoredDefault(_Model):
    """A clearance no tag gave, which a rule supplied, deciding what the persona sees."""

    space_id: str
    persona: Persona
    rule: str


class TreeMeasure(_Model):
    """How far the tree the planner saw was collapsed. Measured each turn, not assumed."""

    # Levels below the root that were shown, and that the visible tree has.
    depth: int
    full_depth: int
    # Spaces shown, and spaces visible.
    nodes: int
    full_nodes: int
    node_cap: int


class TypeTrace(_Model):
    type_id: str
    record_provenance: str
    name: str
    name_provenance: str
    name_rule: str | None
    name_basis: str | None
    omniclass: str
    # The lowest persona cleared for it, and the rule that said so.
    clearance: Persona
    clearance_rule: str
    authored_default: bool
    unrecognised_category: bool


class AssetTrace(_Model):
    """One asset an answer touched: both provenance fields, with the rule or basis."""

    asset_id: str
    type_id: str
    space_id: str
    record_provenance: str
    source: str | None
    name: str
    name_provenance: str
    name_rule: str | None
    name_basis: str | None
    # "recorded", or "position not recorded". Never coordinates.
    position: str
    position_provenance: str | None
    position_rule: str | None
    position_basis: str | None


class ProvenanceClass(_Model):
    """Assets that share every provenance field, and how many of them there are."""

    record_provenance: str
    name_provenance: str
    name_rule: str | None
    name_basis: str | None
    position: str
    position_provenance: str | None
    position_rule: str | None
    position_basis: str | None
    assets: int


class CountRow(_Model):
    space_id: str | None
    type_id: str | None
    label: str | None
    sourced: int
    generated: int
    # False for a space with no recorded installations. It has no rank.
    recorded: bool
    rank: int | None


class AssetsTrace(_Model):
    # The scope node, or None for the whole venue.
    scope: str | None
    types: list[str] | None
    # The scale level counted by, for an aggregation.
    grouped_by: str | None
    # How many assets the answer rests on, and how many of any type the scope holds.
    touched: int
    installed_in_scope: int
    # Each asset, up to TRACE_ASSET_CAP of them.
    listed: list[AssetTrace]
    # Every asset, by provenance class.
    classes: list[ProvenanceClass]
    rows: list[CountRow]
    # Assets in scope under no space of the scale level counted by.
    ungrouped: int = 0


class Trace(_Model):
    snapshot: str
    persona: Persona
    # The shape of the plan that ran to the end, if one did.
    shape: QuestionShape | None = None
    step_cap: int = STEP_CAP
    call_cap: int = CALL_CAP
    tree: TreeMeasure | None = None
    # The first plan and, after a trigger, the second.
    plans: list[PlanVersion] = []
    resolved: list[Resolved] = []
    resolved_types: list[ResolvedTypes] = []
    provenance: list[Provenance] = []
    authored_defaults: list[AuthoredDefault] = []
    asset_types: list[TypeTrace] = []
    assets: AssetsTrace | None = None
    # The persona's graph and what each route or nearest step did on it.
    route: RouteTrace | None = None
    composition_rejection: list[str] = []
    unavailable: str | None = None


class Turn(_Model):
    kind: TurnKind
    cause: AbstentionCause | None = None
    # Which gap, when the cause is a coverage gap.
    gap: CoverageGap | None = None
    segments: list[Segment]
    candidates: list[Candidate] = []
    # Every route the answer rests on, in plan order, each complete in itself.
    routes: list[RouteOut] = []
    state: ConversationState
    trace: Trace
    llm_calls: list[LlmCall]
    # Whether this turn needed a second plan, and whether that one ran its steps.
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED


class TurnRefused(Exception):
    """The request cannot be served as asked: a snapshot not served, a state not its own."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


PLANNING_INSTRUCTIONS = f"""\
You plan how to answer one question about a venue: its buildings, floors and rooms,
and the assets installed in them. Return only a plan, as JSON matching the schema.

"venue" shows the top of the venue's containment tree, with what lies below the
shown part as counts per tier, and "tiers", the venue's scale levels, outermost
first. Use them to read the question. You cannot expand the tree: name a place in
words and a step will find it.

shape:
- "place-location": the question asks where a place is.
- "asset-containment": it asks what is installed in a place, or where assets of a
  type are within a place.
- "asset-aggregation": it asks to compare a count across a scale level ("which
  floor has the most lights").
- "proximity": it asks for the nearest asset of a type ("the nearest extinguisher
  to room 107").
- "route": it asks how to get somewhere, or how long or how far that is.
- "step-dependent": answering needs the result of one step to decide the next
  ("what is in the room with the most lights"). No steps.
- "unsupported": anything else. No steps.

mentions: each place and each asset type the question names, with the words that
name it, verbatim ("107", "50.34 room 107", "Audimax"; "lights", "AC units"), and
its kind, "place" or "asset-type". Ids m1, m2, ... A scale level on its own
("floor" in "which floor") is not a mention.

constraints: {{"kind": "within", "mention": a, "container": b}} when the question
puts place a inside place b ("room 107 in 50.34").

steps: ids s1, s2, ..., at most {STEP_CAP}. Every step but the last feeds a later one.
- resolve_place, one per place mention. Resolve a container before what lies
  within it, and point the inner step at it with "within": {{"step": "<id>"}}.
  A place-location plan ends in the step for the place asked about.
- resolve_asset_type, one per asset-type mention.
- assets_in_scope: every asset under one place. "scope": {{"step": "<the place
  step>"}}, left out for the whole venue. "types": {{"step": "<the type step>"}},
  left out for assets of any type. An asset-containment plan ends in it. It covers
  every space under the scope in one step: never plan a step per room.
- aggregate: {{"assets": {{"step": "<the assets_in_scope step>"}}, "by": "<tier>"}}
  counts those assets per space of that tier. "by" is one of "tiers". A scale level
  named in the question sets both: the place it names is the scope, the tier it
  compares across is "by". An asset-aggregation plan ends in it.
- places_matching, for a place mention that describes rather than names ("the
  nearest toilet", "a lift"): every place that fits, as a set. Only a route reads it.
- route: {{"destination": {{"step": "<id>"}}}}, the step for the place asked for: a
  resolve_place step, a places_matching step, or an assets_in_scope step for a route
  to an asset. It ends at the nearest. "origin": {{"step": "<a resolve_place step>"}}
  only when the question says where to start, otherwise leave it out. "step_free":
  true only when the question asks for no stairs, in whatever words. Never infer it.
  "alternatives": 2 or 3 only when the question asks for more than one option.
  For "to A and then to B", a second route with "after": {{"step": "<the first route>"}}.
  At most {ROUTE_STEP_CAP} route steps, at most {ALTERNATIVES_CAP} alternatives each.
  A route plan ends in a route step.
- nearest: {{"assets": {{"step": "<the assets_in_scope step>"}}}}, and "to": {{"step":
  "<a resolve_place step>"}} for the place to be near, left out when the question
  names none. A proximity plan ends in it.

If "pending" is present, the previous turn asked the user which place they meant
and listed the candidates. Read the question as the answer to that, and resolve
the candidate it picks by its label.
"""

RECOVERY_INSTRUCTIONS = (
    PLANNING_INSTRUCTIONS
    + """
"previous" holds the plan you already wrote for this question, which could not run,
with the trigger it raised and the reasons. Write the whole plan again, fixing what
the reasons name. Keep what they do not name as it was.
"""
)

_COMPOSE_RULES = """\
Return only JSON matching the schema: segments, each with its text and the facts it cites.

Name things only by their placeholders, written in braces: {f1}, {f2}, ... Never
write a place's name, a type's name or a number yourself, and never write any digit.
Cite every fact a segment rests on.
"""

_COUNT_RULES = """\
A count placeholder fills in as a number. Where the assets counted are generated it
fills in as the number followed by the word "generated", and next to one of those a
sourced count is followed by "sourced", so write the noun after the placeholder and
do not add either word yourself. Counts listed together are one count split by
provenance: name all of them or none, and never describe them as a single figure.
Name every fact in "required". Say only what the facts say: do not explain, qualify
or guess at anything they leave out.
"""

COMPOSITION_INSTRUCTIONS = {
    QuestionShape.PLACE_LOCATION: (
        "You answer where a place is, in one or two short sentences, in the question's "
        "language.\n" + _COMPOSE_RULES + "\nThe subject is the place asked about; the other "
        "facts are the places it lies within, innermost first, with their tiers.\n"
    ),
    QuestionShape.ASSET_CONTAINMENT: (
        "You answer what is installed in a place, briefly, in the question's language.\n"
        + _COMPOSE_RULES
        + "\n"
        + _COUNT_RULES
        + """
"scope" is the place asked about and "total" the count of what was found there.
"rows" breaks it down, by asset type or by space as "rows_by" says, most first:
"of" is the type or space and "counts" its count. "more_rows" counts rows left out,
"in_spaces_with_no_label" what sits in spaces that have no label to print, and
"position_not_recorded" the assets found whose position is not recorded.
"""
    ),
    QuestionShape.ROUTE: (
        "You answer how to get somewhere, briefly, in the question's language.\n"
        + _COMPOSE_RULES
        + """
"routes" are the routes found, in order. Each has its "destination", its "duration"
and its "distance", the places it passes through in "via", and how it changes level
in "changes_level_by". "origin" is where it starts, and "origin_is_default" says the
question named no start. A duration placeholder fills in as seconds and says that it
is modelled; a distance fills in as metres. Write no unit and no "about" yourself.
"alternatives" are routes to the next nearest places. Name every fact in "required".
Say only what the facts say: this is not a measurement and you do not know the way
in more detail than "via".
"""
    ),
    QuestionShape.PROXIMITY: (
        "You answer which asset of a kind is nearest, briefly, in the question's language.\n"
        + _COMPOSE_RULES
        + "\n"
        + _COUNT_RULES
        + """
"nearest" lists where the nearest ones are: "in" is the place and "counts" how many
are there. "types" are what they are. "anchor" is the place they are nearest to.
"basis" says what nearest means here. "modelled-duration": a route reaches them,
and "duration" and "distance" give it, already saying they are modelled.
"shared-containment": no route was found, and they are the ones in the same
"shared" place as the anchor. Then say they share that place, and do not call it a
distance.
"""
    ),
    QuestionShape.ASSET_AGGREGATION: (
        "You answer which spaces of one scale level hold the most of something, briefly, "
        "in the question's language.\n"
        + _COMPOSE_RULES
        + "\n"
        + _COUNT_RULES
        + """
"rows" are the spaces of the tier "grouped_by", most first, each with its "counts".
"leaders" hold the most; several share it on a tie, and none means every count is
zero. "no_recorded_installations" are spaces with nothing recorded in them at all:
say so for each, and never rank them or give them a count. "scope" is the place
the question was asked within and "total" the count across it.
"""
    ),
}

ANSWER_UNAVAILABLE = "The language model could not be reached, so there is no answer."
CLARIFY = "More than one place matches. Which one do you mean?"
NO_INSTALLATIONS = "No installations are recorded for {scope}."
NO_INSTALLATIONS_ANYWHERE = "No installations are recorded here."
ABSTAIN = {
    AbstentionCause.NO_SUCH_PLACE: "I don't know a place by that name.",
    AbstentionCause.NO_SUCH_ASSET_TYPE: "I don't know of anything installed by that name.",
    AbstentionCause.NOTHING_AT_SCALE_LEVEL: "I don't know of anything at that scale level there.",
    AbstentionCause.STEP_DEPENDENT_QUESTION: (
        "That needs one answer before the next question can be asked. Ask them one at a time."
    ),
    AbstentionCause.UNSUPPORTED_QUESTION: "I can't answer that kind of question yet.",
    AbstentionCause.PLAN_REJECTED: "I couldn't work out how to answer that.",
    AbstentionCause.COMPOSITION_REJECTED: "I found the place but couldn't word the answer.",
    AbstentionCause.NO_ROUTE: "There is no route on the mapped graph for this persona.",
    AbstentionCause.NOTHING_NEAR: "I don't know of one near there.",
}
# What a step that found nothing makes of the turn, by its tool.
_NOTHING = {
    "resolve_place": AbstentionCause.NO_SUCH_PLACE,
    "resolve_asset_type": AbstentionCause.NO_SUCH_ASSET_TYPE,
    "aggregate": AbstentionCause.NOTHING_AT_SCALE_LEVEL,
    "places_matching": AbstentionCause.NO_SUCH_PLACE,
    "route": AbstentionCause.NO_ROUTE,
    "nearest": AbstentionCause.NOTHING_NEAR,
}
_DECLINED = {
    QuestionShape.STEP_DEPENDENT: AbstentionCause.STEP_DEPENDENT_QUESTION,
    QuestionShape.UNSUPPORTED: AbstentionCause.UNSUPPORTED_QUESTION,
}


@dataclass
class _Turn:
    """A turn being built up."""

    state: ConversationState
    trace: Trace
    calls: list[LlmCall]
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED
    routes: list[RouteOut] = field(default_factory=list)

    async def call(
        self,
        llm: Llm,
        purpose: Purpose,
        *,
        instructions: str,
        payload: dict[str, Any],
        response_format: type[BaseModel],
    ) -> str:
        if len(self.calls) >= CALL_CAP:
            raise RuntimeError(f"a turn makes at most {CALL_CAP} model calls")
        text, call = await llm.call(
            purpose, instructions=instructions, payload=payload, response_format=response_format
        )
        self.calls.append(call)
        return text

    def note(self, **fields: Any) -> None:
        self.trace = self.trace.model_copy(update=fields)

    def add_version(self, version: PlanVersion) -> None:
        self.note(plans=[*self.trace.plans, version])

    def finish(
        self,
        kind: TurnKind,
        segments: list[Segment],
        *,
        cause: AbstentionCause | None = None,
        gap: CoverageGap | None = None,
        candidates: list[Candidate] | None = None,
        state: ConversationState | None = None,
    ) -> Turn:
        return Turn(
            kind=kind,
            cause=cause,
            gap=gap,
            segments=segments,
            candidates=candidates or [],
            # Only an answer carries routes: a turn that says nothing shows none.
            routes=self.routes if kind is TurnKind.ANSWER else [],
            state=state or self.state,
            trace=self.trace,
            llm_calls=self.calls,
            recovery=self.recovery,
        )

    def abstain(self, cause: AbstentionCause) -> Turn:
        state = self.state.model_copy(update={"turns": self.state.turns + 1, "pending": None})
        return self.finish(
            TurnKind.ABSTENTION,
            [Segment(text=ABSTAIN[cause], citations=[])],
            cause=cause,
            state=state,
        )

    def unavailable(self, error: LlmUnavailable) -> Turn:
        # The state goes back as it came, so the client can ask again.
        self.note(unavailable=str(error.purpose))
        return self.finish(TurnKind.UNAVAILABLE, [Segment(text=ANSWER_UNAVAILABLE, citations=[])])


class TurnService:
    def __init__(self, store: Store, llm: Llm, node_cap: int = NODE_CAP) -> None:
        self.store = store
        self.llm = llm
        self.node_cap = node_cap

    async def take(self, request: TurnRequest) -> Turn:
        snapshot = await asyncio.to_thread(self._snapshot, request)
        venue = await asyncio.to_thread(self.store.venue, snapshot, request.persona)
        state = _own_state(request, snapshot, venue)
        shown = collapse(venue, self.node_cap)
        measure = TreeMeasure(
            depth=shown.depth,
            full_depth=shown.full_depth,
            nodes=shown.nodes,
            full_nodes=shown.full_nodes,
            node_cap=shown.node_cap,
        )
        turn = _Turn(state, Trace(snapshot=snapshot, persona=request.persona, tree=measure), [])
        try:
            first = await self._attempt(turn, request.question, venue, shown, None)
            if first.trigger is None:
                return await self._settle(turn, request.question, venue, first)
            # Unrepaired until the second plan runs, so a failed recovery call counts.
            turn.recovery = RecoveryOutcome.UNREPAIRED
            second = await self._attempt(turn, request.question, venue, shown, first)
            if second.trigger is None:
                # A second plan with nothing to run is the model giving up, not a repair.
                if second.execution is not None:
                    turn.recovery = RecoveryOutcome.REPAIRED
                return await self._settle(turn, request.question, venue, second)
        except LlmUnavailable as error:
            return turn.unavailable(error)
        return _give_up(turn, venue)

    async def _attempt(
        self,
        turn: _Turn,
        question: str,
        venue: Venue,
        shown: Collapsed,
        previous: _Attempt | None,
    ) -> _Attempt:
        """One plan from the model, checked and, when it may, run."""
        payload: dict[str, Any] = {
            "question": question,
            "venue": {"tiers": list(shown.tiers), "tree": shown.tree},
        }
        if turn.state.pending is not None:
            payload["pending"] = {
                "mention": turn.state.pending.mention,
                "candidates": [_label(venue.spaces[i]) for i in turn.state.pending.candidates],
            }
        if previous is not None:
            assert previous.trigger is not None
            payload["previous"] = {
                "plan": previous.written,
                "trigger": previous.trigger.model_dump(mode="json", exclude_none=True),
            }
        text = await turn.call(
            self.llm,
            Purpose.PLANNING if previous is None else Purpose.RECOVERY,
            instructions=PLANNING_INSTRUCTIONS if previous is None else RECOVERY_INSTRUCTIONS,
            payload=payload,
            response_format=Plan,
        )
        number = 1 if previous is None else 2
        trigger: Trigger | None
        try:
            plan = parse_plan(text)
        except PlanRejected as rejected:
            written: Any = _json_object(text)
            if written is None:
                written = text[:UNPARSED_CAP]
            trigger = Trigger(kind=rejected.trigger, reasons=rejected.reasons)
            version = PlanVersion(
                version=number,
                plan=written if isinstance(written, dict) else None,
                unparsed=written if isinstance(written, str) else None,
                verdict=Verdict.REJECTED,
                rejection=rejected.reasons,
                trigger=trigger,
            )
            attempt = _Attempt(written, None, None, trigger)
        else:
            written = plan.model_dump(mode="json", exclude_none=True)
            execution = None if plan.shape in DECLINED else execute(plan, venue)
            trigger = execution.trigger if execution is not None else None
            version = PlanVersion(
                version=number,
                plan=written,
                verdict=Verdict.ACCEPTED,
                steps=_steps(execution) if execution is not None else [],
                trigger=trigger,
            )
            attempt = _Attempt(written, plan, execution, trigger)
        if previous is not None:
            version = version.model_copy(update={"changes": diff(previous.written, written)})
        turn.add_version(version)
        return attempt

    async def _settle(self, turn: _Turn, question: str, venue: Venue, attempt: _Attempt) -> Turn:
        """The turn a plan that raised no trigger ends in."""
        assert attempt.plan is not None
        shape = attempt.plan.shape
        turn.note(shape=shape)
        if attempt.execution is None:
            return turn.abstain(_DECLINED[shape])

        execution = attempt.execution
        turn.note(
            resolved=_resolved(execution, venue), resolved_types=_resolved_types(execution, venue)
        )
        turn.note(route=route_answer.trace(venue, execution))
        last = execution.last
        if last.outcome is Outcome.NONE:
            return turn.abstain(_NOTHING[last.tool])
        if last.outcome is Outcome.AMBIGUOUS:
            assert last.text is not None
            return _clarify(turn, venue, last.text, last.ids)

        found: ScopeResult | None = None
        if shape is QuestionShape.PLACE_LOCATION:
            subject = venue.spaces[last.ids[0]]
            answer = compose.location(venue, question, subject)
            state = turn.state.model_copy(update={"place": subject.id, "asset_types": []})
        elif shape is QuestionShape.ROUTE:
            runs = [r.routed for r in execution.by_tool("route") if r.routed is not None]
            answer = compose.route(venue, question, runs)
            turn.routes = route_answer.routes_of(venue, execution)
            # Where the last route ended. The route itself is in the turn.
            state = turn.state.model_copy(update={"place": last.ids[0], "asset_types": []})
        elif shape is QuestionShape.PROXIMITY:
            assert last.near is not None and last.found is not None
            answer = compose.proximity(venue, question, last.near)
            turn.routes = route_answer.routes_of(venue, execution)
            near = [venue.assets[i] for i in last.near.asset_ids]
            types = [venue.asset_types[i] for i in sorted({asset.type_id for asset in near})]
            turn.note(asset_types=[_type_trace(kind) for kind in types])
            state = turn.state.model_copy(
                update={"place": near[0].space_id, "asset_types": [kind.id for kind in types]}
            )
        else:
            found = last.found
            assert found is not None
            if last.aggregation is not None:
                answer = compose.aggregation(venue, question, found, last.aggregation)
            else:
                answer = compose.containment(venue, question, found)
            types = [venue.asset_types[i] for i in found.types or ()]
            turn.note(
                asset_types=[_type_trace(kind) for kind in types],
                assets=_assets_trace(venue, last, answer),
            )
            state = turn.state.model_copy(
                update={"place": found.scope, "asset_types": [kind.id for kind in types]}
            )
        state = state.model_copy(update={"turns": turn.state.turns + 1, "pending": None})
        places = [fact.space for fact in answer.facts if fact.space is not None]
        turn.note(provenance=[_provenance(s) for s in places], authored_defaults=_defaults(places))

        if found is not None and found.installed == 0:
            return _coverage_gap(turn, venue, found, state)

        text = await turn.call(
            self.llm,
            Purpose.COMPOSITION,
            instructions=COMPOSITION_INSTRUCTIONS[shape],
            payload=answer.payload,
            response_format=compose.Draft,
        )
        try:
            segments = compose.fill(text, answer.facts, answer.required, answer.together)
        except CompositionRejected as rejected:
            turn.note(composition_rejection=rejected.reasons)
            return turn.abstain(AbstentionCause.COMPOSITION_REJECTED)
        return turn.finish(TurnKind.ANSWER, segments, state=state)

    def _snapshot(self, request: TurnRequest) -> str:
        current, previous = self.store.served()
        held = request.state.snapshot if request.state else None
        if request.snapshot and held and request.snapshot != held:
            raise TurnRefused(409, f"the conversation is held on {held}, not {request.snapshot}")
        snapshot = request.snapshot or held or current
        if snapshot is None:
            raise TurnRefused(503, "no snapshot is current")
        if snapshot not in (current, previous):
            raise TurnRefused(409, f"{snapshot} is no longer served; start a new conversation")
        if request.state and request.state.persona is not request.persona:
            raise TurnRefused(409, "a conversation keeps the persona it began with")
        return snapshot


@dataclass(frozen=True)
class _Attempt:
    # The plan as written: a checked plan's dump, else the model's JSON or its text.
    written: Any
    plan: Plan | None
    execution: Execution | None
    trigger: Trigger | None


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _give_up(turn: _Turn, venue: Venue) -> Turn:
    """A second trigger: ask again what is pending, or else abstain. Never a third call."""
    pending = turn.state.pending
    if pending is not None:
        return _clarify(turn, venue, pending.mention, tuple(pending.candidates))
    return turn.abstain(AbstentionCause.PLAN_REJECTED)


def _own_state(request: TurnRequest, snapshot: str, venue: Venue) -> ConversationState:
    """The prior state, keeping only ids the persona can see. A fresh one if there is none."""
    if request.state is None:
        return ConversationState(snapshot=snapshot, persona=request.persona)
    state = request.state
    pending = state.pending
    if pending is not None:
        kept = [i for i in pending.candidates if i in venue]
        pending = pending.model_copy(update={"candidates": kept}) if kept else None
    place = state.place if state.place in venue else None
    types = [i for i in state.asset_types if i in venue.asset_types]
    return state.model_copy(update={"pending": pending, "place": place, "asset_types": types})


def _clarify(turn: _Turn, venue: Venue, mention: str, ids: tuple[str, ...]) -> Turn:
    candidates = []
    for space_id in ids:
        space = venue.spaces[space_id]
        building = venue.building_of(space)
        candidates.append(
            Candidate(
                space_id=space_id,
                label=_label(space),
                building=_label(building) if building and building.label else None,
            )
        )
    citations = [_place_citation(venue.spaces[space_id]) for space_id in ids]
    state = turn.state.model_copy(
        update={
            "turns": turn.state.turns + 1,
            "pending": Pending(mention=mention, candidates=list(ids)),
        }
    )
    return turn.finish(
        TurnKind.CLARIFICATION,
        [Segment(text=CLARIFY, citations=citations)],
        candidates=candidates,
        state=state,
    )


def _coverage_gap(turn: _Turn, venue: Venue, found: ScopeResult, state: ConversationState) -> Turn:
    """Nothing visible is installed in scope. Said as that, by the host: no count, no model."""
    scope = venue.spaces[found.scope] if found.scope is not None else None
    if scope is not None and scope.label is not None:
        segment = Segment(
            text=NO_INSTALLATIONS.format(scope=scope.label.value),
            citations=[_place_citation(scope)],
        )
    else:
        segment = Segment(text=NO_INSTALLATIONS_ANYWHERE, citations=[])
    return turn.finish(
        TurnKind.ABSTENTION,
        [segment],
        cause=AbstentionCause.COVERAGE_GAP,
        gap=CoverageGap.NO_RECORDED_INSTALLATIONS,
        state=state,
    )


def _label(space: Space) -> str:
    assert space.label is not None
    return space.label.value


def _place_citation(space: Space) -> Citation:
    assert space.label is not None
    return Citation(
        kind=FactKind.PLACE,
        space_id=space.id,
        label=space.label.value,
        provenance=space.label.provenance.value,
        rule=space.label.rule,
        basis=space.label.basis,
        record_provenance=space.record_provenance.value,
    )


def _steps(execution: Execution) -> list[TraceStep]:
    return [
        TraceStep(
            id=r.step,
            tool=r.tool,
            args=r.args,
            text=r.text,
            outcome=r.outcome,
            ids=list(r.ids),
            how=r.how,
            rejection=list(r.rejection),
        )
        for r in execution.results
    ]


def _resolved(execution: Execution, venue: Venue) -> list[Resolved]:
    return [
        Resolved(
            mention=r.mention or "",
            text=r.text or "",
            step=r.step,
            space_id=r.ids[0],
            label=_label(venue.spaces[r.ids[0]]),
            how=r.how,
        )
        for r in execution.by_tool("resolve_place")
        if r.outcome is Outcome.RESOLVED
    ]


def _resolved_types(execution: Execution, venue: Venue) -> list[ResolvedTypes]:
    return [
        ResolvedTypes(
            mention=r.mention or "",
            text=r.text or "",
            step=r.step,
            type_ids=list(r.ids),
            names=[venue.asset_types[i].name.value for i in r.ids],
        )
        for r in execution.by_tool("resolve_asset_type")
        if r.outcome is Outcome.RESOLVED
    ]


def _provenance(space: Space) -> Provenance:
    assert space.label is not None
    return Provenance(
        space_id=space.id,
        record_provenance=space.record_provenance.value,
        source=space.source,
        label=space.label.value,
        label_provenance=space.label.provenance.value,
        label_rule=space.label.rule,
    )


def _defaults(spaces: list[Space]) -> list[AuthoredDefault]:
    return [
        AuthoredDefault(space_id=s.id, persona=s.clearance.persona, rule=s.clearance.rule)
        for s in spaces
        if s.clearance is not None and s.clearance.authored_default
    ]


def _type_trace(kind: AssetType) -> TypeTrace:
    return TypeTrace(
        type_id=kind.id,
        record_provenance=kind.record_provenance.value,
        name=kind.name.value,
        name_provenance=kind.name.provenance.value,
        name_rule=kind.name.rule,
        name_basis=kind.name.basis,
        omniclass=kind.omniclass.value,
        clearance=kind.clearance.persona,
        clearance_rule=kind.clearance.rule,
        authored_default=kind.clearance.authored_default,
        unrecognised_category=kind.clearance.unrecognised_category,
    )


def _asset_provenance(asset: Asset) -> dict[str, Any]:
    """Both provenance fields of an asset, with the rule or basis. Never where it is."""
    position = asset.position
    return {
        "record_provenance": asset.record_provenance.value,
        "name_provenance": asset.name.provenance.value,
        "name_rule": asset.name.rule,
        "name_basis": asset.name.basis,
        "position": POSITION_RECORDED if position else POSITION_NOT_RECORDED,
        "position_provenance": position.provenance.value if position else None,
        "position_rule": position.rule if position else None,
        "position_basis": position.basis if position else None,
    }


def _assets_trace(venue: Venue, last: StepResult, answer: Answer) -> AssetsTrace:
    found = last.found
    assert found is not None
    assets = [venue.assets[i] for i in found.ids]
    provenance = [_asset_provenance(asset) for asset in assets]
    classes = Counter(tuple(fields.items()) for fields in provenance)
    return AssetsTrace(
        scope=found.scope,
        types=list(found.types) if found.types is not None else None,
        grouped_by=last.aggregation.by if last.aggregation else None,
        touched=len(assets),
        installed_in_scope=found.installed,
        listed=[
            AssetTrace(
                asset_id=asset.id,
                type_id=asset.type_id,
                space_id=asset.space_id,
                source=asset.source,
                name=asset.name.value,
                **fields,
            )
            for asset, fields in list(zip(assets, provenance, strict=True))[:TRACE_ASSET_CAP]
        ],
        classes=[ProvenanceClass(**dict(key), assets=count) for key, count in classes.items()],
        rows=[
            CountRow(
                space_id=row.space_id,
                type_id=row.type_id,
                label=row.label,
                sourced=row.tally.sourced,
                generated=row.tally.generated,
                recorded=row.recorded,
                rank=row.rank,
            )
            for row in answer.rows
        ],
        ungrouped=last.aggregation.ungrouped if last.aggregation else 0,
    )
