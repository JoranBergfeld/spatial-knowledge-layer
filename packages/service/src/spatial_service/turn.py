"""The turn: a persona, a snapshot, prior state and a question in; a turn with its trace out.

    plan        the planning model reads the question and writes a plan, which is
                checked in code before any tool runs
    execute     the steps run in order with the venue the host bound: this persona's
                visible spaces in this snapshot. No model call between them.
    recover     only on a typed trigger (recovery.py): the planning deployment gets
                its plan back with the reasons and writes a second, which is checked
                and run the same way. A second trigger ends the turn.
    compose     the composing model words the answer over facts by reference; the
                host checks it and fills in every label

So a turn makes at most three model calls, however many steps run: CALL_CAP, which
the host enforces rather than trusts.

Every turn is one of four kinds. An answer. A clarification, when a mention names
more than one place the persona can see. An abstention, with its cause. Or
unavailable, when a model could not be reached, which answers nothing at all.

Nothing the persona cannot see reaches a model: the planner gets the question
and the labels of pending candidates, which are visible by construction, and the
composer gets references and tiers only.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spatial_core.records import Persona, Space
from spatial_service import compose
from spatial_service.compose import Citation, CompositionRejected, Segment
from spatial_service.executor import Execution, execute
from spatial_service.llm import Llm, LlmCall, LlmUnavailable, Purpose
from spatial_service.places import Outcome, Venue
from spatial_service.plan import STEP_CAP, Plan, PlanRejected, QuestionShape, parse_plan
from spatial_service.recovery import PlanChange, RecoveryOutcome, Trigger, diff
from spatial_service.store import Store

# Plan, recover, compose.
CALL_CAP = 3
# How much of a plan that is not JSON the trace and the recovery call keep.
UNPARSED_CAP = 2000


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
    # The place the last answer was about.
    place: str | None = None
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
    UNSUPPORTED_QUESTION = "unsupported-question"
    PLAN_REJECTED = "plan-rejected"
    COMPOSITION_REJECTED = "composition-rejected"


class Candidate(_Model):
    space_id: str
    label: str
    # The label of the building it is in, to tell candidates apart.
    building: str | None


class TraceStep(_Model):
    id: str
    tool: str
    args: dict[str, Any]
    text: str
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


class Trace(_Model):
    snapshot: str
    persona: Persona
    # The shape of the plan that ran to the end, if one did.
    shape: QuestionShape | None = None
    step_cap: int = STEP_CAP
    call_cap: int = CALL_CAP
    # The first plan and, after a trigger, the second.
    plans: list[PlanVersion] = []
    resolved: list[Resolved] = []
    provenance: list[Provenance] = []
    authored_defaults: list[AuthoredDefault] = []
    composition_rejection: list[str] = []
    unavailable: str | None = None


class Turn(_Model):
    kind: TurnKind
    cause: AbstentionCause | None = None
    segments: list[Segment]
    candidates: list[Candidate] = []
    state: ConversationState
    trace: Trace
    llm_calls: list[LlmCall]
    # Whether this turn needed a second plan, and whether that one ran.
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED


class TurnRefused(Exception):
    """The request cannot be served as asked: a snapshot not served, a state not its own."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


PLANNING_INSTRUCTIONS = f"""\
You plan how to answer one question about a venue: its buildings, floors and rooms.
Return only a plan, as JSON matching the schema.

shape: "place-location" when the question asks where a place is. Anything else is
"unsupported", with no steps.

mentions: each place the question names, with the words that name it, verbatim
("107", "50.34 room 107", "Audimax"). Ids m1, m2, ...

constraints: {{"kind": "within", "mention": a, "container": b}} when the question
puts mention a inside mention b ("room 107 in 50.34").

steps: one resolve_place step per mention, ids s1, s2, ..., at most {STEP_CAP}.
Resolve a container before what lies within it, and point the inner step at it
with "within": {{"step": "<container step id>"}}. The last step resolves the place
the question asks about; every other step feeds a later one.

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

COMPOSITION_INSTRUCTIONS = """\
You answer where a place is, in one or two short sentences, in the question's language.
Return only JSON matching the schema: segments, each with its text and the facts it cites.

Name places only by their placeholders, written in braces: {f1}, {f2}, ... Never
write a place's name or number yourself, and never write any digit. The subject is
the place asked about; the other facts are the places it lies within, innermost
first, with their tiers. Cite every fact a segment rests on.
"""

ANSWER_UNAVAILABLE = "The language model could not be reached, so there is no answer."
CLARIFY = "More than one place matches. Which one do you mean?"
ABSTAIN = {
    AbstentionCause.NO_SUCH_PLACE: "I don't know a place by that name.",
    AbstentionCause.UNSUPPORTED_QUESTION: "So far I can only say where a place is.",
    AbstentionCause.PLAN_REJECTED: "I couldn't work out how to answer that.",
    AbstentionCause.COMPOSITION_REJECTED: "I found the place but couldn't word the answer.",
}


@dataclass
class _Turn:
    """A turn being built up."""

    state: ConversationState
    trace: Trace
    calls: list[LlmCall]
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED

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

    def add_version(self, version: PlanVersion) -> None:
        self.trace = self.trace.model_copy(update={"plans": [*self.trace.plans, version]})

    def finish(
        self,
        kind: TurnKind,
        segments: list[Segment],
        *,
        cause: AbstentionCause | None = None,
        candidates: list[Candidate] | None = None,
        state: ConversationState | None = None,
    ) -> Turn:
        return Turn(
            kind=kind,
            cause=cause,
            segments=segments,
            candidates=candidates or [],
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
        self.trace = self.trace.model_copy(update={"unavailable": str(error.purpose)})
        return self.finish(TurnKind.UNAVAILABLE, [Segment(text=ANSWER_UNAVAILABLE, citations=[])])


class TurnService:
    def __init__(self, store: Store, llm: Llm) -> None:
        self.store = store
        self.llm = llm

    async def take(self, request: TurnRequest) -> Turn:
        snapshot = await asyncio.to_thread(self._snapshot, request)
        venue = await asyncio.to_thread(self.store.venue, snapshot, request.persona)
        state = _own_state(request, snapshot, venue)
        turn = _Turn(state, Trace(snapshot=snapshot, persona=request.persona), [])
        try:
            first = await self._attempt(turn, request.question, venue, None)
            if first.trigger is None:
                return await self._settle(turn, request.question, venue, first)
            # Unrepaired until the second plan runs, so a failed recovery call counts.
            turn.recovery = RecoveryOutcome.UNREPAIRED
            second = await self._attempt(turn, request.question, venue, first)
            if second.trigger is None:
                turn.recovery = RecoveryOutcome.REPAIRED
                return await self._settle(turn, request.question, venue, second)
        except LlmUnavailable as error:
            return turn.unavailable(error)
        return _give_up(turn, venue)

    async def _attempt(
        self, turn: _Turn, question: str, venue: Venue, previous: _Attempt | None
    ) -> _Attempt:
        """One plan from the model, checked and, when it may, run."""
        payload: dict[str, Any] = {"question": question}
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
            execution = None if plan.shape is QuestionShape.UNSUPPORTED else execute(plan, venue)
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
        turn.trace = turn.trace.model_copy(update={"shape": attempt.plan.shape})
        if attempt.execution is None:
            return turn.abstain(AbstentionCause.UNSUPPORTED_QUESTION)

        execution = attempt.execution
        turn.trace = turn.trace.model_copy(update={"resolved": _resolved(execution, venue)})
        last = execution.last
        if last.outcome is Outcome.NONE:
            return turn.abstain(AbstentionCause.NO_SUCH_PLACE)
        if last.outcome is Outcome.AMBIGUOUS:
            return _clarify(turn, venue, last.text, last.ids)

        subject = venue.spaces[last.ids[0]]
        facts = compose.location_facts(venue, subject)
        turn.trace = turn.trace.model_copy(
            update={
                "provenance": [_provenance(f.space) for f in facts],
                "authored_defaults": _defaults([f.space for f in facts]),
            }
        )
        text = await turn.call(
            self.llm,
            Purpose.COMPOSITION,
            instructions=COMPOSITION_INSTRUCTIONS,
            payload=compose.composer_input(question, facts),
            response_format=compose.Draft,
        )
        try:
            segments = compose.fill(text, facts)
        except CompositionRejected as rejected:
            turn.trace = turn.trace.model_copy(update={"composition_rejection": rejected.reasons})
            return turn.abstain(AbstentionCause.COMPOSITION_REJECTED)

        state = turn.state.model_copy(
            update={"turns": turn.state.turns + 1, "place": subject.id, "pending": None}
        )
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
    return state.model_copy(update={"pending": pending, "place": place})


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
    citations = [
        Citation(
            space_id=space_id,
            label=_label(venue.spaces[space_id]),
            provenance=_value_provenance(venue.spaces[space_id]),
            rule=_rule(venue.spaces[space_id]),
        )
        for space_id in ids
    ]
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


def _label(space: Space) -> str:
    assert space.label is not None
    return space.label.value


def _value_provenance(space: Space) -> str:
    assert space.label is not None
    return space.label.provenance.value


def _rule(space: Space) -> str | None:
    assert space.label is not None
    return space.label.rule


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
            mention=r.mention,
            text=r.text,
            step=r.step,
            space_id=r.ids[0],
            label=_label(venue.spaces[r.ids[0]]),
            how=r.how,
        )
        for r in execution.results
        if r.outcome is Outcome.RESOLVED
    ]


def _provenance(space: Space) -> Provenance:
    return Provenance(
        space_id=space.id,
        record_provenance=space.record_provenance.value,
        source=space.source,
        label=_label(space),
        label_provenance=_value_provenance(space),
        label_rule=_rule(space),
    )


def _defaults(spaces: list[Space]) -> list[AuthoredDefault]:
    return [
        AuthoredDefault(space_id=s.id, persona=s.clearance.persona, rule=s.clearance.rule)
        for s in spaces
        if s.clearance is not None and s.clearance.authored_default
    ]
