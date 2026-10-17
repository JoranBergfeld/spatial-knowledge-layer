"""The turn being built, and the ways it ends without composing anything.

A clarification, an abstention with its cause, a coverage gap, or giving up after a
second trigger. All said by the host: no model call. Plus what an answer leaves in
the state for the next turn.

A coverage gap is an abstention with a cause of its own, and says which gap. There
are two, and nothing else gives a reason:

    no recorded installations   nothing is installed in a building, or in something
                                bigger. Stated only at building level or above. A
                                room or floor with nothing in it is named as a gap
                                only through its building, when the whole building
                                is one. Otherwise the turn abstains without one.
    attribute not recorded      none of the assets asked about carries the attribute
                                filtered on: a fill rate of zero. Names the attribute.

Both only ever speak of what the persona can see.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from spatial_core.records import Space
from spatial_service import attribute_answers
from spatial_service.assets import ScopeResult
from spatial_service.conversation import (
    ASSET_CARRY_CAP,
    Carried,
    ConversationState,
    Pending,
    advance,
    own,
)
from spatial_service.executor import StepResult
from spatial_service.facts import (
    POSITION_NOT_RECORDED,
    POSITION_RECORDED,
    Citation,
    FactKind,
    asset_citation,
)
from spatial_service.grounding import Segment
from spatial_service.llm import Llm, LlmCall, LlmUnavailable, Purpose
from spatial_service.places import BUILDING, Outcome, Venue, labelled, tier
from spatial_service.plan import Filter, FollowUpTarget, MentionKind, QuestionShape
from spatial_service.recovery import RecoveryOutcome, Trigger, TriggerKind
from spatial_service.retrieval import EmbeddingCall, RetrievalUnavailable
from spatial_service.route_answer import RouteOut
from spatial_service.turn_models import (
    CALL_CAP,
    AbstentionCause,
    Candidate,
    CoverageGap,
    PlanVersion,
    Trace,
    Turn,
    TurnKind,
    TurnRequest,
)

ANSWER_UNAVAILABLE = "The language model could not be reached, so there is no answer."
CLARIFY = "More than one place matches. Which one do you mean?"
CLARIFY_ASSET = "That could be more than one asset. Which one do you mean?"
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
    AbstentionCause.NO_DOCUMENT: "I don't know of a document about that.",
    AbstentionCause.UNGROUNDED: ("I found documents but couldn't tie an answer to what they say."),
    AbstentionCause.NOTHING_RECORDED: "I can't answer that from what is recorded.",
    AbstentionCause.NOTHING_REFERRED_TO: "I don't know what that refers to.",
    AbstentionCause.NO_ROUTE: "There is no route on the mapped graph for this persona.",
    AbstentionCause.NOTHING_NEAR: "I don't know of one near there.",
}
# What a step that found nothing makes of the turn, by its tool.
NOTHING_FOUND = {
    "resolve_place": AbstentionCause.NO_SUCH_PLACE,
    "resolve_asset_type": AbstentionCause.NO_SUCH_ASSET_TYPE,
    "aggregate": AbstentionCause.NOTHING_AT_SCALE_LEVEL,
    "places_matching": AbstentionCause.NO_SUCH_PLACE,
    "route": AbstentionCause.NO_ROUTE,
    "nearest": AbstentionCause.NOTHING_NEAR,
}
DECLINED_CAUSE = {
    QuestionShape.STEP_DEPENDENT: AbstentionCause.STEP_DEPENDENT_QUESTION,
    QuestionShape.UNSUPPORTED: AbstentionCause.UNSUPPORTED_QUESTION,
}


@dataclass
class OpenTurn:
    """A turn being built up."""

    state: ConversationState
    trace: Trace
    calls: list[LlmCall]
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED
    embeddings: list[EmbeddingCall] = field(default_factory=list)
    question: str = ""
    routes: list[RouteOut] = field(default_factory=list)

    @property
    def number(self) -> int:
        """Which turn of the conversation this is, from 1."""
        return self.state.turns + 1

    def next(self, **update: Any) -> ConversationState:
        """The state this turn hands back, with whatever it resolved (conversation.advance)."""
        return advance(self.state, self.question, **update)

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
            embedding_calls=self.embeddings,
            recovery=self.recovery,
        )

    def abstain(self, cause: AbstentionCause) -> Turn:
        return self.finish(
            TurnKind.ABSTENTION,
            [Segment(text=ABSTAIN[cause], citations=[])],
            cause=cause,
            state=self.next(),
        )

    def unavailable(self, error: LlmUnavailable | RetrievalUnavailable) -> Turn:
        # The state goes back as it came, so the client can ask again.
        self.note(unavailable=str(error.purpose))
        return self.finish(TurnKind.UNAVAILABLE, [Segment(text=ANSWER_UNAVAILABLE, citations=[])])


def give_up(turn: OpenTurn, venue: Venue, trigger: Trigger | None = None) -> Turn:
    """A second trigger: ask again what is pending, or else abstain. Never a third call."""
    pending = turn.state.pending
    if pending is not None:
        return _ask_again(turn, venue, pending)
    if trigger is not None and trigger.kind is TriggerKind.EMPTY_RETRIEVAL:
        return turn.abstain(AbstentionCause.NO_DOCUMENT)
    return turn.abstain(AbstentionCause.PLAN_REJECTED)


def _ask_again(turn: OpenTurn, venue: Venue, pending: Pending) -> Turn:
    return clarify(
        turn, venue, pending.mention, tuple(pending.candidates), pending.kind, pending.turn
    )


def not_referred(turn: OpenTurn, venue: Venue, last: StepResult) -> Turn:
    """A follow-up reference that did not come to one thing: ask which, or abstain."""
    referred = last.referred
    assert referred is not None and last.text is not None
    if last.outcome is Outcome.AMBIGUOUS:
        return clarify(turn, venue, last.text, referred.ids, referred.kind)
    # "The fifth one" of three. The candidates still stand, so they are offered again.
    if referred.to is FollowUpTarget.CANDIDATE and turn.state.pending is not None:
        return _ask_again(turn, venue, turn.state.pending)
    return turn.abstain(AbstentionCause.NOTHING_REFERRED_TO)


def told_last_route(route: Carried, venue: Venue) -> dict[str, Any]:
    """The last route as the planner hears it. Names, never ids, like the rest of it."""
    labels = [
        space.label.value if space.label else None
        for space in map(venue.spaces.get, route.space_ids)
        if space
    ]
    said: dict[str, Any] = {"step_free": bool(route.args.get("step_free"))}
    if labels and labels[-1]:
        said["to"] = labels[-1]
    if len(labels) > 1 and labels[0]:
        said["from"] = labels[0]
    return said


def last_route(turn: OpenTurn, route: RouteOut) -> Carried:
    """The last route as state carries it: its ends and whether it was step-free.

    Ids only in the id list, origin first, so the next turn's re-filter sees both. A
    default origin is not an id: it is whatever the venue says next time too. Seconds
    and metres are not carried. A follow-up routes again.
    """
    ends = [route.destination.space_id]
    if not route.origin.default:
        ends.insert(0, route.origin.space_id)
    return Carried(
        kind="route",
        turn=turn.number,
        space_ids=ends,
        args={"step_free": route.step_free, "default_origin": route.origin.default},
    )


def carried(venue: Venue, last: StepResult, found: ScopeResult) -> dict[str, Any]:
    """What an asset answer leaves in state. A slot it has nothing for stays as it was."""
    update: dict[str, Any] = {}
    place = found.scope
    counted = last.aggregation
    if counted is not None:
        # "Which floor has the most?" then "what is in there?": there is the one that won.
        leaders = [g.space_id for g in counted.leaders]
        if len(leaders) == 1 and venue.spaces[leaders[0]].label is not None:
            place = leaders[0]
    if place is not None:
        update["place"] = place
    if found.types:
        update["asset_types"] = list(found.types)
    # A count per space is about the spaces. Only a containment answer is about its
    # assets, and only a handful can be pointed back at.
    few = counted is None and len(found.ids) <= ASSET_CARRY_CAP
    update["assets"] = list(found.ids) if few else []
    return update


def own_state(request: TurnRequest, snapshot: str, venue: Venue) -> ConversationState:
    """The prior state, keeping only ids the persona can see. A fresh one if there is none."""
    if request.state is None:
        return ConversationState(snapshot=snapshot, persona=request.persona)
    return own(request.state, venue)


def clarify(
    turn: OpenTurn,
    venue: Venue,
    mention: str,
    ids: tuple[str, ...],
    kind: MentionKind = MentionKind.PLACE,
    asked: int | None = None,
) -> Turn:
    """Ask which of ``ids`` was meant. They come from the venue, so all are visible."""
    candidates = []
    citations = []
    for item in ids:
        asset = venue.assets[item] if kind is MentionKind.ASSET else None
        space = venue.spaces[asset.space_id if asset else item]
        building = venue.building_of(space)
        within = labelled(building) if building and building.label else None
        if asset is None:
            candidates.append(Candidate(space_id=item, label=labelled(space), building=within))
            citations.append(_place_citation(space))
            continue
        candidates.append(
            Candidate(
                space_id=space.id,
                label=asset.name.value,
                building=within,
                asset_id=asset.id,
                place=space.label.value if space.label else None,
                position=POSITION_RECORDED if asset.position else POSITION_NOT_RECORDED,
            )
        )
        citations.append(asset_citation(asset))
    # Asked again, the candidates are still from the turn that first offered them.
    pending = Pending(mention=mention, candidates=list(ids), kind=kind, turn=asked or turn.number)
    text = CLARIFY_ASSET if kind is MentionKind.ASSET else CLARIFY
    return turn.finish(
        TurnKind.CLARIFICATION,
        [Segment(text=text, citations=citations)],
        candidates=candidates,
        state=turn.next(pending=pending),
    )


def nothing_installed(
    turn: OpenTurn, venue: Venue, found: ScopeResult, state: ConversationState
) -> Turn:
    """Nothing visible is installed in scope. A coverage gap only at building level or above.

    A building or anything bigger with nothing in it is a gap and is named. A floor
    or room is not named: if its whole building is a gap the building is, and if not
    the turn abstains and gives no reason. Said by the host: no count, no model.
    """
    if found.scope is None:
        gap = venue.root
    else:
        scope = venue.spaces[found.scope]
        gap = next(
            (s for s in (scope, *venue.ancestors(scope)) if _building_or_above(venue, s)), None
        )
        if gap is None or venue.assets_under.get(gap.id):
            return turn.abstain(AbstentionCause.NOTHING_RECORDED)
    if gap is not None and gap.label is not None:
        segment = Segment(
            text=NO_INSTALLATIONS.format(scope=gap.label.value), citations=[_place_citation(gap)]
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


def _building_or_above(venue: Venue, space: Space) -> bool:
    """Whether a space's tier is the building tier or one outside it."""
    if BUILDING not in venue.tiers:
        return space.parent_id is None
    return tier(space) in venue.tiers[: venue.tiers.index(BUILDING) + 1]


def attribute_gap(
    turn: OpenTurn, venue: Venue, found: ScopeResult, unfilled: Filter, state: ConversationState
) -> Turn:
    """Assets were asked about and none carries the attribute. Named, by the host."""
    scope = venue.spaces[found.scope] if found.scope is not None else None
    has_label = scope is not None and scope.label is not None
    text = attribute_answers.gap_text(unfilled, labelled(scope) if scope and has_label else None)
    citations = [_place_citation(scope)] if scope and has_label else []
    return turn.finish(
        TurnKind.ABSTENTION,
        [Segment(text=text, citations=citations)],
        cause=AbstentionCause.COVERAGE_GAP,
        gap=CoverageGap.ATTRIBUTE_NOT_RECORDED,
        state=state,
    )


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
