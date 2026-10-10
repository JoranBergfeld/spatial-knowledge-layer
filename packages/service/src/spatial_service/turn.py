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
more than one place the persona can see, or a follow-up reference points at more
than one asset. An abstention, with its cause. Or unavailable, when a model could
not be reached, which answers nothing at all.

A later turn can lean on an earlier one. The prior state comes in with the request,
is cut to what the persona can see before anything reads it, and the new state goes
back out (conversation.py). The trace shows each follow-up reference, what it
resolved to and from which turn.

A coverage gap is an abstention with a cause of its own, and says which gap: so
far only that the scope asked about has no recorded installations, which is a gap
in the data and not a count of zero.

Nothing the persona cannot see reaches a model: the planner gets the question,
the collapsed tree of visible spaces, the earlier questions and the names of what
the state still holds once it is cut down, and the composer gets references and
kinds only. Nothing counts what was left out, so
nothing can say so.

A document answer adds a second list to composition, the passages retrieved this
turn (retrieval.py), which the composer does get the text of: they passed the
clearance filter. It is settled apart (_settle_documents), since a segment about a
passage can be dropped on its own where a wrong fact rejects the whole answer.
Each segment goes out with the full text of the passages it cites, for the
entailment check, which is not done here. A query embedding is a paid call but
not a model call of the turn, so it is reported in a list of its own.
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
from spatial_service import compose
from spatial_service.assets import ScopeResult
from spatial_service.compose import (
    POSITION_NOT_RECORDED,
    POSITION_RECORDED,
    Answer,
    Citation,
    CompositionRejected,
    FactKind,
    Segment,
)
from spatial_service.conversation import (
    ASSET_CARRY_CAP,
    ConversationState,
    Pending,
    advance,
    name,
    offered,
    own,
    told,
)
from spatial_service.executor import Execution, StepResult, execute
from spatial_service.llm import Llm, LlmCall, LlmUnavailable, Purpose
from spatial_service.places import Outcome, Venue
from spatial_service.plan import (
    DECLINED,
    STEP_CAP,
    FollowUpTarget,
    MentionKind,
    Plan,
    PlanRejected,
    QuestionShape,
    parse_plan,
)
from spatial_service.recovery import PlanChange, RecoveryOutcome, Trigger, TriggerKind, diff
from spatial_service.retrieval import (
    Documents,
    DocumentsTrace,
    DroppedSegment,
    EmbeddingCall,
    RetrievalUnavailable,
)
from spatial_service.retrieval import trace as documents_trace
from spatial_service.store import Store
from spatial_service.tree import NODE_CAP, Collapsed, collapse

# Plan, recover, compose.
CALL_CAP = 3
# How much of a plan that is not JSON the trace and the recovery call keep.
UNPARSED_CAP = 2000
# How many assets the trace lists one by one. Past that it still accounts for every
# asset, by provenance class, and the steps still carry every id.
TRACE_ASSET_CAP = 200

# What a refusal is, for the caller to act on. A snapshot that was dropped and one that
# never existed are the same thing to a conversation: it ends, with a plain message.
SNAPSHOT_DROPPED = "snapshot-dropped"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
    # Retrieval returned nothing, twice. No document the persona can read is
    # anchored there: none exists, or none within their clearance. One cause.
    NO_DOCUMENT = "no-document"
    # Passages were retrieved, and no segment written about them passed provenance.
    UNGROUNDED = "ungrounded"
    # A follow-up reference with nothing to point at: no earlier turn resolved such a
    # thing, or what it resolved is not something this persona can see.
    NOTHING_REFERRED_TO = "nothing-referred-to"


class CoverageGap(StrEnum):
    """Which gap a coverage-gap abstention found."""

    # Nothing the persona can see is installed in the scope asked about.
    NO_RECORDED_INSTALLATIONS = "no-recorded-installations"


class Candidate(_Model):
    space_id: str
    label: str
    # The label of the building it is in, to tell candidates apart.
    building: str | None
    # An asset candidate: its id, the label of the space it is in, and whether its
    # position is recorded. The space is in space_id. Two of a type differ by id.
    asset_id: str | None = None
    place: str | None = None
    position: str | None = None


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


class Reference(_Model):
    """One follow-up reference: what the words pointed back at, and which turn it is from."""

    mention: str
    text: str
    step: str
    # What it was read as, and where it pointed.
    kind: MentionKind
    to: FollowUpTarget
    ordinal: int | None
    outcome: Outcome
    # What it resolved to, or the candidates when it has to ask. Empty for nothing.
    ids: list[str]
    labels: list[str]
    # The turn that resolved that, or that offered the candidates.
    turn: int | None
    # The assets whose type it took, for "like it".
    through: list[str] = []


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
    references: list[Reference] = []
    provenance: list[Provenance] = []
    authored_defaults: list[AuthoredDefault] = []
    asset_types: list[TypeTrace] = []
    assets: AssetsTrace | None = None
    # What was searched and retrieved, each document with its date.
    documents: DocumentsTrace | None = None
    # Segments the provenance check dropped from the answer.
    dropped_segments: list[DroppedSegment] = []
    composition_rejection: list[str] = []
    unavailable: str | None = None


class Turn(_Model):
    kind: TurnKind
    cause: AbstentionCause | None = None
    # Which gap, when the cause is a coverage gap.
    gap: CoverageGap | None = None
    segments: list[Segment]
    candidates: list[Candidate] = []
    state: ConversationState
    trace: Trace
    llm_calls: list[LlmCall]
    # Query embeddings made for retrieval. Paid, but not model calls of the turn.
    embedding_calls: list[EmbeddingCall] = []
    # Whether this turn needed a second plan, and whether that one ran its steps.
    recovery: RecoveryOutcome = RecoveryOutcome.NOT_NEEDED


class TurnRefused(Exception):
    """The request cannot be served as asked: a snapshot not served, a state not its own."""

    def __init__(self, status: int, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


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
- "asset-document": it asks what a document says about assets: a manual, about the
  assets in a place or of a type ("what does the manual say about servicing the AC
  unit in room 107").
- "place-document": it asks what a place's own page says about it ("what does the
  accessibility page say about the Audimax").
- "asset-location": it asks where one asset from earlier in the conversation is
  ("where is that one").
- "step-dependent": answering needs the result of one step to decide the next
  ("what is in the room with the most lights"). No steps.
- "unsupported": anything else. No steps.

mentions: each place, asset type and asset the question names, with the words that
name it, verbatim ("107", "50.34 room 107", "Audimax"; "lights", "AC units"), and
its kind, "place", "asset-type" or "asset". Ids m1, m2, ... A scale level on its own
("floor" in "which floor") is not a mention.

constraints: {{"kind": "within", "mention": a, "container": b}} when the question
puts place a inside place b ("room 107 in 50.34").

steps: ids s1, s2, ..., at most {STEP_CAP}. Every step but the last feeds a later one.
- resolve_place, one per place mention. Resolve a container before what lies
  within it, and point the inner step at it with "within": {{"step": "<id>"}}.
  A place-location plan ends in the step for the place asked about.
- resolve_asset_type, one per asset-type mention.
- resolve_asset, one per asset mention. An asset-location plan ends in it.
- assets_in_scope: every asset under one place. "scope": {{"step": "<the place
  step>"}}, left out for the whole venue. "types": {{"step": "<the type step>"}},
  left out for assets of any type. An asset-containment plan ends in it. It covers
  every space under the scope in one step: never plan a step per room.
- aggregate: {{"assets": {{"step": "<the assets_in_scope step>"}}, "by": "<tier>"}}
  counts those assets per space of that tier. "by" is one of "tiers". A scale level
  named in the question sets both: the place it names is the scope, the tier it
  compares across is "by". An asset-aggregation plan ends in it.
- retrieve_documents: {{"anchors": {{"step": "<id>"}}, "query": "<a few words>"}}
  searches the documents tied to what that step found. "query" is what to look
  for in them, in English, from the question ("servicing", "step-free access").
  An asset-document plan ends in it reading the assets_in_scope step: it searches
  what is anchored to those assets and their types. A place-document plan ends in
  it reading the resolve_place step for the place asked about.

Follow-ups. "conversation" holds the earlier questions and what earlier turns
resolved: a place, assets, asset types, by name. Words that name nothing themselves
and point back at one of those ("that one", "it", "there", "those", "like it") are
still a mention, with "refers": {{"to": "earlier-turn"}} and the kind of thing meant:
"place" for a place, "asset" for one asset, "asset-type" for things of its type
("more like it"). Never copy a name out of "conversation" into a mention, and never
resolve one yourself. An asset is only ever named this way: a question that names
one asset in words is "unsupported".

If "pending" is present, the previous turn asked the user which one they meant and
listed the candidates, which are of the kind "pending" says. Read the question as
the answer to that, and plan the earlier question it was asked about, with a mention
whose "refers" is {{"to": "candidate", "ordinal": n}}: n is the candidate's position
in the list, from 1 ("the second one" is 2).
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

_PASSAGE_RULES = """
"passages" is a second list, apart from the facts: what was retrieved from the
documents, each with a ref (p1, p2, ...), a title and its text. Then say what the
passages say about the question, one segment per passage you use, and put the
passage's ref in that segment's "cites". A passage ref goes in "cites" only, never
in braces. Say only what the passage says. To quote it, copy its words exactly
inside double quotes. Do not use double quotes for anything else. A passage's
numbers are not yours to write: leave out any wording that needs a digit. Use only
the passages that answer the question, and none if none does.
"""

COMPOSITION_INSTRUCTIONS = {
    QuestionShape.PLACE_LOCATION: (
        "You answer where a place is, in one or two short sentences, in the question's "
        "language.\n" + _COMPOSE_RULES + "\nThe subject is the place asked about; the other "
        "facts are the places it lies within, innermost first, with their tiers.\n"
    ),
    QuestionShape.ASSET_LOCATION: (
        "You answer where one asset is, in one or two short sentences, in the question's "
        "language.\n"
        + _COMPOSE_RULES
        + """
"subject" is the asset asked about. Its placeholder fills in as its name, followed
by "(generated)" when it is, so do not add that word yourself. The place facts are
the space it is in and the places that lies within, innermost first, with their
tiers. Where the subject's "position" is "position not recorded", say that its
position within the space is not recorded.
"""
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
    QuestionShape.ASSET_DOCUMENT: (
        "You answer what the documents say about the assets in a place, briefly, in the "
        "question's language.\n"
        + _COMPOSE_RULES
        + "\n"
        + _COUNT_RULES
        + """
"scope" is the place asked about, "total" the count of the assets found there and
"rows" its breakdown, as "rows_by" says. Say what was found first, from the facts.
"""
        + _PASSAGE_RULES
    ),
    QuestionShape.PLACE_DOCUMENT: (
        "You answer what a place's documents say about it, briefly, in the question's "
        "language.\n"
        + _COMPOSE_RULES
        + """
"subject" is the place asked about; the other facts are the places it lies within.
Name every fact in "required".
"""
        + _PASSAGE_RULES
    ),
}

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
    AbstentionCause.NOTHING_REFERRED_TO: "I don't know what that refers to.",
}
# What a step that found nothing makes of the turn, by its tool.
_NOTHING = {
    "resolve_place": AbstentionCause.NO_SUCH_PLACE,
    "resolve_asset_type": AbstentionCause.NO_SUCH_ASSET_TYPE,
    "aggregate": AbstentionCause.NOTHING_AT_SCALE_LEVEL,
}
# What kind of thing each resolving tool resolves.
_KIND_OF = {
    "resolve_place": MentionKind.PLACE,
    "resolve_asset_type": MentionKind.ASSET_TYPE,
    "resolve_asset": MentionKind.ASSET,
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
    embeddings: list[EmbeddingCall] = field(default_factory=list)
    question: str = ""

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


class TurnService:
    def __init__(
        self,
        store: Store,
        llm: Llm,
        node_cap: int = NODE_CAP,
        *,
        documents: Documents | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.node_cap = node_cap
        # Document retrieval, bound here and not by any plan. None retrieves nothing.
        self.documents = documents

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
        trace = Trace(snapshot=snapshot, persona=request.persona, tree=measure)
        turn = _Turn(state, trace, [], question=request.question)
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
        except (LlmUnavailable, RetrievalUnavailable) as error:
            return turn.unavailable(error)
        return _give_up(turn, venue, second.trigger)

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
        conversation = told(turn.state, venue)
        if conversation:
            payload["conversation"] = conversation
        if turn.state.pending is not None:
            payload["pending"] = {
                "mention": turn.state.pending.mention,
                "kind": turn.state.pending.kind.value,
                "candidates": offered(venue, turn.state.pending),
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
            execution = None
            if plan.shape not in DECLINED:
                # Off the loop: a retrieval step waits on the index.
                execution = await asyncio.to_thread(
                    execute, plan, venue, turn.state, self.documents
                )
                turn.embeddings.extend(
                    EmbeddingCall(
                        deployment=r.retrieval.embedding.deployment,
                        input_tokens=r.retrieval.embedding.input_tokens,
                    )
                    for r in execution.results
                    if r.retrieval is not None and r.retrieval.embedding is not None
                )
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
            resolved=_resolved(execution, venue),
            resolved_types=_resolved_types(execution, venue),
            references=_references(execution, venue),
        )
        last = execution.last
        if last.referred is not None and last.outcome is not Outcome.RESOLVED:
            return _not_referred(turn, venue, last)
        if last.outcome is Outcome.NONE:
            return turn.abstain(_NOTHING[last.tool])
        if last.outcome is Outcome.AMBIGUOUS:
            assert last.text is not None
            return _clarify(turn, venue, last.text, last.ids)
        if last.retrieval is not None:
            return await self._settle_documents(turn, question, venue, shape, last)

        found: ScopeResult | None = None
        if shape is QuestionShape.PLACE_LOCATION:
            subject = venue.spaces[last.ids[0]]
            answer = compose.location(venue, question, subject)
            state = turn.next(place=subject.id)
        elif shape is QuestionShape.ASSET_LOCATION:
            asset = venue.assets[last.ids[0]]
            answer = compose.asset_location(venue, question, asset)
            turn.note(
                asset_types=[_type_trace(venue.asset_types[asset.type_id])],
                assets=_one_asset_trace(venue, asset),
            )
            state = turn.next(place=asset.space_id, assets=[asset.id])
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
            state = turn.next(**_carried(venue, last, found))
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

    async def _settle_documents(
        self, turn: _Turn, question: str, venue: Venue, shape: QuestionShape, last: StepResult
    ) -> Turn:
        """An answer over two lists: the facts as before, and the passages retrieved."""
        retrieved = last.retrieval
        assert retrieved is not None
        found = last.found
        if found is not None:
            answer = compose.asset_documents(venue, question, found, retrieved)
            # The types the anchored assets are of, whether or not the question named any.
            types = [venue.asset_types[i] for i in retrieved.anchors.asset_type_ids]
            types.sort(key=lambda kind: kind.id)
            turn.note(
                asset_types=[_type_trace(kind) for kind in types],
                assets=_assets_trace(venue, last, answer),
            )
            state = turn.next(**_carried(venue, last, found))
        else:
            assert retrieved.anchors.scope is not None
            subject = venue.spaces[retrieved.anchors.scope]
            answer = compose.place_documents(venue, question, subject, retrieved)
            state = turn.next(place=subject.id)
        places = [fact.space for fact in answer.facts if fact.space is not None]
        turn.note(
            documents=documents_trace(last.step, retrieved),
            provenance=[_provenance(s) for s in places],
            authored_defaults=_defaults(places),
        )

        text = await turn.call(
            self.llm,
            Purpose.COMPOSITION,
            instructions=COMPOSITION_INSTRUCTIONS[shape],
            payload=answer.payload,
            response_format=compose.Draft,
        )
        try:
            segments, dropped = compose.ground(
                text, answer.facts, answer.required, answer.together, answer.passages
            )
        except CompositionRejected as rejected:
            turn.note(composition_rejection=rejected.reasons)
            return turn.abstain(AbstentionCause.COMPOSITION_REJECTED)
        turn.note(dropped_segments=dropped)
        # A document answer with no document segment left is not one.
        if not any(segment.passages for segment in segments):
            return turn.abstain(AbstentionCause.UNGROUNDED)
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
            # Retired, or never there at all. Told apart nowhere.
            raise TurnRefused(
                409, f"{snapshot} is not served; start a new conversation", SNAPSHOT_DROPPED
            )
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


def _give_up(turn: _Turn, venue: Venue, trigger: Trigger | None = None) -> Turn:
    """A second trigger: ask again what is pending, or else abstain. Never a third call."""
    pending = turn.state.pending
    if pending is not None:
        return _ask_again(turn, venue, pending)
    if trigger is not None and trigger.kind is TriggerKind.EMPTY_RETRIEVAL:
        return turn.abstain(AbstentionCause.NO_DOCUMENT)
    return turn.abstain(AbstentionCause.PLAN_REJECTED)


def _ask_again(turn: _Turn, venue: Venue, pending: Pending) -> Turn:
    return _clarify(
        turn, venue, pending.mention, tuple(pending.candidates), pending.kind, pending.turn
    )


def _not_referred(turn: _Turn, venue: Venue, last: StepResult) -> Turn:
    """A follow-up reference that did not come to one thing: ask which, or abstain."""
    referred = last.referred
    assert referred is not None and last.text is not None
    if last.outcome is Outcome.AMBIGUOUS:
        return _clarify(turn, venue, last.text, referred.ids, referred.kind)
    # "The fifth one" of three. The candidates still stand, so they are offered again.
    if referred.to is FollowUpTarget.CANDIDATE and turn.state.pending is not None:
        return _ask_again(turn, venue, turn.state.pending)
    return turn.abstain(AbstentionCause.NOTHING_REFERRED_TO)


def _carried(venue: Venue, last: StepResult, found: ScopeResult) -> dict[str, Any]:
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


def _own_state(request: TurnRequest, snapshot: str, venue: Venue) -> ConversationState:
    """The prior state, keeping only ids the persona can see. A fresh one if there is none."""
    if request.state is None:
        return ConversationState(snapshot=snapshot, persona=request.persona)
    return own(request.state, venue)


def _clarify(
    turn: _Turn,
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
        within = _label(building) if building and building.label else None
        if asset is None:
            candidates.append(Candidate(space_id=item, label=_label(space), building=within))
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
        citations.append(compose.asset_citation(asset))
    # Asked again, the candidates are still from the turn that first offered them.
    pending = Pending(mention=mention, candidates=list(ids), kind=kind, turn=asked or turn.number)
    text = CLARIFY_ASSET if kind is MentionKind.ASSET else CLARIFY
    return turn.finish(
        TurnKind.CLARIFICATION,
        [Segment(text=text, citations=citations)],
        candidates=candidates,
        state=turn.next(pending=pending),
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


def _references(execution: Execution, venue: Venue) -> list[Reference]:
    found = []
    for r in execution.results:
        referred = r.referred
        if referred is None:
            continue
        found.append(
            Reference(
                mention=r.mention or "",
                text=r.text or "",
                step=r.step,
                kind=_KIND_OF[r.tool],
                to=referred.to,
                ordinal=referred.ordinal,
                outcome=r.outcome,
                ids=list(r.ids),
                labels=[name(venue, referred.kind, i) for i in r.ids],
                turn=referred.turn,
                through=list(referred.through),
            )
        )
    return found


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


def _one_asset_trace(venue: Venue, asset: Asset) -> AssetsTrace:
    """The trace of an answer about one asset: itself, and how much its space holds."""
    fields = _asset_provenance(asset)
    return AssetsTrace(
        scope=asset.space_id,
        types=[asset.type_id],
        grouped_by=None,
        touched=1,
        installed_in_scope=len(venue.assets_under.get(asset.space_id, [])),
        listed=[
            AssetTrace(
                asset_id=asset.id,
                type_id=asset.type_id,
                space_id=asset.space_id,
                source=asset.source,
                name=asset.name.value,
                **fields,
            )
        ],
        classes=[ProvenanceClass(**fields, assets=1)],
        rows=[],
    )


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
