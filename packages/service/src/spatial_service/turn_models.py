"""What a turn takes and gives back, and the trace that goes with it.

Models and the caps they record, nothing else. turn.py fills them in, traces.py
builds the trace's parts and outcomes.py the turns that end early.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from spatial_core.records import Persona
from spatial_service.attributes import FilterTrace
from spatial_service.conversation import ConversationState
from spatial_service.grounding import Segment
from spatial_service.llm import LlmCall
from spatial_service.maintenance import MaintenanceOut, MaintenanceTrace
from spatial_service.places import Outcome
from spatial_service.plan import STEP_CAP, FollowUpTarget, MentionKind, QuestionShape
from spatial_service.recovery import PlanChange, RecoveryOutcome, Trigger
from spatial_service.retrieval import DocumentsTrace, DroppedSegment, EmbeddingCall
from spatial_service.route_answer import RouteOut, RouteTrace

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
    # Nothing is recorded in the scope asked about, and it is too small a place to
    # call a gap in the venue's data. No reason is given.
    NOTHING_RECORDED = "nothing-recorded"
    # A follow-up reference with nothing to point at: no earlier turn resolved such a
    # thing, or what it resolved is not something this persona can see.
    NOTHING_REFERRED_TO = "nothing-referred-to"
    # No route on the mapped graph for this persona. One cause for every reason
    # there might be none, on purpose.
    NO_ROUTE = "no-route"
    # Nothing of the kind asked for can be placed near there, by route or by containment.
    NOTHING_NEAR = "nothing-near"
    # A question only staff and maintenance get an answer to, asked as a visitor.
    # Said for the shape, before anything is looked up.
    STAFF_ONLY = "staff-only"


class CoverageGap(StrEnum):
    """Which gap a coverage-gap abstention found."""

    # Nothing the persona can see is installed in a building, or in something bigger.
    NO_RECORDED_INSTALLATIONS = "no-recorded-installations"
    # None of the assets asked about carries the attribute filtered on.
    ATTRIBUTE_NOT_RECORDED = "attribute-not-recorded"
    # None of the types asked about has a planned job on record.
    NO_PLANNED_MAINTENANCE = "no-planned-maintenance"


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
    # Each attribute filtered on, with its fill rate over the assets asked about.
    filters: list[FilterTrace] = []


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
    # The persona's graph and what each route or nearest step did on it.
    route: RouteTrace | None = None
    # What a maintenance step found: the types, their jobs, and what was counted nearby.
    maintenance: MaintenanceTrace | None = None
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
    # The planned jobs a maintenance answer rests on, as rows, and the same kit nearby.
    maintenance: MaintenanceOut | None = None
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
