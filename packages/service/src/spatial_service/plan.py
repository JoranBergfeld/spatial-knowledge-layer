"""The plan: what the planning model says a question needs, checked in code before anything runs.

A plan names the question's shape, the mentions in it, the constraints between
them from a closed vocabulary, and a flat ordered list of typed tool steps. A
step's arguments may point at an earlier step, whose ids it then works within.
Persona and snapshot are not arguments: the host binds them, and a plan that
tries to carry them does not parse.

The tools, and what each produces:

    resolve_place        a place mention           -> places (one, or the turn stops)
    resolve_asset_type   an asset-type mention     -> asset types (a set)
    resolve_asset        an asset mention          -> an asset (one, or the turn stops)
    assets_in_scope      a scope and types, both optional references,
                         and attribute filters                       -> assets
    aggregate            assets, and a scale level to count them by  -> groups
    retrieve_documents   anchors (a step's assets, its one asset, or its place)
                         and a query                                 -> passages
    places_matching      a description of places   -> a place set (one or many)
    route                an origin and a destination -> a route
    nearest              assets, and a place to be near -> the nearest of them

A reference binds only to an earlier step that produces what it reads. Nothing
reads groups or passages, and nothing that resolves a mention reads assets, so a
plan cannot say "take whichever came out on top and look inside it". A question
like that is its own shape, step-dependent, and gets an abstention.

A filter is an attribute by name and one of four tests: recorded, not-recorded,
before a date, after a date. That is the whole vocabulary. There is no field for a
free predicate, and an attribute's name is a name and nothing else, so text that
compares or computes does not parse. Which names count is the venue's to say (its
attribute catalogue, attributes.py), so that is checked with the other arguments,
before any step runs.

The step cap is six. The longest place chain worth reading in one question is
three deep ("room 107 on level 1 of 50.34"), then a type, the assets and a count
over them. Plans over the cap are rejected, never truncated: running a shortened
plan would answer a different question.

A mention may be a follow-up reference: words that name nothing themselves and point
back into the conversation ("that one", "like it", "the second one"). It says where it
points, at what an earlier turn resolved or at a candidate a clarification offered,
and the step that resolves it reads conversation state and looks nothing up
(conversation.py). An asset mention is only ever one of these so far: nothing finds an
asset by its name yet.

A route's destination is a place, a place set or assets, and it ends at the nearest.
Its origin is a place, or where an earlier route ended, or left out for the persona's
default origin. Step-free is a flag on the step, set only when the question asks.
A plan holds at most two route steps, each with at most three alternatives.

To add a tool: write its step model with READS for its references, add it to Step
and STEPS, give it a row in PRODUCES (plan_check.py), and teach the executor to run
it.

Checking a plan, and the rules it is checked against, is plan_check.py.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

STEP_CAP = 6
# How long a mention's words may be.
MENTION_CAP = 200
# Route caps. A route step may return runners-up to other places of a set.
ROUTE_STEP_CAP = 2
ALTERNATIVES_CAP = 3


class QuestionShape(StrEnum):
    # "Where is X?": one place, and the places it lies within.
    PLACE_LOCATION = "place-location"
    # "What is installed in X?", "Where are the lights on floor 2?": the assets
    # under one scope, of some types or of any.
    ASSET_CONTAINMENT = "asset-containment"
    # "Which floor has the most lights?": assets counted per space of a scale level.
    ASSET_AGGREGATION = "asset-aggregation"
    # "What does the manual say about servicing the unit in X?": the assets under a
    # scope, and passages of the documents anchored to them or to their types.
    ASSET_DOCUMENT = "asset-document"
    # "What does the accessibility page say about X?": passages anchored to a place.
    PLACE_DOCUMENT = "place-document"
    # "Which lights have no service record?", "What falls due before June?": the
    # assets under one scope that pass a filter on an attribute.
    ATTRIBUTE_FILTER = "attribute-filter"
    # "Where is that one?": one asset an earlier turn resolved, and where it sits.
    ASSET_LOCATION = "asset-location"
    # "Where is the nearest extinguisher to room 107?": the nearest asset of some types.
    PROXIMITY = "proximity"
    # "How do I get to the Audimax?": a route between places, by modelled duration.
    ROUTE = "route"
    # A question whose next step depends on what an earlier one finds. No steps:
    # the turn abstains, it does not loop.
    STEP_DEPENDENT = "step-dependent"
    # Anything else. The plan stops there and the turn abstains.
    UNSUPPORTED = "unsupported"


# Shapes the layer declines by name. Their plans carry no steps.
DECLINED = (QuestionShape.STEP_DEPENDENT, QuestionShape.UNSUPPORTED)


class MentionKind(StrEnum):
    PLACE = "place"
    ASSET_TYPE = "asset-type"
    ASSET = "asset"


class FollowUpTarget(StrEnum):
    # What an earlier turn resolved: "that one", "there", "like it".
    EARLIER_TURN = "earlier-turn"
    # A candidate the pending clarification offered: "the second one".
    CANDIDATE = "candidate"


class ConstraintKind(StrEnum):
    # One mention lies inside another: "room 107 in 50.34".
    WITHIN = "within"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Refers(_Model):
    """Where a follow-up reference points. The model says where, code finds what."""

    to: FollowUpTarget
    # Which candidate, counted from 1 in the order the clarification offered them.
    ordinal: int | None = None

    @model_validator(mode="after")
    def _ordinal_goes_with_candidate(self) -> Refers:
        # Checked here and not with ge=1, to keep number keywords out of the schema.
        if self.to is FollowUpTarget.CANDIDATE and (self.ordinal is None or self.ordinal < 1):
            raise ValueError("a candidate reference needs an ordinal, counted from 1")
        if self.to is FollowUpTarget.EARLIER_TURN and self.ordinal is not None:
            raise ValueError("only a candidate reference takes an ordinal")
        return self


class Mention(_Model):
    id: str = Field(pattern=r"^m[0-9]+$")
    # The words of the question that name it, as written.
    text: str = Field(min_length=1, max_length=MENTION_CAP)
    kind: MentionKind
    # Set when the words are a follow-up reference. Then nothing is looked up by them.
    refers: Refers | None = None


class Constraint(_Model):
    kind: ConstraintKind
    mention: str
    container: str


class StepRef(_Model):
    """The ids an earlier step produced."""

    step: str


class _Step(_Model):
    id: str = Field(pattern=r"^s[0-9]+$")
    # Each reference field, and what the step it names must produce. Several
    # words split by "|" where it takes any of them.
    READS: ClassVar[dict[str, str]] = {}

    def references(self) -> list[tuple[str, StepRef]]:
        """The references this step makes, as (what it reads, reference)."""
        found = [(reads, getattr(self, name)) for name, reads in self.READS.items()]
        return [(reads, ref) for reads, ref in found if ref is not None]


class ResolvePlace(_Step):
    """Find the place a mention names, among what the persona can see."""

    READS: ClassVar[dict[str, str]] = {"within": "places"}
    RESOLVES: ClassVar[MentionKind] = MentionKind.PLACE

    tool: Literal["resolve_place"]
    mention: str
    # Only places inside what that step resolved.
    within: StepRef | None = None


class ResolveAssetType(_Step):
    """Find the asset types a mention names, among the types the persona can see."""

    RESOLVES: ClassVar[MentionKind] = MentionKind.ASSET_TYPE

    tool: Literal["resolve_asset_type"]
    mention: str


class FilterTest(StrEnum):
    # The asset carries the attribute, or does not. Not carrying it is a gap in
    # the data and says nothing about the asset.
    RECORDED = "recorded"
    NOT_RECORDED = "not-recorded"
    # The attribute is a date, and falls before or after the one given.
    BEFORE = "before"
    AFTER = "after"


# The tests that compare against a date.
DATED = (FilterTest.BEFORE, FilterTest.AFTER)
FILTER_CAP = 3


class Filter(_Model):
    """One test on one attribute. The attribute is one the venue's catalogue names."""

    # A colon too: a fire asset's attributes are OSM keys, "fire_extinguisher:type".
    attribute: str = Field(pattern=r"^[a-z][a-z0-9_:]{0,59}$")
    test: FilterTest
    # The date to compare against, for before and after. None for the other two.
    date: datetime.date | None = None


class ResolveAsset(_Step):
    """The one asset a mention points back at. By follow-up reference only, so far."""

    RESOLVES: ClassVar[MentionKind] = MentionKind.ASSET

    tool: Literal["resolve_asset"]
    mention: str


class AssetsInScope(_Step):
    """Every visible asset under one scope node. One step, however many spaces lie under it."""

    READS: ClassVar[dict[str, str]] = {
        "scope": "places",
        "types": "asset-types",
    }

    tool: Literal["assets_in_scope"]
    # The place to look under. None is the whole venue.
    scope: StepRef | None = None
    # Only assets of the types that step resolved. None is any type.
    types: StepRef | None = None
    # Only assets passing every one of these.
    filters: list[Filter] = Field(default=[], max_length=FILTER_CAP)


class Aggregate(_Step):
    """Count a step's assets per space of one scale level under its scope."""

    READS: ClassVar[dict[str, str]] = {"assets": "assets"}

    tool: Literal["aggregate"]
    assets: StepRef
    # The scale level to count by, as the venue names its tiers: "floor", "building".
    by: str = Field(min_length=1, max_length=40)


class RetrieveDocuments(_Step):
    """Search the documents anchored to what a step found: its assets, or its place."""

    # "asset" is the one a follow-up points at.
    READS: ClassVar[dict[str, str]] = {"anchors": "assets|asset|places"}

    tool: Literal["retrieve_documents"]
    anchors: StepRef
    # What to look for in them, in words. Only the index ever reads it.
    query: str = Field(min_length=1, max_length=200)


class PlacesMatching(_Step):
    """Find every visible place a description fits. A set: nobody is asked which."""

    RESOLVES: ClassVar[MentionKind] = MentionKind.PLACE

    tool: Literal["places_matching"]
    mention: str


class Route(_Step):
    """The quickest route to a destination, by modelled duration, over what the persona may walk."""

    READS: ClassVar[dict[str, str]] = {
        "origin": "places",
        "after": "route",
        "destination": "places|place-set|assets|asset",
    }

    tool: Literal["route"]
    # Where it starts. None, with no `after`, is the persona's default origin.
    origin: StepRef | None = None
    # Start where that route ended, for a second route.
    after: StepRef | None = None
    # One place, a set of places, assets, or the one asset a follow-up points at.
    # The route ends at the nearest.
    destination: StepRef
    # Take every stair out. Only when the question asks for it.
    step_free: bool = False
    # How many places of a set to route to, nearest first.
    alternatives: int = Field(default=1, ge=1, le=ALTERNATIVES_CAP)


class Nearest(_Step):
    """The nearest of a step's assets to a place, or to the default origin."""

    READS: ClassVar[dict[str, str]] = {
        "assets": "assets",
        "to": "places|asset",
    }

    tool: Literal["nearest"]
    assets: StepRef
    # The place to be near, or an asset, which stands for its space. None is the
    # persona's default origin.
    to: StepRef | None = None


# Every tool a step can call. A plain union, not a discriminated one: that would
# put a oneOf in the schema, which strict structured output does not take.
Step = (
    ResolvePlace
    | ResolveAssetType
    | AssetsInScope
    | Aggregate
    | ResolveAsset
    | RetrieveDocuments
    | PlacesMatching
    | Route
    | Nearest
)
# The steps that resolve a mention.
Resolving = (ResolvePlace, ResolveAssetType, ResolveAsset, PlacesMatching)
STEPS: dict[str, type[_Step]] = {
    "resolve_place": ResolvePlace,
    "resolve_asset_type": ResolveAssetType,
    "assets_in_scope": AssetsInScope,
    "aggregate": Aggregate,
    "retrieve_documents": RetrieveDocuments,
    "resolve_asset": ResolveAsset,
    "places_matching": PlacesMatching,
    "route": Route,
    "nearest": Nearest,
}


def resolves(tool: str) -> MentionKind:
    """The kind of mention a resolving tool resolves, read off its step class."""
    step = STEPS[tool]
    assert issubclass(step, Resolving)
    return step.RESOLVES


class Plan(_Model):
    shape: QuestionShape
    mentions: list[Mention]
    constraints: list[Constraint] = []
    steps: list[Step]
