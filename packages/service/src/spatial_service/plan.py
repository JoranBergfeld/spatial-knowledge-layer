"""The plan: what the planning model says a question needs, checked in code before anything runs.

A plan names the question's shape, the mentions in it, the constraints between
them from a closed vocabulary, and a flat ordered list of typed tool steps. A
step's arguments may point at an earlier step, whose ids it then works within.
Persona and snapshot are not arguments: the host binds them, and a plan that
tries to carry them does not parse.

The tools, and what each produces:

    resolve_place        a place mention           -> places (one, or the turn stops)
    resolve_asset_type   an asset-type mention     -> asset types (a set)
    assets_in_scope      a scope and types, both optional references -> assets
    aggregate            assets, and a scale level to count them by  -> groups
    places_matching      a description of places   -> a place set (one or many)
    route                an origin and a destination -> a route
    nearest              assets, and a place to be near -> the nearest of them

A reference binds only to an earlier step that produces what it reads. Nothing
reads groups, and nothing that resolves a mention reads assets, so a plan cannot
say "take whichever came out on top and look inside it". A question like that is
its own shape, step-dependent, and gets an abstention.

The step cap is six. The longest place chain worth reading in one question is
three deep ("room 107 on level 1 of 50.34"), then a type, the assets and a count
over them. Plans over the cap are rejected, never truncated: running a shortened
plan would answer a different question.

A route's destination is a place, a place set or assets, and it ends at the nearest.
Its origin is a place, or where an earlier route ended, or left out for the persona's
default origin. Step-free is a flag on the step, set only when the question asks.
A plan holds at most two route steps, each with at most three alternatives.

A plan that may not run raises a trigger, unbound-reference when a reference in
it binds to nothing and plan-invalid otherwise, which buys it one recovery call.

To add a tool: write its step model with READS for its references, add it to Step
and STEPS, give it a row in PRODUCES, and teach the executor to run it.
"""

from __future__ import annotations

import json
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_service.recovery import TriggerKind

STEP_CAP = 6
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


class ConstraintKind(StrEnum):
    # One mention lies inside another: "room 107 in 50.34".
    WITHIN = "within"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Mention(_Model):
    id: str = Field(pattern=r"^m[0-9]+$")
    # The words of the question that name it, as written.
    text: str = Field(min_length=1, max_length=200)
    kind: MentionKind


class Constraint(_Model):
    kind: ConstraintKind
    mention: str
    container: str


class StepRef(_Model):
    """The ids an earlier step produced."""

    step: str


class _Step(_Model):
    id: str = Field(pattern=r"^s[0-9]+$")
    # Each reference field, and what the step it names must produce. A tuple where
    # it may be one of several.
    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {}

    def references(self) -> list[tuple[str | tuple[str, ...], StepRef]]:
        """The references this step makes, as (what it reads, reference)."""
        found = [(reads, getattr(self, name)) for name, reads in self.READS.items()]
        return [(reads, ref) for reads, ref in found if ref is not None]


class ResolvePlace(_Step):
    """Find the place a mention names, among what the persona can see."""

    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {"within": "places"}
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


class AssetsInScope(_Step):
    """Every visible asset under one scope node. One step, however many spaces lie under it."""

    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {
        "scope": "places",
        "types": "asset-types",
    }

    tool: Literal["assets_in_scope"]
    # The place to look under. None is the whole venue.
    scope: StepRef | None = None
    # Only assets of the types that step resolved. None is any type.
    types: StepRef | None = None


class Aggregate(_Step):
    """Count a step's assets per space of one scale level under its scope."""

    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {"assets": "assets"}

    tool: Literal["aggregate"]
    assets: StepRef
    # The scale level to count by, as the venue names its tiers: "floor", "building".
    by: str = Field(min_length=1, max_length=40)


class PlacesMatching(_Step):
    """Find every visible place a description fits. A set: nobody is asked which."""

    RESOLVES: ClassVar[MentionKind] = MentionKind.PLACE

    tool: Literal["places_matching"]
    mention: str


class Route(_Step):
    """The quickest route to a destination, by modelled duration, over what the persona may walk."""

    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {
        "origin": "places",
        "after": "route",
        "destination": ("places", "place-set", "assets"),
    }

    tool: Literal["route"]
    # Where it starts. None, with no `after`, is the persona's default origin.
    origin: StepRef | None = None
    # Start where that route ended, for a second route.
    after: StepRef | None = None
    # One place, a set of places or assets. The route ends at the nearest.
    destination: StepRef
    # Take every stair out. Only when the question asks for it.
    step_free: bool = False
    # How many places of a set to route to, nearest first.
    alternatives: int = Field(default=1, ge=1, le=ALTERNATIVES_CAP)


class Nearest(_Step):
    """The nearest of a step's assets to a place, or to the default origin."""

    READS: ClassVar[dict[str, str | tuple[str, ...]]] = {"assets": "assets", "to": "places"}

    tool: Literal["nearest"]
    assets: StepRef
    # The place to be near. None is the persona's default origin.
    to: StepRef | None = None


# Every tool a step can call. A plain union, not a discriminated one: that would
# put a oneOf in the schema, which strict structured output does not take.
Step = (
    ResolvePlace | ResolveAssetType | AssetsInScope | Aggregate | PlacesMatching | Route | Nearest
)
# The steps that resolve a mention.
Resolving = (ResolvePlace, ResolveAssetType, PlacesMatching)
STEPS: dict[str, type[_Step]] = {
    "resolve_place": ResolvePlace,
    "resolve_asset_type": ResolveAssetType,
    "assets_in_scope": AssetsInScope,
    "aggregate": Aggregate,
    "places_matching": PlacesMatching,
    "route": Route,
    "nearest": Nearest,
}

# What each tool's steps produce. A reference binds only to an earlier step whose
# output is the kind it reads.
PRODUCES = {
    "resolve_place": "places",
    "resolve_asset_type": "asset-types",
    "assets_in_scope": "assets",
    "aggregate": "groups",
    "places_matching": "place-set",
    "route": "route",
    "nearest": "nearest",
}

# The tool a shape's last step calls: the one whose result is the answer.
ENDS_IN = {
    QuestionShape.PLACE_LOCATION: "resolve_place",
    QuestionShape.ASSET_CONTAINMENT: "assets_in_scope",
    QuestionShape.ASSET_AGGREGATION: "aggregate",
    QuestionShape.PROXIMITY: "nearest",
    QuestionShape.ROUTE: "route",
}


class Plan(_Model):
    shape: QuestionShape
    mentions: list[Mention]
    constraints: list[Constraint] = []
    steps: list[Step]


class PlanRejected(Exception):
    """A plan that may not run, and the trigger it raises."""

    def __init__(self, trigger: TriggerKind, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.trigger = trigger
        self.reasons = reasons


def parse_plan(text: str) -> Plan:
    """The planning model's output as a checked plan. Raises PlanRejected otherwise.

    A plan that does not parse, or parses and breaks a rule, is invalid. One with a
    reference that binds to nothing is unbound, even if it breaks rules too: a
    dangling reference breaks others in its wake (the step it should have read now
    feeds nothing), and it is the one to fix. The reasons list all of them.
    """
    try:
        plan = Plan.model_validate_json(text)
    except ValidationError as error:
        raise PlanRejected(TriggerKind.PLAN_INVALID, _parse_reasons(text, error)) from error
    unbound, invalid = bind(plan), check(plan)
    if unbound:
        raise PlanRejected(TriggerKind.UNBOUND_REFERENCE, unbound + invalid)
    if invalid:
        raise PlanRejected(TriggerKind.PLAN_INVALID, invalid)
    return plan


def _parse_reasons(text: str, error: ValidationError) -> list[str]:
    """Why a plan did not parse, with each step judged only as the tool it names.

    A step that fits no tool fails against every one of them, and the errors for
    the tools it never meant to be are noise. So a step naming a known tool keeps
    that tool's errors, and one naming none gets a single line saying so.
    """
    try:
        written = json.loads(text)
    except ValueError:
        written = None
    steps = written.get("steps") if isinstance(written, dict) else None
    by_class = {model.__name__: tool for tool, model in STEPS.items()}
    reasons: list[str] = []
    for problem in error.errors():
        loc: tuple[Any, ...] = problem["loc"]
        if len(loc) >= 3 and loc[0] == "steps" and loc[2] in by_class:
            step = steps[loc[1]] if isinstance(steps, list) and loc[1] < len(steps) else None
            named = step.get("tool") if isinstance(step, dict) else None
            if named not in STEPS:
                reason = f"steps.{loc[1]}.tool: {named!r} is not a tool"
                if reason not in reasons:
                    reasons.append(reason)
                continue
            if by_class[loc[2]] != named:
                continue
            loc = (*loc[:2], *loc[3:])
        reasons.append(f"{'.'.join(map(str, loc)) or 'plan'}: {problem['msg']}")
    return reasons


def bind(plan: Plan) -> list[str]:
    """Every reference in a plan that names nothing it can bind to. Empty when all bind."""
    reasons: list[str] = []
    mentions = {mention.id for mention in plan.mentions}
    earlier: dict[str, Step] = {}
    for step in plan.steps:
        if isinstance(step, Resolving) and step.mention not in mentions:
            reasons.append(f"step {step.id} resolves {step.mention}, which is not a mention")
        for reads, ref in step.references():
            source = earlier.get(ref.step)
            if source is None:
                reasons.append(f"step {step.id} reads {ref.step}, which is not earlier")
            elif PRODUCES[source.tool] not in ((reads,) if isinstance(reads, str) else reads):
                wanted = reads if isinstance(reads, str) else " or ".join(reads)
                reasons.append(
                    f"step {step.id} reads {source.id}, which produces "
                    f"{PRODUCES[source.tool]}, not {wanted}"
                )
        earlier.setdefault(step.id, step)
    for constraint in plan.constraints:
        for named in (constraint.mention, constraint.container):
            if named not in mentions:
                reasons.append(f"a {constraint.kind} constraint names {named}, not a mention")
    return reasons


def check(plan: Plan) -> list[str]:
    """What is wrong with a plan that parses, references aside. Empty when it may run."""
    reasons: list[str] = []
    mentions = {mention.id: mention for mention in plan.mentions}
    if len(mentions) != len(plan.mentions):
        reasons.append("mention ids repeat")

    if plan.shape in DECLINED:
        if plan.steps:
            reasons.append(f"a question that is {plan.shape} has no steps")
        return reasons

    if not plan.steps:
        reasons.append("the plan has no steps")
    if len(plan.steps) > STEP_CAP:
        reasons.append(f"the plan has {len(plan.steps)} steps, over the cap of {STEP_CAP}")
    if plan.steps and plan.steps[-1].tool != ENDS_IN[plan.shape]:
        reasons.append(
            f"a {plan.shape} plan ends in {ENDS_IN[plan.shape]}, not {plan.steps[-1].tool}"
        )

    routes = [step for step in plan.steps if isinstance(step, Route)]
    if len(routes) > ROUTE_STEP_CAP:
        reasons.append(f"the plan has {len(routes)} route steps, over the cap of {ROUTE_STEP_CAP}")
    for route in routes:
        if route.origin is not None and route.after is not None:
            reasons.append(f"step {route.id} starts at a place or after a route, not both")

    seen: set[str] = set()
    referenced: set[str] = set()
    for step in plan.steps:
        if step.id in seen:
            reasons.append(f"step {step.id} repeats")
        seen.add(step.id)
        referenced.update(ref.step for _, ref in step.references())

    resolving = [step for step in plan.steps if isinstance(step, Resolving)]
    resolved = [step.mention for step in resolving]
    for mention in plan.mentions:
        if resolved.count(mention.id) != 1:
            reasons.append(f"{mention.id} is resolved by {resolved.count(mention.id)} steps")
    for step in resolving:
        named = mentions.get(step.mention)
        if named is not None and named.kind is not step.RESOLVES:
            reasons.append(
                f"step {step.id} is {step.tool}, but {named.id} is a mention of kind {named.kind}"
            )

    # Every step but the last feeds a later one: the last is what the question asks for.
    for step in plan.steps[:-1]:
        if step.id not in referenced:
            reasons.append(f"step {step.id} feeds no later step")

    by_mention = {step.mention: step for step in resolving}
    for constraint in plan.constraints:
        if constraint.mention not in mentions or constraint.container not in mentions:
            continue
        if constraint.mention == constraint.container:
            reasons.append(f"{constraint.mention} cannot lie within itself")
            continue
        inner, outer = by_mention.get(constraint.mention), by_mention.get(constraint.container)
        if inner is None or outer is None:
            continue
        within = inner.within if isinstance(inner, ResolvePlace) else None
        if within is None or within.step != outer.id:
            reasons.append(
                f"{constraint.mention} lies within {constraint.container}, "
                f"but step {inner.id} does not read step {outer.id}"
            )
    return reasons
