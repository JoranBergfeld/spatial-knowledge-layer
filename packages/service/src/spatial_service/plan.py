"""The plan: what the planning model says a question needs, checked in code before anything runs.

A plan names the question's shape, the mentions in it, the constraints between
them from a closed vocabulary, and a flat ordered list of typed tool steps. A
step's arguments may point at an earlier step, whose ids it then works within.
Persona and snapshot are not arguments: the host binds them, and a plan that
tries to carry them does not parse.

The tools, and what each produces:

    resolve_place        a place mention           -> places (one, or the turn stops)
    resolve_asset_type   an asset-type mention     -> asset types (a set)
    assets_in_scope      a scope and types, both optional references,
                         and attribute filters                       -> assets
    aggregate            assets, and a scale level to count them by  -> groups

A reference binds only to an earlier step that produces what it reads. Nothing
reads groups, and nothing that resolves a mention reads assets, so a plan cannot
say "take whichever came out on top and look inside it". A question like that is
its own shape, step-dependent, and gets an abstention.

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

A plan that may not run raises a trigger, unbound-reference when a reference in
it binds to nothing and plan-invalid otherwise, which buys it one recovery call.

To add a tool: write its step model with READS for its references, add it to Step
and STEPS, give it a row in PRODUCES, and teach the executor to run it.
"""

from __future__ import annotations

import datetime
import json
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_service.recovery import TriggerKind

STEP_CAP = 6


class QuestionShape(StrEnum):
    # "Where is X?": one place, and the places it lies within.
    PLACE_LOCATION = "place-location"
    # "What is installed in X?", "Where are the lights on floor 2?": the assets
    # under one scope, of some types or of any.
    ASSET_CONTAINMENT = "asset-containment"
    # "Which floor has the most lights?": assets counted per space of a scale level.
    ASSET_AGGREGATION = "asset-aggregation"
    # "Which lights have no service record?", "What falls due before June?": the
    # assets under one scope that pass a filter on an attribute.
    ATTRIBUTE_FILTER = "attribute-filter"
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
    # Each reference field, and what the step it names must produce.
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

    attribute: str = Field(pattern=r"^[a-z][a-z0-9_]{0,59}$")
    test: FilterTest
    # The date to compare against, for before and after. None for the other two.
    date: datetime.date | None = None


class AssetsInScope(_Step):
    """Every visible asset under one scope node. One step, however many spaces lie under it."""

    READS: ClassVar[dict[str, str]] = {"scope": "places", "types": "asset-types"}

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


# Every tool a step can call. A plain union, not a discriminated one: that would
# put a oneOf in the schema, which strict structured output does not take.
Step = ResolvePlace | ResolveAssetType | AssetsInScope | Aggregate
# The steps that resolve a mention.
Resolving = (ResolvePlace, ResolveAssetType)
STEPS: dict[str, type[_Step]] = {
    "resolve_place": ResolvePlace,
    "resolve_asset_type": ResolveAssetType,
    "assets_in_scope": AssetsInScope,
    "aggregate": Aggregate,
}

# What each tool's steps produce. A reference binds only to an earlier step whose
# output is the kind it reads.
PRODUCES = {
    "resolve_place": "places",
    "resolve_asset_type": "asset-types",
    "assets_in_scope": "assets",
    "aggregate": "groups",
}

# The tool a shape's last step calls: the one whose result is the answer.
ENDS_IN = {
    QuestionShape.PLACE_LOCATION: "resolve_place",
    QuestionShape.ASSET_CONTAINMENT: "assets_in_scope",
    QuestionShape.ASSET_AGGREGATION: "aggregate",
    QuestionShape.ATTRIBUTE_FILTER: "assets_in_scope",
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
            elif PRODUCES[source.tool] != reads:
                reasons.append(
                    f"step {step.id} reads {source.id}, which produces "
                    f"{PRODUCES[source.tool]}, not {reads}"
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

    seen: set[str] = set()
    referenced: set[str] = set()
    for step in plan.steps:
        if step.id in seen:
            reasons.append(f"step {step.id} repeats")
        seen.add(step.id)
        referenced.update(ref.step for _, ref in step.references())

    reasons.extend(_filter_reasons(plan))

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


def _filter_reasons(plan: Plan) -> list[str]:
    """What is wrong with a plan's filters, short of whether the venue has the attribute."""
    reasons: list[str] = []
    filtering = False
    for step in plan.steps:
        if not isinstance(step, AssetsInScope):
            continue
        filtering = filtering or bool(step.filters)
        for number, one in enumerate(step.filters, start=1):
            if one.test in DATED and one.date is None:
                reasons.append(f"step {step.id} filter {number}: {one.test} needs a date")
            if one.test not in DATED and one.date is not None:
                reasons.append(f"step {step.id} filter {number}: {one.test} takes no date")
    if plan.shape is QuestionShape.ATTRIBUTE_FILTER and not filtering:
        reasons.append("an attribute-filter plan filters by at least one attribute")
    if plan.shape is QuestionShape.ASSET_CONTAINMENT and filtering:
        reasons.append("an asset-containment plan has no filters; with one it is attribute-filter")
    return reasons
