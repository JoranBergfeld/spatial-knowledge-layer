"""The plan: what the planning model says a question needs, checked in code before anything runs.

A plan names the question's shape, the mentions in it, the constraints between
them from a closed vocabulary, and a flat ordered list of typed tool steps. A
step's arguments may point at an earlier step, whose ids it then works within.
Persona and snapshot are not arguments: the host binds them, and a plan that
tries to carry them does not parse.

The step cap is four. The one supported shape needs a step per place mention,
and the longest place mention chain worth reading in one question is three deep
("room 107 on level 1 of 50.34"), plus one to spare. Plans over the cap are
rejected, never truncated: running a shortened plan would answer a different
question.

A plan that may not run raises a trigger, unbound-reference when a reference in
it binds to nothing and plan-invalid otherwise, which buys it one recovery call.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_service.recovery import TriggerKind

STEP_CAP = 4


class QuestionShape(StrEnum):
    # "Where is X?": one place, and the places it lies within.
    PLACE_LOCATION = "place-location"
    # Anything else. The plan stops there and the turn abstains.
    UNSUPPORTED = "unsupported"


class MentionKind(StrEnum):
    PLACE = "place"


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


class ResolvePlace(_Model):
    """Find the place a mention names, among what the persona can see."""

    id: str = Field(pattern=r"^s[0-9]+$")
    tool: Literal["resolve_place"]
    mention: str
    # Only places inside what that step resolved.
    within: StepRef | None = None


# Every tool a step can call. One so far; the next joins as a union on `tool`.
Step = ResolvePlace


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


# What each tool's steps produce, and what a step's `within` must read. A reference
# binds only to an earlier step whose output is the kind it reads.
PRODUCES = {"resolve_place": "places"}
WITHIN_READS = {"resolve_place": "places"}


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
        raise PlanRejected(
            TriggerKind.PLAN_INVALID,
            [f"{'.'.join(map(str, e['loc'])) or 'plan'}: {e['msg']}" for e in error.errors()],
        ) from error
    unbound, invalid = bind(plan), check(plan)
    if unbound:
        raise PlanRejected(TriggerKind.UNBOUND_REFERENCE, unbound + invalid)
    if invalid:
        raise PlanRejected(TriggerKind.PLAN_INVALID, invalid)
    return plan


def bind(plan: Plan) -> list[str]:
    """Every reference in a plan that names nothing it can bind to. Empty when all bind."""
    reasons: list[str] = []
    mentions = {mention.id for mention in plan.mentions}
    earlier: dict[str, Step] = {}
    for step in plan.steps:
        if step.mention not in mentions:
            reasons.append(f"step {step.id} resolves {step.mention}, which is not a mention")
        if step.within is not None:
            source = earlier.get(step.within.step)
            if source is None:
                reasons.append(f"step {step.id} reads {step.within.step}, which is not earlier")
            elif PRODUCES[source.tool] != WITHIN_READS[step.tool]:
                reasons.append(
                    f"step {step.id} reads {source.id}, which produces "
                    f"{PRODUCES[source.tool]}, not {WITHIN_READS[step.tool]}"
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

    if plan.shape is QuestionShape.UNSUPPORTED:
        if plan.steps:
            reasons.append("an unsupported question has no steps")
        return reasons

    if not plan.steps:
        reasons.append("the plan has no steps")
    if len(plan.steps) > STEP_CAP:
        reasons.append(f"the plan has {len(plan.steps)} steps, over the cap of {STEP_CAP}")

    seen: set[str] = set()
    referenced: set[str] = set()
    for step in plan.steps:
        if step.id in seen:
            reasons.append(f"step {step.id} repeats")
        seen.add(step.id)
        if step.within is not None:
            referenced.add(step.within.step)

    resolved = [step.mention for step in plan.steps]
    for mention in plan.mentions:
        if resolved.count(mention.id) != 1:
            reasons.append(f"{mention.id} is resolved by {resolved.count(mention.id)} steps")

    # Every step but the last feeds a later one: the last is the place asked about.
    for step in plan.steps[:-1]:
        if step.id not in referenced:
            reasons.append(f"step {step.id} feeds no later step")

    by_mention = {step.mention: step for step in plan.steps}
    for constraint in plan.constraints:
        if constraint.mention not in mentions or constraint.container not in mentions:
            continue
        if constraint.mention == constraint.container:
            reasons.append(f"{constraint.mention} cannot lie within itself")
            continue
        inner, outer = by_mention.get(constraint.mention), by_mention.get(constraint.container)
        if inner is None or outer is None:
            continue
        if inner.within is None or inner.within.step != outer.id:
            reasons.append(
                f"{constraint.mention} lies within {constraint.container}, "
                f"but step {inner.id} does not read step {outer.id}"
            )
    return reasons
