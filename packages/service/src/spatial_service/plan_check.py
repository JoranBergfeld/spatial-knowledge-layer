"""Checking a plan: whether it parses, whether every reference binds, and the rules.

A plan that may not run raises a trigger, unbound-reference when a reference in
it binds to nothing and plan-invalid otherwise, which buys it one recovery call.

A document shape says what its retrieval is anchored by: assets for asset-document,
or the one asset a follow-up points at ("what does its manual say"), and a place
for place-document. A plan that mixes them up is invalid.
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from spatial_service.plan import (
    DATED,
    DECLINED,
    ROUTE_STEP_CAP,
    STEP_CAP,
    STEPS,
    AssetsInScope,
    Plan,
    QuestionShape,
    ResolvePlace,
    Resolving,
    RetrieveDocuments,
    Route,
    Step,
)
from spatial_service.recovery import TriggerKind

# What each tool's steps produce. A reference binds only to an earlier step whose
# output is the kind it reads.
PRODUCES = {
    "resolve_place": "places",
    "resolve_asset_type": "asset-types",
    "assets_in_scope": "assets",
    "aggregate": "groups",
    "retrieve_documents": "passages",
    # A route reads one asset as where to go, a nearest step as where to be near.
    "resolve_asset": "asset",
    "places_matching": "place-set",
    "route": "route",
    "nearest": "nearest",
    "maintenance_tasks": "jobs",
}

# The tool a shape's last step calls: the one whose result is the answer.
ENDS_IN = {
    QuestionShape.PLACE_LOCATION: "resolve_place",
    QuestionShape.ASSET_CONTAINMENT: "assets_in_scope",
    QuestionShape.ASSET_AGGREGATION: "aggregate",
    QuestionShape.ASSET_DOCUMENT: "retrieve_documents",
    QuestionShape.PLACE_DOCUMENT: "retrieve_documents",
    QuestionShape.ATTRIBUTE_FILTER: "assets_in_scope",
    QuestionShape.ASSET_LOCATION: "resolve_asset",
    QuestionShape.PROXIMITY: "nearest",
    QuestionShape.ROUTE: "route",
    QuestionShape.ASSET_MAINTENANCE: "maintenance_tasks",
}

# What a document shape's retrieval is anchored by: what the step it reads produces.
ANCHORED_BY = {
    QuestionShape.ASSET_DOCUMENT: ("assets", "asset"),
    QuestionShape.PLACE_DOCUMENT: ("places",),
}


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
            elif PRODUCES[source.tool] not in reads.split("|"):
                reasons.append(
                    f"step {step.id} reads {source.id}, which produces "
                    f"{PRODUCES[source.tool]}, not {reads.replace('|', ' or ')}"
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

    last = plan.steps[-1] if plan.steps else None
    if isinstance(last, RetrieveDocuments) and plan.shape in ANCHORED_BY:
        anchored = {step.id: PRODUCES[step.tool] for step in plan.steps}.get(last.anchors.step)
        wanted = ANCHORED_BY[plan.shape]
        # Anything else it might read is not something a retrieval reads at all,
        # and bind() has said so already.
        if anchored in RetrieveDocuments.READS["anchors"].split("|") and anchored not in wanted:
            reasons.append(
                f"a {plan.shape} plan retrieves by {wanted[0]}, but step {last.id} reads {anchored}"
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
    if plan.shape is QuestionShape.ASSET_MAINTENANCE and filtering:
        reasons.append("an asset-maintenance plan has no filters: jobs are planned per type")
    return reasons
