"""The executor: runs a checked plan's steps in order, with no model call between them.

The host binds the venue (one persona, one snapshot) before the first step. A step
whose `within` points at an earlier step works inside the ids that step produced.
The first step that does not resolve to exactly one place ends the run: a
clarification or an abstention, decided by what it found, or a trigger when the
tool rejected its arguments.

Every step's arguments are checked before the first step runs. Checked as each
step came up, a malformed argument in a later step would be reached only when the
earlier ones resolved, and whether a turn buys a recovery call would then tell
what the persona can see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spatial_service.places import How, Outcome, Venue, check_arguments, resolve_place
from spatial_service.plan import Plan, Step
from spatial_service.recovery import Trigger, TriggerKind


@dataclass(frozen=True)
class StepResult:
    step: str
    tool: str
    mention: str
    text: str
    args: dict[str, Any]
    outcome: Outcome
    ids: tuple[str, ...]
    how: How | None
    matched: int
    # Why the tool rejected its arguments, when it did.
    rejection: tuple[str, ...] = ()


@dataclass(frozen=True)
class Execution:
    results: tuple[StepResult, ...]

    @property
    def last(self) -> StepResult:
        return self.results[-1]

    @property
    def complete(self) -> bool:
        """Every step ran and resolved one place."""
        return all(r.outcome is Outcome.RESOLVED for r in self.results)

    @property
    def trigger(self) -> Trigger | None:
        """A trigger the run raised, which only a step whose arguments were rejected does."""
        if self.last.outcome is not Outcome.REJECTED:
            return None
        return Trigger(
            kind=TriggerKind.MALFORMED_ARGUMENTS,
            step=self.last.step,
            reasons=list(self.last.rejection),
        )


def execute(plan: Plan, venue: Venue) -> Execution:
    mentions = {mention.id: mention for mention in plan.mentions}
    for step in plan.steps:
        text = mentions[step.mention].text
        problems = check_arguments(text)
        if problems:
            rejected = StepResult(
                step=step.id,
                tool=step.tool,
                mention=step.mention,
                text=text,
                args=_args(step),
                outcome=Outcome.REJECTED,
                ids=(),
                how=None,
                matched=0,
                rejection=tuple(problems),
            )
            return Execution((rejected,))

    produced: dict[str, tuple[str, ...]] = {}
    results: list[StepResult] = []
    for step in plan.steps:
        text = mentions[step.mention].text
        within = produced[step.within.step] if step.within is not None else None
        found = resolve_place(venue, text, within)
        results.append(
            StepResult(
                step=step.id,
                tool=step.tool,
                mention=step.mention,
                text=text,
                args=_args(step),
                outcome=found.outcome,
                ids=found.ids,
                how=found.how,
                matched=found.matched,
            )
        )
        if found.outcome is not Outcome.RESOLVED:
            break
        produced[step.id] = found.ids
    return Execution(tuple(results))


def _args(step: Step) -> dict[str, Any]:
    return step.model_dump(mode="json", exclude={"id", "tool"})
