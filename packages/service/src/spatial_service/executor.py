"""The executor: runs a checked plan's steps in order, with no model call between them.

The host binds the venue (one persona, one snapshot) before the first step. A step
whose `within` points at an earlier step works inside the ids that step produced.
The first step that does not resolve to exactly one place ends the run: a
clarification or an abstention, decided by what it found.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spatial_service.places import How, Outcome, Venue, resolve_place
from spatial_service.plan import Plan


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


def execute(plan: Plan, venue: Venue) -> Execution:
    mentions = {mention.id: mention for mention in plan.mentions}
    produced: dict[str, tuple[str, ...]] = {}
    results: list[StepResult] = []
    for step in plan.steps:
        text = mentions[step.mention].text
        within = produced[step.within.step] if step.within is not None else None
        found = resolve_place(venue, text, within)
        result = StepResult(
            step=step.id,
            tool=step.tool,
            mention=step.mention,
            text=text,
            args=step.model_dump(mode="json", exclude={"id", "tool"}),
            outcome=found.outcome,
            ids=found.ids,
            how=found.how,
            matched=found.matched,
        )
        results.append(result)
        if found.outcome is not Outcome.RESOLVED:
            break
        produced[step.id] = found.ids
    return Execution(tuple(results))
