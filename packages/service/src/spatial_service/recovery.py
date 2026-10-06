"""Recovery: the one second chance a plan gets, and only on a typed trigger.

A trigger is something the model got wrong that a second plan could fix:

    plan-invalid          the plan failed validation: it does not parse, breaks the
                          step cap, repeats an id, leaves a step feeding nothing
    unbound-reference     a reference in the plan names nothing it can bind to: a
                          step resolving a mention the plan lacks, a step reading one
                          that is not earlier or whose output it cannot take
    empty-retrieval       document retrieval returned nothing. No tool retrieves
                          documents yet, so nothing raises it so far.
    malformed-arguments   a tool rejected its arguments before looking anything up

Outcomes that are answers are never triggers, and never buy a model call: nothing
the persona can see matching (which covers a place beyond clearance), several
places matching (a clarification), a question of a shape the layer does not
answer. A tool's arguments are judged before it reads the venue, so whether one is
malformed never depends on what the persona can see.

The trace keeps both versions of a plan. The second is also given as a diff
against the first, so a misreading of the question shows as the lines that moved.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class TriggerKind(StrEnum):
    PLAN_INVALID = "plan-invalid"
    UNBOUND_REFERENCE = "unbound-reference"
    EMPTY_RETRIEVAL = "empty-retrieval"
    MALFORMED_ARGUMENTS = "malformed-arguments"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Trigger(_Model):
    kind: TriggerKind
    # The step that raised it, for a trigger raised while the plan ran.
    step: str | None = None
    reasons: list[str]


class RecoveryOutcome(StrEnum):
    """Per turn, so the repair rate is a count over turns."""

    # The first plan ran without a trigger.
    NOT_NEEDED = "not-needed"
    # The first plan raised a trigger and the second ran its steps without one.
    REPAIRED = "repaired"
    # The second plan raised a trigger too, or ran nothing because it called the
    # question unsupported, or the recovery call failed.
    UNREPAIRED = "unrepaired"


class PlanChange(_Model):
    """One difference between two versions of a plan, at a JSON Pointer path."""

    op: Literal["add", "remove", "replace"]
    path: str
    before: Any = None
    after: Any = None


def diff(before: Any, after: Any, path: str = "") -> list[PlanChange]:
    """What changed from ``before`` to ``after``, both JSON values, in document order.

    Lists of objects that each carry an ``id`` (steps, mentions) are matched by it,
    so a step added or dropped at the front is one change and not every step after
    it. What a list lost comes first, at its old index; the rest sits at its new
    one. Any other list, and one whose shared ids changed order, goes by position.
    """
    if before == after:
        return []
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[PlanChange] = []
        for key in [*before, *(k for k in after if k not in before)]:
            at = f"{path}/{_escape(str(key))}"
            if key not in after:
                changes.append(PlanChange(op="remove", path=at, before=before[key]))
            elif key not in before:
                changes.append(PlanChange(op="add", path=at, after=after[key]))
            else:
                changes.extend(diff(before[key], after[key], at))
        return changes
    if isinstance(before, list) and isinstance(after, list):
        changes = []
        old, new = _by_id(before), _by_id(after)
        if old is not None and new is not None and _same_order(old, new):
            for index, key in enumerate(old):
                if key not in new:
                    changes.append(
                        PlanChange(op="remove", path=f"{path}/{index}", before=before[index])
                    )
            for index, key in enumerate(new):
                at = f"{path}/{index}"
                if key not in old:
                    changes.append(PlanChange(op="add", path=at, after=after[index]))
                else:
                    changes.extend(diff(old[key], new[key], at))
            return changes
        for index in range(max(len(before), len(after))):
            at = f"{path}/{index}"
            if index >= len(after):
                changes.append(PlanChange(op="remove", path=at, before=before[index]))
            elif index >= len(before):
                changes.append(PlanChange(op="add", path=at, after=after[index]))
            else:
                changes.extend(diff(before[index], after[index], at))
        return changes
    return [PlanChange(op="replace", path=path, before=before, after=after)]


def _by_id(items: list[Any]) -> dict[str, Any] | None:
    """A list's items under their ids, when each is an object with one of its own."""
    keyed: dict[str, Any] = {}
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            return None
        keyed[item["id"]] = item
    return keyed if len(keyed) == len(items) else None


def _same_order(old: dict[str, Any], new: dict[str, Any]) -> bool:
    """Whether the ids two lists share come in the same order in both."""
    return [k for k in old if k in new] == [k for k in new if k in old]


def _escape(key: str) -> str:
    return key.replace("~", "~0").replace("/", "~1")
