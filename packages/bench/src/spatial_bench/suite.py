"""The deterministic checks, run in one go over one bundle: tool correctness and both leak checks.

No model is called. The planner and the composer are scripted (stub.py), so this
costs nothing and gives the same result every time. It is what CI runs on the
hand-made handover and on the mini campus, and what an operator runs on a real
bundle before an evaluation run spends anything.

It needs a Postgres it may create databases on. It makes three: the bundle as it
is, a copy with the restricted things removed and a copy with more of them
added. All three are dropped when it is done.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_bench.bundle import Bundle
from spatial_bench.check import Verdict, compare_all
from spatial_bench.leaks import Difference, Hidden, Leak, LeakReport, input_leaks, turn_differences
from spatial_bench.questions import QuestionSet
from spatial_bench.run import loaded
from spatial_bench.variants import with_restricted_added, without_restricted
from spatial_core.records import Persona
from spatial_service.retrieval import Embedder

# The persona the counterfactual check holds still: the one who sees least.
LOWEST = Persona.VISITOR
VARIANTS: dict[str, Callable[[Bundle, Persona], Bundle]] = {
    "removed": without_restricted,
    "added": with_restricted_added,
}


@dataclass(frozen=True)
class Report:
    venue: str
    snapshot: str
    gold_source: dict[str, str]
    checked_on: str
    verdicts: tuple[Verdict, ...]
    leaks: LeakReport

    @property
    def tools_passed(self) -> bool:
        return all(verdict.ok for verdict in self.verdicts)

    @property
    def passed(self) -> bool:
        return self.tools_passed and self.leaks.passed

    def as_json(self) -> dict[str, Any]:
        wrong = [verdict for verdict in self.verdicts if not verdict.ok]
        missed = [verdict for verdict in self.verdicts if verdict.missed_documents]
        return {
            "format": "spatial-checks@1",
            "venue": self.venue,
            "snapshot": self.snapshot,
            "gold_source": self.gold_source,
            "checked_on": self.checked_on,
            "model": "scripted",
            "questions": len(self.verdicts),
            "tool_correctness": {
                "passed": self.tools_passed,
                "mismatched": [
                    {"question": v.question, "mismatches": list(v.mismatches)} for v in wrong
                ],
                "missed_documents": [
                    {"question": v.question, "documents": list(v.missed_documents)} for v in missed
                ],
            },
            "leak_checks": {
                "input": {
                    "passed": not self.leaks.inputs,
                    "turns": self.leaks.turns.get("input", 0),
                    "leaks": [
                        {
                            "question": leak.question,
                            "persona": leak.persona.value,
                            "kind": leak.kind,
                            # The hidden thing itself. This report is the operator's
                            # to read, not something to publish as it is.
                            "value": leak.value,
                            "where": leak.where,
                        }
                        for leak in self.leaks.inputs
                    ],
                },
                "counterfactual": {
                    "passed": not self.leaks.counterfactual,
                    "turns": self.leaks.turns.get("counterfactual", 0),
                    "differences": [
                        {"question": d.question, "variant": d.variant, "paths": list(d.paths)}
                        for d in self.leaks.counterfactual
                    ],
                },
            },
        }


def run(
    bundle: Path,
    gold: QuestionSet,
    admin_url: str,
    work: Path,
    *,
    embedder: Embedder | None = None,
    counterfactual: bool = True,
) -> Report:
    """Ask every question of ``bundle`` and check what comes back.

    ``work`` is a directory for the index files and the bundle's copies. With no
    ``embedder`` retrieval is keyword search only, which is what a scripted run
    over a bundle with real vectors has to settle for.
    """
    records = Bundle.read(bundle)
    leaks: list[Leak] = []
    differences: list[Difference] = []
    turns = {"input": 0, "counterfactual": 0}
    with loaded(bundle, admin_url, work / "index") as base:
        asked = base.bench(embedder).ask_all(gold.questions)
        verdicts = compare_all(asked)
        for persona in gold.personas:
            leaks += input_leaks(asked, Hidden.of(records, persona))
        turns["input"] += len(asked)
        lowest = [one for one in asked if one.question.persona is LOWEST]
        for name, change in VARIANTS.items() if counterfactual else ():
            copy = change(records, LOWEST).write(work / name)
            with loaded(copy, admin_url, work / f"index-{name}") as other:
                again = other.bench(embedder).ask_all(gold.asked_as(LOWEST))
            differences += turn_differences(lowest, again, name)
            # The copy has hidden things the bundle never had. They may not show up either.
            leaks += input_leaks(again, Hidden.of(Bundle.read(copy), LOWEST))
            turns["input"] += len(again)
            turns["counterfactual"] += len(again)
    return Report(
        venue=str(records.manifest["venue"]),
        snapshot=str(records.manifest["snapshot"]),
        gold_source=dict(gold.source),
        checked_on=datetime.datetime.now(datetime.UTC).date().isoformat(),
        verdicts=tuple(verdicts),
        leaks=LeakReport(tuple(leaks), tuple(differences), turns),
    )
