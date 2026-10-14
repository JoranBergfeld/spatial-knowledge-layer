"""The deterministic checks, run in one go over one bundle: tool correctness and both leak checks.

No model is called. The planner and the composer are scripted (stub.py), so this
costs nothing and gives the same result every time. It is what CI runs on the
hand-made handover and on the mini campus, and what an operator runs on a real
bundle before an evaluation run spends anything.

Routes have no gold, so tool correctness says nothing of them. The counterfactual
check does not need gold though: it only needs a turn to stay the same. So on a
venue with edges it also asks the visitor the way to rooms they can see, on the
bundle and on each copy. Those are the route probes, and they are held to the same
two leak checks as any other turn.

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
from spatial_bench.questions import Expected, ExpectedKind, Question, QuestionSet, Slots
from spatial_bench.run import loaded
from spatial_bench.variants import with_restricted_added, without_restricted
from spatial_core.records import PERSONAS, Persona
from spatial_service.retrieval import Embedder

# The persona the counterfactual check holds still: the one who sees least.
LOWEST = Persona.VISITOR
VARIANTS: dict[str, Callable[[Bundle, Persona], Bundle]] = {
    "removed": without_restricted,
    "added": with_restricted_added,
}


# How many rooms the counterfactual check asks the way to.
PROBE_CAP = 25
PROBE_TEMPLATE = "counterfactual-route-probe"


def route_probes(bundle: Bundle, persona: Persona = LOWEST) -> list[Question]:
    """The way to rooms ``persona`` can see, as questions. None on a venue with no edges.

    Not gold: what comes back is only ever compared with itself across the copies.
    """
    if not bundle.edges:
        return []
    seen = bundle.visible(persona).spaces
    labels = sorted(
        {
            space.label.value
            for space in bundle.spaces
            if space.id in seen and space.label is not None and space.clearance is not None
        }
    )
    step = max(1, -(-len(labels) // PROBE_CAP))
    return [
        Question(
            id=f"probe/route/{label}/{persona.value}",
            pair=f"probe/route/{label}",
            shape="route",
            template=PROBE_TEMPLATE,
            persona=persona,
            question=f"How do I get to {label}?",
            slots=Slots(place=label),
            # Never scored. A question has to carry one.
            expected=Expected(kind=ExpectedKind.ANSWER, reason="a probe, not gold"),
        )
        for label in labels[::step][:PROBE_CAP]
    ]


def unpaired(gold: QuestionSet) -> list[Question]:
    """Gold's questions again, once per pair, as each persona gold has no copy for.

    Gold pairs the visitor with the technician, which leaves staff unasked. The
    input leak check needs no gold, only the turn, so staff get the same words and
    slots. Nothing here is ever scored.
    """
    once = {question.pair: question for question in reversed(gold.questions)}
    return [
        question.model_copy(update={"id": f"{pair}/{persona.value}", "persona": persona})
        for persona in PERSONAS
        if persona not in gold.personas
        for pair, question in sorted(once.items())
    ]


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
                    # Of the bundle itself. The copies are only ever asked as a visitor.
                    "turns_by_persona": {
                        persona.value: self.leaks.asked_as.get(persona, 0) for persona in PERSONAS
                    },
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
                    # Routes asked of each copy on top of those. No gold behind them.
                    "route_probes": self.leaks.turns.get("route_probes", 0),
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
    turns = {"input": 0, "counterfactual": 0, "route_probes": 0}
    asked_as: dict[Persona, int] = {}
    probes = route_probes(records) if counterfactual else []
    with loaded(bundle, admin_url, work / "index") as base:
        asked = base.bench(embedder).ask_all(gold.questions)
        verdicts = compare_all(asked)
        probed = base.bench(embedder).ask_all(probes)
        # Every persona, whether or not gold has a copy for them. No gold is needed here.
        read = [*asked, *probed, *base.bench(embedder).ask_all(unpaired(gold))]
        for persona in PERSONAS:
            leaks += input_leaks(read, Hidden.of(records, persona))
            asked_as[persona] = sum(1 for one in read if one.question.persona is persona)
        turns["input"] += len(read)
        lowest = [one for one in asked if one.question.persona is LOWEST]
        for name, change in VARIANTS.items() if counterfactual else ():
            copy = change(records, LOWEST).write(work / name)
            with loaded(copy, admin_url, work / f"index-{name}") as other:
                again = other.bench(embedder).ask_all(gold.asked_as(LOWEST))
                routed = other.bench(embedder).ask_all(probes)
            differences += turn_differences(lowest, again, name)
            differences += turn_differences(probed, routed, name)
            # The copy has hidden things the bundle never had. They may not show up either.
            leaks += input_leaks([*again, *routed], Hidden.of(Bundle.read(copy), LOWEST))
            turns["input"] += len(again) + len(routed)
            turns["counterfactual"] += len(again)
            turns["route_probes"] += len(routed)
    return Report(
        venue=str(records.manifest["venue"]),
        snapshot=str(records.manifest["snapshot"]),
        gold_source=dict(gold.source),
        checked_on=datetime.datetime.now(datetime.UTC).date().isoformat(),
        verdicts=tuple(verdicts),
        leaks=LeakReport(tuple(leaks), tuple(differences), turns, asked_as),
    )
