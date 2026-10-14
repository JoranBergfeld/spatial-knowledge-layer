"""An evaluation run: the question set put to real models, scored, and written down.

A run is tied to one snapshot (the current one of the database it is pointed at)
and one set of deployments. It goes in this order:

    prepare    read everything, refuse what cannot be published or costed, and
               work out the worst-case cost. Nothing is called yet.
    execute    ask every question through the service's turn endpoint, score
               each turn as it comes back, and stop before the ceiling.
    write      the eval snapshot (what the site reads) and the run report (what
               the owner reads, every question in it).

A run that stops early writes no eval snapshot. Figures over the questions it
happened to reach would not be figures over the question set. It still writes
its report, with what it spent.

The models, the index and the NLI model come in through `Wiring`. cli.py builds
the real ones from the environment. Tests pass a scripted model and a stand-in
NLI, and no test starts a real run.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi.testclient import TestClient

from spatial_bench import stats
from spatial_bench.evalmetrics import EvalMetrics
from spatial_bench.nli import Nli
from spatial_bench.questions import HAND_WRITTEN, ExpectedKind, Question, QuestionSet
from spatial_bench.run import Asked
from spatial_bench.score import DECLINES, PRODUCED, UNAVAILABLE, Scored, score
from spatial_bench.spend import Assumed, Estimate, Ledger, Models, Rates, SpendError, estimate
from spatial_bench.stats import Rule
from spatial_core.records import Persona
from spatial_service.app import create_app
from spatial_service.evaluation import (
    FORMAT,
    LABEL,
    UNMEASURED,
    CompositionStage,
    Declined,
    Deployments,
    EvalSnapshot,
    Figures,
    LeakCheck,
    LeakChecks,
    RetrievalStage,
    Run,
    Section,
    Stages,
    TurnKinds,
)
from spatial_service.llm import Llm
from spatial_service.places import Venue
from spatial_service.retrieval import Anchors, Documents, Retrieval
from spatial_service.store import Store

REPORT_FORMAT = "spatial-eval-run@1"
CHECKS_FORMAT = "spatial-checks@1"
TEMPLATE = "template"
SNAPSHOT_FILE = "eval-snapshot.json"
# This many turns in a row with no model or no index behind them, and the run gives up.
UNREACHABLE_STREAK = 5

Say = Callable[[str], None]


class RunError(Exception):
    pass


@dataclass
class Wiring:
    """Everything a run calls out to."""

    store: Store
    llm: Llm
    nli: Nli
    documents: Documents | None = None
    # The same retrieval with the semantic ranker on, when the run compares it.
    reranked: Documents | None = None
    # Whether ``documents`` embeds its queries. Keyword-only retrieval names no deployment.
    embeds: bool = True
    semantic_ranker: bool = False
    metrics: EvalMetrics = field(default_factory=EvalMetrics)


@dataclass(frozen=True)
class Checks:
    """The leak checks' last result, from `bench check`'s report."""

    venue: str
    snapshot: str
    leaks: LeakChecks

    @classmethod
    def load(cls, path: Path) -> Checks:
        try:
            found = json.loads(path.read_text(encoding="utf-8"))
            if found["format"] != CHECKS_FORMAT:
                raise ValueError(f"it is {found['format']}, this reads {CHECKS_FORMAT}")
            date = found["checked_on"]
            both = found["leak_checks"]
            if not (both["input"]["turns"] and both["counterfactual"]["turns"]):
                raise ValueError("a leak check was skipped. Run bench check with both")
            return cls(
                venue=found["venue"],
                snapshot=found["snapshot"],
                leaks=LeakChecks(
                    input=LeakCheck(passed=both["input"]["passed"], date=date),
                    counterfactual=LeakCheck(passed=both["counterfactual"]["passed"], date=date),
                ),
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise RunError(f"cannot use the checks report {path}: {error}") from error


@dataclass(frozen=True)
class Request:
    """What the owner asked for."""

    gold: QuestionSet
    gold_sha256: str
    hand_written: Sequence[Question]
    checks: Checks
    rates: Rates
    ceiling: float
    assumed: Assumed = Assumed()
    # An earlier run's report, to weigh this run's model pair against.
    baseline: dict[str, Any] | None = None

    @property
    def questions(self) -> list[tuple[str, Question]]:
        template = [(TEMPLATE, question) for question in self.gold.questions]
        return template + [(HAND_WRITTEN, question) for question in self.hand_written]


@dataclass(frozen=True)
class Prepared:
    run: Run
    models: Models
    estimate: Estimate


@dataclass(frozen=True)
class Outcome:
    scored: tuple[Scored, ...]
    ledger: Ledger
    # Why the run stopped before the last question. None when it got through.
    stopped: str | None

    @property
    def completed(self) -> bool:
        return self.stopped is None


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(request: Request, wiring: Wiring) -> Prepared:
    """Name the run and cost it. Raises RunError for a run that should not start."""
    if request.gold.source.get("kind") != "cobie":
        # The demo venue's gold is a regression check. Nothing from it is published.
        raise RunError("this gold is not the test bench's. A run measures the test bench only.")
    current, _ = wiring.store.served()
    if current is None:
        raise RunError("no snapshot is current in this database")
    if request.checks.snapshot != current:
        raise RunError(
            f"the checks report is for {request.checks.snapshot} and the current snapshot is"
            f" {current}. Run bench check on the bundle that is loaded."
        )
    index = wiring.store.venue(current, Persona.VISITOR).index
    embedding = index.embedding_deployment if index and wiring.documents and wiring.embeds else None
    models = Models(
        planning=wiring.llm.planning.name,
        composition=wiring.llm.composition.name,
        embedding=embedding,
        embeddings_per_turn=2 if wiring.reranked is not None else 1,
    )
    try:
        cost = estimate(len(request.questions), models, request.rates, request.assumed)
    except SpendError as error:
        raise RunError(str(error)) from error
    if cost.amount > request.ceiling:
        raise RunError(
            f"the estimate, {cost.amount:.2f} {request.rates.currency}, is over the ceiling of"
            f" {request.ceiling:.2f}. Raise the ceiling or ask fewer questions."
        )
    if request.baseline is not None:
        then = request.baseline.get("run", {})
        if request.baseline.get("format") != REPORT_FORMAT or not request.baseline.get("completed"):
            raise RunError("the baseline is not the report of a run that finished")
        if (then.get("snapshot"), then.get("gold_sha256")) != (current, request.gold_sha256):
            raise RunError("the baseline run asked another question set or another snapshot")
    today = datetime.datetime.now(datetime.UTC).date().isoformat()
    run = Run(
        id=f"{today}-{uuid4().hex[:8]}",
        date=today,
        venue=request.checks.venue,
        snapshot=current,
        deployments=Deployments(
            planning=models.planning,
            composition=models.composition,
            embedding=models.embedding,
            nli=wiring.nli.name,
        ),
        semantic_ranker=wiring.semantic_ranker,
        gold_sha256=request.gold_sha256,
    )
    return Prepared(run, models, cost)


class _Paired:
    """Retrieval as the turn uses it, with the ranker's version of each one kept beside it."""

    def __init__(self, plain: Documents, reranked: Documents) -> None:
        self.plain = plain
        self.reranked = reranked
        self.last: list[str] | None = None
        self.extra = 0

    def retrieve(self, venue: Venue, anchors: Anchors, query: str) -> Retrieval:
        found = self.plain.retrieve(venue, anchors, query)
        again = self.reranked.retrieve(venue, anchors, query)
        self.last = list(dict.fromkeys(passage.chunk.document_id for passage in again.passages))
        self.extra += again.embedding is not None
        return found

    def take(self) -> tuple[list[str] | None, int]:
        last, extra = self.last, self.extra
        self.last, self.extra = None, 0
        return last, extra


def execute(request: Request, wiring: Wiring, prepared: Prepared) -> Outcome:
    """Ask and score every question, or as many as the ceiling allows."""
    ledger = Ledger(request.rates, request.assumed)
    scored: list[Scored] = []
    paired = None
    documents = wiring.documents
    if wiring.documents is not None and wiring.reranked is not None:
        documents = paired = _Paired(wiring.documents, wiring.reranked)
    app = create_app(wiring.store, wiring.llm, documents=documents)
    stopped = None
    streak = 0
    # One client for the whole run, so the model clients keep one event loop.
    with TestClient(app) as client:
        for set_name, question in request.questions:
            # Before the call, not after: the next turn may cost as much as the
            # dearest one so far, or as much as the estimate thought a turn could.
            worst = max(prepared.estimate.per_turn, ledger.largest_turn)
            if ledger.amount + worst > request.ceiling:
                stopped = (
                    f"the ceiling: {ledger.amount:.4f} spent, and one more turn could pass"
                    f" {request.ceiling:.2f}"
                )
                break
            response = client.post(
                "/turn",
                json={
                    "persona": question.persona.value,
                    "snapshot": prepared.run.snapshot,
                    "question": question.question,
                },
            )
            turn = response.json()
            reranked, extra = paired.take() if paired else (None, 0)
            ledger.add_turn(turn, extra)
            asked = Asked(question, response.status_code, turn, (), (), ())
            scored.append(score(asked, wiring.nli, set_name=set_name, reranked=reranked))
            if ledger.unpriced:
                # Same as before the run starts: no rate, no run. This turn is already
                # paid for, so it is counted and scored, and nothing more is asked.
                stopped = (
                    f"no rate for {', '.join(sorted(ledger.unpriced))}, so the run cannot be costed"
                )
                break
            streak = streak + 1 if scored[-1].produced == UNAVAILABLE else 0
            if streak >= UNREACHABLE_STREAK:
                stopped = f"{streak} turns in a row came back unavailable"
                break
    return Outcome(tuple(scored), ledger, stopped)


def section(scored: Sequence[Scored]) -> Section:
    """The figures over one set of questions. An empty set gives figures over nothing."""
    n = len(scored)
    documents = [one for one in scored if one.recall_at_5 is not None]
    composed = [one for one in scored if one.numbers is not None]
    clarified = [one for one in scored if one.candidates is not None]
    matrix = {
        expected.value: {
            produced: sum(
                1 for one in scored if (one.expected, one.produced) == (expected, produced)
            )
            for produced in PRODUCED
        }
        for expected in ExpectedKind
    }
    personas = sorted({one.persona for one in scored})
    return Section(
        n=n,
        correct=stats.share(sum(one.correct for one in scored), n),
        stages=Stages(
            tools=stats.share(sum(one.tools for one in scored), n),
            retrieval=RetrievalStage(
                recall_at_5=stats.mean([one.recall_at_5 or 0.0 for one in documents]),
                mrr=stats.mean([one.reciprocal_rank or 0.0 for one in documents]),
            ),
            composition=CompositionStage(
                holds=stats.share(sum(bool(one.composition) for one in composed), len(composed)),
                numbers=stats.share(sum(bool(one.numbers) for one in composed), len(composed)),
                entailment=stats.share(
                    sum(one.entailed for one in scored), sum(one.segments for one in scored)
                ),
            ),
        ),
        turn_kinds=TurnKinds(
            matrix=matrix if n else {},
            answered_when_it_should_have_declined=Declined(
                count=sum(one.answered_when_it_should_have_declined for one in scored),
                of=sum(one.expected in DECLINES for one in scored),
            ),
        ),
        clarification_candidates=stats.share(
            sum(bool(one.candidates) for one in clarified), len(clarified)
        ),
        by_persona={
            persona: stats.share(
                sum(one.correct for one in scored if one.persona == persona),
                sum(one.persona == persona for one in scored),
            )
            for persona in personas
        },
    )


def snapshot_of(run: Run, scored: Sequence[Scored], checks: Checks) -> EvalSnapshot:
    return EvalSnapshot(
        format=FORMAT,
        label=LABEL,
        run=run,
        figures=Figures(
            template=section([one for one in scored if one.set == TEMPLATE]),
            hand_written=section([one for one in scored if one.set == HAND_WRITTEN]),
        ),
        leak_checks=checks.leaks,
        unmeasured=list(UNMEASURED),
    )


def rules(request: Request, outcome: Outcome) -> list[Rule]:
    """The two adoption rules, worked out. Printed and written down, applied by nobody."""
    compared = [one for one in outcome.scored if one.reranked_recall_at_5 is not None]
    gains = [(one.reranked_recall_at_5 or 0.0) - (one.recall_at_5 or 0.0) for one in compared]
    ranker = stats.ranker_rule(gains if compared else None)
    if request.baseline is None:
        return [ranker, stats.pair_rule(None)]
    then = {one["id"]: bool(one["correct"]) for one in request.baseline["questions"]}
    differences = [
        float(one.correct) - float(then[one.question])
        for one in outcome.scored
        if one.question in then
    ]
    cheaper = outcome.ledger.amount < float(request.baseline["spend"]["amount"])
    return [ranker, stats.pair_rule(differences, cheaper)]


def segments_file(run: Run) -> str:
    return f"segments-{run.id}.jsonl"


def write_segments(path: Path, outcome: Outcome) -> None:
    """Every segment the NLI model scored, one a line, with what it was scored against.

    For a person to label, so the NLI model's verdicts can be checked by hand. It
    holds passage text of every clearance class, so it is the owner's alone: not
    the eval snapshot, not the metrics, and readable by nobody else on the machine.
    """
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    with path.open("w", encoding="utf-8") as out:
        for one in outcome.scored:
            for segment in one.scored_segments:
                line = {"question": one.question, "persona": one.persona, **asdict(segment)}
                out.write(json.dumps(line, ensure_ascii=False) + "\n")


def report_of(request: Request, prepared: Prepared, outcome: Outcome) -> dict[str, Any]:
    """The owner's report: everything, every question in it. Not for publishing."""
    return {
        "format": REPORT_FORMAT,
        # Every scored segment with its passage text and NLI score, beside this file.
        "segments_file": segments_file(prepared.run),
        "completed": outcome.completed,
        "stopped": outcome.stopped,
        "run": prepared.run.model_dump(mode="json"),
        "asked": len(outcome.scored),
        "of": len(request.questions),
        "estimate": {**asdict(prepared.estimate), "assumed_tokens": asdict(request.assumed)},
        "ceiling": request.ceiling,
        # Evaluation spend, in its own ledger category. Never part of a visitor's day.
        "spend": outcome.ledger.as_json(),
        # Which of a question's two kinds each figure reads.
        "turn_kinds_are": {
            "produced": "what the site would send, after the NLI drop. The matrix is over this",
            "layer_produced": (
                "what the layer produced, before the NLI drop."
                " Answered when it should have declined is over this"
            ),
        },
        "adoption_rules": [
            {
                "rule": rule.name,
                "difference": rule.difference.model_dump(mode="json"),
                "met": rule.met,
                "said": rule.said,
                "applied": False,
            }
            for rule in rules(request, outcome)
        ],
        "questions": [
            {
                "id": one.question,
                "set": one.set,
                "persona": one.persona,
                "shape": one.shape,
                "expected": one.expected,
                "produced": one.produced,
                "layer_produced": one.layer_produced,
                "correct": one.correct,
                "tools": one.tools,
                "retrieval": one.retrieval,
                "composition": one.composition,
                "recall_at_5": one.recall_at_5,
                "reciprocal_rank": one.reciprocal_rank,
                "reranked_recall_at_5": one.reranked_recall_at_5,
                "segments": one.segments,
                "entailed": one.entailed,
                "mismatches": list(one.mismatches),
            }
            for one in outcome.scored
        ],
    }
