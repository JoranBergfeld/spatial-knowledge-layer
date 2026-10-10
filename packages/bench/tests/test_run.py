"""Evaluation runs, through `bench run`, on the hand-made handover.

No real run is started here. The planner and the composer are the scripted ones,
with token usage the test sets. The NLI model is a stand-in that says what the
test tells it to. The stores are real: Postgres and the stand-in index.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import fixture
import prose
from spatial_bench.cli import STOPPED, main
from spatial_bench.evalmetrics import EvalMetrics
from spatial_bench.evalrun import Wiring
from spatial_bench.questions import QuestionSet
from spatial_bench.run import Loaded
from spatial_bench.stub import COMPOSER, PLANNER, scripted_llm
from spatial_service.app import create_app
from spatial_service.evaluation import EvalSnapshot
from spatial_service.places import Venue
from spatial_service.retrieval import Anchors, Embedded, Retrieval, Retriever
from spatial_service.store import PostgresStore

# Per million tokens. Made up, and round on purpose.
RATES = {
    "currency": "USD",
    "as_of": "2026-10-01",
    "per_million_tokens": {
        PLANNER: {"input": 2.0, "output": 20.0},
        COMPOSER: {"input": 1.0, "output": 10.0},
        prose.EMBEDDING_DEPLOYMENT: {"input": 0.0},
    },
}
# What each scripted call reports: 1000 tokens in, 100 out.
USAGE = (1000, 100)
# So a planning call is 0.002 + 0.002 and a composition call 0.001 + 0.001.
PLANNING_CALL, COMPOSITION_CALL = 0.004, 0.002


class Embedder:
    def embed(self, deployment: str, text: str) -> Embedded:
        return Embedded(vector=tuple(prose.embedding(text)), deployment=deployment, input_tokens=1)


@dataclass
class Says:
    """An NLI model that gives every pair the same score, and keeps what it was asked."""

    score: float = 0.9
    pairs: list[tuple[str, str]] = field(default_factory=list)
    name: str = "stand-in-nli"

    def entailment(self, premise: str, hypothesis: str) -> float:
        self.pairs.append((premise, hypothesis))
        return self.score


class SameAgain:
    """Retrieval with the ranker on, played by a ranker that changes nothing."""

    def __init__(self, behind: Retriever) -> None:
        self.behind = behind

    def retrieve(self, venue: Venue, anchors: Anchors, query: str) -> Retrieval:
        return self.behind.retrieve(venue, anchors, query)


@dataclass
class Run:
    """One `bench run`, with everything it wrote and said."""

    code: int
    out: Path
    said: str
    prompts: list[str]
    calls: list[str]

    @property
    def snapshot(self) -> dict[str, Any]:
        found: dict[str, Any] = json.loads((self.out / "eval-snapshot.json").read_text())
        return found

    @property
    def report(self) -> dict[str, Any]:
        (path,) = self.out.glob("run-*.json")
        found: dict[str, Any] = json.loads(path.read_text())
        return found

    @property
    def template(self) -> dict[str, Any]:
        found: dict[str, Any] = self.snapshot["figures"]["template"]
        return found


Start = Callable[..., Run]


@pytest.fixture
def start(
    bench_loaded: Loaded,
    bench_gold: QuestionSet,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> Start:
    def run(
        *extra: str,
        answer: str = "yes",
        ceiling: str = "20",
        usage: tuple[int, int] = USAGE,
        nli: Says | None = None,
        gold: Callable[[dict[str, Any]], None] | None = None,
        hand_written: Sequence[dict[str, Any]] | None = None,
        checks: dict[str, Any] | None = None,
        metrics: EvalMetrics | None = None,
        compare_ranker: bool = False,
    ) -> Run:
        asked = bench_gold.model_dump(mode="json")
        if gold is not None:
            gold(asked)
        files = {
            "gold.json": asked,
            "rates.json": RATES,
            "checks.json": {
                "format": "spatial-checks@1",
                "venue": "mini-bench",
                "snapshot": bench_loaded.snapshot,
                "checked_on": "2026-10-09",
                "leak_checks": {
                    "input": {"passed": True, "turns": 332},
                    "counterfactual": {"passed": True, "turns": 166},
                },
                **(checks or {}),
            },
        }
        if hand_written is not None:
            files["hand-written.json"] = {
                "format": "spatial-hand-written@1",
                "venue": "mini-bench",
                "questions": list(hand_written),
            }
        for name, body in files.items():
            (tmp_path / name).write_text(json.dumps(body))
        calls: list[str] = []
        prompts: list[str] = []
        retriever = Retriever(bench_loaded.index, Embedder())
        wiring = Wiring(
            store=PostgresStore(bench_loaded.database_url),
            llm=scripted_llm(bench_gold.questions, calls, usage=usage),
            nli=nli or Says(),
            documents=retriever,
            reranked=SameAgain(retriever) if compare_ranker else None,
            metrics=metrics or EvalMetrics(),
        )

        def confirm(prompt: str) -> str:
            prompts.append(prompt)
            return answer

        out = tmp_path / f"out-{len(list(tmp_path.glob('out-*')))}"
        argv = ["run", "--gold", str(tmp_path / "gold.json")]
        argv += ["--checks", str(tmp_path / "checks.json"), "--rates", str(tmp_path / "rates.json")]
        argv += ["--hand-written", str(tmp_path / "hand-written.json")]
        argv += ["--ceiling", ceiling, "--out", str(out), *extra]
        code = main(argv, wiring=wiring, confirm=confirm)
        captured = capsys.readouterr()
        return Run(code, out, captured.out + captured.err, prompts, calls)

    return run


@pytest.fixture(scope="module")
def finished(
    bench_loaded: Loaded,
    bench_gold: QuestionSet,
    tmp_path_factory: pytest.TempPathFactory,
) -> Run:
    """One whole run with the scripted model, shared by the tests that only read it."""
    capture = pytest.MonkeyPatch()
    lines: list[str] = []
    capture.setattr("builtins.print", lambda *said, **_: lines.append(" ".join(map(str, said))))
    tmp = tmp_path_factory.mktemp("finished")
    (tmp / "gold.json").write_text(json.dumps(bench_gold.model_dump(mode="json")))
    (tmp / "rates.json").write_text(json.dumps(RATES))
    checks = {
        "format": "spatial-checks@1",
        "venue": "mini-bench",
        "snapshot": bench_loaded.snapshot,
        "checked_on": "2026-10-09",
        "leak_checks": {
            "input": {"passed": True, "turns": 332},
            "counterfactual": {"passed": False, "turns": 166},
        },
    }
    (tmp / "checks.json").write_text(json.dumps(checks))
    calls: list[str] = []
    wiring = Wiring(
        store=PostgresStore(bench_loaded.database_url),
        llm=scripted_llm(bench_gold.questions, calls, usage=USAGE),
        nli=Says(),
        documents=Retriever(bench_loaded.index, Embedder()),
    )
    argv = ["run", "--gold", str(tmp / "gold.json"), "--checks", str(tmp / "checks.json")]
    argv += ["--rates", str(tmp / "rates.json"), "--hand-written", str(tmp / "none.json")]
    argv += ["--ceiling", "20", "--out", str(tmp / "out")]
    try:
        code = main(argv, wiring=wiring, confirm=lambda _: "yes")
    finally:
        capture.undo()
    return Run(code, tmp / "out", "\n".join(lines), [], calls)


# Starting a run


def test_a_run_prints_its_cost_estimate_and_asks_before_it_calls_anything(start: Start) -> None:
    run = start(answer="no")

    # 166 questions, three calls each at most. At the default assumed sizes a turn is
    # two planning calls (8000 in, 600 out) and one composition call (3000 in, 400 out):
    # 2 * (0.016 + 0.012) + (0.003 + 0.004) = 0.063, times 166.
    assert "at most 498 model calls, 10.46 USD at the rates of 2026-10-01" in run.said
    assert run.prompts == ["Start the run? Type yes to spend: "]
    assert run.code == 1
    assert "Nothing was spent" in run.said
    assert run.calls == []
    assert not run.out.exists()


def test_a_run_whose_estimate_is_over_the_ceiling_does_not_even_ask(start: Start) -> None:
    run = start(ceiling="10")

    assert run.code == 1
    assert "is over the ceiling of 10.00" in run.said
    assert run.prompts == []
    assert run.calls == []


def test_a_run_stops_before_the_ceiling_and_writes_no_eval_snapshot(start: Start) -> None:
    # Told a call is tiny, so the estimate fits. The calls then come in at a thousand times that.
    run = start("--planning-tokens", "1:0", "--composition-tokens", "1:0", ceiling="0.05")

    assert run.code == STOPPED
    assert not (run.out / "eval-snapshot.json").exists()
    report = run.report
    assert report["completed"] is False
    assert 0 < report["asked"] < report["of"] == 166
    assert 0 < report["spend"]["amount"] <= 0.05
    assert "No eval snapshot was written" in run.said


def test_a_deployment_with_no_rate_stops_the_run_before_it_starts(start: Start) -> None:
    del RATES["per_million_tokens"][COMPOSER]
    try:
        run = start()
    finally:
        RATES["per_million_tokens"][COMPOSER] = {"input": 1.0, "output": 10.0}

    assert run.code == 1
    assert f"no rate for {COMPOSER}" in run.said
    assert run.prompts == []


def test_a_run_needs_leak_checks_that_ran_on_the_snapshot_it_measures(start: Start) -> None:
    other = start(checks={"snapshot": "2020-01-01"})
    skipped = start(
        checks={
            "leak_checks": {
                "input": {"passed": True, "turns": 166},
                "counterfactual": {"passed": True, "turns": 0},
            }
        }
    )

    assert (other.code, skipped.code) == (1, 1)
    assert "the checks report is for 2020-01-01" in other.said
    assert "a leak check was skipped" in skipped.said
    assert other.prompts == skipped.prompts == []


def test_gold_that_is_not_the_test_benchs_is_never_measured(start: Start) -> None:
    def from_a_bundle(gold: dict[str, Any]) -> None:
        gold["source"]["kind"] = "bundle"

    run = start(gold=from_a_bundle)

    assert run.code == 1
    assert "A run measures the test bench only" in run.said
    assert run.prompts == []


# What a run writes


def test_a_finished_run_writes_an_eval_snapshot_the_contract_accepts(finished: Run) -> None:
    assert finished.code == 0
    snapshot = EvalSnapshot.model_validate(finished.snapshot)

    assert snapshot.label == "Measured on a test bench: a real building handover, not this campus"
    assert snapshot.run.venue == "mini-bench"
    assert snapshot.run.snapshot == "2026-09-15"
    assert snapshot.run.deployments.model_dump() == {
        "planning": PLANNER,
        "composition": COMPOSER,
        "embedding": prose.EMBEDDING_DEPLOYMENT,
        "nli": "stand-in-nli",
    }
    assert snapshot.run.id.startswith(snapshot.run.date)
    assert snapshot.figures.template.n == 166
    assert snapshot.figures.template.correct.n == 166
    assert snapshot.figures.template.correct.ci95 is not None


def test_the_service_serves_what_the_run_wrote(finished: Run) -> None:
    app = create_app(
        fixture.store(), fixture.llm(None, None, []), evaluation=finished.out / "eval-snapshot.json"
    )

    response = TestClient(app).get("/eval")

    assert response.status_code == 200
    assert response.json() == finished.snapshot


def test_leak_checks_are_passed_on_as_pass_or_fail_with_their_date(finished: Run) -> None:
    assert finished.snapshot["leak_checks"] == {
        "input": {"passed": True, "date": "2026-10-09"},
        "counterfactual": {"passed": False, "date": "2026-10-09"},
    }
    assert "counterfactual leak check: FAIL on 2026-10-09" in finished.said


def test_routes_proximity_and_timing_leaks_are_listed_as_unmeasured(finished: Run) -> None:
    assert finished.snapshot["unmeasured"] == ["routes", "proximity", "timing-leaks", "follow-ups"]
    assert "Unmeasured: routes, proximity, timing-leaks, follow-ups" in finished.said


# Scoring


def test_with_a_scripted_model_and_a_willing_nli_every_stage_holds(finished: Run) -> None:
    template = finished.template

    assert template["stages"]["tools"]["value"] == 1.0
    # 13 questions are after a document that is there to find.
    assert template["stages"]["retrieval"]["recall_at_5"] == {
        "value": 1.0,
        "n": 13,
        "ci95": [1.0, 1.0],
    }
    assert template["stages"]["retrieval"]["mrr"]["n"] == 13
    # 96 answers were composed, 13 of them with a segment citing a passage.
    assert template["stages"]["composition"]["holds"]["n"] == 96
    assert template["stages"]["composition"]["numbers"]["value"] == 1.0
    assert template["stages"]["composition"]["entailment"]["value"] == 1.0
    assert template["correct"]["value"] == 1.0


def test_the_stages_are_reported_one_by_one(finished: Run) -> None:
    for line in (
        "correct, all three stages:",
        "tools, exact match:",
        "retrieval Recall@5:",
        "retrieval MRR:",
        "composition holds:",
        "every number a fact:",
        "segments entailed:",
    ):
        assert line in finished.said


def test_paired_questions_are_scored_per_persona_each_against_its_own_gold(finished: Run) -> None:
    by_persona = finished.template["by_persona"]

    assert {name: figure["n"] for name, figure in by_persona.items()} == {
        "maintenance": 83,
        "visitor": 83,
    }


def test_a_segment_the_nli_model_does_not_back_fails_composition_and_nothing_else(
    start: Start,
) -> None:
    nli = Says(score=0.49)

    template = start(nli=nli).template

    # Scored against the passage it cites: the passage is the premise.
    premise, hypothesis = nli.pairs[0]
    assert hypothesis == "The cited passage covers this."
    assert "maintenance" in premise or len(premise) > len(hypothesis)
    assert template["stages"]["composition"]["entailment"]["value"] == 0.0
    assert template["stages"]["tools"]["value"] == 1.0
    assert template["stages"]["retrieval"]["recall_at_5"]["value"] == 1.0
    # The 13 document answers are no longer correct. The other 153 still are.
    assert template["correct"]["value"] == round(153 / 166, 4)


def test_an_answer_with_no_backed_segment_counts_as_the_abstention_the_site_would_send(
    start: Start,
) -> None:
    matrix = start(nli=Says(score=0.1)).template["turn_kinds"]["matrix"]

    assert matrix["answer"]["abstention"] == 13
    assert matrix["answer"]["answer"] == 83


def test_a_score_of_exactly_one_half_keeps_the_segment(start: Start) -> None:
    template = start(nli=Says(score=0.5)).template

    assert template["stages"]["composition"]["entailment"]["value"] == 1.0


def test_results_are_a_confusion_matrix_over_expected_turn_kinds(finished: Run) -> None:
    kinds = finished.template["turn_kinds"]

    assert kinds["matrix"] == {
        "answer": {
            "answer": 96,
            "clarification": 0,
            "abstention": 0,
            "coverage-gap": 0,
            "unavailable": 0,
        },
        "clarification": {
            "answer": 0,
            "clarification": 11,
            "abstention": 0,
            "coverage-gap": 0,
            "unavailable": 0,
        },
        "abstention": {
            "answer": 0,
            "clarification": 0,
            "abstention": 53,
            "coverage-gap": 0,
            "unavailable": 0,
        },
        "coverage-gap": {
            "answer": 0,
            "clarification": 0,
            "abstention": 0,
            "coverage-gap": 6,
            "unavailable": 0,
        },
    }
    assert kinds["answered_when_it_should_have_declined"] == {"count": 0, "of": 59}


def test_answering_what_should_have_been_declined_is_reported_on_its_own(start: Start) -> None:
    def expect_declines(gold: dict[str, Any]) -> None:
        # Gold now says two questions the layer answers should have been declined.
        answers = [q for q in gold["questions"] if q["expected"]["kind"] == "answer"]
        answers[0]["expected"] = {"kind": "abstention"}
        answers[1]["expected"] = {"kind": "coverage-gap"}

    run = start(gold=expect_declines)
    kinds = run.template["turn_kinds"]

    assert kinds["answered_when_it_should_have_declined"] == {"count": 2, "of": 61}
    assert kinds["matrix"]["abstention"]["answer"] == 1
    assert kinds["matrix"]["coverage-gap"]["answer"] == 1
    assert "answered when it should have declined: 2 of 61" in run.said
    assert run.template["correct"]["value"] == round(164 / 166, 4)


def test_clarification_candidates_are_held_to_golds_cut_to_the_personas_clearance(
    start: Start, finished: Run
) -> None:
    def one_more(gold: dict[str, Any]) -> None:
        # As if gold had not cut its candidates to the persona: one more than the layer offers.
        clarifying = [q for q in gold["questions"] if q["expected"]["kind"] == "clarification"]
        clarifying[0]["expected"]["candidates"].append("space:not-for-this-persona")

    assert finished.template["clarification_candidates"]["value"] == 1.0
    assert finished.template["clarification_candidates"]["n"] == 11

    candidates = start(gold=one_more).template["clarification_candidates"]

    assert (candidates["n"], candidates["value"]) == (11, round(10 / 11, 4))


# Hand-written questions


def test_with_no_hand_written_questions_their_figures_are_over_nothing(finished: Run) -> None:
    hand_written = finished.snapshot["figures"]["hand_written"]

    assert hand_written["n"] == 0
    assert hand_written["correct"] == {"value": None, "n": 0, "ci95": None}
    assert hand_written["turn_kinds"]["matrix"] == {}
    assert "Hand-written questions: 0" in finished.said


def test_hand_written_questions_are_reported_apart_from_template_questions(
    start: Start, bench_gold: QuestionSet
) -> None:
    answered = next(q for q in bench_gold.questions if q.expected.kind.value == "answer")
    written = [
        {
            "id": "right",
            "persona": answered.persona.value,
            "shape": answered.shape,
            "question": answered.question,
            "expected": answered.expected.model_dump(mode="json", exclude_none=True),
        },
        {
            "id": "wrong",
            "persona": answered.persona.value,
            "shape": answered.shape,
            "question": answered.question,
            "expected": {"kind": "abstention"},
        },
    ]

    run = start(hand_written=written)
    figures = run.snapshot["figures"]

    assert figures["template"]["n"] == 166
    assert figures["template"]["correct"]["value"] == 1.0
    assert figures["hand_written"]["n"] == 2
    assert figures["hand_written"]["correct"]["value"] == 0.5
    assert "166 template questions, 2 hand-written" in run.said
    assert [q["id"] for q in run.report["questions"] if q["set"] == "hand-written"] == [
        "hand-written/right",
        "hand-written/wrong",
    ]


# Spend


def test_the_run_reports_its_spend_as_evaluation_spend_by_deployment(finished: Run) -> None:
    spend = finished.report["spend"]
    entries = {(e["deployment"], e["purpose"]): e for e in spend["entries"]}
    planning = entries[(PLANNER, "planning")]
    composition = entries[(COMPOSER, "composition")]

    assert spend["category"] == "evaluation"
    assert spend["currency"] == "USD"
    assert spend["rates_as_of"] == "2026-10-01"
    # One planning call a turn. The scripted planner never needs a second.
    assert planning["calls"] + entries.get((PLANNER, "recovery"), {"calls": 0})["calls"] >= 166
    assert planning["input_tokens"] == planning["calls"] * 1000
    assert planning["cost"] == pytest.approx(planning["calls"] * PLANNING_CALL)
    assert composition["cost"] == pytest.approx(composition["calls"] * COMPOSITION_CALL)
    assert spend["amount"] == pytest.approx(sum(e["cost"] for e in spend["entries"]))
    assert 'Spend, category "evaluation" (not visitor spend)' in finished.said


def test_the_eval_snapshot_says_nothing_about_money(finished: Run) -> None:
    assert "spend" not in json.dumps(finished.snapshot)
    assert "ceiling" not in json.dumps(finished.snapshot)


# Metrics


def test_the_confusion_matrix_is_pushed_labelled_with_run_snapshot_and_deployments(
    start: Start,
) -> None:
    reader = InMemoryMetricReader()

    run = start(metrics=EvalMetrics(MeterProvider(metric_readers=[reader])))

    data = reader.get_metrics_data()
    points = [
        (dict(point.attributes or {}), point.value)
        for resource in (data.resource_metrics if data else [])
        for scope in resource.scope_metrics
        for metric in scope.metrics
        if metric.name == "spatial.eval.turns"
        for point in metric.data.data_points
    ]
    common = {
        "run_id": run.snapshot["run"]["id"],
        "snapshot": "2026-09-15",
        "planning_deployment": PLANNER,
        "composition_deployment": COMPOSER,
        "question_set": "template",
    }
    cells = {(p["expected_turn_kind"], p["turn_kind"]): count for p, count in points}
    assert cells == {
        ("answer", "answer"): 96,
        ("clarification", "clarification"): 11,
        ("abstention", "abstention"): 53,
        ("coverage-gap", "coverage-gap"): 6,
    }
    for labels, _ in points:
        assert {name: labels[name] for name in common} == common
        assert len(labels) == len(common) + 2


# Adoption rules


def test_the_adoption_rules_are_printed_and_say_when_there_was_nothing_to_compare(
    finished: Run,
) -> None:
    assert "Adoption rules, worked out and not applied:" in finished.said
    assert "semantic ranker: not compared in this run (--compare-ranker)" in finished.said
    assert "model pair: no baseline run to compare with (--baseline)" in finished.said
    assert [rule["applied"] for rule in finished.report["adoption_rules"]] == [False, False]


def test_the_ranker_rule_is_worked_out_from_a_second_retrieval_of_each_question(
    start: Start,
) -> None:
    run = start(compare_ranker=True)
    (ranker, _) = run.report["adoption_rules"]

    # The stand-in ranker returns what plain retrieval does, so there is no gain to adopt.
    assert ranker["difference"] == {"value": 0.0, "n": 13, "ci95": [0.0, 0.0]}
    assert ranker["met"] is False
    assert "semantic ranker: no gain the interval can show: keep off" in run.said
    assert run.snapshot["run"]["semantic_ranker"] is False


def test_the_model_pair_rule_compares_this_run_with_a_baseline_run_question_by_question(
    start: Start, finished: Run, tmp_path: Path
) -> None:
    baseline = finished.report
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))

    # Same answers as the baseline, at a tenth of the tokens: cheaper and no worse.
    cheaper = start("--baseline", str(tmp_path / "baseline.json"), usage=(100, 10))
    (_, pair) = cheaper.report["adoption_rules"]

    assert pair["difference"] == {"value": 0.0, "n": 166, "ci95": [0.0, 0.0]}
    assert pair["met"] is True
    assert "it may replace the baseline" in cheaper.said

    # Same answers again, for more money: the baseline stays, whatever the difference.
    dearer = start("--baseline", str(tmp_path / "baseline.json"), usage=(5000, 500))
    assert dearer.report["adoption_rules"][1]["met"] is False
    assert "not the cheaper one" in dearer.said


def test_a_baseline_from_another_question_set_is_refused_before_anything_is_spent(
    start: Start, finished: Run, tmp_path: Path
) -> None:
    baseline = finished.report
    baseline["run"]["gold_sha256"] = "0" * 64
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))

    run = start("--baseline", str(tmp_path / "baseline.json"))

    assert run.code == 1
    assert "another question set or another snapshot" in run.said
    assert run.prompts == []
