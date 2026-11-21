"""bench: gold for a venue, the checks that need no model, and evaluation runs.

    bench gold cobie --workbook handover.xlsx --ocr ocr/ --out gold.json
    bench gold bundle --bundle bundles/2026-09-15 --out gold.json
    bench check --bundle bundles/2026-09-15 --gold gold.json
    bench fresh --gold gold.json --snapshot 2026-09-15 --database-url ...
    bench run --gold gold.json --checks checks.json --rates rates.json --ceiling 20 --out runs/
    bench label runs/segments-<run id>.jsonl ...

`gold` runs the gold scripts in an interpreter of their own. `check` loads the
bundle into throwaway databases and runs tool correctness and both leak checks
with a scripted model. It exits 1 if anything fails, a leak check above all.

`fresh` asks the question set of a snapshot that is loaded and not promoted,
with the scripted model. It reports and exits 0 whatever it finds.

`run` is an evaluation run, and the only command here that spends money. It
prints what the run could cost, asks, and stops before the ceiling. See
evalrun.py. It reads the service's own environment (the database, the
deployments, the index) plus SPATIAL_NLI_MODEL_DIR.

`label` asks a person about each segment a run scored, then works out the NLI
model's precision on what they said. See label.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

import psycopg

from spatial_bench import evalmetrics, evalrun, fresh, label, questions, suite
from spatial_bench.evalrun import Outcome, Prepared, Request, RunError, Wiring
from spatial_bench.label import LabelError
from spatial_bench.nli import MODEL_DIR_ENV, NliError, OnnxNli
from spatial_bench.questions import GoldError
from spatial_bench.run import BenchError
from spatial_bench.spend import CATEGORY, Assumed, Rates, SpendError
from spatial_core.search import LocalIndex
from spatial_service.evaluation import Figure, Section
from spatial_service.store import StoreError

# A Postgres this may create databases on. Never the layer's own login.
ADMIN_URL_ENV = "SPATIAL_BENCH_ADMIN_URL"
DATABASE_URL_ENV = "SPATIAL_SERVICE_DATABASE_URL"
# The run stopped before the last question. No eval snapshot was written.
STOPPED = 3


def _tokens(text: str) -> tuple[int, int]:
    try:
        tokens_in, tokens_out = text.split(":")
        return int(tokens_in), int(tokens_out)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not IN:OUT, as in 8000:600") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gold = commands.add_parser("gold", help="generate a question set with its gold")
    sources = gold.add_subparsers(dest="source", required=True)
    cobie = sources.add_parser("cobie", help="the test bench: gold from its COBie workbook")
    cobie.add_argument("--workbook", type=Path, required=True)
    cobie.add_argument("--ocr", type=Path, help="the OCR'd text of its documents")
    bundle = sources.add_parser("bundle", help="the demo venue: gold read off its bundle")
    bundle.add_argument("--bundle", type=Path, required=True)
    for source in (cobie, bundle):
        source.add_argument("--out", type=Path, required=True)
        source.add_argument("--per-template", type=int, default=6)

    check = commands.add_parser("check", help="tool correctness and both leak checks, no model")
    check.add_argument("--bundle", type=Path, required=True, help="the bundle to check")
    check.add_argument("--gold", type=Path, required=True, help="its question set")
    check.add_argument(
        "--admin-url",
        default=os.environ.get(ADMIN_URL_ENV),
        help=f"a Postgres login that may create databases (default: ${ADMIN_URL_ENV})",
    )
    check.add_argument("--report", type=Path, help="write the result here as JSON")
    check.add_argument(
        "--no-counterfactual",
        action="store_true",
        help="skip the counterfactual leak check, which loads the bundle twice more",
    )

    new = commands.add_parser("fresh", help="the question set on a loaded snapshot, no model")
    new.add_argument("--gold", type=Path, required=True)
    new.add_argument("--snapshot", required=True, help="the snapshot load just wrote")
    new.add_argument(
        "--database-url",
        default=os.environ.get(DATABASE_URL_ENV),
        help=f"a reader login on the database it was loaded into (default: ${DATABASE_URL_ENV})",
    )
    new.add_argument("--search-dir", type=Path, help="the stand-in index, if that is where it went")

    run = commands.add_parser("run", help="an evaluation run. Spends money, asks first")
    run.add_argument("--gold", type=Path, required=True, help="the test bench's question set")
    run.add_argument(
        "--hand-written",
        type=Path,
        default=questions.HAND_WRITTEN_FILE,
        help="the hand-written questions (default: %(default)s). No file is none",
    )
    run.add_argument("--checks", type=Path, required=True, help="bench check's report")
    run.add_argument("--rates", type=Path, required=True, help="what each deployment costs")
    run.add_argument("--ceiling", type=float, required=True, help="the most the run may spend")
    run.add_argument("--out", type=Path, required=True, help="a directory for what it writes")
    run.add_argument("--baseline", type=Path, help="an earlier run's report, for the pair rule")
    run.add_argument(
        "--compare-ranker",
        action="store_true",
        help="retrieve a second time with the semantic ranker on, for its adoption rule",
    )
    assumed = Assumed()
    run.add_argument(
        "--planning-tokens",
        type=_tokens,
        default=assumed.planning,
        metavar="IN:OUT",
        help="tokens one planning call is assumed to take, for the estimate",
    )
    run.add_argument(
        "--composition-tokens", type=_tokens, default=assumed.composition, metavar="IN:OUT"
    )

    labelling = commands.add_parser("label", help="label a run's segments, then NLI precision")
    labelling.add_argument(
        "segments", type=Path, nargs="+", help="segments-<run id>.jsonl, as a run wrote it"
    )
    labelling.add_argument(
        "--precision-only", action="store_true", help="ask nothing, just work out the figure"
    )
    return parser


def _gold(args: argparse.Namespace) -> int:
    if args.source == "cobie":
        made = questions.from_cobie(args.workbook, args.out, args.ocr, args.per_template)
    else:
        made = questions.from_bundle(args.bundle, args.out, args.per_template)
    print(f"{len(made.questions)} questions with gold in {args.out}")
    return 0


def _check(args: argparse.Namespace) -> int:
    if not args.admin_url:
        print(f"bench check: pass --admin-url or set {ADMIN_URL_ENV}", file=sys.stderr)
        return 2
    gold = questions.load(args.gold)
    # The framework warns on every turn that a scripted client calls no functions.
    logging.getLogger("agent_framework").setLevel(logging.ERROR)
    with tempfile.TemporaryDirectory(prefix="spatial-bench-") as work:
        report = suite.run(
            args.bundle,
            gold,
            args.admin_url,
            Path(work),
            counterfactual=not args.no_counterfactual,
        )
    said = report.as_json()
    if args.report:
        args.report.write_text(json.dumps(said, indent=1) + "\n", encoding="utf-8")
    tools, leaks = said["tool_correctness"], said["leak_checks"]
    print(f"{report.venue} {report.snapshot}: {said['questions']} questions, scripted model")
    print(
        f"  tool correctness: {'pass' if tools['passed'] else 'FAIL'}"
        f" ({len(tools['mismatched'])} mismatched)"
    )
    for wrong in tools["mismatched"][:10]:
        print(f"    {wrong['question']}: {'; '.join(wrong['mismatches'])}")
    for name in ("input", "counterfactual"):
        found = leaks[name].get("leaks", leaks[name].get("differences"))
        probes = leaks[name].get("route_probes")
        print(
            f"  {name} leak check: {'pass' if leaks[name]['passed'] else 'FAIL'}"
            f" over {leaks[name]['turns']} turns"
            + (f" and {probes} route probes" if probes else "")
        )
        for one in found[:10]:
            print(f"    {json.dumps(one)}")
    return 0 if report.passed else 1


def _fresh(args: argparse.Namespace) -> int:
    if not args.database_url:
        print(f"bench fresh: pass --database-url or set {DATABASE_URL_ENV}", file=sys.stderr)
        return 2
    index = LocalIndex(args.search_dir) if args.search_dir else None
    fresh.report(args.gold, args.database_url, args.snapshot, index)
    # Whatever it found. It is a report.
    return 0


def wiring_from_env(compare_ranker: bool) -> Wiring:
    """The real models, index and NLI model, as the environment names them."""
    from spatial_service import search
    from spatial_service.app import ConfigError, Settings, azure_llm
    from spatial_service.retrieval import Retriever
    from spatial_service.store import PostgresStore

    try:
        settings = Settings.from_env()
        index = search.from_env(os.environ, settings.database_url)
        embedder = search.embedder_from_env(
            os.environ,
            endpoint=settings.endpoint,
            api_key=settings.api_key,
            api_version=settings.api_version,
            timeout_seconds=settings.timeout_seconds,
        )
    except (ConfigError, ValueError) as error:
        raise RunError(str(error)) from error
    if not os.environ.get(MODEL_DIR_ENV):
        raise RunError(f"set {MODEL_DIR_ENV} to the directory with the site's NLI model")
    if compare_ranker and (not isinstance(index, search.AzureSearch) or settings.semantic_ranker):
        raise RunError(
            "--compare-ranker needs an Azure index, and the ranker off for the run itself"
        )
    documents = Retriever(index, embedder, semantic_ranker=settings.semantic_ranker)
    reranked = Retriever(index, embedder, semantic_ranker=True) if compare_ranker else None
    return Wiring(
        store=PostgresStore(settings.database_url),
        llm=azure_llm(settings),
        nli=OnnxNli(Path(os.environ[MODEL_DIR_ENV])),
        documents=documents,
        reranked=reranked,
        semantic_ranker=settings.semantic_ranker,
        metrics=evalmetrics.from_env(),
    )


def _figure(figure: Figure, percent: bool = True) -> str:
    if figure.value is None or figure.ci95 is None:
        return "no questions"
    low, high = figure.ci95
    if percent:
        return f"{figure.value:.1%} (n={figure.n}, 95% CI {low:.1%} to {high:.1%})"
    return f"{figure.value:.3f} (n={figure.n}, 95% CI {low:.3f} to {high:.3f})"


def _say_section(name: str, found: Section) -> None:
    print(f"{name} questions: {found.n}")
    if not found.n:
        print("  none asked, so no figures")
        return
    stages, kinds = found.stages, found.turn_kinds
    print(f"  correct, all three stages:  {_figure(found.correct)}")
    for persona, figure in found.by_persona.items():
        print(f"    as {persona}: {_figure(figure)}")
    print(f"  tools, exact match:         {_figure(stages.tools)}")
    print(f"  retrieval Recall@5:         {_figure(stages.retrieval.recall_at_5, percent=False)}")
    print(f"  retrieval MRR:              {_figure(stages.retrieval.mrr, percent=False)}")
    print(f"  composition holds:          {_figure(stages.composition.holds)}")
    print(f"    every number a fact:      {_figure(stages.composition.numbers)}")
    print(f"    segments entailed:        {_figure(stages.composition.entailment)}")
    print(f"  clarification candidates:   {_figure(found.clarification_candidates)}")
    declined = kinds.answered_when_it_should_have_declined
    print(f"  answered when it should have declined: {declined.count} of {declined.of}")
    print("    counted on what the layer produced, before the NLI drop")
    print("  turn kinds, expected down and produced across:")
    print("    produced is what the site would send, after the NLI drop")
    columns = list(next(iter(kinds.matrix.values())))
    print("    " + " " * 14 + "".join(f"{column:>15}" for column in columns))
    for expected, row in kinds.matrix.items():
        print(f"    {expected:<14}" + "".join(f"{row[column]:>15}" for column in columns))


def _say_plan(request: Request, prepared: Prepared) -> None:
    run, cost, currency = prepared.run, prepared.estimate, request.rates.currency
    assumed = request.assumed
    print(f"Evaluation run {run.id}")
    print(f"  venue {run.venue}, snapshot {run.snapshot}")
    print(f"  planning {run.deployments.planning}, composition {run.deployments.composition}")
    print(f"  embedding {run.deployments.embedding or 'none'}, NLI {run.deployments.nli}")
    print(
        f"  {len(request.gold.questions)} template questions,"
        f" {len(request.hand_written)} hand-written"
    )
    print(
        f"  estimate: at most {cost.calls} model calls, {cost.amount:.2f} {currency}"
        f" at the rates of {request.rates.as_of}"
    )
    print(
        "    worst case, on assumed call sizes: planning"
        f" {assumed.planning[0]}:{assumed.planning[1]} tokens twice a turn,"
        f" composition {assumed.composition[0]}:{assumed.composition[1]} once"
    )
    print(f"  ceiling: {request.ceiling:.2f} {currency}. The run stops before it.")
    print(f'  spend is reported under "{CATEGORY}", apart from visitor spend')


def _say_outcome(request: Request, prepared: Prepared, outcome: Outcome) -> None:
    currency = request.rates.currency
    spent = outcome.ledger.as_json()
    print(f'Spend, category "{CATEGORY}" (not visitor spend): {spent["amount"]:.4f} {currency}')
    print(f"  estimate was {prepared.estimate.amount:.2f}, ceiling {request.ceiling:.2f}")
    for entry in spent["entries"]:
        print(
            f"  {entry['deployment']} {entry['purpose']}: {entry['calls']} calls,"
            f" {entry['input_tokens']} in, {entry['output_tokens']} out, {entry['cost']:.4f}"
        )
    if spent["unpriced"]:
        print(f"  no rate for {', '.join(spent['unpriced'])}, so its cost is NOT in that amount")
    if spent["unmetered_calls"]:
        print(f"  {spent['unmetered_calls']} calls reported no usage and were costed as assumed")


def _label(args: argparse.Namespace, confirm: Callable[[str], str]) -> int:
    finished = args.precision_only or label.label(args.segments, ask=confirm)
    found = label.precision(args.segments)
    written = label.write_precision(args.segments, found)
    print()
    for name, counted in [*found["runs"].items(), ("all of them", found["all"])]:
        print(
            f"{name}: {counted['labelled']} of {counted['segments']} labelled,"
            f" {counted['kept_unfaithful']} kept and unfaithful,"
            f" {counted['dropped_faithful']} dropped and faithful"
        )
        figure = Figure.model_validate(counted["precision"])
        print(f"  precision: {_figure(figure) if figure.n else 'nothing kept is labelled yet'}")
    if found["enough"] is False:
        print(
            f"Precision is under {label.ENOUGH:.0%}. That is a decision to make:"
            " fall back to an LLM judge and re-cost the turn."
        )
    if not finished:
        print("Stopped before the end. Run it again to carry on where you left off.")
    print(f"Written to {written}")
    return 0


def _run(args: argparse.Namespace, wiring: Wiring | None, confirm: Callable[[str], str]) -> int:
    request = Request(
        gold=questions.load(args.gold),
        gold_sha256=evalrun.sha256_of(args.gold),
        hand_written=questions.load_hand_written(args.hand_written),
        checks=evalrun.Checks.load(args.checks),
        rates=Rates.load(args.rates),
        ceiling=args.ceiling,
        assumed=Assumed(planning=args.planning_tokens, composition=args.composition_tokens),
        baseline=json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None,
    )
    wiring = wiring or wiring_from_env(args.compare_ranker)
    try:
        prepared = evalrun.prepare(request, wiring)
        _say_plan(request, prepared)
        if confirm("Start the run? Type yes to spend: ").strip().lower() != "yes":
            print("Not started. Nothing was spent.")
            return 1
        logging.getLogger("agent_framework").setLevel(logging.ERROR)
        outcome = evalrun.execute(request, wiring, prepared)

        args.out.mkdir(parents=True, exist_ok=True)
        report = args.out / f"run-{prepared.run.id}.json"
        said = evalrun.report_of(request, prepared, outcome)
        report.write_text(json.dumps(said, indent=1) + "\n", encoding="utf-8")
        segments = args.out / evalrun.segments_file(prepared.run)
        evalrun.write_segments(segments, outcome)
        _say_outcome(request, prepared, outcome)
        print(f"Scored segments, private, in {segments}")
        if not outcome.completed:
            print(
                f"Stopped after {len(outcome.scored)} of {len(request.questions)} questions:"
                f" {outcome.stopped}."
            )
            print(f"No eval snapshot was written. The report is in {report}")
            return STOPPED

        snapshot = evalrun.snapshot_of(prepared.run, outcome.scored, request.checks)
        print(snapshot.label)
        _say_section("Template", snapshot.figures.template)
        _say_section("Hand-written", snapshot.figures.hand_written)
        for name, check in snapshot.leak_checks:
            print(f"{name} leak check: {'pass' if check.passed else 'FAIL'} on {check.date}")
        print(f"Unmeasured: {', '.join(snapshot.unmeasured)}")
        print("Adoption rules, worked out and not applied:")
        for rule in said["adoption_rules"]:
            difference = Figure.model_validate(rule["difference"])
            print(f"  {rule['rule']}: {rule['said']}")
            if difference.value is not None:
                print(f"    paired difference {_figure(difference, percent=False)}")
        wiring.metrics.push(prepared.run, outcome.scored)
        written = args.out / evalrun.SNAPSHOT_FILE
        written.write_text(snapshot.model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"Eval snapshot in {written}, report in {report}")
        return 0
    finally:
        wiring.metrics.close()


def main(
    argv: Sequence[str] | None = None,
    *,
    wiring: Wiring | None = None,
    confirm: Callable[[str], str] = input,
) -> int:
    """``wiring`` and ``confirm`` are for tests: the models a run calls, and who says yes."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "gold":
            return _gold(args)
        if args.command == "check":
            return _check(args)
        if args.command == "fresh":
            return _fresh(args)
        if args.command == "label":
            return _label(args, confirm)
        return _run(args, wiring, confirm)
    except (
        GoldError,
        BenchError,
        RunError,
        SpendError,
        NliError,
        LabelError,
        psycopg.Error,
        StoreError,
        OSError,
        ValueError,
    ) as error:
        print(f"bench {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
