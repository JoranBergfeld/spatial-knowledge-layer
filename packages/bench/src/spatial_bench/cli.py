"""bench: gold for a venue, and the checks that need no model.

    bench gold cobie --workbook handover.xlsx --ocr ocr/ --out gold.json
    bench gold bundle --bundle bundles/2026-09-15 --out gold.json
    bench check --bundle bundles/2026-09-15 --gold gold.json

`gold` runs the gold scripts in an interpreter of their own. `check` loads the
bundle into throwaway databases and runs tool correctness and both leak checks
with a scripted model. It exits 1 if anything fails, a leak check above all.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import psycopg

from spatial_bench import questions, suite
from spatial_bench.questions import GoldError
from spatial_bench.run import BenchError

# A Postgres this may create databases on. Never the layer's own login.
ADMIN_URL_ENV = "SPATIAL_BENCH_ADMIN_URL"


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
        print(
            f"  {name} leak check: {'pass' if leaks[name]['passed'] else 'FAIL'}"
            f" over {leaks[name]['turns']} turns"
        )
        for one in found[:10]:
            print(f"    {json.dumps(one)}")
    return 0 if report.passed else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _gold(args) if args.command == "gold" else _check(args)
    except (GoldError, BenchError, psycopg.Error, OSError) as error:
        print(f"bench {args.command} failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
