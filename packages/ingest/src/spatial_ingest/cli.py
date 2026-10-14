"""The ingestion CLI: build a snapshot bundle, load it into the stores, promote it.

Load and promote write to the layer's database, named by --database-url or the
SPATIAL_DATABASE_URL environment variable. Ingestion holds those credentials; the
service never does. Load also writes the search index, which it finds through
AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_API_KEY. Build translates and embeds, and
finds its deployments through the AZURE_OPENAI_* variables the service uses.
"""

from __future__ import annotations

import argparse
import datetime
import importlib
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import psycopg

from spatial_ingest.build import MANIFEST_FILE, BuildError, build, build_handover
from spatial_ingest.corpus import CorpusError
from spatial_ingest.documents import DocumentsError
from spatial_ingest.handover.cobie import CobieError
from spatial_ingest.handover.definition import Definition, DefinitionError, load_definition
from spatial_ingest.handover.fetch import fetch_handover
from spatial_ingest.handover.ifc import IfcError
from spatial_ingest.handover.records import HandoverError
from spatial_ingest.load import LoadError, load
from spatial_ingest.models import ModelError
from spatial_ingest.promote import PromoteError, promote
from spatial_ingest.pull import PullError
from spatial_ingest.seed.generate import InventoryError
from spatial_ingest.structure import StructureError
from spatial_ingest.summaries import SummaryError
from spatial_ingest.venue import load_venue

DATABASE_URL_ENV = "SPATIAL_DATABASE_URL"
# The built-in handover venue: the test bench.
TEST_BENCH = "test-bench"
# Each load and promote leaves its log here, inside the bundle it ran for.
RUNS_DIR = "runs"


def _date(text: str) -> str:
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a date like 2026-09-15") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ingest", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    build_cmd = commands.add_parser("build", help="build a snapshot bundle from pinned inputs")
    build_cmd.add_argument("--date", required=True, type=_date, help="the snapshot date")
    build_cmd.add_argument(
        "--venue", default="kit-campus-sued", help="a built-in venue, or a venue .json file"
    )
    build_cmd.add_argument("--out", type=Path, default=Path("bundles"), help="where bundles live")
    build_cmd.add_argument(
        "--extract",
        type=Path,
        help="a dated regional extract (.osm.pbf) to cut down, for when Overpass rate limits",
    )
    build_cmd.add_argument(
        "--refetch", action="store_true", help="replace the stored pull instead of reusing it"
    )
    build_cmd.add_argument(
        "--inventory",
        type=Path,
        help="the seed inventory's pinned inputs (default: the published inventory.json)",
    )

    build_cmd.add_argument(
        "--documents",
        type=Path,
        help="the document corpus (default: the venue's published corpus, if it has one)",
    )
    build_cmd.add_argument(
        "--refetch-documents",
        action="store_true",
        help="fetch every document source again, which checks its link again too",
    )
    build_cmd.add_argument(
        "--handover",
        type=Path,
        help="for a handover venue (the test bench): the directory its pinned files are in",
    )

    fetch_cmd = commands.add_parser(
        "fetch-handover", help="fetch a handover venue's pinned files and check their hashes"
    )
    fetch_cmd.add_argument(
        "--venue", default=TEST_BENCH, help="a handover venue, or a definition .json file"
    )
    fetch_cmd.add_argument("--to", type=Path, required=True, help="the directory to fetch into")

    for name, help_text in (
        ("load", "load a bundle into the stores and check it; it does not become current"),
        ("promote", "make a loaded snapshot current; promote the previous one to roll back"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--date", required=True, type=_date, help="the snapshot date")
        command.add_argument("--out", type=Path, default=Path("bundles"), help="where bundles live")
        command.add_argument(
            "--database-url",
            default=os.environ.get(DATABASE_URL_ENV),
            help=f"the layer's database, with write credentials (default: ${DATABASE_URL_ENV})",
        )
        if name == "load":
            command.add_argument(
                "--gold",
                type=Path,
                help="a question set to ask of the loaded snapshot. Reported, never blocking",
            )
    return parser


@contextmanager
def _run_log(bundle: Path, command: str) -> Iterator[Path | None]:
    """Log to stderr, and to a file of this run's own in the bundle when there is one."""
    logger = logging.getLogger("spatial_ingest")
    logger.setLevel(logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    path = None
    if bundle.is_dir():
        stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
        path = bundle / RUNS_DIR / f"{stamp}-{command}.log"
        path.parent.mkdir(exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    try:
        yield path
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()


_HANDOVER_ERRORS = (
    BuildError,
    CobieError,
    DefinitionError,
    DocumentsError,
    HandoverError,
    IfcError,
    ModelError,
    SummaryError,
    ValueError,
    KeyError,
    OSError,
)


def _build_handover(args: argparse.Namespace, definition: Definition) -> int:
    if args.handover is None:
        print(
            f"ingest build: {definition.id} is built from a handover. "
            "Pass --handover with the directory its files are in.",
            file=sys.stderr,
        )
        return 2
    try:
        bundle = build_handover(
            definition=definition,
            date=args.date,
            out=args.out,
            handover=args.handover,
            refetch_documents=args.refetch_documents,
        )
    except _HANDOVER_ERRORS as error:
        print(f"ingest build failed: {error}", file=sys.stderr)
        return 1
    manifest = json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))
    print(f"built {bundle}")
    print(json.dumps(manifest["counts"], indent=2))
    print(f"visible assets: {json.dumps(manifest['inventory']['visible_assets'])}")
    prose = manifest["documents"]
    print(
        f"documents: {prose['documents']} in {prose['chunks']} chunks, "
        f"{len(prose['skipped'])} skipped, ${prose['spend']['total_usd']} spent"
    )
    for skipped in prose["skipped"]:
        print(f"  skipped {skipped['id']}: {skipped['reason']}")
    return 0


def _fetch_handover(args: argparse.Namespace) -> int:
    try:
        definition = load_definition(args.venue)
        if definition is None:
            raise DefinitionError(f"{args.venue} is not a handover venue")
        done = fetch_handover(definition, args.to)
    except (DefinitionError, ValueError, OSError) as error:
        print(f"ingest fetch-handover failed: {error}", file=sys.stderr)
        return 1
    for name, what in done.items():
        print(f"{what}: {args.to / name}")
    return 0


def _build(args: argparse.Namespace) -> int:
    try:
        definition = load_definition(args.venue)
    except (DefinitionError, ValueError, OSError) as error:
        print(f"ingest build failed: {error}", file=sys.stderr)
        return 1
    if definition is not None:
        return _build_handover(args, definition)
    try:
        bundle = build(
            venue=load_venue(args.venue),
            date=args.date,
            out=args.out,
            extract=args.extract,
            refetch=args.refetch,
            inventory_file=args.inventory,
            corpus_file=args.documents,
            refetch_documents=args.refetch_documents,
        )
    except (
        BuildError,
        CorpusError,
        DocumentsError,
        ModelError,
        SummaryError,
        InventoryError,
        PullError,
        StructureError,
        ValueError,
        KeyError,
        OSError,
    ) as error:
        print(f"ingest build failed: {error}", file=sys.stderr)
        return 1
    manifest = json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))
    print(f"built {bundle}")
    print(json.dumps(manifest["counts"], indent=2))
    stock = manifest["inventory"]
    print(
        f"inventory: {stock['sourced']['assets']} sourced fire assets, "
        f"{stock['generated']['assets']} generated assets (seed {stock['seed']})"
    )
    prose = manifest["documents"]
    print(
        f"documents: {prose['documents']} in {prose['chunks']} chunks, "
        f"{prose['summaries']['documents']} summaries, "
        f"{len(prose['skipped'])} skipped, {prose['offline']} with their source offline, "
        f"${prose['spend']['total_usd']} spent"
    )
    for skipped in prose["skipped"]:
        print(f"  skipped {skipped['id']}: {skipped['reason']}")
    if prose["spend"]["unpriced"]:
        print(f"  no rate for {', '.join(prose['spend']['unpriced'])}, so its cost is not counted")
    return 0


def _load(args: argparse.Namespace) -> int:
    result = load(bundle=args.out / args.date, database_url=args.database_url)
    if not result.ok:
        print(f"ingest load: {result.snapshot} failed its checks and cannot be promoted")
        for failure in result.failures:
            print(f"  {failure}")
        return 1
    print(f"loaded {result.snapshot} with index {result.index}. It is not current until promoted.")
    print(json.dumps(result.rows, indent=2))
    if result.dropped:
        print(f"dropped {', '.join(result.dropped)}")
    if args.gold:
        _question_set(args.gold, args.database_url, result.snapshot)
    return 0


def _question_set(gold: Path, database_url: str, snapshot: str) -> None:
    """Ask the question set of the snapshot just loaded, with the bench's scripted model.

    It is a report. Whatever it finds, and even if it cannot run at all, load has
    already succeeded and the snapshot can be promoted. The bench is looked up by
    name because it depends on ingestion, not the other way round. The ingest
    image does not carry it. There, run `bench fresh` from a checkout.
    """
    try:
        fresh = importlib.import_module("spatial_bench.fresh")
    except ImportError:
        print("question set not run: the bench is not installed here. Use `bench fresh`.")
        return
    try:
        fresh.report(gold, database_url, snapshot)
    except Exception as error:  # noqa: BLE001
        print(f"question set not run: {error}")


def _promote(args: argparse.Namespace) -> int:
    pointer = promote(snapshot=args.date, database_url=args.database_url)
    print(f"current snapshot {pointer.current}, previous {pointer.previous}")
    print(f"current index {pointer.index}, previous {pointer.previous_index}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        return _build(args)
    if args.command == "fetch-handover":
        return _fetch_handover(args)
    if not args.database_url:
        print(
            f"ingest {args.command}: pass --database-url or set {DATABASE_URL_ENV}",
            file=sys.stderr,
        )
        return 2
    with _run_log(args.out / args.date, args.command):
        try:
            return _load(args) if args.command == "load" else _promote(args)
        except (LoadError, PromoteError, psycopg.Error, OSError) as error:
            logging.getLogger("spatial_ingest").error("ingest %s failed: %s", args.command, error)
            return 1


if __name__ == "__main__":
    sys.exit(main())
