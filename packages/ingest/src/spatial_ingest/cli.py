"""The ingestion CLI: build a snapshot bundle, load it into the stores, promote it.

Load and promote write to the layer's database, named by --database-url or the
SPATIAL_DATABASE_URL environment variable. Ingestion holds those credentials; the
service never does.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import psycopg

from spatial_ingest.build import MANIFEST_FILE, BuildError, build
from spatial_ingest.load import LoadError, load
from spatial_ingest.promote import PromoteError, promote
from spatial_ingest.pull import PullError
from spatial_ingest.structure import StructureError
from spatial_ingest.venue import load_venue

DATABASE_URL_ENV = "SPATIAL_DATABASE_URL"
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


def _build(args: argparse.Namespace) -> int:
    try:
        bundle = build(
            venue=load_venue(args.venue),
            date=args.date,
            out=args.out,
            extract=args.extract,
            refetch=args.refetch,
        )
    except (BuildError, PullError, StructureError, ValueError, KeyError, OSError) as error:
        print(f"ingest build failed: {error}", file=sys.stderr)
        return 1
    manifest = json.loads((bundle / MANIFEST_FILE).read_text(encoding="utf-8"))
    print(f"built {bundle}")
    print(json.dumps(manifest["counts"], indent=2))
    return 0


def _load(args: argparse.Namespace) -> int:
    result = load(bundle=args.out / args.date, database_url=args.database_url)
    if not result.ok:
        print(f"ingest load: {result.snapshot} failed its checks and cannot be promoted")
        for failure in result.failures:
            print(f"  {failure}")
        return 1
    print(f"loaded {result.snapshot}. It is not current until promoted.")
    print(json.dumps(result.rows, indent=2))
    if result.dropped:
        print(f"dropped {', '.join(result.dropped)}")
    return 0


def _promote(args: argparse.Namespace) -> int:
    pointer = promote(snapshot=args.date, database_url=args.database_url)
    print(f"current snapshot {pointer.current}, previous {pointer.previous}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        return _build(args)
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
