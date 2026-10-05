"""The ingestion CLI. One command so far: build. Load and promote come later."""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from spatial_ingest.build import MANIFEST_FILE, BuildError, build
from spatial_ingest.pull import PullError
from spatial_ingest.structure import StructureError
from spatial_ingest.venue import load_venue


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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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


if __name__ == "__main__":
    sys.exit(main())
