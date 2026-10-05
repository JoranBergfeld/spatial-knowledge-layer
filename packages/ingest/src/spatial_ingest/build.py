"""ingest build: pinned inputs in, snapshot bundle out.

The bundle for a snapshot date:

    <out>/<date>/
      pull/            the pinned pull, private
      osm/             everything sourced from OSM or derived from it by a rule
      manifest.json    input hashes and what produced each part

Nothing in the bundle depends on when the build ran, so building twice from the
same stored pull gives the same bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spatial_ingest import pull
from spatial_ingest.structure import RULES, TREE_RULES, build_structure
from spatial_ingest.venue import Venue

MANIFEST_FILE = "manifest.json"


class BuildError(Exception):
    pass


def _jsonl(records: Iterable[BaseModel]) -> bytes:
    lines = [
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for record in records
    ]
    return "".join(line + "\n" for line in lines).encode()


def _venue_input(venue: Venue) -> dict[str, Any]:
    definition = asdict(venue)
    canonical = json.dumps(definition, ensure_ascii=False, sort_keys=True).encode()
    return {"definition": definition, "sha256": hashlib.sha256(canonical).hexdigest()}


def build(
    *,
    venue: Venue,
    date: str,
    out: Path,
    extract: Path | None = None,
    refetch: bool = False,
) -> Path:
    bundle = out / date
    pull_dir = bundle / "pull"
    manifest_path = bundle / MANIFEST_FILE

    if refetch or not pull.has_pull(pull_dir):
        if extract is not None:
            pull.cut_extract(venue, date, extract, pull_dir)
        else:
            pull.fetch_overpass(venue, date, pull_dir)
    elif extract is not None:
        raise BuildError(
            f"{pull_dir} already holds a pull. Pass --refetch to replace it with the extract."
        )

    stored = pull.load_pull(pull_dir, venue, date)
    structure = build_structure(stored.data, venue)

    tool = f"spatial-ingest@{version('spatial-ingest')}"
    parts: dict[str, Any] = {}
    for name, records, rules in (
        ("osm/spaces.jsonl", structure.spaces, RULES),
        ("osm/spans.jsonl", structure.spans, TREE_RULES),
    ):
        content = _jsonl(records)
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        parts[name] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "records": len(records),
            "produced_by": [tool, *rules],
        }

    manifest = {
        "snapshot": date,
        "venue": venue.id,
        "inputs": {"pull": stored.pin, "venue": _venue_input(venue)},
        "parts": parts,
        "counts": structure.counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle
