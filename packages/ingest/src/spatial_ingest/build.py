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

from spatial_core import clearance
from spatial_core.records import PERSONAS, Space, VisibleSpace
from spatial_ingest import pull
from spatial_ingest.rules import tiers
from spatial_ingest.structure import EDGE_RULES, RULES, TREE_RULES, build_structure
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


def _clearance_counts(
    spaces: Iterable[Space], visible: Iterable[VisibleSpace], unrecognised: int
) -> dict[str, Any]:
    """The share of rooms whose clearance is sourced, and what each persona sees.

    Rooms are `indoor=room` spaces, so corridors and areas do not count.
    """
    rooms = [
        space
        for space in spaces
        if space.tier.value == tiers.ROOM
        and space.attributes.get("indoor") is not None
        and space.attributes["indoor"].value == "room"
    ]
    sourced = sum(1 for room in rooms if room.clearance and not room.clearance.authored_default)
    seen = [entry.persona for entry in visible]
    return {
        "rooms": len(rooms),
        "rooms_sourced": sourced,
        "sourced_share": round(sourced / len(rooms), 4) if rooms else 0.0,
        "unrecognised_access": unrecognised,
        "visible_spaces": {persona.value: seen.count(persona) for persona in PERSONAS},
    }


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
    sets = clearance.visible_sets(structure.spaces, structure.edges)
    visible = [
        VisibleSpace(persona=persona, space_id=space_id)
        for persona in PERSONAS
        for space_id in sorted(sets[persona])
    ]
    # Each element once: a door makes an edge each way.
    door_clearances = {edge.door: edge.door_clearance for edge in structure.edges}
    unrecognised = sum(
        1
        for cleared in [
            *(space.clearance for space in structure.spaces),
            *door_clearances.values(),
        ]
        if cleared is not None
        and cleared.access is not None
        and cleared.access not in clearance.ACCESS
    )
    visible_rules = tuple(dict.fromkeys((*RULES, *EDGE_RULES, clearance.VISIBLE_RULE)))

    tool = f"spatial-ingest@{version('spatial-ingest')}"
    parts: dict[str, Any] = {}
    for name, records, rules in (
        ("osm/spaces.jsonl", structure.spaces, RULES),
        ("osm/spans.jsonl", structure.spans, TREE_RULES),
        ("osm/edges.jsonl", structure.edges, EDGE_RULES),
        ("osm/visible_spaces.jsonl", visible, visible_rules),
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
        "clearance": _clearance_counts(structure.spaces, visible, unrecognised),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle
