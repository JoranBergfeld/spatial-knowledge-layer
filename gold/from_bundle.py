"""A snapshot bundle as a World, for the demo venue's internal regression.

The demo venue has no handover to compute gold from. Its bundle is the only
record of what it holds, so this reads the bundle's JSON lines as data: the
spaces, the assets, the documents and the three visible sets as build wrote
them. It does not work clearance out again, so on the demo venue this checks the
tools against the stores and not the clearance rules. Those have the test bench.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from world import PERSONAS, Asset, AssetType, Document, Space, World

HALVES = ("osm", "seed")


def _lines(bundle: Path, name: str) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for path in (bundle / half / name for half in HALVES):
        if path.is_file():
            found += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return found


def _value(field: dict[str, Any] | None) -> str | None:
    return field["value"] if field else None


def read(bundle: Path) -> World:
    spaces = {
        row["id"]: Space(
            id=row["id"],
            tier=row["tier"]["value"],
            parent=row["parent_id"],
            label=_value(row.get("label")),
            # A room with several numbers prints the first.
            ref=(_value(row.get("ref")) or "").split(";")[0].strip() or None,
            name=_value(row.get("name")),
            lowest=(row.get("clearance") or {}).get("persona"),
        )
        for row in _lines(bundle, "spaces.jsonl")
    }
    types = {
        row["id"]: AssetType(row["id"], row["name"]["value"], row["clearance"]["persona"])
        for row in _lines(bundle, "asset_types.jsonl")
    }
    serviced: dict[str, list[str]] = {}
    for row in _lines(bundle, "service_events.jsonl"):
        serviced.setdefault(row["asset_id"], []).append(row["start"]["value"])
    assets = {
        row["id"]: Asset(
            id=row["id"],
            type=row["type_id"],
            space=row["space_id"],
            attributes={key: value["value"] for key, value in row["attributes"].items()},
            serviced=tuple(serviced.get(row["id"], ())),
        )
        for row in _lines(bundle, "assets.jsonl")
    }
    documents = {}
    listed = bundle / "documents" / "documents.jsonl"
    if listed.is_file():
        for line in listed.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            documents[row["id"]] = Document(
                id=row["id"],
                title=row["title"],
                kind=row["kind"],
                anchors=tuple(sorted((a["kind"], a["id"]) for a in row["anchors"])),
                lowest=row["clearance_class"],
            )

    world = World(spaces, types, assets, documents)
    seen: dict[str, dict[str, set[str]]] = {
        persona: {"spaces": set(), "assets": set(), "types": set()} for persona in PERSONAS
    }
    for name, kind, key in (
        ("visible_spaces.jsonl", "spaces", "space_id"),
        ("visible_assets.jsonl", "assets", "asset_id"),
        ("visible_asset_types.jsonl", "types", "type_id"),
    ):
        for row in _lines(bundle, name):
            seen[row["persona"]][kind].add(row[key])
    world.visible = {
        persona: {kind: frozenset(ids) for kind, ids in kinds.items()}
        for persona, kinds in seen.items()
    }
    return world
