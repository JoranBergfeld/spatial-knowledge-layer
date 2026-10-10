"""ingest build: pinned inputs in, snapshot bundle out.

The bundle for a snapshot date:

    <out>/<date>/
      pull/            the pinned pull, private
      osm/             everything sourced from OSM or derived from it by a rule
      seed/            the generated seed inventory
      documents/       the corpus: originals as fetched (private), their English
                       chunks, and the translations and vectors behind them
      manifest.json    input hashes and what produced each part

The pull, the fetched originals, the translations and the vectors are kept in the
bundle the first time they are made, and a link check keeps the day it ran. So
building twice from the same bundle gives the same bytes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spatial_core import clearance
from spatial_core.records import (
    PERSONAS,
    Asset,
    AssetType,
    Persona,
    Space,
    VisibleAsset,
    VisibleAssetType,
    VisibleSpace,
)
from spatial_core.stores import index_name
from spatial_ingest import documents, models, pull
from spatial_ingest.corpus import default_corpus, read_corpus
from spatial_ingest.rules import fire, tiers
from spatial_ingest.seed import generate as inventory
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


def _visible_assets(
    assets: list[Asset], types: list[AssetType], sets: dict[Persona, frozenset[str]]
) -> tuple[list[VisibleAsset], list[VisibleAssetType]]:
    asset_sets, type_sets = clearance.visible_assets(assets, types, sets)
    return (
        [
            VisibleAsset(persona=persona, asset_id=asset_id)
            for persona in PERSONAS
            for asset_id in sorted(asset_sets[persona])
        ],
        [
            VisibleAssetType(persona=persona, type_id=type_id)
            for persona in PERSONAS
            for type_id in sorted(type_sets[persona])
        ],
    )


def _catalogue(assets: Iterable[Asset]) -> dict[str, list[str]]:
    """The attribute catalogue: per asset type, the attributes at least one asset carries."""
    catalogue: dict[str, set[str]] = {}
    for asset in assets:
        catalogue.setdefault(asset.type_id, set()).update(asset.attributes)
    return {type_id: sorted(keys) for type_id, keys in sorted(catalogue.items())}


def build(
    *,
    venue: Venue,
    date: str,
    out: Path,
    extract: Path | None = None,
    refetch: bool = False,
    inventory_file: Path | None = None,
    corpus_file: Path | None = None,
    refetch_documents: bool = False,
    model_client: Callable[[], models.Models] = models.from_env,
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

    inputs = inventory.read_inputs(inventory_file or inventory.default_inventory())
    corpus_path = corpus_file or default_corpus(venue.id)
    corpus = read_corpus(corpus_path) if corpus_path is not None else None
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

    placer = fire.Placer(structure.spaces, f"site:{venue.id}")
    sourced = fire.fire_assets(stored.data, placer)
    seeded = inventory.generate(
        inputs=inputs,
        snapshot=date,
        spaces=structure.spaces,
        edges=structure.edges,
        osm=stored.data,
        sourced=sourced,
    )
    all_types = [*sourced.types, *seeded.types]
    all_assets = [*sourced.assets, *seeded.assets]
    visible_assets, visible_types = _visible_assets(all_assets, all_types, sets)
    sourced_assets = {asset.id for asset in sourced.assets}
    sourced_types = {kind.id for kind in sourced.types}
    fire_rules = (fire.RULE, fire.SPACE_RULE, clearance.TYPE_RULE)
    asset_visible_rules = (*visible_rules, clearance.TYPE_RULE, clearance.ASSET_VISIBLE_RULE)
    seed_rules = (*inventory.RULES, *fire_rules)

    prose = documents.ingest(
        corpus=corpus,
        bundle=bundle,
        spaces=structure.spaces,
        types=all_types,
        assets=all_assets,
        models=model_client,
        today=datetime.datetime.now(datetime.UTC).date().isoformat(),
        refetch=refetch_documents,
    )
    translation_rules = (models.PROMPT_VERSION,) if prose.translations else ()

    tool = f"spatial-ingest@{version('spatial-ingest')}"
    parts: dict[str, Any] = {}
    for name, records, rules in (
        ("osm/spaces.jsonl", structure.spaces, RULES),
        ("osm/spans.jsonl", structure.spans, TREE_RULES),
        ("osm/edges.jsonl", structure.edges, EDGE_RULES),
        ("osm/visible_spaces.jsonl", visible, visible_rules),
        ("osm/asset_types.jsonl", sourced.types, fire_rules),
        ("osm/assets.jsonl", sourced.assets, (*fire_rules, *TREE_RULES)),
        (
            "osm/visible_assets.jsonl",
            [v for v in visible_assets if v.asset_id in sourced_assets],
            asset_visible_rules,
        ),
        (
            "osm/visible_asset_types.jsonl",
            [v for v in visible_types if v.type_id in sourced_types],
            asset_visible_rules,
        ),
        ("seed/asset_types.jsonl", seeded.types, inventory.RULES),
        ("seed/assets.jsonl", seeded.assets, inventory.RULES),
        ("seed/maintenance_tasks.jsonl", seeded.tasks, seed_rules),
        ("seed/service_events.jsonl", seeded.events, seed_rules),
        (
            "seed/visible_assets.jsonl",
            [v for v in visible_assets if v.asset_id not in sourced_assets],
            asset_visible_rules,
        ),
        (
            "seed/visible_asset_types.jsonl",
            [v for v in visible_types if v.type_id not in sourced_types],
            asset_visible_rules,
        ),
        (documents.DOCUMENTS_PART, prose.documents, (*documents.RULES, *translation_rules)),
        (documents.CHUNKS_PART, prose.chunks, (*documents.RULES, *translation_rules)),
        (documents.TRANSLATIONS_PART, prose.translations, translation_rules),
        (documents.VECTORS_PART, prose.vectors, ()),
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
        "inputs": {
            "pull": stored.pin,
            "venue": _venue_input(venue),
            "inventory": inputs.pin(),
            # None for a venue with no corpus. Its document parts are then empty.
            "documents": corpus.pin() if corpus is not None else None,
        },
        "parts": parts,
        "counts": structure.counts,
        "clearance": _clearance_counts(structure.spaces, visible, unrecognised),
        "inventory": {
            "seed": inputs.seed,
            "sourced": {
                "asset_types": len(sourced.types),
                "assets": len(sourced.assets),
                "dropped_outside_site": sourced.dropped_outside_site,
            },
            "generated": seeded.counts,
            # Types whose Table 23 code matched nothing and failed closed. Reported,
            # never a reason to block load.
            "unrecognised_categories": sum(
                1 for kind in all_types if kind.clearance.unrecognised_category
            ),
            "visible_assets": {
                persona.value: sum(1 for v in visible_assets if v.persona is persona)
                for persona in PERSONAS
            },
            "attribute_catalogue": _catalogue(all_assets),
        },
        # The index load writes the chunks to, and what promote points at.
        "documents": {"index": index_name(venue.id, date), **prose.summary},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return bundle
