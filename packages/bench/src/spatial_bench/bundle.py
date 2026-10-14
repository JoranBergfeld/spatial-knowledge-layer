"""A snapshot bundle held in memory: its records by part, and writing it back out.

The checks read a bundle to know what exists and who sees it, and the
counterfactual leak check writes changed copies of one. Only the parts load reads
are kept. Originals, the pull and the run logs stay where they are.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from spatial_core.records import (
    PERSONAS,
    AnchorKind,
    Asset,
    AssetType,
    Chunk,
    Document,
    Edge,
    Persona,
    ServiceEvent,
    Space,
    VisibleAsset,
    VisibleAssetType,
    VisibleSpace,
)
from spatial_ingest import summaries
from spatial_ingest.build import MANIFEST_FILE
from spatial_ingest.documents import CHUNKS_PART, DOCUMENTS_PART
from spatial_ingest.load import DOCUMENT_PARTS, HALVES, TABLES
from spatial_ingest.summaries import Summary


def _models() -> dict[str, type[BaseModel]]:
    tables = {f"{half}/{table}.jsonl": model for half in HALVES for table, model in TABLES.items()}
    return {**tables, **DOCUMENT_PARTS}


@dataclass
class Bundle:
    manifest: dict[str, Any]
    # Part name ("osm/assets.jsonl") to its records. A part the bundle lacks is not here.
    parts: dict[str, list[BaseModel]]

    @classmethod
    def read(cls, path: Path) -> Bundle:
        manifest = json.loads((path / MANIFEST_FILE).read_text(encoding="utf-8"))
        parts: dict[str, list[BaseModel]] = {}
        for name, model in _models().items():
            if name in manifest.get("parts", {}):
                lines = (path / name).read_text(encoding="utf-8").splitlines()
                parts[name] = [model.model_validate_json(line) for line in lines]
        return cls(manifest, parts)

    def copy(self) -> Bundle:
        """A bundle of its own to change. The records are frozen, so they are shared."""
        return Bundle(json.loads(json.dumps(self.manifest)), dict(self.parts))

    def write(self, path: Path) -> Path:
        """Write the parts and a manifest that agrees with them."""
        manifest = json.loads(json.dumps(self.manifest))
        for name, records in self.parts.items():
            content = "".join(
                json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                + "\n"
                for record in records
            ).encode()
            (path / name).parent.mkdir(parents=True, exist_ok=True)
            (path / name).write_bytes(content)
            listed = manifest["parts"].setdefault(name, {})
            listed["sha256"] = hashlib.sha256(content).hexdigest()
            listed["records"] = len(records)
        said = manifest.setdefault("documents", {})
        said["documents"] = len(self.documents)
        said["chunks"] = len(self.chunks)
        said["chunks_by_clearance_class"] = {
            persona.value: sum(1 for chunk in self.chunks if chunk.clearance_class is persona)
            for persona in PERSONAS
        }
        said["summaries"] = summaries.manifest(self.summaries)
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        return path

    def _both(self, table: str) -> list[Any]:
        return [r for half in HALVES for r in self.parts.get(f"{half}/{table}.jsonl", [])]

    def half_of(self, table: str, record_id: str) -> str:
        """Which half of the bundle holds a record: the sourced one or the seed one."""
        for half in HALVES:
            if any(r.id == record_id for r in self.parts.get(f"{half}/{table}.jsonl", [])):  # type: ignore[attr-defined]
                return half
        raise KeyError(record_id)

    @property
    def spaces(self) -> list[Space]:
        return cast(list[Space], self._both("spaces"))

    @property
    def edges(self) -> list[Edge]:
        return cast(list[Edge], self._both("edges"))

    @property
    def types(self) -> list[AssetType]:
        return cast(list[AssetType], self._both("asset_types"))

    @property
    def assets(self) -> list[Asset]:
        return cast(list[Asset], self._both("assets"))

    @property
    def events(self) -> list[ServiceEvent]:
        return cast(list[ServiceEvent], self._both("service_events"))

    @property
    def documents(self) -> list[Document]:
        return cast(list[Document], self.parts.get(DOCUMENTS_PART, []))

    @property
    def chunks(self) -> list[Chunk]:
        return cast(list[Chunk], self.parts.get(CHUNKS_PART, []))

    @property
    def summaries(self) -> list[Summary]:
        return cast(list[Summary], self.parts.get(summaries.PART, []))

    def visible(self, persona: Persona) -> Visible:
        """What exists for a persona, as the bundle's three visible sets say."""
        spaces = cast(list[VisibleSpace], self._both("visible_spaces"))
        assets = cast(list[VisibleAsset], self._both("visible_assets"))
        types = cast(list[VisibleAssetType], self._both("visible_asset_types"))
        return Visible(
            persona=persona,
            spaces=frozenset(row.space_id for row in spaces if row.persona is persona),
            assets=frozenset(row.asset_id for row in assets if row.persona is persona),
            types=frozenset(row.type_id for row in types if row.persona is persona),
        )


@dataclass(frozen=True)
class Visible:
    persona: Persona
    spaces: frozenset[str]
    assets: frozenset[str]
    types: frozenset[str]

    def anchored(self, kind: AnchorKind, anchor_id: str) -> bool:
        pool = {
            AnchorKind.SPACE: self.spaces,
            AnchorKind.ASSET_TYPE: self.types,
            AnchorKind.ASSET: self.assets,
        }
        return anchor_id in pool[kind]

    def reads(self, document: Document) -> bool:
        """A document exists for a persona who clears its class and sees one of its anchors."""
        return self.persona.rank >= document.clearance_class.rank and any(
            self.anchored(anchor.kind, anchor.id) for anchor in document.anchors
        )


def replace_part(bundle: Bundle, name: str, records: Iterable[BaseModel]) -> None:
    """Set a part's records. A part the bundle did not have is added to it."""
    bundle.parts[name] = list(records)
    bundle.manifest.setdefault("parts", {}).setdefault(name, {"produced_by": []})
