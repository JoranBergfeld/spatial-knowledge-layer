"""The document corpus of a venue: which originals to ingest and what each is about.

A corpus is a published JSON file, pinned by its sha256 like every other input.
It lists sources, never text: the text is fetched at build and the originals stay
in the private bundle.

    {
      "venue": "kit-campus-sued",
      "html_content_between": ["<!-- CONTENT-AREA START -->", "<!-- CONTENT-AREA END -->"],
      "documents": [
        {"id": "kit:hoersaal/141", "kind": "accessibility-page", "language": "de",
         "title": "30.22 Gaede Hs", "url": "https://...",
         "anchors": [{"building": "30.22"}]},
        {"id": "manual:b1", "kind": "manual", "language": "en",
         "title": "...", "url": "https://...",
         "anchors": [{"asset_type": "seed:type:light-fixture-b1"}]}
      ]
    }

An anchor names one thing: a space, an asset type or an asset by its id in the
snapshot, or a building by the number the venue gives it, which
anchor-building-by-ref resolves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from spatial_core.records import DocumentKind


class CorpusError(Exception):
    pass


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AnchorSpec(_Entry):
    building: str | None = None
    space: str | None = None
    asset_type: str | None = None
    asset: str | None = None
    # Set when a published rule found the anchor and the source did not name it:
    # a handover document keyed to a system, anchored to that system's components.
    rule: str | None = None

    @model_validator(mode="after")
    def _names_one_thing(self) -> Self:
        named = (self.building, self.space, self.asset_type, self.asset)
        if len([value for value in named if value]) != 1:
            raise ValueError("an anchor names exactly one of building, space, asset_type or asset")
        return self


class Source(_Entry):
    id: str
    kind: DocumentKind
    title: str
    url: str
    # The original's language, as a BCP 47 tag. "en" is not translated.
    language: str
    anchors: tuple[AnchorSpec, ...]

    @model_validator(mode="after")
    def _is_anchored(self) -> Self:
        if not self.anchors:
            raise ValueError(f"{self.id} has no anchors")
        return self


class _File(_Entry):
    venue: str
    # Only what lies between these two markers of an HTML page is read.
    html_content_between: tuple[str, str] | None = None
    documents: tuple[Source, ...]


@dataclass(frozen=True)
class Corpus:
    venue: str
    html_content_between: tuple[str, str] | None
    sources: tuple[Source, ...]
    sha256: str
    name: str

    def pin(self) -> dict[str, Any]:
        return {"file": self.name, "sha256": self.sha256, "sources": len(self.sources)}


def default_corpus(venue_id: str) -> Path | None:
    """The published corpus of a built-in venue, if it has one."""
    path = Path(str(resources.files("spatial_ingest.corpora") / f"{venue_id}.json"))
    return path if path.is_file() else None


def read_corpus(path: Path) -> Corpus:
    try:
        content = path.read_bytes()
        parsed = _File.model_validate(json.loads(content))
    except (OSError, ValueError, ValidationError) as error:
        raise CorpusError(f"cannot read the corpus {path}: {error}") from error
    ids = [source.id for source in parsed.documents]
    repeated = sorted({source_id for source_id in ids if ids.count(source_id) > 1})
    if repeated:
        raise CorpusError(f"the corpus {path} lists {', '.join(repeated)} more than once")
    return Corpus(
        venue=parsed.venue,
        html_content_between=parsed.html_content_between,
        sources=parsed.documents,
        sha256=hashlib.sha256(content).hexdigest(),
        name=path.name,
    )
