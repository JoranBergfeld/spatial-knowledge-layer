"""A handover venue's definition: which files make it up, and the hash each is pinned to.

The files are public and big, so they are not in this repository and no test
fetches them. The operator fetches them once (`ingest fetch-handover`) into a
directory, and build reads them from there and stops if one does not match its pin.
The OCR'd text of the documents is the operator's to make. Build pins each text
file in the manifest the first time it reads it, the way it pins a fetched page.

    {
      "id": "test-bench",
      "source": "handover",
      "handover": {"name": "...", "licence": "CC BY 4.0", "repository": "..."},
      "cobie": {"file": "handover.xlsx", "sha256": "...", "url": "https://..."},
      "ifc": [{"file": "model.ifc", "sha256": "...", "url": "https://..."}],
      "documents": {"file": "documents.zip", "sha256": "...", "url": "https://..."},
      "ocr_dir": "ocr",
      "origin": {"lon": 0.0, "lat": 0.0},
      "placeholder_manufacturers": ["supplycompany@usace.army.mil"]
    }
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

SOURCE = "handover"


class DefinitionError(Exception):
    pass


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pinned(_Entry):
    file: str
    sha256: str
    # Where the operator fetches it from. None for a file that only exists locally.
    url: str | None = None
    size: int | None = None


class Origin(_Entry):
    lon: float
    lat: float


class Definition(_Entry):
    id: str
    source: str
    # Who to credit, and under what terms. Goes into the manifest as it is.
    handover: dict[str, str]
    cobie: Pinned
    ifc: tuple[Pinned, ...] = ()
    # The archive the documents came in. Build never opens it: it reads the OCR'd text.
    documents: Pinned | None = None
    ocr_dir: str = "ocr"
    # Where the model's own (0, 0) is put on the globe. Authored: a redacted building
    # has no address. It only exists so a placement fits the schema's WGS84 point.
    origin: Origin = Origin(lon=0.0, lat=0.0)
    # Contact addresses that name no manufacturer.
    placeholder_manufacturers: tuple[str, ...] = ()

    def pinned(self) -> list[Pinned]:
        return [self.cobie, *self.ifc, *([self.documents] if self.documents else [])]

    def pin(self) -> dict[str, Any]:
        canonical = json.dumps(self.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return {
            "definition": self.model_dump(mode="json"),
            "sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        }


def _raw(name_or_path: str) -> str | None:
    path = Path(name_or_path)
    if path.suffix == ".json":
        return path.read_text(encoding="utf-8")
    builtin = resources.files("spatial_ingest.venues") / f"{name_or_path}.json"
    return builtin.read_text(encoding="utf-8") if builtin.is_file() else None


def load_definition(name_or_path: str) -> Definition | None:
    """A handover venue by id or by file. None when it is some other kind of venue."""
    raw = _raw(name_or_path)
    if raw is None:
        return None
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("source") != SOURCE:
        return None
    try:
        return Definition.model_validate(data)
    except ValidationError as error:
        raise DefinitionError(f"{name_or_path} is not a handover definition: {error}") from error


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_pins(definition: Definition, directory: Path) -> dict[str, str]:
    """Every file build reads, checked against its pin. Returns file name to sha256.

    The document archive is checked only when it is there, since build does not read it.
    """
    found: dict[str, str] = {}
    for pinned in definition.pinned():
        path = directory / pinned.file
        if not path.is_file():
            if pinned is definition.documents:
                continue
            raise DefinitionError(
                f"{path} is missing. Fetch the handover first: ingest fetch-handover"
            )
        actual = sha256_of(path)
        if actual != pinned.sha256:
            raise DefinitionError(
                f"{path} does not match the hash it is pinned to ({actual} is not {pinned.sha256})"
            )
        found[pinned.file] = actual
    return found
