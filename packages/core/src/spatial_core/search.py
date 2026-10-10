"""The search index: where a snapshot's document chunks live. One index per snapshot.

Ingestion writes an index at load and the service will search it. Both go
through SearchIndex, so neither cares whether the index is Azure AI Search or
the stand-in below. Structured facts never come from here, only prose.

LocalIndex is that stand-in: one JSON file per index in a directory. It is what
the tests use and what a laptop uses. It keeps the vectors but ranks nothing
yet; querying it is the retrieval tool's job.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Chunk

# Names a directory for LocalIndex. Ingestion and the service read the same one.
SEARCH_DIR_ENV = "SPATIAL_SEARCH_DIR"


class IndexEntry(BaseModel):
    """A chunk and the embedding of its text."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: Chunk
    vector: tuple[float, ...]


class SearchIndex(Protocol):
    def create(self, name: str, dimensions: int) -> None:
        """A fresh, empty index of that name, replacing any index already there."""

    def upload(self, name: str, entries: Sequence[IndexEntry]) -> None: ...

    def chunks(self, name: str) -> list[Chunk]:
        """Every chunk the index holds, in id order."""

    def delete(self, name: str) -> None:
        """Drop the index. Dropping one that is not there is fine."""

    def exists(self, name: str) -> bool: ...


class LocalIndex:
    """An index as a file: <directory>/<name>.json. For tests and for running locally."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def _read(self, name: str) -> dict[str, object]:
        stored: dict[str, object] = json.loads(self._path(name).read_text(encoding="utf-8"))
        return stored

    def _write(self, name: str, dimensions: int, entries: Sequence[IndexEntry]) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        body = {
            "dimensions": dimensions,
            "entries": [entry.model_dump(mode="json") for entry in entries],
        }
        self._path(name).write_text(
            json.dumps(body, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )

    def create(self, name: str, dimensions: int) -> None:
        self._write(name, dimensions, [])

    def entries(self, name: str) -> list[IndexEntry]:
        """Chunks with their vectors, in id order. The stand-in's own extra."""
        stored = self._read(name)["entries"]
        assert isinstance(stored, list)
        return [IndexEntry.model_validate(entry) for entry in stored]

    def upload(self, name: str, entries: Sequence[IndexEntry]) -> None:
        dimensions = self._read(name)["dimensions"]
        assert isinstance(dimensions, int)
        for entry in entries:
            if len(entry.vector) != dimensions:
                raise ValueError(
                    f"{entry.chunk.id} has a vector of {len(entry.vector)}, "
                    f"the index takes {dimensions}"
                )
        merged = {entry.chunk.id: entry for entry in [*self.entries(name), *entries]}
        self._write(name, dimensions, [merged[key] for key in sorted(merged)])

    def chunks(self, name: str) -> list[Chunk]:
        return [entry.chunk for entry in self.entries(name)]

    def delete(self, name: str) -> None:
        self._path(name).unlink(missing_ok=True)

    def exists(self, name: str) -> bool:
        return self._path(name).is_file()
