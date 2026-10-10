"""The search index: where a snapshot's document chunks live. One index per snapshot.

Ingestion writes an index at load and the service searches it. Writing goes
through SearchIndex and searching through SearchQueries, so neither side cares
whether the index is Azure AI Search or the stand-in below. Structured facts
never come from here, only prose.

LocalIndex is that stand-in: one JSON file per index in a directory. It is what
the tests use and what a laptop uses.

A query is words, a vector and a filter, and the answer is hybrid: a keyword
ranking and a vector ranking fused with RRF, inside document search and nowhere
else. The filter is a value, not a string, so the stand-in can apply it and the
Azure client can render it, and there is one definition of which chunk passes:
Filter.matches.

An index holds passages and one summary per document, told apart by role. A
search returns both unless the filter names a role. What to do with a summary hit
is the caller's business (the service's retrieval.py): the index only ranks.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Chunk, ChunkRole, Persona

# Names a directory for LocalIndex. Ingestion and the service read the same one.
SEARCH_DIR_ENV = "SPATIAL_SEARCH_DIR"
# The semantic configuration an Azure index carries, for when the ranker is switched on.
SEMANTIC_CONFIGURATION = "chunks"
# The constant in reciprocal rank fusion. Azure AI Search uses the same.
RRF_K = 60


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


@dataclass(frozen=True)
class IndexRef:
    """A snapshot's index, and the deployment whose embeddings it holds."""

    name: str
    # None when the manifest names none. Then a query is keywords only.
    embedding_deployment: str | None = None


@dataclass(frozen=True)
class Filter:
    """Which chunks a query may return. Every clause holds, and the anchors are any-of.

    The anchor ids are the caller's to choose, and the caller only ever holds ids
    its persona can see. So a chunk that passes has an anchor in the visible set.
    """

    clearance_classes: tuple[Persona, ...]
    space_ids: frozenset[str] = frozenset()
    asset_type_ids: frozenset[str] = frozenset()
    asset_ids: frozenset[str] = frozenset()
    # Passages and summaries both, unless narrowed.
    roles: tuple[ChunkRole, ...] = (ChunkRole.PASSAGE, ChunkRole.SUMMARY)
    # Only chunks of these documents, when any are named.
    document_ids: frozenset[str] = frozenset()

    @property
    def anchored(self) -> bool:
        """Whether any anchor is named. A filter naming none passes nothing."""
        return bool(self.space_ids or self.asset_type_ids or self.asset_ids)

    def matches(self, chunk: Chunk) -> bool:
        return (
            chunk.role in self.roles
            and (not self.document_ids or chunk.document_id in self.document_ids)
            and chunk.clearance_class in self.clearance_classes
            and bool(
                self.space_ids.intersection(chunk.space_ids)
                or self.asset_type_ids.intersection(chunk.asset_type_ids)
                or self.asset_ids.intersection(chunk.asset_ids)
            )
        )


@dataclass(frozen=True)
class Query:
    # The words to rank by. None ranks by nothing: whatever passes the filter, in id order.
    text: str | None
    filter: Filter
    # The embedding of the text. None searches by keyword alone.
    vector: tuple[float, ...] | None = None
    top: int = 5
    # Rerank the fused results with the semantic ranker. Off unless asked for.
    semantic_ranker: bool = False


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    # From 1, best first.
    rank: int
    score: float


class SearchQueries(Protocol):
    def search(self, name: str, query: Query) -> list[Hit]:
        """The chunks of index ``name`` that pass the filter, best first, at most ``top``."""


# Words the stand-in's keyword half skips, as an English analyzer would.
_STOPWORDS = frozenset("a an and are as at be by for from in is it of on or the to with".split())


def _terms(text: str) -> list[str]:
    return [word for word in re.findall(r"\w+", text.casefold()) if word not in _STOPWORDS]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b, strict=False)) / norm if norm else 0.0


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

    def search(self, name: str, query: Query) -> list[Hit]:
        """Hybrid, roughly as Azure does it: BM25-ish keywords and cosine, fused with RRF.

        There is no semantic ranker here, so that switch changes nothing.
        """
        if not self.exists(name):
            return []
        pool = [entry for entry in self.entries(name) if query.filter.matches(entry.chunk)]
        if not pool:
            return []
        if query.text is None and query.vector is None:
            return [
                Hit(chunk=entry.chunk, rank=rank, score=1.0)
                for rank, entry in enumerate(pool[: query.top], start=1)
            ]
        words = set(_terms(query.text or ""))
        counts = {e.chunk.id: Counter(_terms(f"{e.chunk.title} {e.chunk.text}")) for e in pool}
        spread = {word: sum(1 for c in counts.values() if c[word]) for word in words}

        def keyword(chunk_id: str) -> float:
            tf = counts[chunk_id]
            return sum(
                tf[word]
                / (tf[word] + 1.2)
                * math.log(1 + (len(pool) - spread[word] + 0.5) / (spread[word] + 0.5))
                for word in words
                if tf[word]
            )

        by_keyword = {e.chunk.id: keyword(e.chunk.id) for e in pool}
        rankings = [
            sorted((i for i, s in by_keyword.items() if s > 0), key=lambda i: (-by_keyword[i], i))
        ]
        if query.vector is not None:
            by_vector = {e.chunk.id: _cosine(query.vector, e.vector) for e in pool}
            rankings.append(sorted(by_vector, key=lambda i: (-by_vector[i], i)))
        fused: dict[str, float] = {}
        for ranking in rankings:
            for position, chunk_id in enumerate(ranking, start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1 / (RRF_K + position)
        chunks = {entry.chunk.id: entry.chunk for entry in pool}
        ordered = sorted(fused, key=lambda i: (-fused[i], i))[: query.top]
        return [
            Hit(chunk=chunks[chunk_id], rank=rank, score=fused[chunk_id])
            for rank, chunk_id in enumerate(ordered, start=1)
        ]
