"""Asking a question set of a loaded snapshot, through the service's turn endpoint.

A bundle is loaded into a database of its own and a search index directory of its
own, promoted, and asked through `POST /turn` with the scripted model. Each
question is a conversation of one turn. What comes back is kept with everything
that went into a model on the way, which is what the checks read.

The database is made for the run and dropped after it. It is never the demo
venue's, and never one somebody else made.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo

from spatial_bench.questions import Question
from spatial_bench.stub import scripted_llm
from spatial_core.search import Hit, LocalIndex, Query, SearchQueries
from spatial_ingest.load import load
from spatial_ingest.promote import promote
from spatial_service.app import create_app
from spatial_service.retrieval import Embedded, Embedder, Retriever
from spatial_service.store import PostgresStore, Store


class BenchError(Exception):
    pass


@dataclass(frozen=True)
class Asked:
    """One question, the turn it got, and everything a model or the index was sent for it."""

    question: Question
    status: int
    turn: dict[str, Any]
    # Instructions and payloads of every model call, in order.
    inputs: tuple[str, ...]
    # Every query the index got, filter included.
    queries: tuple[Query, ...]
    # Every text sent to the embedding deployment.
    embedded: tuple[str, ...]


class _RecordingIndex:
    def __init__(self, behind: SearchQueries) -> None:
        self.behind = behind
        self.queries: list[Query] = []

    def search(self, name: str, query: Query) -> list[Hit]:
        self.queries.append(query)
        return self.behind.search(name, query)


class _RecordingEmbedder:
    def __init__(self, behind: Embedder) -> None:
        self.behind = behind
        self.texts: list[str] = []

    def embed(self, deployment: str, text: str) -> Embedded:
        self.texts.append(text)
        return self.behind.embed(deployment, text)


@dataclass
class Bench:
    """A loaded snapshot to put questions to."""

    store: Store
    # None retrieves nothing. No embedder is keyword search only.
    index: SearchQueries | None = None
    embedder: Embedder | None = None
    asked: int = field(default=0, init=False)

    def ask(self, question: Question, known: Sequence[Question] | None = None) -> Asked:
        """One turn. ``known`` is what the scripted planner has plans for."""
        inputs: list[str] = []
        index = _RecordingIndex(self.index) if self.index is not None else None
        embedder = _RecordingEmbedder(self.embedder) if self.embedder is not None else None
        app = create_app(
            self.store,
            scripted_llm(known or [question], inputs),
            documents=Retriever(index, embedder) if index is not None else None,
        )
        response = TestClient(app).post(
            "/turn", json={"persona": question.persona.value, "question": question.question}
        )
        self.asked += 1
        return Asked(
            question=question,
            status=response.status_code,
            turn=response.json(),
            inputs=tuple(inputs),
            queries=tuple(index.queries) if index is not None else (),
            embedded=tuple(embedder.texts) if embedder is not None else (),
        )

    def ask_all(self, questions: Sequence[Question]) -> list[Asked]:
        return [self.ask(question, questions) for question in questions]


@dataclass(frozen=True)
class Loaded:
    """A bundle in stores of its own."""

    bundle: Path
    database_url: str
    index: LocalIndex
    snapshot: str

    def bench(self, embedder: Embedder | None = None) -> Bench:
        return Bench(PostgresStore(self.database_url), self.index, embedder)


@contextmanager
def loaded(bundle: Path, admin_url: str, search_dir: Path) -> Iterator[Loaded]:
    """Load and promote ``bundle`` into a fresh database and index, and drop both afterwards.

    ``admin_url`` is a login that may create databases, on a Postgres for tests and
    runs like this one. A snapshot that fails an invariant is an error here: there
    is nothing worth asking of it.
    """
    name = f"spatial_bench_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = make_conninfo(admin_url, dbname=name)
    try:
        index = LocalIndex(search_dir)
        result = load(bundle=bundle, database_url=url, index=index)
        if not result.ok:
            raise BenchError(f"{bundle} failed its invariants: " + "; ".join(result.failures[:5]))
        promote(snapshot=result.snapshot, database_url=url)
        yield Loaded(bundle, url, index, result.snapshot)
    finally:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
            )
