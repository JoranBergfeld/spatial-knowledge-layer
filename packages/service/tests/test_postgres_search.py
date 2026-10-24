"""The search index as tables in Postgres: load writes one, the service searches it.

Needs SPATIAL_TEST_DATABASE_URL, like the store tests, on a server that has
pgvector. The fixture creates the extension as the superuser, since load won't.
The same snapshot as test_store.py, loaded with the index in Postgres this time.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from itertools import combinations
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

import fixture
import papers
from spatial_core.records import ChunkRole, Persona
from spatial_core.search import Filter, LocalIndex, Query
from spatial_core.stores import READER_ROLE
from spatial_ingest.load import load
from spatial_ingest.promote import promote
from spatial_ingest.search import PostgresSearchIndex
from spatial_service.app import create_app
from spatial_service.retrieval import Retriever
from spatial_service.search import PostgresSearch, keywords
from spatial_service.store import PostgresStore
from test_store import ADMIN_URL_ENV, _bundle


def _database(admin_url: str, *, vector: bool) -> str:
    name = f"spatial_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = make_conninfo(admin_url, dbname=name)
    if vector:
        with psycopg.connect(url, autocommit=True) as conn:
            conn.execute("CREATE EXTENSION vector")
    return url


def _drop(admin_url: str, url: str) -> None:
    name = conninfo_to_dict(url)["dbname"]
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(str(name))))


@pytest.fixture(scope="module")
def stores(admin_url: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str]]:
    """The fixture snapshot, current, its index in Postgres; the owner's URL and a reader's."""
    owner_url = _database(admin_url, vector=True)
    reader = f"spatial_test_reader_{uuid4().hex[:8]}"
    bundle = _bundle(tmp_path_factory.mktemp("bundles"))
    result = load(bundle=bundle, database_url=owner_url, index=PostgresSearchIndex(owner_url))
    assert result.ok, result.failures
    promote(snapshot=fixture.SNAPSHOT, database_url=owner_url)
    with psycopg.connect(owner_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'reader' IN ROLE {}").format(
                sql.Identifier(reader), sql.Identifier(READER_ROLE)
            )
        )
    yield owner_url, make_conninfo(owner_url, user=reader, password="reader")
    _drop(admin_url, owner_url)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(reader)))


@pytest.fixture(scope="module")
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the store tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return url


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory: pytest.TempPathFactory) -> LocalIndex:
    """The same chunks in the stand-in, which is the definition of what passes a filter."""
    index = LocalIndex(tmp_path_factory.mktemp("stand-in"))
    index.create(papers.INDEX, papers.DIMENSIONS)
    index.upload(papers.INDEX, papers.entries())
    return index


@pytest.mark.parametrize(
    ("persona", "expected"),
    [("visitor", ["manual:b1#0"]), ("maintenance", ["manual:p1#0", "manual:b1#0"])],
)
def test_a_manual_is_retrieved_from_postgres_through_the_reader(
    stores: tuple[str, str], persona: str, expected: list[str]
) -> None:
    _, reader_url = stores
    store = PostgresStore(reader_url)
    store.check_role()
    plan = papers.manual("isolate the supply before servicing", "Audimax")
    app = create_app(
        store,
        fixture.llm(lambda _p: plan, papers.quoting, []),
        documents=Retriever(PostgresSearch(reader_url), papers.Embedder()),
    )

    turn = TestClient(app).post("/turn", json={"persona": persona, "question": "?"}).json()

    assert turn["kind"] == "answer"
    assert papers.retrieved(turn) == expected
    assert "Precis" not in json.dumps(turn)


def _filters() -> list[Filter]:
    """Every persona's classes against anchors from one chunk, two, or all of them."""
    chunks = papers.passages()
    anchor_sets = [
        (frozenset(a.space_ids), frozenset(a.asset_type_ids), frozenset(a.asset_ids))
        for a in chunks
    ]
    anchor_sets += [
        (x[0] | y[0], x[1] | y[1], x[2] | y[2]) for x, y in combinations(anchor_sets[:4], 2)
    ]
    filters = []
    for count in range(1, len(Persona) + 1):
        classes = tuple(Persona)[:count]
        for spaces, types, assets in anchor_sets:
            filters.append(Filter(classes, spaces, types, assets))
        # An anchor nothing carries: nothing passes.
        filters.append(Filter(classes, space_ids=frozenset({"osm:way/0"})))
    first = chunks[0]
    filters.append(
        Filter(
            tuple(Persona),
            asset_type_ids=frozenset(first.asset_type_ids),
            roles=(ChunkRole.SUMMARY,),
            document_ids=frozenset({first.document_id}),
        )
    )
    return filters


@pytest.mark.parametrize("chosen", _filters())
def test_a_fetch_by_filter_returns_what_the_stand_in_does(
    stores: tuple[str, str], stand_in: LocalIndex, chosen: Filter
) -> None:
    _, reader_url = stores
    query = Query(text=None, filter=chosen, top=50)

    found = PostgresSearch(reader_url).search(papers.INDEX, query)

    assert [hit.chunk for hit in found] == [
        hit.chunk for hit in stand_in.search(papers.INDEX, query)
    ]


def test_hybrid_ranks_the_passage_that_shares_the_words_first(stores: tuple[str, str]) -> None:
    _, reader_url = stores
    everything = Filter(
        tuple(Persona),
        space_ids=frozenset(s for c in papers.passages() for s in c.space_ids),
        asset_type_ids=frozenset(t for c in papers.passages() for t in c.asset_type_ids),
        asset_ids=frozenset(a for c in papers.passages() for a in c.asset_ids),
        roles=(ChunkRole.PASSAGE,),
    )
    text = "torque the busbar bolts"

    for vector in (None, papers.vector(text)):
        hits = PostgresSearch(reader_url).search(
            papers.INDEX, Query(text, everything, vector=vector, top=3)
        )
        assert hits[0].chunk.id == "manual:p1#0"
        assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))
        assert all(a.score >= b.score for a, b in zip(hits, hits[1:], strict=False))


def test_the_reader_reads_the_index_and_cannot_write_it(stores: tuple[str, str]) -> None:
    _, reader_url = stores
    table = sql.Identifier("search", papers.INDEX)

    with psycopg.connect(reader_url, autocommit=True) as conn:
        [(count,)] = conn.execute(sql.SQL("SELECT count(*) FROM {}").format(table)).fetchall()
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            conn.execute(sql.SQL("DELETE FROM {}").format(table))
    assert count == len(papers.entries())


def test_the_owner_side_reads_back_what_it_wrote(stores: tuple[str, str]) -> None:
    owner_url, _ = stores
    index = PostgresSearchIndex(owner_url)

    assert index.exists(papers.INDEX)
    assert index.chunks(papers.INDEX) == sorted(
        (entry.chunk for entry in papers.entries()), key=lambda chunk: chunk.id
    )
    assert not index.exists("spatial-mini-19990101")


def test_load_without_the_vector_extension_fails_and_says_why(
    admin_url: str, tmp_path: Path
) -> None:
    url = _database(admin_url, vector=False)
    try:
        result = load(bundle=_bundle(tmp_path), database_url=url, index=PostgresSearchIndex(url))
        assert not result.ok
        assert "no vector extension" in result.failures[0]
        with psycopg.connect(url) as conn:
            installed = conn.execute("SELECT FROM pg_extension WHERE extname = 'vector'").fetchall()
        assert installed == []
    finally:
        _drop(admin_url, url)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Torque the busbar bolts", "torque | the | busbar | bolts"),
        ("don't & (drop) | it!", "don | t | drop | it"),
        ("snake_case süd süd", "snake | case | süd"),
        ("?!", ""),
    ],
)
def test_a_question_becomes_an_any_of_tsquery_with_no_syntax_in_it(
    text: str, expected: str
) -> None:
    assert keywords(text) == expected
