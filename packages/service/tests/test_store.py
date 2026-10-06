"""The service against a real Postgres, reading through a login that only holds the reader role.

Set SPATIAL_TEST_DATABASE_URL to a superuser connection. The fixture snapshot is
written as a bundle, loaded and promoted by ingestion, and read back by the
service. Without the URL these tests skip, except in CI, where they fail instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import BaseModel

import fixture
from fixture import located, where
from spatial_core.records import Persona
from spatial_core.stores import READER_ROLE
from spatial_ingest.load import load
from spatial_ingest.promote import promote
from spatial_service.app import create_app
from spatial_service.store import PostgresStore, StoreError

ADMIN_URL_ENV = "SPATIAL_TEST_DATABASE_URL"


@pytest.fixture(scope="module")
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the store tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return url


def _bundle(root: Path) -> Path:
    bundle = root / fixture.SNAPSHOT
    parts = {}
    records: dict[str, list[BaseModel]] = {
        "osm/spaces.jsonl": list(fixture.spaces()),
        "osm/visible_spaces.jsonl": list(fixture.visible()),
    }
    for name, rows in records.items():
        content = "".join(row.model_dump_json() + "\n" for row in rows).encode()
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_bytes(content)
        parts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "records": len(rows)}
    manifest = {"snapshot": fixture.SNAPSHOT, "venue": "mini", "parts": parts}
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


@pytest.fixture(scope="module")
def stores(admin_url: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[str, str]]:
    """A database holding the fixture snapshot, current; its owner's URL and a reader's."""
    name = f"spatial_test_{uuid4().hex[:12]}"
    reader = f"spatial_test_reader_{uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    owner_url = make_conninfo(admin_url, dbname=name)
    result = load(bundle=_bundle(tmp_path_factory.mktemp("bundles")), database_url=owner_url)
    assert result.ok, result.failures
    promote(snapshot=fixture.SNAPSHOT, database_url=owner_url)
    with psycopg.connect(owner_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'reader' IN ROLE {}").format(
                sql.Identifier(reader), sql.Identifier(READER_ROLE)
            )
        )
    yield owner_url, make_conninfo(owner_url, user=reader, password="reader")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(reader)))


def test_the_service_reads_through_the_reader_role(stores: tuple[str, str]) -> None:
    _, reader_url = stores
    store = PostgresStore(reader_url)
    store.check_role()
    app = create_app(store, fixture.llm(lambda _p: where("107"), located, []))

    turn = TestClient(app).post("/turn", json={"persona": "visitor", "question": "107?"}).json()

    assert turn["kind"] == "clarification"
    assert turn["state"]["snapshot"] == fixture.SNAPSHOT
    assert [c["space_id"] for c in turn["candidates"]] == ["osm:way/2001", "osm:way/1003"]


def test_a_persona_reads_its_visible_set_and_nothing_else(stores: tuple[str, str]) -> None:
    _, reader_url = stores
    store = PostgresStore(reader_url)

    for persona in Persona:
        venue = store.venue(fixture.SNAPSHOT, persona)
        expected = {row.space_id for row in fixture.visible() if row.persona is persona}
        assert set(venue.spaces) == expected
        assert all(venue.spaces[i] == s for s in fixture.spaces() if (i := s.id) in expected)


def test_a_login_that_can_write_is_refused(stores: tuple[str, str]) -> None:
    owner_url, _ = stores

    with pytest.raises(StoreError, match="can write"):
        PostgresStore(owner_url).check_role()


def test_the_reader_cannot_write(stores: tuple[str, str]) -> None:
    _, reader_url = stores

    with psycopg.connect(reader_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("UPDATE control.current_snapshot SET snapshot = NULL")
