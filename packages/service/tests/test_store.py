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
from fixture import counted, filtered, having, located, most, where
from spatial_core.records import PERSONAS, Persona, VisibleAsset, VisibleAssetType
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
    seen_assets, seen_types = fixture.visible_assets()
    visible_assets = [
        VisibleAsset(persona=persona, asset_id=asset_id)
        for persona in PERSONAS
        for asset_id in sorted(seen_assets[persona])
    ]
    visible_types = [
        VisibleAssetType(persona=persona, type_id=type_id)
        for persona in PERSONAS
        for type_id in sorted(seen_types[persona])
    ]
    records: dict[str, list[BaseModel]] = {
        "osm/spaces.jsonl": list(fixture.spaces()),
        "osm/visible_spaces.jsonl": list(fixture.visible()),
    }
    # Sourced records to the osm half, generated ones to the seed half, as build does.
    for half in ("osm", "seed"):
        records |= {
            f"{half}/asset_types.jsonl": [
                t for t in fixture.asset_types() if t.id.startswith(half)
            ],
            f"{half}/assets.jsonl": [a for a in fixture.assets() if a.id.startswith(half)],
            f"{half}/service_events.jsonl": [
                e for e in fixture.service_events() if e.id.startswith(half)
            ],
            f"{half}/visible_assets.jsonl": [
                v for v in visible_assets if v.asset_id.startswith(half)
            ],
            f"{half}/visible_asset_types.jsonl": [
                v for v in visible_types if v.type_id.startswith(half)
            ],
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


def test_a_persona_reads_its_visible_assets_and_types_and_every_tier(
    stores: tuple[str, str],
) -> None:
    _, reader_url = stores
    store = PostgresStore(reader_url)
    seen_assets, seen_types = fixture.visible_assets()

    for persona in Persona:
        venue = store.venue(fixture.SNAPSHOT, persona)
        assert set(venue.assets) == seen_assets[persona]
        assert set(venue.asset_types) == seen_types[persona]
        assert all(venue.assets[a.id] == a for a in fixture.assets() if a.id in venue.assets)
        # Tier words are the snapshot's, whoever asks.
        assert venue.tiers == ("site", "building", "floor", "room")
    assert len(store.venue(fixture.SNAPSHOT, Persona.VISITOR).assets) == 8
    assert len(store.venue(fixture.SNAPSHOT, Persona.MAINTENANCE).assets) == 13


@pytest.mark.parametrize(
    ("persona", "expected"),
    [
        ("visitor", ["Mini Campus holds 7 generated.", "Informatik: 6 generated."]),
        ("maintenance", ["Mini Campus holds 9 generated.", "Informatik: 8 generated."]),
    ],
)
def test_a_count_read_from_postgres_differs_by_persona(
    stores: tuple[str, str], persona: str, expected: list[str]
) -> None:
    _, reader_url = stores
    app = create_app(
        PostgresStore(reader_url), fixture.llm(lambda _p: most("building", "lights"), counted, [])
    )

    body = {"persona": persona, "question": "Which building has the most lights?"}
    turn = TestClient(app).post("/turn", json=body).json()

    assert turn["kind"] == "answer"
    assert [s["text"] for s in turn["segments"]][:2] == expected


def test_a_persona_reads_the_service_events_of_its_visible_assets_only(
    stores: tuple[str, str],
) -> None:
    _, reader_url = stores
    store = PostgresStore(reader_url)

    def events(persona: Persona) -> set[str]:
        held = store.venue(fixture.SNAPSHOT, persona).service_events
        return {event.id for found in held.values() for event in found}

    everyone = {e.id for e in fixture.service_events()}
    serverraum = {e.id for e in fixture.service_events() if e.asset_id == "seed:asset:P2"}
    assert events(Persona.MAINTENANCE) == everyone
    assert events(Persona.VISITOR) == everyone - serverraum


def test_service_due_is_worked_out_from_what_postgres_holds(stores: tuple[str, str]) -> None:
    _, reader_url = stores
    plan = having("service_due", "before", "2026-01-01", place="Audimax", kind="lights")
    app = create_app(PostgresStore(reader_url), fixture.llm(lambda _p: plan, filtered, []))

    body = {"persona": "visitor", "question": "Which lights in the Audimax fall due before 2026?"}
    turn = TestClient(app).post("/turn", json=body).json()

    assert turn["kind"] == "answer"
    assert [s["text"] for s in turn["segments"]] == [
        "Audimax holds 1 generated.",
        "Audimax: 1 generated.",
        "Counted: service due before 2026-01-01.",
        "1 generated with no service event or installation date recorded.",
    ]
    [fill] = turn["trace"]["assets"]["filters"]
    assert (fill["carried"], fill["of"], fill["default_interval"]) == (2, 3, True)
    assert {d["asset_id"]: d["counted_from"] for d in fill["due"]} == {
        "seed:asset:L1": "service-event",
        "seed:asset:L2": "installation-date",
    }


def test_a_login_that_can_write_is_refused(stores: tuple[str, str]) -> None:
    owner_url, _ = stores

    with pytest.raises(StoreError, match="can write"):
        PostgresStore(owner_url).check_role()


def test_the_reader_cannot_write(stores: tuple[str, str]) -> None:
    _, reader_url = stores

    with psycopg.connect(reader_url) as conn, pytest.raises(psycopg.errors.InsufficientPrivilege):
        conn.execute("UPDATE control.current_snapshot SET snapshot = NULL")
