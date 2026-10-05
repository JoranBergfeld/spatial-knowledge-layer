"""Ingestion's side of the stores: its connection, its lock, and applying the DDL."""

from __future__ import annotations

import datetime
from typing import Any

import psycopg
from psycopg import sql

from spatial_core.stores import READER_ROLE, SnapshotSchemas, ddl

# Held for a whole load or promote, so two ingestion runs never interleave.
_LOCK_KEY = 0x5B47_1A7E


def connect(database_url: str) -> psycopg.Connection[Any]:
    """An autocommit connection holding the ingestion lock. Transactions are explicit."""
    conn = psycopg.connect(database_url, autocommit=True)
    conn.execute("SELECT pg_advisory_lock(%s)", (_LOCK_KEY,))
    return conn


def _filled(text: str, **identifiers: str) -> sql.Composed:
    return sql.SQL(text).format(
        reader=sql.Identifier(READER_ROLE),
        reader_name=sql.Literal(READER_ROLE),
        **{key: sql.Identifier(value) for key, value in identifiers.items()},
    )


def apply_control(conn: psycopg.Connection[Any]) -> None:
    with conn.transaction():
        conn.execute(_filled(ddl("control")))


def create_snapshot(conn: psycopg.Connection[Any], schemas: SnapshotSchemas) -> None:
    conn.execute(_filled(ddl("snapshot"), osm=schemas.osm, seed=schemas.seed, views=schemas.views))


def drop_snapshot(conn: psycopg.Connection[Any], schemas: SnapshotSchemas) -> None:
    for schema in schemas.all():
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))


def pointer(conn: psycopg.Connection[Any]) -> tuple[str | None, str | None]:
    """The current snapshot and the previous one, as dates like 2026-09-15."""
    row = conn.execute("SELECT snapshot, previous FROM control.current_snapshot").fetchone()
    assert row is not None
    return _iso(row[0]), _iso(row[1])


def _iso(day: datetime.date | None) -> str | None:
    return day.isoformat() if day is not None else None
