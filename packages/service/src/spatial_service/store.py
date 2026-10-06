"""The service's side of the stores: read only, through the reader role.

The service reads the control tables and each snapshot's views, never the schema
pair behind them. Its login must be a member of the reader role and must not be
able to write; it refuses to start otherwise, and runs every transaction read only
besides.

A persona's view of a snapshot is read once and kept. Snapshots do not change once
loaded, and load refuses to rebuild the current or the previous one, which are the
only two the service serves.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from spatial_core.records import Persona, Space, VisibleSpace
from spatial_core.stores import CONTROL_SCHEMA, READER_ROLE, snapshot_schemas
from spatial_service.places import Venue


class StoreError(Exception):
    pass


class Store(Protocol):
    def served(self) -> tuple[str | None, str | None]:
        """The current snapshot and the previous one. Only these are served."""
        ...

    def venue(self, snapshot: str, persona: Persona) -> Venue:
        """What ``persona`` can see in ``snapshot``, and nothing else."""
        ...


class MemoryStore:
    """A snapshot held in memory, for tests and fixtures."""

    def __init__(
        self,
        snapshot: str,
        spaces: Iterable[Space],
        visible: Iterable[VisibleSpace],
        previous: str | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.previous = previous
        self._spaces = {space.id: space for space in spaces}
        self._visible: dict[Persona, set[str]] = {}
        for row in visible:
            self._visible.setdefault(row.persona, set()).add(row.space_id)

    def served(self) -> tuple[str | None, str | None]:
        return self.snapshot, self.previous

    def venue(self, snapshot: str, persona: Persona) -> Venue:
        ids = self._visible.get(persona, set())
        return Venue(snapshot, persona, {i: s for i, s in self._spaces.items() if i in ids})


class PostgresStore:
    def __init__(self, database_url: str) -> None:
        self._url = database_url
        self._venues: dict[tuple[str, Persona], Venue] = {}

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        return psycopg.connect(
            self._url, row_factory=dict_row, options="-c default_transaction_read_only=on"
        )

    def check_role(self) -> None:
        """Refuse a login that is not a reader, or that could write."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT pg_has_role(current_user, %s, 'MEMBER') AS reader,"
                " has_schema_privilege(%s, 'CREATE') AS writer",
                (READER_ROLE, CONTROL_SCHEMA),
            ).fetchone()
        assert row is not None
        if not row["reader"]:
            raise StoreError(f"the service's login is not a member of {READER_ROLE}")
        if row["writer"]:
            raise StoreError("the service's login can write to the stores; give it a reader login")

    def served(self) -> tuple[str | None, str | None]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT snapshot::text, previous::text FROM control.current_snapshot"
            ).fetchone()
        if row is None:
            return None, None
        return row["snapshot"], row["previous"]

    def venue(self, snapshot: str, persona: Persona) -> Venue:
        key = (snapshot, persona)
        if key not in self._venues:
            # Two snapshots are served, so anything older than those is dropped.
            current, previous = self.served()
            for stale in [k for k in self._venues if k[0] not in (current, previous)]:
                del self._venues[stale]
            self._venues[key] = self._read_venue(snapshot, persona)
        return self._venues[key]

    def _read_venue(self, snapshot: str, persona: Persona) -> Venue:
        views = snapshot_schemas(snapshot).views
        query = sql.SQL(
            "SELECT space.* FROM {views}.spaces space"
            " JOIN {views}.visible_spaces visible ON visible.space_id = space.id"
            " WHERE visible.persona = %s"
        ).format(views=sql.Identifier(views))
        with self._connect() as conn:
            loaded = conn.execute(
                "SELECT status FROM control.snapshots WHERE snapshot = %s", (snapshot,)
            ).fetchone()
            if loaded is None or loaded["status"] != "loaded":
                raise StoreError(f"{snapshot} is not a loaded snapshot")
            rows = conn.execute(query, (persona.value,)).fetchall()
        spaces = [Space.model_validate(row) for row in rows]
        return Venue(snapshot, persona, {space.id: space for space in spaces})
