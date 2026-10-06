"""ingest load: a snapshot bundle into the stores, checked, but not current.

Load writes the bundle's osm/ half into the snapshot's osm schema and its seed/
half into its seed schema, joins them in a schema of views, then runs the
invariants over what it wrote. A snapshot that fails them is recorded as failed
and cannot be promoted. Load never touches the current-snapshot pointer, and it
refuses to overwrite the current or the previous snapshot, so it never changes
what a visitor reads. It does drop every snapshot older than those two, so exactly
one previous snapshot is ever kept.

Loading the same bundle again rebuilds the snapshot's schemas from scratch, so it
yields the same stores.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ValidationError

from spatial_core.records import Edge, Space, Span, VisibleSpace
from spatial_core.stores import snapshot_schemas
from spatial_ingest import db, invariants
from spatial_ingest.build import MANIFEST_FILE

log = logging.getLogger(__name__)

# Which record each table holds. Both halves of a bundle have the same tables.
TABLES: dict[str, type[BaseModel]] = {
    "spaces": Space,
    "spans": Span,
    "edges": Edge,
    "visible_spaces": VisibleSpace,
}
HALVES = ("osm", "seed")


class LoadError(Exception):
    """The load was refused, or the bundle could not be read. Nothing was written."""


class BundleError(Exception):
    """The bundle does not match its manifest, or holds a record that does not parse."""


@dataclass(frozen=True)
class LoadResult:
    snapshot: str
    # Empty when every invariant holds. Otherwise the snapshot cannot be promoted.
    failures: tuple[str, ...]
    rows: dict[str, int]
    dropped: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class _Part:
    table: str  # "osm.spaces"
    records: Sequence[BaseModel]


def read_manifest(bundle: Path) -> dict[str, Any]:
    path = bundle / MANIFEST_FILE
    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise LoadError(f"cannot read {path}: {error}") from error
    if not isinstance(manifest.get("snapshot"), str) or not isinstance(manifest.get("venue"), str):
        raise LoadError(f"{path} names no snapshot or venue")
    try:
        snapshot_schemas(manifest["snapshot"])
    except ValueError as error:
        raise LoadError(f"{path} names a snapshot that is not a date: {error}") from error
    return manifest


def _read_parts(bundle: Path, manifest: dict[str, Any]) -> list[_Part]:
    listed: dict[str, Any] = dict(manifest.get("parts", {}))
    known = {f"{half}/{table}.jsonl" for half in HALVES for table in TABLES}
    unknown = sorted(set(listed) - known)
    if unknown:
        raise BundleError(f"the manifest lists parts load does not know: {', '.join(unknown)}")
    for half in HALVES:
        for path in sorted((bundle / half).glob("*")):
            name = f"{half}/{path.name}"
            if name not in listed:
                raise BundleError(f"{name} is in the bundle but not in its manifest")

    parts = []
    for half in HALVES:
        for table, model in TABLES.items():
            name = f"{half}/{table}.jsonl"
            if name not in listed:
                parts.append(_Part(f"{half}.{table}", ()))
                continue
            content = (bundle / name).read_bytes()
            if hashlib.sha256(content).hexdigest() != listed[name]["sha256"]:
                raise BundleError(f"{name} does not match the sha256 in the manifest")
            records = []
            for number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
                try:
                    records.append(model.model_validate_json(line))
                except ValidationError as error:
                    raise BundleError(f"{name} line {number}: {error}") from error
            parts.append(_Part(f"{half}.{table}", records))
    return parts


def _expected_rows(manifest: dict[str, Any]) -> dict[str, int]:
    listed = manifest.get("parts", {})
    return {
        f"{half}.{table}": int(listed.get(f"{half}/{table}.jsonl", {}).get("records", 0))
        for half in HALVES
        for table in TABLES
    }


def _insert(conn: psycopg.Connection[Any], schema: str, part: _Part) -> None:
    if not part.records:
        return
    model = type(part.records[0])
    columns = list(model.model_fields)
    query = sql.SQL("INSERT INTO {}.{} ({}) VALUES ({})").format(
        sql.Identifier(schema),
        sql.Identifier(part.table.split(".")[1]),
        sql.SQL(", ").join(sql.Identifier(column) for column in columns),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    rows = []
    for record in part.records:
        dumped = record.model_dump(mode="json")
        rows.append(
            [
                Jsonb(dumped[column]) if isinstance(dumped[column], dict | list) else dumped[column]
                for column in columns
            ]
        )
    with conn.cursor() as cursor:
        cursor.executemany(query, rows)


def _record(
    conn: psycopg.Connection[Any],
    manifest: dict[str, Any],
    failures: Sequence[str],
) -> None:
    schemas = snapshot_schemas(manifest["snapshot"])
    conn.execute(
        """
        INSERT INTO control.snapshots AS row (
            snapshot, venue, osm_schema, seed_schema, views_schema,
            status, failures, manifest, loaded_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (snapshot) DO UPDATE SET
            venue = excluded.venue,
            osm_schema = excluded.osm_schema,
            seed_schema = excluded.seed_schema,
            views_schema = excluded.views_schema,
            status = excluded.status,
            failures = excluded.failures,
            manifest = excluded.manifest,
            loaded_at = excluded.loaded_at
        """,
        (
            manifest["snapshot"],
            manifest["venue"],
            schemas.osm,
            schemas.seed,
            schemas.views,
            "failed" if failures else "loaded",
            Jsonb(list(failures)),
            Jsonb(manifest),
        ),
    )


def _drop_older(
    conn: psycopg.Connection[Any], current: str | None, previous: str | None, loading: str
) -> tuple[str, ...]:
    """Drop every snapshot older than both the current and the previous one.

    Those two are kept: conversations may be held on either. So is anything newer,
    which has been loaded but not yet promoted, and the snapshot being loaded.
    """
    kept = [snapshot for snapshot in (current, previous) if snapshot]
    if not kept:
        return ()
    rows = conn.execute(
        "SELECT snapshot::text FROM control.snapshots WHERE snapshot < %s AND snapshot <> %s"
        " ORDER BY snapshot",
        (min(kept), loading),
    ).fetchall()
    dropped = tuple(snapshot for (snapshot,) in rows)
    for snapshot in dropped:
        db.drop_snapshot(conn, snapshot_schemas(snapshot))
        conn.execute("DELETE FROM control.snapshots WHERE snapshot = %s", (snapshot,))
        log.info("dropped snapshot %s", snapshot)
    return dropped


def load(*, bundle: Path, database_url: str) -> LoadResult:
    manifest = read_manifest(bundle)
    snapshot = manifest["snapshot"]
    schemas = snapshot_schemas(snapshot)
    log.info("loading %s (venue %s) from %s", snapshot, manifest["venue"], bundle)

    with db.connect(database_url) as conn:
        db.apply_control(conn)
        current, previous = db.pointer(conn)
        log.info("current snapshot %s, previous %s", current, previous)
        if snapshot in (current, previous):
            role = "current" if snapshot == current else "previous"
            raise LoadError(
                f"{snapshot} is the {role} snapshot and conversations may be reading it. "
                "Promote another snapshot before loading this one again."
            )

        with conn.transaction():
            dropped = _drop_older(conn, current, previous, snapshot)

        failures: list[str]
        rows: dict[str, int] = {}
        try:
            parts = _read_parts(bundle, manifest)
            with conn.transaction():
                db.drop_snapshot(conn, schemas)
                db.create_snapshot(conn, schemas)
                for part in parts:
                    _insert(conn, getattr(schemas, part.table.split(".")[0]), part)
                    rows[part.table] = len(part.records)
                    log.info("wrote %d rows to %s", len(part.records), part.table)
                failures = invariants.check(conn, schemas, _expected_rows(manifest))
                _record(conn, manifest, failures)
        except (BundleError, psycopg.Error) as error:
            # Whatever an earlier load of this snapshot left behind is no longer
            # what the bundle says, so it cannot be promoted either.
            failures = [f"the bundle could not be written: {error}"]
            with conn.transaction():
                _record(conn, manifest, failures)

    for failure in failures:
        log.error("invariant failed: %s", failure)
    log.info("%s %s", snapshot, "failed its checks" if failures else "loaded, not current")
    return LoadResult(snapshot, tuple(failures), rows, dropped)
