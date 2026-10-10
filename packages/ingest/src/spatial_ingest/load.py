"""ingest load: a snapshot bundle into the stores, checked, but not current.

Load writes the bundle's osm/ half into the snapshot's osm schema and its seed/
half into its seed schema, joins them in a schema of views, writes the document
chunks and summaries with their vectors to a search index of the snapshot's own,
then runs the invariants over what it wrote. A snapshot that fails them is recorded as failed
and cannot be promoted. Load never touches the current-snapshot pointer, and it
refuses to overwrite the current or the previous snapshot, so it never changes
what a visitor reads. It does drop every snapshot older than those two, index
included, so exactly one previous snapshot is ever kept.

Loading the same bundle again rebuilds the snapshot's schemas and its index from
scratch, so it yields the same stores. It makes no model call: the translations,
the summaries and the vectors are the bundle's.
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

from spatial_core.records import (
    Asset,
    AssetType,
    Chunk,
    Document,
    Edge,
    MaintenanceTask,
    ServiceEvent,
    Space,
    Span,
    VisibleAsset,
    VisibleAssetType,
    VisibleSpace,
)
from spatial_core.search import IndexEntry, SearchIndex
from spatial_core.stores import index_name, snapshot_schemas
from spatial_ingest import db, invariants, search, summaries
from spatial_ingest.build import MANIFEST_FILE
from spatial_ingest.documents import (
    CHUNKS_PART,
    DOCUMENTS_PART,
    TRANSLATIONS_PART,
    VECTORS_PART,
    Translation,
    Vector,
)
from spatial_ingest.documents import DIR as DOCUMENTS_DIR
from spatial_ingest.summaries import Summary

log = logging.getLogger(__name__)

# Which record each table holds. Both halves of a bundle have the same tables.
TABLES: dict[str, type[BaseModel]] = {
    "spaces": Space,
    "spans": Span,
    "edges": Edge,
    "visible_spaces": VisibleSpace,
    "asset_types": AssetType,
    "assets": Asset,
    "maintenance_tasks": MaintenanceTask,
    "service_events": ServiceEvent,
    "visible_assets": VisibleAsset,
    "visible_asset_types": VisibleAssetType,
}
HALVES = ("osm", "seed")
# The bundle's document parts. Only the chunks, the document summaries and their
# vectors leave the bundle, and they go to the search index, not to Postgres.
DOCUMENT_PARTS: dict[str, type[BaseModel]] = {
    DOCUMENTS_PART: Document,
    CHUNKS_PART: Chunk,
    TRANSLATIONS_PART: Translation,
    summaries.PART: Summary,
    VECTORS_PART: Vector,
}
# What an index with no chunks is created with: text-embedding-3-small's size.
DEFAULT_DIMENSIONS = 1536


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
    # The search index the snapshot's chunks went to.
    index: str | None = None

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


def _parse(bundle: Path, name: str, listed: dict[str, Any], model: type[BaseModel]) -> list[Any]:
    content = (bundle / name).read_bytes()
    if hashlib.sha256(content).hexdigest() != listed[name]["sha256"]:
        raise BundleError(f"{name} does not match the sha256 in the manifest")
    records = []
    for number, line in enumerate(content.decode("utf-8").splitlines(), start=1):
        try:
            records.append(model.model_validate_json(line))
        except ValidationError as error:
            raise BundleError(f"{name} line {number}: {error}") from error
    return records


def _read_entries(
    bundle: Path, manifest: dict[str, Any]
) -> tuple[list[IndexEntry], list[Summary], list[Document]]:
    """The bundle's chunks and summary chunks, each with its vector, its summaries and documents.

    Every document part is checked on the way.
    """
    listed: dict[str, Any] = dict(manifest.get("parts", {}))
    read: dict[str, list[Any]] = {}
    for name, model in DOCUMENT_PARTS.items():
        # A part the manifest does not list is empty, as it is for the tables.
        read[name] = _parse(bundle, name, listed, model) if name in listed else []
    for path in sorted((bundle / DOCUMENTS_DIR).glob("*.jsonl")):
        if f"{DOCUMENTS_DIR}/{path.name}" not in listed:
            raise BundleError(
                f"{DOCUMENTS_DIR}/{path.name} is in the bundle but not in its manifest"
            )
    vectors: dict[str, Vector] = {vector.chunk_id: vector for vector in read[VECTORS_PART]}
    try:
        summarised = summaries.chunks_of(read[summaries.PART], read[CHUNKS_PART])
    except summaries.SummaryError as error:
        raise BundleError(str(error)) from error
    entries = []
    for chunk in [*read[CHUNKS_PART], *summarised]:
        if chunk.id not in vectors:
            raise BundleError(f"chunk {chunk.id} has no vector in {VECTORS_PART}")
        entries.append(IndexEntry(chunk=chunk, vector=vectors[chunk.id].vector))
    return entries, read[summaries.PART], read[DOCUMENTS_PART]


def _read_parts(bundle: Path, manifest: dict[str, Any]) -> list[_Part]:
    listed: dict[str, Any] = dict(manifest.get("parts", {}))
    known = {f"{half}/{table}.jsonl" for half in HALVES for table in TABLES}
    unknown = sorted(set(listed) - known - set(DOCUMENT_PARTS))
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
            parts.append(_Part(f"{half}.{table}", _parse(bundle, name, listed, model)))
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
    index: str,
) -> None:
    schemas = snapshot_schemas(manifest["snapshot"])
    conn.execute(
        """
        INSERT INTO control.snapshots AS row (
            snapshot, venue, osm_schema, seed_schema, views_schema,
            status, failures, manifest, search_index, loaded_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (snapshot) DO UPDATE SET
            venue = excluded.venue,
            osm_schema = excluded.osm_schema,
            seed_schema = excluded.seed_schema,
            views_schema = excluded.views_schema,
            status = excluded.status,
            failures = excluded.failures,
            manifest = excluded.manifest,
            search_index = excluded.search_index,
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
            index,
        ),
    )


def _drop_older(
    conn: psycopg.Connection[Any],
    index: SearchIndex,
    current: str | None,
    previous: str | None,
    loading: str,
) -> tuple[str, ...]:
    """Drop every snapshot older than both the current and the previous one, index and all.

    Those two are kept: conversations may be held on either. So is anything newer,
    which has been loaded but not yet promoted, and the snapshot being loaded.
    """
    kept = [snapshot for snapshot in (current, previous) if snapshot]
    if not kept:
        return ()
    rows = conn.execute(
        "SELECT snapshot::text, search_index FROM control.snapshots"
        " WHERE snapshot < %s AND snapshot <> %s ORDER BY snapshot",
        (min(kept), loading),
    ).fetchall()
    dropped = tuple(snapshot for snapshot, _ in rows)
    for snapshot, held in rows:
        if held:
            index.delete(held)
        db.drop_snapshot(conn, snapshot_schemas(snapshot))
        conn.execute("DELETE FROM control.snapshots WHERE snapshot = %s", (snapshot,))
        log.info("dropped snapshot %s", snapshot)
    return dropped


def _report_links(manifest: dict[str, Any]) -> None:
    """Say which sources were offline when build asked. Reported, never a reason to fail."""
    links = manifest.get("documents", {}).get("links", [])
    offline = [link for link in links if link.get("status") == "offline"]
    log.info("link check: %d of %d sources were offline at build", len(offline), len(links))
    for link in offline:
        log.info(
            "  offline as of %s: %s (%s)", link.get("checked_on"), link.get("url"), link["document"]
        )


def load(*, bundle: Path, database_url: str, index: SearchIndex | None = None) -> LoadResult:
    """Write a bundle to the stores and check it. ``index`` defaults to the environment's."""
    manifest = read_manifest(bundle)
    snapshot = manifest["snapshot"]
    schemas = snapshot_schemas(snapshot)
    name = index_name(manifest["venue"], snapshot)
    if index is None:
        try:
            index = search.from_env()
        except search.SearchError as error:
            raise LoadError(str(error)) from error
    log.info("loading %s (venue %s) from %s", snapshot, manifest["venue"], bundle)
    _report_links(manifest)

    with db.connect(database_url) as conn:
        db.apply_control(conn)
        current, previous = db.pointer(conn)
        log.info("current snapshot %s, previous %s", current, previous)
        # A database holds one venue. That is what keeps the test bench out of the
        # demo venue's stores, and the demo venue out of the bench's.
        others = conn.execute(
            "SELECT DISTINCT venue FROM control.snapshots WHERE venue <> %s ORDER BY venue",
            (manifest["venue"],),
        ).fetchall()
        if others:
            raise LoadError(
                f"this database holds {', '.join(venue for (venue,) in others)}. "
                f"{manifest['venue']} gets a database of its own."
            )
        if snapshot in (current, previous):
            role = "current" if snapshot == current else "previous"
            raise LoadError(
                f"{snapshot} is the {role} snapshot and conversations may be reading it. "
                "Promote another snapshot before loading this one again."
            )

        try:
            with conn.transaction():
                dropped = _drop_older(conn, index, current, previous, snapshot)
        except search.SearchError as error:
            raise LoadError(f"could not drop an old snapshot's index: {error}") from error

        failures: list[str]
        rows: dict[str, int] = {}
        try:
            parts = _read_parts(bundle, manifest)
            entries, made, documents = _read_entries(bundle, manifest)
            embedding = manifest.get("documents", {}).get("embedding") or {}
            with conn.transaction():
                db.drop_snapshot(conn, schemas)
                db.create_snapshot(conn, schemas)
                for part in parts:
                    _insert(conn, getattr(schemas, part.table.split(".")[0]), part)
                    rows[part.table] = len(part.records)
                    log.info("wrote %d rows to %s", len(part.records), part.table)
                index.create(name, int(embedding.get("dimensions") or DEFAULT_DIMENSIONS))
                index.upload(name, entries)
                rows["index.chunks"] = len(entries)
                log.info("wrote %d chunks to the index %s", len(entries), name)
                failures = invariants.check(conn, schemas, _expected_rows(manifest))
                # Read back from the index, so this checks what the service will search.
                failures += invariants.check_chunks(
                    conn, schemas, index.chunks(name), manifest, made, documents
                )
                _record(conn, manifest, failures, name)
        except (BundleError, psycopg.Error, search.SearchError) as error:
            # Whatever an earlier load of this snapshot left behind is no longer
            # what the bundle says, so it cannot be promoted either.
            failures = [f"the bundle could not be written: {error}"]
            with conn.transaction():
                _record(conn, manifest, failures, name)

    for failure in failures:
        log.error("invariant failed: %s", failure)
    log.info("%s %s", snapshot, "failed its checks" if failures else "loaded, not current")
    return LoadResult(snapshot, tuple(failures), rows, dropped, name)
