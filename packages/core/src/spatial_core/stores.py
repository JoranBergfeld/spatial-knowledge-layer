"""Where a snapshot lives in Postgres. Ingestion writes there, the service will read it.

The layer has a database of its own on the site's instance. In it, the control
schema says which snapshots are loaded and which is current, and each snapshot is
a schema pair joined in a third schema of views (ADR 0001 in the site's repository).
The DDL is in ddl/. Ingestion applies it; the service only reads.

A snapshot's document chunks live in a search index of its own, named here too.
In Postgres that index is a table in the search schema (ddl/search.sql).
"""

from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from importlib import resources

CONTROL_SCHEMA = "control"
# Where a snapshot's search index is a table, when the index is in Postgres.
SEARCH_SCHEMA = "search"
# The service's role: SELECT on the control tables and on each snapshot's views.
READER_ROLE = "spatial_reader"


@dataclass(frozen=True)
class SnapshotSchemas:
    osm: str
    seed: str
    views: str

    def all(self) -> tuple[str, str, str]:
        return (self.osm, self.seed, self.views)


def snapshot_schemas(snapshot: str) -> SnapshotSchemas:
    """The schemas of the snapshot dated ``snapshot`` (2026-09-15 -> snap_20260915_*)."""
    stem = f"snap_{datetime.date.fromisoformat(snapshot):%Y%m%d}"
    return SnapshotSchemas(osm=f"{stem}_osm", seed=f"{stem}_seed", views=stem)


def index_name(venue: str, snapshot: str) -> str:
    """The search index of a venue's snapshot (kit-campus-sued, 2026-09-15 ->
    spatial-kit-campus-sued-20260915). The venue is in the name so the test
    bench can never share an index with the demo venue.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", venue.lower()).strip("-")
    if not slug:
        raise ValueError(f"venue {venue!r} has nothing an index can be named after")
    return f"spatial-{slug}-{datetime.date.fromisoformat(snapshot):%Y%m%d}"


def ddl(name: str) -> str:
    """The text of ddl/<name>.sql, with its placeholders still in braces."""
    return (resources.files("spatial_core") / "ddl" / f"{name}.sql").read_text(encoding="utf-8")
