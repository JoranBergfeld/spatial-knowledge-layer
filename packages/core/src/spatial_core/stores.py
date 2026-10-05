"""Where a snapshot lives in Postgres. Ingestion writes there, the service will read it.

The layer has a database of its own on the site's instance. In it, the control
schema says which snapshots are loaded and which is current, and each snapshot is
a schema pair joined in a third schema of views (ADR 0001 in the site's repository).
The DDL is in ddl/. Ingestion applies it; the service only reads.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from importlib import resources

CONTROL_SCHEMA = "control"
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


def ddl(name: str) -> str:
    """The text of ddl/<name>.sql, with its placeholders still in braces."""
    return (resources.files("spatial_core") / "ddl" / f"{name}.sql").read_text(encoding="utf-8")
