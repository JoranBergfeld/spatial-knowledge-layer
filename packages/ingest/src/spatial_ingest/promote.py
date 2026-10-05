"""ingest promote: make a loaded snapshot current. Rollback is the same command.

Promotion is one write to the current-snapshot pointer row. The snapshot it
replaces becomes the previous one, which the next load keeps. Rolling back is
promoting that previous snapshot again.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from spatial_core.stores import snapshot_schemas
from spatial_ingest import db

log = logging.getLogger(__name__)


class PromoteError(Exception):
    pass


@dataclass(frozen=True)
class Pointer:
    current: str | None
    previous: str | None


def promote(*, snapshot: str, database_url: str) -> Pointer:
    snapshot_schemas(snapshot)  # rejects anything that is not a date
    with db.connect(database_url) as conn:
        db.apply_control(conn)
        with conn.transaction():
            row = conn.execute(
                "SELECT status, failures FROM control.snapshots WHERE snapshot = %s", (snapshot,)
            ).fetchone()
            if row is None:
                raise PromoteError(f"{snapshot} is not loaded")
            status, failures = row
            if status != "loaded":
                raise PromoteError(
                    f"{snapshot} failed its checks and cannot be promoted: " + "; ".join(failures)
                )
            before = db.pointer(conn)
            conn.execute(
                """
                UPDATE control.current_snapshot SET
                    previous = CASE WHEN snapshot = %(s)s THEN previous ELSE snapshot END,
                    snapshot = %(s)s,
                    promoted_at = now()
                """,
                {"s": snapshot},
            )
            after = Pointer(*db.pointer(conn))
    log.info("current snapshot %s -> %s, previous %s", before[0], after.current, after.previous)
    return after
