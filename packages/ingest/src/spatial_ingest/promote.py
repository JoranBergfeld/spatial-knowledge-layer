"""ingest promote: make a loaded snapshot current. Rollback is the same command.

Promotion is one write to the current-snapshot pointer row, which names the
snapshot (and through it the schema pair) and its search index. The snapshot it
replaces becomes the previous one, index and all, which the next load keeps.
Rolling back is promoting that previous snapshot again.
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
    # The search indexes of those two.
    index: str | None = None
    previous_index: str | None = None


def promote(*, snapshot: str, database_url: str) -> Pointer:
    snapshot_schemas(snapshot)  # rejects anything that is not a date
    with db.connect(database_url) as conn:
        db.apply_control(conn)
        with conn.transaction():
            row = conn.execute(
                "SELECT status, failures, search_index FROM control.snapshots WHERE snapshot = %s",
                (snapshot,),
            ).fetchone()
            if row is None:
                raise PromoteError(f"{snapshot} is not loaded")
            status, failures, index = row
            if status != "loaded":
                raise PromoteError(
                    f"{snapshot} failed its checks and cannot be promoted: " + "; ".join(failures)
                )
            if not index:
                raise PromoteError(
                    f"{snapshot} was loaded before snapshots had a search index. Load it again."
                )
            before = db.pointer(conn)
            conn.execute(
                """
                UPDATE control.current_snapshot SET
                    previous = CASE WHEN snapshot = %(s)s THEN previous ELSE snapshot END,
                    previous_search_index = CASE
                        WHEN snapshot = %(s)s THEN previous_search_index ELSE search_index
                    END,
                    snapshot = %(s)s,
                    search_index = %(index)s,
                    promoted_at = now()
                """,
                {"s": snapshot, "index": index},
            )
            indexes = conn.execute(
                "SELECT search_index, previous_search_index FROM control.current_snapshot"
            ).fetchone()
            assert indexes is not None
            after = Pointer(*db.pointer(conn), *indexes)
    log.info(
        "current snapshot %s -> %s (index %s), previous %s",
        before[0],
        after.current,
        after.index,
        after.previous,
    )
    return after
