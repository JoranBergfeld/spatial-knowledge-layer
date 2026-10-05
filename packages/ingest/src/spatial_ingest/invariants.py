"""The checks a loaded snapshot must pass before it can be promoted.

Each runs over what was written, not over the bundle, so it checks the stores the
service will read. Each query returns the ids that break it; none means it holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from spatial_core.records import RecordProvenance
from spatial_core.stores import SnapshotSchemas

# How many offending ids a failure names. The count is always complete.
EXAMPLES = 5


@dataclass(frozen=True)
class Invariant:
    name: str
    query: str


INVARIANTS = (
    Invariant(
        "the tree has exactly one root",
        """
        SELECT id FROM {views}.spaces
        WHERE parent_id IS NULL
            AND (SELECT count(*) FROM {views}.spaces WHERE parent_id IS NULL) <> 1
        UNION ALL
        SELECT '(no root)' WHERE NOT EXISTS (SELECT FROM {views}.spaces WHERE parent_id IS NULL)
        """,
    ),
    Invariant(
        "every space has one parent",
        # Each id once across the pair, and each parent a space of this snapshot.
        """
        SELECT id FROM {views}.spaces GROUP BY id HAVING count(*) > 1
        UNION ALL
        SELECT space.id FROM {views}.spaces space
        LEFT JOIN {views}.spaces parent ON parent.id = space.parent_id
        WHERE space.parent_id IS NOT NULL AND parent.id IS NULL
        """,
    ),
    Invariant(
        "ancestor paths agree with the tree",
        # A space's path is its parent's path plus its parent. Holding for every
        # space also rules out cycles, since no path can be one longer than itself.
        """
        SELECT id FROM {views}.spaces
        WHERE parent_id IS NULL AND ancestor_path <> '[]'::jsonb
        UNION ALL
        SELECT space.id FROM {views}.spaces space
        JOIN {views}.spaces parent ON parent.id = space.parent_id
        WHERE space.ancestor_path <> parent.ancestor_path || jsonb_build_array(
            jsonb_build_object('tier', parent.tier ->> 'value', 'id', parent.id)
        )
        """,
    ),
    Invariant(
        "spans name spaces of the snapshot",
        """
        SELECT span.space_id || ' -> ' || span.floor_id FROM {views}.spans span
        WHERE NOT EXISTS (SELECT FROM {views}.spaces WHERE id = span.space_id)
            OR NOT EXISTS (SELECT FROM {views}.spaces WHERE id = span.floor_id)
        """,
    ),
    Invariant(
        "the OSM schema holds no seed rows",
        f"""
        SELECT id FROM {{osm}}.spaces
        WHERE record_provenance <> '{RecordProvenance.SOURCED.value}'
        """,
    ),
)


def check(
    conn: psycopg.Connection[Any], schemas: SnapshotSchemas, expected_rows: dict[str, int]
) -> list[str]:
    """Every invariant that does not hold, described. Empty when the snapshot is sound."""
    failures = []
    for invariant in INVARIANTS:
        query = sql.SQL(invariant.query).format(
            osm=sql.Identifier(schemas.osm),
            seed=sql.Identifier(schemas.seed),
            views=sql.Identifier(schemas.views),
        )
        offending = sorted(str(row[0]) for row in conn.execute(query))
        if offending:
            shown = ", ".join(offending[:EXAMPLES])
            more = f" and {len(offending) - EXAMPLES} more" if len(offending) > EXAMPLES else ""
            failures.append(f"{invariant.name}: fails for {shown}{more}")

    for table, expected in sorted(expected_rows.items()):
        schema, name = table.split(".")
        query = sql.SQL("SELECT count(*) FROM {}.{}").format(
            sql.Identifier(getattr(schemas, schema)), sql.Identifier(name)
        )
        row = conn.execute(query).fetchone()
        stored = row[0] if row else 0
        if stored != expected:
            failures.append(f"{table} holds {stored} rows, the manifest says {expected}")
    return failures
