"""The checks a loaded snapshot must pass before it can be promoted.

Each runs over what was written, not over the bundle, so it checks the stores the
service will read. Each query returns the ids that break it; none means it holds.
The chunk checks do the same over the chunks read back from the search index.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg import sql

from spatial_core import clearance
from spatial_core.records import (
    PERSONAS,
    AnchorKind,
    Chunk,
    ChunkRole,
    Persona,
    RecordProvenance,
)
from spatial_core.stores import SnapshotSchemas
from spatial_ingest.summaries import Summary

# How many offending ids a failure names. The count is always complete.
EXAMPLES = 5


# Tables whose rows carry a record provenance.
_PROVENANCED = ("spaces", "asset_types", "assets", "maintenance_tasks", "service_events")
# A persona's rank in SQL, least cleared first.
_RANK = "array_position(ARRAY['visitor', 'staff', 'maintenance'], {})"


def _union(select: str) -> str:
    return "\nUNION ALL\n".join(select.format(table=table) for table in _PROVENANCED)


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
        "edges name spaces of the snapshot",
        """
        SELECT edge.id FROM {views}.edges edge
        WHERE NOT EXISTS (SELECT FROM {views}.spaces WHERE id = edge.to_id)
            OR (edge.from_id IS NOT NULL
                AND NOT EXISTS (SELECT FROM {views}.spaces WHERE id = edge.from_id))
        """,
    ),
    Invariant(
        "visible sets name spaces of the snapshot",
        """
        SELECT visible.persona || ' ' || visible.space_id FROM {views}.visible_spaces visible
        WHERE NOT EXISTS (SELECT FROM {views}.spaces WHERE id = visible.space_id)
        """,
    ),
    Invariant(
        "a visible space's parent is visible",
        """
        SELECT visible.persona || ' ' || visible.space_id FROM {views}.visible_spaces visible
        JOIN {views}.spaces space ON space.id = visible.space_id
        WHERE space.parent_id IS NOT NULL AND NOT EXISTS (
            SELECT FROM {views}.visible_spaces parent
            WHERE parent.persona = visible.persona AND parent.space_id = space.parent_id
        )
        """,
    ),
    Invariant(
        "the visible sets nest",
        # Visitor within staff within maintenance.
        """
        SELECT narrower.persona || ' ' || narrower.space_id
        FROM {views}.visible_spaces narrower
        JOIN (VALUES ('visitor', 'staff'), ('staff', 'maintenance')) AS chain (persona, wider)
            ON chain.persona = narrower.persona
        WHERE NOT EXISTS (
            SELECT FROM {views}.visible_spaces wider
            WHERE wider.persona = chain.wider AND wider.space_id = narrower.space_id
        )
        """,
    ),
    Invariant(
        "the OSM schema holds no seed rows",
        _union(
            "SELECT id FROM {{osm}}.{table}"
            f" WHERE record_provenance <> '{RecordProvenance.SOURCED.value}'"
        ),
    ),
    Invariant(
        "the seed schema holds only seed rows",
        _union(
            "SELECT id FROM {{seed}}.{table}"
            f" WHERE record_provenance <> '{RecordProvenance.SEED.value}'"
        ),
    ),
    Invariant(
        "the seed schema holds no edges",
        # Edges come from doors in OSM. A generated door asset is bound to one, never one.
        "SELECT id FROM {seed}.edges",
    ),
    Invariant(
        "every asset sits in exactly one space",
        # Each asset once across the pair, and its space a space of this snapshot.
        """
        SELECT id FROM {views}.assets GROUP BY id HAVING count(*) > 1
        UNION ALL
        SELECT asset.id FROM {views}.assets asset
        WHERE NOT EXISTS (SELECT FROM {views}.spaces WHERE id = asset.space_id)
        """,
    ),
    Invariant(
        "an asset's ancestor path agrees with the tree",
        # An asset's path is its space's path plus its space.
        """
        SELECT asset.id FROM {views}.assets asset
        JOIN {views}.spaces space ON space.id = asset.space_id
        WHERE asset.ancestor_path <> space.ancestor_path || jsonb_build_array(
            jsonb_build_object('tier', space.tier ->> 'value', 'id', space.id)
        )
        """,
    ),
    Invariant(
        "every asset is of a type of the snapshot",
        """
        SELECT id FROM {views}.asset_types GROUP BY id HAVING count(*) > 1
        UNION ALL
        SELECT asset.id FROM {views}.assets asset
        WHERE NOT EXISTS (SELECT FROM {views}.asset_types WHERE id = asset.type_id)
        """,
    ),
    Invariant(
        "a door asset is bound to a door of the snapshot",
        """
        SELECT asset.id FROM {views}.assets asset
        WHERE asset.door IS NOT NULL
            AND NOT EXISTS (SELECT FROM {views}.edges WHERE door = asset.door)
        """,
    ),
    Invariant(
        "tasks and events name what they belong to",
        """
        SELECT task.id FROM {views}.maintenance_tasks task
        WHERE NOT EXISTS (SELECT FROM {views}.asset_types WHERE id = task.type_id)
        UNION ALL
        SELECT event.id FROM {views}.service_events event
        WHERE NOT EXISTS (SELECT FROM {views}.assets WHERE id = event.asset_id)
            OR (event.task_id IS NOT NULL
                AND NOT EXISTS (SELECT FROM {views}.maintenance_tasks WHERE id = event.task_id))
        """,
    ),
    Invariant(
        "a visible asset's type is cleared and its space is visible",
        f"""
        SELECT visible.persona || ' ' || visible.asset_id FROM {{views}}.visible_assets visible
        LEFT JOIN {{views}}.assets asset ON asset.id = visible.asset_id
        LEFT JOIN {{views}}.asset_types kind ON kind.id = asset.type_id
        WHERE asset.id IS NULL
            OR kind.id IS NULL
            OR {_RANK.format("kind.clearance ->> 'persona'")} > {_RANK.format("visible.persona")}
            OR NOT EXISTS (
                SELECT FROM {{views}}.visible_spaces space
                WHERE space.persona = visible.persona AND space.space_id = asset.space_id
            )
        """,
    ),
    Invariant(
        "a type is visible only through a visible asset",
        """
        SELECT visible.persona || ' ' || visible.type_id FROM {views}.visible_asset_types visible
        WHERE NOT EXISTS (
            SELECT FROM {views}.visible_assets seen
            JOIN {views}.assets asset ON asset.id = seen.asset_id
            WHERE seen.persona = visible.persona AND asset.type_id = visible.type_id
        )
        """,
    ),
    Invariant(
        "the visible assets nest",
        # Visitor within staff within maintenance, as the spaces do.
        """
        SELECT narrower.persona || ' ' || narrower.asset_id
        FROM {views}.visible_assets narrower
        JOIN (VALUES ('visitor', 'staff'), ('staff', 'maintenance')) AS chain (persona, wider)
            ON chain.persona = narrower.persona
        WHERE NOT EXISTS (
            SELECT FROM {views}.visible_assets wider
            WHERE wider.persona = chain.wider AND wider.asset_id = narrower.asset_id
        )
        """,
    ),
    Invariant(
        "the visible asset types nest",
        """
        SELECT narrower.persona || ' ' || narrower.type_id
        FROM {views}.visible_asset_types narrower
        JOIN (VALUES ('visitor', 'staff'), ('staff', 'maintenance')) AS chain (persona, wider)
            ON chain.persona = narrower.persona
        WHERE NOT EXISTS (
            SELECT FROM {views}.visible_asset_types wider
            WHERE wider.persona = chain.wider AND wider.type_id = narrower.type_id
        )
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


def _failure(name: str, offending: Sequence[str]) -> list[str]:
    if not offending:
        return []
    listed = sorted(offending)
    shown = ", ".join(listed[:EXAMPLES])
    more = f" and {len(listed) - EXAMPLES} more" if len(listed) > EXAMPLES else ""
    return [f"{name}: fails for {shown}{more}"]


def check_chunks(
    conn: psycopg.Connection[Any],
    schemas: SnapshotSchemas,
    chunks: Sequence[Chunk],
    manifest: dict[str, Any],
    summaries: Sequence[Summary] = (),
) -> list[str]:
    """What does not hold of the chunks the index holds, against the stores and the manifest.

    ``summaries`` is the bundle's record of what each summary was made from.
    """

    def rows(query: str) -> list[tuple[Any, ...]]:
        return conn.execute(sql.SQL(query).format(views=sql.Identifier(schemas.views))).fetchall()

    def persona(value: str | None) -> Persona | None:
        return Persona(value) if value else None

    def path(steps: list[dict[str, str]]) -> list[str]:
        return [step["id"] for step in steps]

    spaces = {
        space_id: (persona(cleared), path(steps))
        for space_id, cleared, steps in rows(
            "SELECT id, clearance ->> 'persona', ancestor_path FROM {views}.spaces"
        )
    }
    types = {
        type_id: persona(cleared)
        for type_id, cleared in rows("SELECT id, clearance ->> 'persona' FROM {views}.asset_types")
    }
    assets = {
        asset_id: (type_id, path(steps))
        for asset_id, type_id, steps in rows(
            "SELECT id, type_id, ancestor_path FROM {views}.assets"
        )
    }

    unanchored, misfiled, mispathed = [], [], []
    for chunk in chunks:
        known = (
            all(space_id in spaces for space_id in chunk.space_ids)
            and all(type_id in types for type_id in chunk.asset_type_ids)
            and all(
                asset_id in assets and assets[asset_id][0] in types for asset_id in chunk.asset_ids
            )
        )
        if not known or not (chunk.space_ids or chunk.asset_type_ids or chunk.asset_ids):
            unanchored.append(chunk.id)
            continue
        anchors = [
            *((AnchorKind.SPACE, spaces[space_id][0]) for space_id in chunk.space_ids),
            *((AnchorKind.ASSET_TYPE, types[type_id]) for type_id in chunk.asset_type_ids),
            *((AnchorKind.ASSET, types[assets[asset_id][0]]) for asset_id in chunk.asset_ids),
        ]
        if chunk.clearance_class is not clearance.document_clearance(chunk.kind, anchors):
            misfiled.append(chunk.id)
        above: set[str] = set()
        for space_id in chunk.space_ids:
            above.update([space_id, *spaces[space_id][1]])
        for asset_id in chunk.asset_ids:
            above.update(assets[asset_id][1])
        if sorted(above) != sorted(chunk.ancestor_ids):
            mispathed.append(chunk.id)

    failures = [
        *_failure("a chunk's anchors are spaces, types and assets of the snapshot", unanchored),
        *_failure("a chunk's clearance class is its strictest anchor's", misfiled),
        *_failure("a chunk's ancestor paths agree with the tree", mispathed),
    ]

    passages = {chunk.id: chunk for chunk in chunks if chunk.role is ChunkRole.PASSAGE}
    summarised = [chunk for chunk in chunks if chunk.role is ChunkRole.SUMMARY]
    # A summary may only have read passages of its own document, which are all of
    # its own class. This is the check that no summary crosses a clearance class.
    crossed = [
        made.id
        for made in summaries
        if not made.source_chunk_ids
        or any(
            source not in passages
            or passages[source].document_id != made.document_id
            or passages[source].clearance_class is not made.clearance_class
            for source in made.source_chunk_ids
        )
    ]
    recorded = {made.id for made in summaries if made.indexed}
    unrecorded = [chunk.id for chunk in summarised if chunk.id not in recorded]
    counted: dict[str, int] = {}
    for chunk in summarised:
        counted[chunk.document_id] = counted.get(chunk.document_id, 0) + 1
    documents = {chunk.document_id for chunk in passages.values()} | set(counted)
    failures += [
        *_failure(
            "a summary is built from passages of its own document and clearance class", crossed
        ),
        *_failure("a summary chunk is a summary the bundle recorded", unrecorded),
        *_failure(
            "every document in the index has one summary",
            [document for document in documents if counted.get(document, 0) != 1],
        ),
    ]

    said = manifest.get("documents", {})
    parts = manifest.get("parts", {})
    told = said.get("summaries", {})
    expected = int(parts.get("documents/chunks.jsonl", {}).get("records", 0))
    expected += int(told.get("documents", 0))
    if len(chunks) != expected:
        failures.append(f"the index holds {len(chunks)} chunks, the manifest says {expected}")
    for what, held, by_class in (
        ("passages", list(passages.values()), said.get("chunks_by_clearance_class", {})),
        ("summaries", summarised, told.get("by_clearance_class", {})),
    ):
        for cleared in PERSONAS:
            found = sum(1 for chunk in held if chunk.clearance_class is cleared)
            listed = int(by_class.get(cleared.value, 0))
            if found != listed:
                failures.append(
                    f"the index holds {found} {cleared.value} {what}, the manifest says {listed}"
                )
    return failures
