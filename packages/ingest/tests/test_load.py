"""ingest load and ingest promote, end to end against a real Postgres.

Set SPATIAL_TEST_DATABASE_URL to a superuser connection. Each test gets a fresh
database of its own on that server. Without it these tests skip, except in CI,
where they fail instead.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from spatial_core.stores import READER_ROLE
from spatial_ingest.build import MANIFEST_FILE
from spatial_ingest.cli import main
from spatial_ingest.load import TABLES, load

ADMIN_URL_ENV = "SPATIAL_TEST_DATABASE_URL"
SERVICE_ROLE = "spatial_test_service"
A, B, C, D, E = "2026-09-01", "2026-09-08", "2026-09-15", "2026-09-22", "2026-09-29"


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the store tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return url


@pytest.fixture
def database_url(admin_url: str) -> Iterator[str]:
    name = f"spatial_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield make_conninfo(admin_url, dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))


Bundle = Callable[[str], Path]


@pytest.fixture
def bundle(tmp_path: Path, venue_file: Path) -> Bundle:
    """Builds the mini campus as the snapshot of a date, once, and returns its bundle."""
    out = tmp_path / "bundles"

    def built(date: str) -> Path:
        if not (out / date / MANIFEST_FILE).exists():
            args = ["build", "--date", date, "--venue", str(venue_file), "--out", str(out)]
            assert main(args) == 0
        return out / date

    return built


def ingest(command: str, bundle: Path, database_url: str) -> int:
    return main(
        [command, "--date", bundle.name, "--out", str(bundle.parent)]
        + ["--database-url", database_url]
    )


def query(database_url: str, text: str, *params: Any) -> list[tuple[Any, ...]]:
    with psycopg.connect(database_url) as conn:
        return conn.execute(text.encode(), params or None).fetchall()  # type: ignore[call-overload]


def pointer(database_url: str) -> tuple[str | None, str | None]:
    ((current, previous),) = query(
        database_url, "SELECT snapshot::text, previous::text FROM control.current_snapshot"
    )
    return current, previous


def snapshots(database_url: str) -> dict[str, str]:
    rows = query(database_url, "SELECT snapshot::text, status FROM control.snapshots")
    return dict(rows)  # type: ignore[arg-type]


def schemas(database_url: str) -> list[str]:
    rows = query(
        database_url,
        "SELECT nspname FROM pg_namespace WHERE nspname LIKE 'snap\\_%' ORDER BY nspname",
    )
    return [name for (name,) in rows]


def stores(database_url: str, date: str) -> dict[str, list[Any]]:
    """Every row of a snapshot, by table, in a stable order."""
    stem = "snap_" + date.replace("-", "")
    dumped = {}
    for schema in (stem, f"{stem}_osm", f"{stem}_seed"):
        for table in TABLES:
            rows = query(
                database_url,
                f"SELECT row_to_json(t)::text AS r FROM {schema}.{table} t ORDER BY r",
            )
            dumped[f"{schema}.{table}"] = [json.loads(row) for (row,) in rows]
    return dumped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def rewrite(bundle: Path, part: str, records: list[dict[str, Any]]) -> None:
    """Replace a part and update the manifest to match, as a broken build would."""
    content = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records)
    (bundle / part).parent.mkdir(exist_ok=True)
    (bundle / part).write_text(content)
    manifest = json.loads((bundle / MANIFEST_FILE).read_text())
    manifest["parts"][part] = {
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "records": len(records),
        "produced_by": ["a test"],
    }
    (bundle / MANIFEST_FILE).write_text(json.dumps(manifest))


def edit(bundle: Path, part: str, record_id: str, **changes: Any) -> None:
    records = read_jsonl(bundle / part)
    for record in records:
        if record["id"] == record_id:
            record.update(changes)
    rewrite(bundle, part, records)


def edit_space(bundle: Path, space_id: str, **changes: Any) -> None:
    records = read_jsonl(bundle / "osm" / "spaces.jsonl")
    for record in records:
        if record["id"] == space_id:
            record.update(changes)
    rewrite(bundle, "osm/spaces.jsonl", records)


def test_a_clean_load_writes_the_schema_pair_and_does_not_promote(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0

    assert "not current until promoted" in capsys.readouterr().out
    assert snapshots(database_url) == {C: "loaded"}
    assert pointer(database_url) == (None, None)
    assert schemas(database_url) == ["snap_20260915", "snap_20260915_osm", "snap_20260915_seed"]
    stored = stores(database_url, C)
    assert len(stored["snap_20260915_osm.spaces"]) == 29
    assert len(stored["snap_20260915_osm.spans"]) == 2
    assert stored["snap_20260915_seed.spaces"] == []
    # The views join the pair, and every record comes back as the bundle has it.
    by_id = {space["id"]: space for space in stored["snap_20260915.spaces"]}
    for record in read_jsonl(built / "osm" / "spaces.jsonl"):
        assert by_id[record["id"]] == record
    assert stored["snap_20260915.spans"] == read_jsonl(built / "osm" / "spans.jsonl")
    edges = sorted(read_jsonl(built / "osm" / "edges.jsonl"), key=lambda edge: edge["id"])
    assert sorted(stored["snap_20260915.edges"], key=lambda edge: edge["id"]) == edges
    personas = [entry["persona"] for entry in stored["snap_20260915.visible_spaces"]]
    assert {persona: personas.count(persona) for persona in set(personas)} == {
        "visitor": 8,
        "staff": 26,
        "maintenance": 28,
    }
    # The inventory: generated records in the seed schema, the sourced fire assets
    # in the OSM one, and the views joining them.
    for half in ("osm", "seed"):
        for table in ("asset_types", "assets", "maintenance_tasks", "service_events"):
            expected = (
                read_jsonl(built / half / f"{table}.jsonl")
                if (built / half / f"{table}.jsonl").exists()
                else []
            )
            assert len(stored[f"snap_20260915_{half}.{table}"]) == len(expected), (half, table)
    assert len(stored["snap_20260915_osm.assets"]) == 3
    assert len(stored["snap_20260915_seed.assets"]) > 100
    assets = {asset["id"]: asset for asset in stored["snap_20260915.assets"]}
    for record in read_jsonl(built / "seed" / "assets.jsonl"):
        assert assets[record["id"]] == record


def test_each_run_leaves_its_log_beside_the_bundle(bundle: Bundle, database_url: str) -> None:
    built = bundle(C)

    assert ingest("load", built, database_url) == 0
    assert ingest("promote", built, database_url) == 0

    (load_log,) = (built / "runs").glob("*-load.log")
    (promote_log,) = (built / "runs").glob("*-promote.log")
    assert "wrote 29 rows to osm.spaces" in load_log.read_text()
    assert f"current snapshot None -> {C}" in promote_log.read_text()
    # A run's log is not part of the bundle, so the manifest is untouched.
    assert "runs" not in json.loads((built / MANIFEST_FILE).read_text())["parts"]


def test_loading_the_same_bundle_twice_yields_the_same_stores(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    first = stores(database_url, C)

    assert ingest("load", built, database_url) == 0

    assert stores(database_url, C) == first
    assert snapshots(database_url) == {C: "loaded"}


BREAKAGES: dict[str, tuple[Callable[[Path], None], str]] = {
    "an ancestor path that skips the floor": (
        lambda b: edit_space(
            b,
            "osm:way/1001",
            ancestor_path=[
                {"tier": "site", "id": "site:mini-campus"},
                {"tier": "building", "id": "osm:way/100"},
            ],
        ),
        "ancestor paths agree with the tree: fails for osm:way/1001",
    ),
    "a seed row in the OSM schema": (
        lambda b: edit_space(b, "osm:way/1001", record_provenance="seed"),
        "the OSM schema holds no seed rows: fails for osm:way/1001",
    ),
    "a parent that is not in the snapshot": (
        lambda b: edit_space(b, "osm:way/1001", parent_id="floor:way/100:9"),
        "every space has one parent: fails for osm:way/1001",
    ),
    "a second root": (
        lambda b: edit_space(b, "osm:way/600", parent_id=None, ancestor_path=[]),
        "the tree has exactly one root: fails for osm:way/600, site:mini-campus",
    ),
    "a space in both halves of the pair": (
        lambda b: rewrite(
            b,
            "seed/spaces.jsonl",
            [
                {**space, "record_provenance": "seed", "parent_id": "osm:way/200"}
                for space in read_jsonl(b / "osm" / "spaces.jsonl")
                if space["id"] == "osm:way/1001"
            ],
        ),
        "every space has one parent: fails for osm:way/1001",
    ),
    "a span to a floor that is not there": (
        lambda b: rewrite(
            b,
            "osm/spans.jsonl",
            [
                *read_jsonl(b / "osm" / "spans.jsonl"),
                {
                    "space_id": "osm:way/1004",
                    "floor_id": "floor:way/100:7",
                    "provenance": "derived",
                    "rule": "level-parse@1",
                },
            ],
        ),
        "spans name spaces of the snapshot: fails for osm:way/1004 -> floor:way/100:7",
    ),
    "visible sets that do not nest": (
        lambda b: rewrite(
            b,
            "osm/visible_spaces.jsonl",
            [
                entry
                for entry in read_jsonl(b / "osm" / "visible_spaces.jsonl")
                if (entry["persona"], entry["space_id"]) != ("staff", "osm:way/2001")
            ],
        ),
        "the visible sets nest: fails for visitor osm:way/2001",
    ),
    "a visible space whose parent is not": (
        lambda b: rewrite(
            b,
            "osm/visible_spaces.jsonl",
            [
                entry
                for entry in read_jsonl(b / "osm" / "visible_spaces.jsonl")
                if (entry["persona"], entry["space_id"]) != ("maintenance", "floor:way/100:2")
            ],
        ),
        "a visible space's parent is visible: fails for maintenance osm:way/1009",
    ),
    "an edge into a space that is not there": (
        lambda b: rewrite(
            b,
            "osm/edges.jsonl",
            [
                {**edge, "to_id": "osm:way/9999"} if edge["door"] == "node/1104" else edge
                for edge in read_jsonl(b / "osm" / "edges.jsonl")
            ],
        ),
        "edges name spaces of the snapshot: fails for node/1104:outside>osm:way/1001",
    ),
    "an asset in a space that is not there": (
        lambda b: edit(b, "seed/assets.jsonl", first_seed_asset(b), space_id="osm:way/9999"),
        "every asset sits in exactly one space: fails for seed:asset:",
    ),
    "an asset in both halves of the pair": (
        lambda b: rewrite(
            b,
            "osm/assets.jsonl",
            [
                *read_jsonl(b / "osm" / "assets.jsonl"),
                {
                    **read_jsonl(b / "seed" / "assets.jsonl")[0],
                    "record_provenance": "sourced",
                    "space_id": "osm:way/200",
                },
            ],
        ),
        "every asset sits in exactly one space: fails for seed:asset:",
    ),
    "an asset of a type that is not there": (
        lambda b: edit(b, "seed/assets.jsonl", first_seed_asset(b), type_id="seed:type:nothing"),
        "every asset is of a type of the snapshot: fails for seed:asset:",
    ),
    "a generated record in the OSM schema": (
        lambda b: edit(b, "osm/assets.jsonl", "osm:asset:node/1201", record_provenance="seed"),
        "the OSM schema holds no seed rows: fails for osm:asset:node/1201",
    ),
    "a sourced record in the seed schema": (
        lambda b: edit(
            b, "seed/asset_types.jsonl", "seed:type:door-type-a", record_provenance="sourced"
        ),
        "the seed schema holds only seed rows: fails for seed:type:door-type-a",
    ),
    "a generated edge": (
        lambda b: rewrite(
            b,
            "seed/edges.jsonl",
            [{**read_jsonl(b / "osm" / "edges.jsonl")[0], "id": "seed:edge"}],
        ),
        "the seed schema holds no edges: fails for seed:edge",
    ),
    "a door asset bound to no door": (
        lambda b: edit(
            b,
            "seed/assets.jsonl",
            "seed:asset:node/1101:door-type-a",
            door="node/9999",
        ),
        "a door asset is bound to a door of the snapshot: fails for seed:asset:node/1101",
    ),
    "a service event on an asset that is not there": (
        lambda b: rewrite(
            b,
            "seed/service_events.jsonl",
            [
                {**event, "asset_id": "seed:asset:nothing"} if index == 0 else event
                for index, event in enumerate(read_jsonl(b / "seed" / "service_events.jsonl"))
            ],
        ),
        "tasks and events name what they belong to: fails for seed:event:",
    ),
    "staff shown maintenance equipment": (
        lambda b: rewrite(
            b,
            "seed/visible_assets.jsonl",
            [
                *read_jsonl(b / "seed" / "visible_assets.jsonl"),
                {"persona": "staff", "asset_id": "seed:asset:osm:way/1005:vav-200-mm:1"},
            ],
        ),
        "a visible asset's type is cleared and its space is visible: "
        "fails for staff seed:asset:osm:way/1005:vav-200-mm:1",
    ),
    "a type visible with no visible asset": (
        lambda b: rewrite(
            b,
            "seed/visible_asset_types.jsonl",
            [
                *read_jsonl(b / "seed" / "visible_asset_types.jsonl"),
                {"persona": "staff", "type_id": "seed:type:vav-200-mm"},
            ],
        ),
        "a type is visible only through a visible asset: fails for staff seed:type:vav-200-mm",
    ),
    "an asset whose ancestor path skips its space": (
        lambda b: edit(b, "seed/assets.jsonl", first_seed_asset(b), ancestor_path=[]),
        "an asset's ancestor path agrees with the tree: fails for seed:asset:",
    ),
    "visible assets that do not nest": (
        lambda b: rewrite(
            b,
            "osm/visible_assets.jsonl",
            [
                entry
                for entry in read_jsonl(b / "osm" / "visible_assets.jsonl")
                if (entry["persona"], entry["asset_id"]) != ("staff", "osm:asset:node/1201")
            ],
        ),
        "the visible assets nest: fails for visitor osm:asset:node/1201",
    ),
    "visible asset types that do not nest": (
        lambda b: rewrite(
            b,
            "osm/visible_asset_types.jsonl",
            [
                entry
                for entry in read_jsonl(b / "osm" / "visible_asset_types.jsonl")
                if (entry["persona"], entry["type_id"]) != ("staff", "osm:type:fire_extinguisher")
            ],
        ),
        "the visible asset types nest: fails for visitor osm:type:fire_extinguisher",
    ),
    "a part that does not match its manifest": (
        lambda b: (b / "osm" / "spans.jsonl").write_text(""),
        "osm/spans.jsonl does not match the sha256 in the manifest",
    ),
    "a record that does not parse": (
        lambda b: rewrite(b, "osm/spans.jsonl", [{"space_id": "osm:way/1004"}]),
        "osm/spans.jsonl line 1",
    ),
}


def first_seed_asset(bundle: Path) -> str:
    asset_id: str = read_jsonl(bundle / "seed" / "assets.jsonl")[0]["id"]
    return asset_id


@pytest.mark.parametrize("breakage", list(BREAKAGES))
def test_a_broken_bundle_fails_load_and_cannot_be_promoted(
    breakage: str, bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    built = bundle(C)
    breaks, failure = BREAKAGES[breakage]
    breaks(built)

    assert ingest("load", built, database_url) == 1
    assert failure in capsys.readouterr().out
    assert snapshots(database_url) == {C: "failed"}

    assert ingest("promote", built, database_url) == 1
    assert "failed its checks and cannot be promoted" in capsys.readouterr().err
    assert pointer(database_url) == (None, None)


def test_a_failed_reload_leaves_the_snapshot_unpromotable(
    bundle: Bundle, database_url: str
) -> None:
    built = bundle(C)
    assert ingest("load", built, database_url) == 0
    edit_space(built, "osm:way/1001", record_provenance="seed")

    assert ingest("load", built, database_url) == 1

    assert snapshots(database_url) == {C: "failed"}
    assert ingest("promote", built, database_url) == 1


def test_promote_moves_the_pointer_and_rollback_is_the_same_command(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    for date in (A, B):
        assert ingest("load", bundle(date), database_url) == 0

    assert ingest("promote", bundle(A), database_url) == 0
    assert pointer(database_url) == (A, None)
    assert ingest("promote", bundle(B), database_url) == 0
    assert pointer(database_url) == (B, A)

    # Rollback: the same command, pointed at the previous snapshot.
    assert ingest("promote", bundle(A), database_url) == 0
    assert pointer(database_url) == (A, B)
    assert f"current snapshot {A}, previous {B}" in capsys.readouterr().out

    # Promoting the current snapshot again changes nothing.
    assert ingest("promote", bundle(A), database_url) == 0
    assert pointer(database_url) == (A, B)


def test_a_snapshot_that_was_never_loaded_cannot_be_promoted(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest("promote", bundle(C), database_url) == 1
    assert f"{C} is not loaded" in capsys.readouterr().err


def test_load_never_changes_what_visitors_read(
    bundle: Bundle, database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert ingest("load", bundle(A), database_url) == 0
    assert ingest("promote", bundle(A), database_url) == 0
    serving = stores(database_url, A)

    assert ingest("load", bundle(B), database_url) == 0
    assert pointer(database_url) == (A, None)
    assert stores(database_url, A) == serving

    # Not even by loading the current snapshot again.
    assert ingest("load", bundle(A), database_url) == 1
    assert f"{A} is the current snapshot" in capsys.readouterr().err
    assert stores(database_url, A) == serving

    # Nor the previous one, which conversations may still be held on.
    assert ingest("promote", bundle(B), database_url) == 0
    assert ingest("load", bundle(A), database_url) == 1
    assert f"{A} is the previous snapshot" in capsys.readouterr().err
    assert snapshots(database_url) == {A: "loaded", B: "loaded"}


def test_the_next_load_drops_every_snapshot_older_than_current_and_previous(
    bundle: Bundle, database_url: str
) -> None:
    for date in (A, B, C):
        assert ingest("load", bundle(date), database_url) == 0
        assert ingest("promote", bundle(date), database_url) == 0
    assert pointer(database_url) == (C, B)
    assert set(snapshots(database_url)) == {A, B, C}

    result = load(bundle=bundle(D), database_url=database_url)

    assert result.dropped == (A,)
    assert snapshots(database_url) == {B: "loaded", C: "loaded", D: "loaded"}
    assert not [name for name in schemas(database_url) if name.startswith("snap_20260901")]

    # A newer snapshot not yet promoted is kept, and so is an older one being loaded.
    result = load(bundle=bundle(E), database_url=database_url)
    assert result.dropped == ()
    result = load(bundle=bundle(A), database_url=database_url)
    assert result.dropped == ()
    assert set(snapshots(database_url)) == {A, B, C, D, E}

    # Until the next load, which drops it again, failed or not.
    edit_space(bundle(E), "osm:way/1001", record_provenance="seed")
    result = load(bundle=bundle(E), database_url=database_url)
    assert not result.ok
    assert result.dropped == (A,)
    assert snapshots(database_url) == {B: "loaded", C: "loaded", D: "loaded", E: "failed"}
    assert pointer(database_url) == (C, B)


def test_the_service_role_reads_the_views_and_writes_nothing(
    bundle: Bundle, database_url: str, admin_url: str
) -> None:
    assert ingest("load", bundle(C), database_url) == 0
    assert ingest("promote", bundle(C), database_url) == 0
    with psycopg.connect(admin_url, autocommit=True) as conn:
        exists = conn.execute("SELECT FROM pg_roles WHERE rolname = %s", (SERVICE_ROLE,))
        if exists.fetchone() is None:
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(sql.Identifier(SERVICE_ROLE)))
        conn.execute(
            sql.SQL("ALTER ROLE {} PASSWORD 'service'").format(sql.Identifier(SERVICE_ROLE))
        )
        conn.execute(
            sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(READER_ROLE), sql.Identifier(SERVICE_ROLE)
            )
        )
    service_url = make_conninfo(database_url, user=SERVICE_ROLE, password="service")

    with psycopg.connect(service_url, autocommit=True) as conn:
        row = conn.execute(
            "SELECT s.views_schema FROM control.current_snapshot p "
            "JOIN control.snapshots s ON s.snapshot = p.snapshot"
        ).fetchone()
        assert row == ("snap_20260915",)
        count = conn.execute("SELECT count(*) FROM snap_20260915.spaces").fetchone()
        assert count == (29,)
        catalogue = conn.execute(
            "SELECT attribute, assets FROM snap_20260915.attribute_catalogue"
            " WHERE type_id = 'osm:type:fire_extinguisher'"
        ).fetchall()
        assert catalogue == [("fire_extinguisher:type", 1)]
        for view in ("asset_types", "assets", "maintenance_tasks", "service_events"):
            row = conn.execute(f"SELECT count(*) FROM snap_20260915.{view}".encode()).fetchone()
            assert row is not None and row[0] > 0, view

        for statement in (
            "SELECT count(*) FROM snap_20260915_osm.spaces",
            "SELECT count(*) FROM snap_20260915_seed.spaces",
            "SELECT count(*) FROM snap_20260915_seed.assets",
            "UPDATE control.current_snapshot SET snapshot = NULL",
            "DELETE FROM control.snapshots",
            "CREATE TABLE snap_20260915.mine (id int)",
            "CREATE TABLE control.mine (id int)",
        ):
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement.encode())
