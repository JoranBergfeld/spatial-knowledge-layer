"""The question set on a snapshot that is loaded and not promoted: reported, never blocking."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

from spatial_bench.cli import main as bench
from spatial_bench.questions import QuestionSet
from spatial_core.search import SEARCH_DIR_ENV
from spatial_ingest.cli import main as ingest
from spatial_service.store import PostgresStore

DATE = "2026-09-15"


@pytest.fixture
def database(admin_url: str) -> Iterator[str]:
    """An empty database: nothing loaded, nothing current."""
    name = f"spatial_fresh_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield make_conninfo(admin_url, dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        )


def gold_file(gold: QuestionSet, path: Path, *, break_one: bool = False) -> Path:
    asked = gold.model_dump(mode="json")
    if break_one:
        answer = next(q for q in asked["questions"] if q["expected"]["kind"] == "answer")
        answer["expected"] = {"kind": "abstention"}
    path.write_text(json.dumps(asked))
    return path


def test_load_asks_the_question_set_of_the_new_snapshot_and_promotes_nothing(
    bench_bundle: Path,
    bench_gold: QuestionSet,
    database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(SEARCH_DIR_ENV, str(tmp_path / "index"))
    load = ["load", "--date", DATE, "--out", str(bench_bundle.parent), "--database-url", database]

    code = ingest([*load, "--gold", str(gold_file(bench_gold, tmp_path / "gold.json"))])

    assert code == 0
    said = capsys.readouterr().out
    assert f"question set on {DATE}, scripted model: 166 of 166 turns match gold" in said
    assert "It is not current until promoted" in said
    assert PostgresStore(database).served() == (None, None)


def test_a_question_set_that_does_not_match_is_reported_and_load_still_succeeds(
    bench_bundle: Path,
    bench_gold: QuestionSet,
    database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(SEARCH_DIR_ENV, str(tmp_path / "index"))
    load = ["load", "--date", DATE, "--out", str(bench_bundle.parent), "--database-url", database]
    broken = gold_file(bench_gold, tmp_path / "gold.json", break_one=True)

    code = ingest([*load, "--gold", str(broken)])

    assert code == 0
    said = capsys.readouterr().out
    assert "165 of 166 turns match gold. Reported only, this blocks nothing." in said
    assert "turn kind: got answer, gold says abstention" in said
    # And it can still be promoted.
    assert ingest(["promote", "--date", DATE, "--database-url", database]) == 0


def test_a_question_set_that_cannot_be_read_does_not_fail_load_either(
    bench_bundle: Path,
    database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(SEARCH_DIR_ENV, str(tmp_path / "index"))
    load = ["load", "--date", DATE, "--out", str(bench_bundle.parent), "--database-url", database]

    code = ingest([*load, "--gold", str(tmp_path / "no-such-gold.json")])

    assert code == 0
    assert "question set not run: cannot read the question set" in capsys.readouterr().out


def test_bench_fresh_asks_a_loaded_snapshot_by_name_and_exits_zero_whatever_it_finds(
    bench_bundle: Path,
    bench_gold: QuestionSet,
    database: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(SEARCH_DIR_ENV, str(tmp_path / "index"))
    load = ["load", "--date", DATE, "--out", str(bench_bundle.parent), "--database-url", database]
    assert ingest(load) == 0
    capsys.readouterr()
    fresh = ["fresh", "--snapshot", DATE, "--database-url", database]
    fresh += ["--search-dir", str(tmp_path / "index")]

    matching = bench([*fresh, "--gold", str(gold_file(bench_gold, tmp_path / "gold.json"))])
    said = capsys.readouterr().out
    broken = gold_file(bench_gold, tmp_path / "broken.json", break_one=True)
    mismatching = bench([*fresh, "--gold", str(broken)])

    assert (matching, mismatching) == (0, 0)
    assert "166 of 166 turns match gold" in said
    assert "165 of 166 turns match gold" in capsys.readouterr().out
