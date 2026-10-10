"""The eval snapshot over HTTP: the last run's figures, or one exact empty answer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import fixture
from spatial_service import evaluation
from spatial_service.app import create_app


def served(path: Path | None) -> TestClient:
    return TestClient(create_app(fixture.store(), fixture.llm(None, None, []), evaluation=path))


def contract(name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((evaluation.CONTRACT_DIR / name).read_text())
    return found


def test_with_no_run_the_service_gives_the_one_empty_answer(tmp_path: Path) -> None:
    for nowhere in (None, tmp_path / "never-written.json"):
        response = served(nowhere).get("/eval")

        assert response.status_code == 200
        assert response.json() == {"format": "spatial-eval@1", "run": None}


def test_the_empty_answer_is_the_one_in_the_contract_fixture() -> None:
    assert contract("empty.json") == served(None).get("/eval").json()


def test_a_snapshot_a_run_wrote_is_served_as_it_was_written(tmp_path: Path) -> None:
    written = tmp_path / "eval-snapshot.json"
    written.write_text((evaluation.CONTRACT_DIR / "snapshot.json").read_text())

    response = served(written).get("/eval")

    assert response.status_code == 200
    body = response.json()
    assert body == contract("snapshot.json")
    # What a published figure has to name, straight off the body.
    assert body["label"] == "Measured on a test bench: a real building handover, not this campus"
    assert body["run"]["date"] == "2026-10-10"
    assert body["run"]["snapshot"] == "2026-09-15"
    assert body["run"]["deployments"]["planning"] == "gpt-5.4"
    correct = body["figures"]["template"]["correct"]
    assert (correct["n"], correct["value"], correct["ci95"]) == (154, 0.7532, [0.6795, 0.8146])
    assert body["leak_checks"]["input"] == {"passed": True, "date": "2026-10-09"}
    assert body["unmeasured"] == ["routes", "proximity", "timing-leaks", "follow-ups"]


def test_zero_hand_written_questions_is_a_normal_snapshot() -> None:
    hand_written = contract("snapshot.json")["figures"]["hand_written"]

    assert hand_written["n"] == 0
    assert hand_written["correct"] == {"value": None, "n": 0, "ci95": None}


def test_a_file_that_is_not_the_contract_is_an_error_and_never_an_empty_answer(
    tmp_path: Path,
) -> None:
    renamed = contract("snapshot.json")
    renamed["figure"] = renamed.pop("figures")
    for body in ("not json", json.dumps(renamed), json.dumps({"format": "spatial-eval@2"})):
        written = tmp_path / "eval-snapshot.json"
        written.write_text(body)

        response = served(written).get("/eval")

        assert response.status_code == 500
        assert response.json()["code"] == "eval-snapshot-unreadable"
