"""`bench label`: asking about segments, keeping the answers, and the precision they give."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from spatial_bench import label
from spatial_bench.cli import main
from spatial_bench.label import LabelError


def written(directory: Path, run: str, *segments: dict[str, Any]) -> Path:
    path = directory / f"segments-{run}.jsonl"
    path.write_text("".join(json.dumps(one) + "\n" for one in segments), encoding="utf-8")
    return path


def segment(question: str, number: int, kept: bool, score: float = 0.9) -> dict[str, Any]:
    passage = {"chunk_id": "doc:a#1", "document_id": "doc:a", "text": "It says so.", "score": score}
    return {
        "question": question,
        "persona": "visitor",
        "segment": number,
        "text": f"Segment {number} says so.",
        "passages": [passage],
        "score": score,
        "kept": kept,
    }


def answers(*said: str) -> Iterator[str]:
    return iter(said)


def test_a_segments_file_has_to_be_named_after_its_run(tmp_path: Path) -> None:
    assert label.run_of(tmp_path / "segments-2026-10-17-6f60db21.jsonl") == "2026-10-17-6f60db21"
    with pytest.raises(LabelError, match="segments-<run id>"):
        label.run_of(tmp_path / "whatever.jsonl")


def test_the_score_is_never_shown_while_labelling(tmp_path: Path) -> None:
    path = written(tmp_path, "r1", segment("q/1", 1, kept=True, score=0.4321))
    shown: list[str] = []
    said = answers("f")

    label.label([path], ask=lambda _: next(said), say=shown.append)

    assert "Segment 1 says so." in shown[0]
    assert "It says so." in shown[0]
    assert "0.4321" not in shown[0]
    assert "kept" not in shown[0].lower()


def test_quitting_keeps_what_was_said_and_the_next_go_carries_on(tmp_path: Path) -> None:
    path = written(tmp_path, "r1", segment("q/1", 1, True), segment("q/1", 2, True))
    first = answers("nonsense", "u", "q")

    assert label.label([path], ask=lambda _: next(first), say=lambda _: None) is False
    assert label.load_labels(path) == {("r1", "q/1", 1): False}

    asked: list[str] = []
    second = answers("f")
    assert label.label([path], ask=lambda _: next(second), say=asked.append) is True
    assert len(asked) == 1
    assert "segment 2" in asked[0]
    assert label.load_labels(path) == {("r1", "q/1", 1): False, ("r1", "q/1", 2): True}
    assert label.labels_file(path).stat().st_mode & 0o777 == 0o600


def test_end_of_input_is_quitting(tmp_path: Path) -> None:
    path = written(tmp_path, "r1", segment("q/1", 1, True))

    def closed(_: str) -> str:
        raise EOFError

    assert label.label([path], ask=closed, say=lambda _: None) is False


def test_precision_is_the_faithful_share_of_what_the_model_kept(tmp_path: Path) -> None:
    one = written(
        tmp_path,
        "r1",
        segment("q/1", 1, kept=True),
        segment("q/1", 2, kept=True),
        segment("q/2", 1, kept=False),
        segment("q/3", 1, kept=True),
    )
    two = written(tmp_path, "r2", segment("q/1", 1, kept=True))
    # Kept and faithful, kept and not, dropped and faithful, skipped. Then r2's one, faithful.
    said = answers("f", "u", "f", "s", "f")
    label.label([one, two], ask=lambda _: next(said), say=lambda _: None)

    found = label.precision([one, two])

    first = found["runs"]["r1"]
    assert (first["segments"], first["labelled"], first["kept"]) == (4, 3, 2)
    assert (first["kept_unfaithful"], first["dropped"], first["dropped_faithful"]) == (1, 1, 1)
    assert first["precision"]["value"] == 0.5
    # The same question id in two runs is two segments, not one.
    assert found["all"]["kept"] == 3
    assert found["all"]["precision"]["value"] == pytest.approx(2 / 3, abs=1e-4)
    assert found["enough"] is False


def test_bench_label_writes_the_figure_beside_the_segments(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = written(tmp_path, "r1", segment("q/1", 1, True), segment("q/2", 1, True))
    said = answers("f", "f")

    code = main(["label", str(path)], confirm=lambda _: next(said))

    assert code == 0
    found = json.loads((tmp_path / label.PRECISION_FILE).read_text(encoding="utf-8"))
    assert found["format"] == label.PRECISION_FORMAT
    assert found["runs"]["r1"]["precision"]["value"] == 1.0
    assert found["enough"] is True
    assert "precision: 100.0%" in capsys.readouterr().out


def test_precision_only_asks_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = written(tmp_path, "r1", segment("q/1", 1, True))

    def never(_: str) -> str:
        raise AssertionError("asked")

    assert main(["label", str(path), "--precision-only"], confirm=never) == 0
    assert "nothing kept is labelled yet" in capsys.readouterr().out
