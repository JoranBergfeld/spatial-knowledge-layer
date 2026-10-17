"""Labelling scored segments by hand, and the NLI model's precision on them.

    bench label runs/segments-2026-10-17-3ddaadac.jsonl runs/segments-2026-10-17-6f60db21.jsonl

It shows one segment at a time with the passages it cites and asks whether those
passages say what the segment says. It never shows the NLI score, so the score
can't lean on the answer. Each answer is written straight away to
`labels-<run id>.jsonl` beside the segments, so quitting halfway loses nothing and
the next go starts at the first segment without a label.

Then it works out precision: of the segments the NLI model kept, the share a
person called faithful. Per run, since a figure goes with the run it came from,
and over all of them. It writes that to `nli-precision.json` beside the segments.
Kept and unfaithful is the one that matters: it is what goes out to a visitor.
Dropped and faithful is counted too, it costs answers but nobody gets misled.

The labels are the owner's, like the segments. They name questions and nothing
else, but they sit next to passage text, so they get the same file mode.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_bench import stats

PRECISION_FILE = "nli-precision.json"
PRECISION_FORMAT = "spatial-nli-precision@1"
# Under this, issue #17 says to raise it: fall back to an LLM judge and re-cost the turn.
ENOUGH = 0.85
_SEGMENTS = re.compile(r"segments-(?P<run>.+)\.jsonl")
_ANSWERS = {"f": True, "u": False}


class LabelError(Exception):
    pass


@dataclass(frozen=True)
class Segment:
    run: str
    question: str
    persona: str
    number: int
    text: str
    passages: tuple[dict[str, Any], ...]
    kept: bool

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.run, self.question, self.number)


def run_of(segments: Path) -> str:
    """The run id a segments file is named after."""
    found = _SEGMENTS.fullmatch(segments.name)
    if not found:
        raise LabelError(f"{segments.name} is not a segments-<run id>.jsonl a run wrote")
    return found["run"]


def labels_file(segments: Path) -> Path:
    return segments.with_name(f"labels-{run_of(segments)}.jsonl")


def load_segments(path: Path) -> list[Segment]:
    run = run_of(path)
    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        one = json.loads(line)
        found.append(
            Segment(
                run=run,
                question=one["question"],
                persona=one["persona"],
                number=one["segment"],
                text=one["text"],
                passages=tuple(one["passages"]),
                kept=one["kept"],
            )
        )
    return found


def load_labels(segments: Path) -> dict[tuple[str, str, int], bool]:
    """What a person said so far about a segments file. A later line for the same segment wins."""
    path, run = labels_file(segments), run_of(segments)
    if not path.exists():
        return {}
    said = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            one = json.loads(line)
            said[(run, one["question"], one["segment"])] = one["faithful"]
    return said


def _append(path: Path, segment: Segment, faithful: bool) -> None:
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    line = {"question": segment.question, "segment": segment.number, "faithful": faithful}
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(line) + "\n")


def _show(segment: Segment, place: str) -> str:
    lines = [
        "",
        "=" * 78,
        f"{place}  {segment.question} ({segment.persona}), segment {segment.number}",
        "",
        "The segment says:",
        f"  {segment.text}",
    ]
    for number, passage in enumerate(segment.passages, start=1):
        lines += ["", f"Passage {number}, {passage['chunk_id']}:", passage["text"]]
    return "\n".join(lines)


def label(
    paths: Sequence[Path],
    ask: Callable[[str], str] = input,
    say: Callable[[str], None] = print,
) -> bool:
    """Ask about every segment with no label yet. False if the person quit early."""
    for path in paths:
        out = labels_file(path)
        said = load_labels(path)
        segments = load_segments(path)
        for place, segment in enumerate(segments, start=1):
            if segment.key in said:
                continue
            say(_show(segment, f"[{run_of(path)} {place}/{len(segments)}]"))
            while True:
                try:
                    answer = ask("Faithful? f yes, u no, s skip, q quit: ").strip().lower()
                except EOFError:
                    answer = "q"
                if answer in _ANSWERS or answer in ("s", "q"):
                    break
            if answer == "q":
                return False
            if answer == "s":
                continue
            _append(out, segment, _ANSWERS[answer])
            said[segment.key] = _ANSWERS[answer]
    return True


def _count(segments: Sequence[Segment], said: dict[tuple[str, str, int], bool]) -> dict[str, Any]:
    labelled = [one for one in segments if one.key in said]
    kept = [one for one in labelled if one.kept]
    dropped = [one for one in labelled if not one.kept]
    faithful_kept = sum(said[one.key] for one in kept)
    return {
        "segments": len(segments),
        "labelled": len(labelled),
        "kept": len(kept),
        "kept_unfaithful": len(kept) - faithful_kept,
        "dropped": len(dropped),
        "dropped_faithful": sum(said[one.key] for one in dropped),
        # Of what the NLI model kept, the share a person called faithful.
        "precision": stats.share(faithful_kept, len(kept)).model_dump(mode="json"),
    }


def precision(paths: Sequence[Path]) -> dict[str, Any]:
    """Precision per run and over all of them, from the labels so far."""
    runs, everything, said = {}, [], {}
    for path in paths:
        segments, labels = load_segments(path), load_labels(path)
        runs[run_of(path)] = _count(segments, labels)
        everything += segments
        said.update(labels)
    over_all = _count(everything, said)
    value = over_all["precision"]["value"]
    return {
        "format": PRECISION_FORMAT,
        "threshold": ENOUGH,
        "runs": runs,
        "all": over_all,
        "enough": None if value is None else value >= ENOUGH,
    }


def write_precision(paths: Sequence[Path], found: dict[str, Any]) -> Path:
    out = paths[0].with_name(PRECISION_FILE)
    out.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
    return out
