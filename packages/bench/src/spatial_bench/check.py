"""Tool correctness: a turn against its gold, by exact match on kind, ids and counts.

This is the first of the three stages an answer is scored on. It reads the turn's
trace, since that is where the ids are, and it never reads the composed text.
Retrieval is held to less here: every document retrieved has to be one gold says
the persona may get, which is exact, and whether the one asked for is among them
is reported apart. Recall and MRR are an evaluation run's to work out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from spatial_bench.questions import Expected, ExpectedKind
from spatial_bench.run import Asked

# How many candidates a clarification names at most.
CANDIDATE_CAP = 6


@dataclass(frozen=True)
class Verdict:
    question: str
    expected: ExpectedKind
    # The turn's kind, with a coverage gap as a kind of its own. None if it never came back.
    got: ExpectedKind | None
    # What does not match gold. Empty when the tools were right.
    mismatches: tuple[str, ...]
    # Documents gold says the question is after that retrieval did not return.
    missed_documents: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.mismatches


def kind_of(turn: dict[str, Any]) -> ExpectedKind | None:
    """A turn's kind as gold names kinds. An unavailable turn has none."""
    if turn.get("kind") == "abstention" and turn.get("cause") == "coverage-gap":
        return ExpectedKind.COVERAGE_GAP
    try:
        return ExpectedKind(str(turn.get("kind")))
    except ValueError:
        return None


def _step(turn: dict[str, Any], tool: str) -> dict[str, Any] | None:
    plans = turn["trace"]["plans"]
    steps = plans[-1]["steps"] if plans else []
    return next((step for step in reversed(steps) if step["tool"] == tool), None)


def _same(what: str, got: Any, want: Any) -> list[str]:
    return [] if got == want else [f"{what}: got {got!r}, gold says {want!r}"]


def _assets(turn: dict[str, Any], expected: Expected) -> list[str]:
    found: list[str] = []
    step = _step(turn, "assets_in_scope")
    trace = turn["trace"].get("assets") or {}
    rows = trace.get("rows", [])
    if expected.ids is not None:
        found += _same("asset ids", sorted(step["ids"]) if step else None, sorted(expected.ids))
    if expected.total is not None:
        found += _same("total", trace.get("touched"), expected.total)
    if expected.counts is not None:
        key = "space_id" if expected.not_recorded is not None else "type_id"
        counts = {
            row[key]: row["sourced"] + row["generated"]
            for row in rows
            if row["recorded"] and row[key] is not None
        }
        found += _same("counts", counts, expected.counts)
    if expected.not_recorded is not None:
        apart = sorted(row["space_id"] for row in rows if not row["recorded"])
        found += _same("nothing recorded in", apart, sorted(expected.not_recorded))
    if expected.most is not None:
        most = sorted(
            row["space_id"]
            for row in rows
            if row["rank"] == 1 and row["sourced"] + row["generated"] > 0
        )
        found += _same("the most in", most, sorted(expected.most))
    if expected.ungrouped is not None:
        found += _same("ungrouped", trace.get("ungrouped"), expected.ungrouped)
    return found


def _fill(turn: dict[str, Any], expected: Expected) -> list[str]:
    if expected.fill is None:
        return []
    filters = (turn["trace"].get("assets") or {}).get("filters", [])
    told = next((f for f in filters if f["attribute"] == expected.fill.attribute), None)
    if told is None:
        return [f"no fill rate in the trace for {expected.fill.attribute}"]
    got = {key: told[key] for key in ("carried", "of", "not_recorded")}
    want = expected.fill.model_dump(exclude={"attribute"})
    return _same("fill rate", got, want)


def compare(asked: Asked) -> Verdict:
    """One turn against its gold."""
    expected, turn = asked.question.expected, asked.turn
    got = kind_of(turn) if asked.status == 200 else None
    mismatches: list[str] = []
    missed: list[str] = []
    if got is not expected.kind and expected.allows(got, turn.get("cause")):
        return Verdict(asked.question.id, expected.kind, got, ())
    if got is not expected.kind:
        said = got.value if got else f"HTTP {asked.status}"
        mismatches.append(f"turn kind: got {said}, gold says {expected.kind.value}")
        return Verdict(asked.question.id, expected.kind, got, tuple(mismatches))

    if expected.kind is ExpectedKind.CLARIFICATION and expected.candidates is not None:
        offered = [c["asset_id"] or c["space_id"] for c in turn["candidates"]]
        if len(expected.candidates) <= CANDIDATE_CAP:
            mismatches += _same("candidates", sorted(offered), sorted(expected.candidates))
        elif len(offered) != CANDIDATE_CAP or not set(offered) <= set(expected.candidates):
            mismatches.append(f"candidates: got {offered!r}, not {CANDIDATE_CAP} of gold's")

    if expected.kind is ExpectedKind.COVERAGE_GAP:
        mismatches += _fill(turn, expected)

    if expected.kind is ExpectedKind.ANSWER:
        if asked.question.shape == "place-location" and expected.ids is not None:
            step = _step(turn, "resolve_place")
            mismatches += _same("place", step["ids"] if step else None, expected.ids)
        elif asked.question.shape != "place-document":
            mismatches += _assets(turn, expected)
            mismatches += _fill(turn, expected)
        if expected.retrievable is not None:
            trace = turn["trace"].get("documents") or {}
            retrieved = [row["document_id"] for row in trace.get("retrieved", [])]
            beyond = sorted(set(retrieved) - set(expected.retrievable))
            if beyond:
                mismatches.append(f"retrieved documents gold says are out of reach: {beyond!r}")
            if not retrieved:
                mismatches.append("an answer with no document retrieved")
            missed = [d for d in expected.documents or [] if d not in retrieved]
    return Verdict(asked.question.id, expected.kind, got, tuple(mismatches), tuple(missed))


def compare_all(asked: Sequence[Asked]) -> list[Verdict]:
    return [compare(one) for one in asked]
