"""The eval snapshot: what the last evaluation run measured, as the service hands it on.

An evaluation run (the bench's `bench run`) writes one JSON file. The service
reads it from SPATIAL_EVAL_SNAPSHOT and serves it at `GET /eval`. The backend
reads that strictly, so this module is the contract:

    a snapshot      the models below, "format": "spatial-eval@1"
    no run yet      EMPTY, exactly, and nothing else counts as empty

A file that is there and does not fit is an error (500), never the empty
answer. An empty card is a claim that nothing was measured, so it is only made
when no file was written.

eval_contract/ next to this file holds one body of each kind. Tests here pin
them to the models, and the backend's contract tests copy them from there.

Every figure has its n and its 95% interval beside it, and the run block names
the date, the snapshot and the deployments for all of them. A figure over no
questions has a null value and a null interval. That is normal: the hand-written
set starts out empty.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

FORMAT = "spatial-eval@1"
SNAPSHOT_ENV = "SPATIAL_EVAL_SNAPSHOT"
CONTRACT_DIR = Path(__file__).parent / "eval_contract"
# The label every published figure carries. The spec's words.
LABEL = "Measured on a test bench: a real building handover, not this campus"
# No run has written a snapshot. This body and no other.
EMPTY: dict[str, Any] = {"format": FORMAT, "run": None}
# Stated, so the absence of a figure is not read as a pass. Follow-ups are on the
# list because the question set is one turn each.
UNMEASURED = ("routes", "proximity", "timing-leaks", "follow-ups")


class EvalSnapshotError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Figure(_Model):
    """One measured number. ``value`` and ``ci95`` are null when ``n`` is zero."""

    value: float | None
    n: int
    ci95: tuple[float, float] | None


class Deployments(_Model):
    planning: str
    composition: str
    # None when the run retrieved by keyword alone.
    embedding: str | None
    # The NLI model that scored entailment. The same one the site runs.
    nli: str


class Run(_Model):
    id: str
    date: str
    venue: str
    snapshot: str
    deployments: Deployments
    semantic_ranker: bool
    # The question set the run asked, by the hash of its gold file.
    gold_sha256: str


class RetrievalStage(_Model):
    recall_at_5: Figure
    mrr: Figure


class CompositionStage(_Model):
    # Answers where every number is a fact and every document segment is entailed.
    holds: Figure
    numbers: Figure
    # Over segments, not answers: the share the NLI model let through.
    entailment: Figure


class Stages(_Model):
    tools: Figure
    retrieval: RetrievalStage
    composition: CompositionStage


class Declined(_Model):
    """Answered when gold says to decline. ``of`` is how many should have been declined.

    Counted on what the layer produced, before the site's NLI model drops anything.
    """

    count: int
    of: int


class TurnKinds(_Model):
    # Expected turn kind, then the kind produced, then how many. Produced is what the
    # site would send: an answer whose document segments the NLI model all drops is
    # an abstention here.
    matrix: dict[str, dict[str, int]]
    answered_when_it_should_have_declined: Declined


class Section(_Model):
    """The figures over one set of questions."""

    n: int
    # All three stages hold and the turn kind is the expected one.
    correct: Figure
    stages: Stages
    turn_kinds: TurnKinds
    # Clarifications whose candidates are gold's, cut to the persona's clearance.
    clarification_candidates: Figure
    by_persona: dict[str, Figure]


class Figures(_Model):
    template: Section
    # Reported apart, always. Zero questions until somebody writes some.
    hand_written: Section


class LeakCheck(_Model):
    passed: bool
    date: str


class LeakChecks(_Model):
    input: LeakCheck
    counterfactual: LeakCheck


class EvalSnapshot(_Model):
    format: Literal["spatial-eval@1"]
    label: str
    run: Run
    figures: Figures
    leak_checks: LeakChecks
    unmeasured: list[str]


def read(path: Path | None) -> dict[str, Any]:
    """The body to serve: the snapshot a run wrote, or EMPTY when none has."""
    if path is None or not path.exists():
        return EMPTY
    try:
        found = EvalSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise EvalSnapshotError(f"{path} is not an eval snapshot: {error}") from error
    return found.model_dump(mode="json")
