"""The question set and its gold, as the gold script writes them.

Gold is made by the scripts in gold/ at the top of the repository. They share no
code with the layer and are never imported here: `generate` runs them in an
interpreter of their own with site-packages switched off, and this module only
reads the file that comes out.

    format        "spatial-gold@1"
    source        what gold was computed from: a COBie workbook or a bundle, with its sha256
    unmeasured    the shapes with no questions, by design: proximity and route
    questions     one per (template instance, persona)

A question's `slots` are the words a plan needs, picked apart: the place, the
asset type, the scale level, the filter, the document query. Evaluation runs with
a real model ignore them. The scripted planner builds its plan from them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Persona

FORMAT = "spatial-gold@1"
GOLD_DIR_ENV = "SPATIAL_GOLD_DIR"


class GoldError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExpectedKind(StrEnum):
    """The expected turn kind. A coverage gap is its own kind here, as it is in the metrics."""

    ANSWER = "answer"
    CLARIFICATION = "clarification"
    ABSTENTION = "abstention"
    COVERAGE_GAP = "coverage-gap"


class Fill(_Model):
    """The fill rate behind an attribute answer: how many of the assets asked about carry it."""

    attribute: str
    carried: int
    of: int
    not_recorded: int


class Expected(_Model):
    kind: ExpectedKind
    # Why gold expects a turn that is not an answer. For reading, not for scoring:
    # the layer's own causes are coarser on purpose.
    reason: str | None = None
    # The ids the answer rests on: the place located, or the assets found.
    ids: list[str] | None = None
    total: int | None = None
    # Assets per type for a containment answer, per space of the scale level for a count.
    counts: dict[str, int] | None = None
    # A place's enclosing spaces, outermost first.
    within: list[str] | None = None
    # A count's spaces with nothing recorded at all. Listed apart, never ranked.
    not_recorded: list[str] | None = None
    most: list[str] | None = None
    ungrouped: int | None = None
    # A clarification's candidates, already cut to what the persona can see.
    candidates: list[str] | None = None
    matched: int | None = None
    fill: Fill | None = None
    # Every document retrieval may return here, and the ones the question is after.
    retrievable: list[str] | None = None
    documents: list[str] | None = None


class FilterSlot(_Model):
    attribute: str
    test: str
    date: str | None = None


class Slots(_Model):
    place: str | None = None
    asset_type: str | None = None
    by: str | None = None
    filter: FilterSlot | None = None
    query: str | None = None


class Question(_Model):
    id: str
    # Shared by the visitor's and the maintenance technician's copy of one question.
    pair: str
    shape: str
    template: str
    persona: Persona
    question: str
    slots: Slots
    expected: Expected


class QuestionSet(_Model):
    format: str
    source: dict[str, str]
    per_template: int
    personas: list[Persona]
    unmeasured: list[str]
    world: dict[str, Any]
    questions: list[Question]

    def asked_as(self, persona: Persona) -> list[Question]:
        return [question for question in self.questions if question.persona is persona]


def load(path: Path) -> QuestionSet:
    try:
        found = QuestionSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GoldError(f"cannot read the question set {path}: {error}") from error
    if found.format != FORMAT:
        raise GoldError(f"{path} is {found.format}, this reads {FORMAT}")
    return found


def gold_dir() -> Path:
    """Where the gold scripts are: gold/ at the top of the repository, or wherever the env says."""
    named = os.environ.get(GOLD_DIR_ENV)
    found = Path(named) if named else Path(__file__).resolve().parents[4] / "gold"
    if not (found / "__main__.py").is_file():
        raise GoldError(f"no gold scripts in {found}. Set {GOLD_DIR_ENV}.")
    return found


def generate(*arguments: str, out: Path) -> QuestionSet:
    """Run the gold script and read what it wrote.

    ``-I -S`` is the wall: no site-packages and no PYTHONPATH, so nothing of the
    layer can be imported over there, by accident or otherwise.
    """
    command = [sys.executable, "-I", "-S", str(gold_dir()), *arguments, "--out", str(out)]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise GoldError(f"gold generation failed: {done.stderr.strip() or done.stdout.strip()}")
    return load(out)


def from_cobie(
    workbook: Path, out: Path, ocr: Path | None = None, per_template: int = 6
) -> QuestionSet:
    """Gold for the test bench, from its handover workbook."""
    extra = ["--ocr", str(ocr)] if ocr is not None else []
    return generate(
        "cobie", "--workbook", str(workbook), *extra, "--per-template", str(per_template), out=out
    )


def from_bundle(bundle: Path, out: Path, per_template: int = 6) -> QuestionSet:
    """Gold for the demo venue's regression, read off its bundle."""
    return generate("bundle", "--bundle", str(bundle), "--per-template", str(per_template), out=out)


# The hand-written questions. A person writes these, about fifty of them, and an
# evaluation run reports them apart from the template ones. There are none yet.
HAND_WRITTEN_FORMAT = "spatial-hand-written@1"
HAND_WRITTEN = "hand-written"
HAND_WRITTEN_FILE = Path("questions") / "hand-written.json"


class HandWritten(_Model):
    """One hand-written question. Gold is written by hand too, in the template questions' terms."""

    id: str
    persona: Persona
    shape: str
    question: str
    expected: Expected
    # Set it on both copies when the same words are asked under two personas.
    pair: str | None = None


class HandWrittenSet(_Model):
    format: str
    # The venue the gold was written against. The test bench, by its venue id.
    venue: str
    questions: list[HandWritten]


def load_hand_written(path: Path) -> list[Question]:
    """The hand-written questions in ``path``. No file is no questions, which is fine."""
    if not path.exists():
        return []
    try:
        found = HandWrittenSet.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GoldError(f"cannot read the hand-written questions {path}: {error}") from error
    if found.format != HAND_WRITTEN_FORMAT:
        raise GoldError(f"{path} is {found.format}, this reads {HAND_WRITTEN_FORMAT}")
    ids = [question.id for question in found.questions]
    if len(ids) != len(set(ids)):
        raise GoldError(f"{path} has the same question id twice")
    return [
        Question(
            id=f"{HAND_WRITTEN}/{question.id}",
            pair=f"{HAND_WRITTEN}/{question.pair or question.id}",
            shape=question.shape,
            template=HAND_WRITTEN,
            persona=question.persona,
            question=question.question,
            slots=Slots(),
            expected=question.expected,
        )
        for question in found.questions
    ]
