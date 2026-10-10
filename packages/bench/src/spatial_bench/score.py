"""One turn scored against its gold, a stage at a time. No model judges anything.

    tools         check.compare: turn kind, id sets, counts, fill rates and a
                  clarification's candidates, all by exact match
    retrieval     Recall@5 and reciprocal rank of the documents gold says the
                  question is after, over the documents retrieved, in rank order
    composition   every number in the composed text is a fact the tools counted,
                  and every segment that cites a passage is entailed by it

An answer is correct only when all three hold and the turn kind is the expected
one. A stage that does not apply to a turn (no document asked for, nothing
composed) is None and holds by default.

Three things decided here, since the spec leaves them open:

- Retrieval holds when every document the question is after is in the top five.
- A segment citing several passages is scored against each and keeps its best.
- The kind produced is what the site would send. The backend drops a segment the
  NLI model does not back and abstains when none is left, so an answer whose
  document segments all fail counts as an abstention in the confusion matrix.

The numbers check looks for digits. The composer is not allowed to write any,
so every digit in an answer came out of a placeholder. A number written out as a
word by the model would get past it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from spatial_bench.check import compare, kind_of
from spatial_bench.nli import THRESHOLD, Nli
from spatial_bench.questions import ExpectedKind
from spatial_bench.run import Asked

TOP = 5
# What a turn that never came back counts as. Not a kind gold can expect.
UNAVAILABLE = "unavailable"
PRODUCED = (*(kind.value for kind in ExpectedKind), UNAVAILABLE)
DECLINES = (ExpectedKind.ABSTENTION.value, ExpectedKind.COVERAGE_GAP.value)


@dataclass(frozen=True)
class Scored:
    question: str
    # "template" or "hand-written".
    set: str
    pair: str
    persona: str
    shape: str
    expected: str
    produced: str
    tools: bool
    mismatches: tuple[str, ...]
    # None when the question is after no document.
    recall_at_5: float | None
    reciprocal_rank: float | None
    # The same with the semantic ranker on, when the run compared it.
    reranked_recall_at_5: float | None
    # None when the turn composed no answer.
    numbers: bool | None
    # Segments citing a passage, and how many of them the NLI model backs.
    segments: int
    entailed: int
    # None unless gold expects a clarification and one came back.
    candidates: bool | None

    @property
    def retrieval(self) -> bool | None:
        return None if self.recall_at_5 is None else self.recall_at_5 == 1.0

    @property
    def composition(self) -> bool | None:
        if self.numbers is None:
            return None
        return self.numbers and self.entailed == self.segments

    @property
    def correct(self) -> bool:
        return (
            self.produced == self.expected
            and self.tools
            and self.retrieval is not False
            and self.composition is not False
        )

    @property
    def answered_when_it_should_have_declined(self) -> bool:
        return self.expected in DECLINES and self.produced == ExpectedKind.ANSWER.value


def ranked_documents(turn: dict[str, Any]) -> list[str]:
    """The documents a turn retrieved, best first, each once."""
    trace = (turn.get("trace") or {}).get("documents") or {}
    rows = sorted(trace.get("retrieved", []), key=lambda row: row["rank"])
    return list(dict.fromkeys(row["document_id"] for row in rows))


def recall_at(ranked: Sequence[str], relevant: Sequence[str], top: int = TOP) -> float:
    return len(set(ranked[:top]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: Sequence[str], top: int = TOP) -> float:
    return next((1 / rank for rank, doc in enumerate(ranked[:top], 1) if doc in relevant), 0.0)


def numbers_are_facts(turn: dict[str, Any]) -> bool:
    """Whether every number in the composed text is a count a citation carries."""
    for segment in turn["segments"]:
        text: str = segment["text"]
        for citation in segment["citations"]:
            if citation.get("count") is not None and citation["label"] != str(citation["count"]):
                return False
        cited = [citation["label"] for citation in segment["citations"]]
        quoted = [quote for passage in segment["passages"] for quote in passage["quotes"]]
        for known in sorted([*cited, *quoted], key=len, reverse=True):
            text = text.replace(known, " ")
        if re.search(r"\d", text):
            return False
    return True


def entailed(turn: dict[str, Any], nli: Nli) -> tuple[int, int]:
    """How many segments cite a passage, and how many of those their passage entails."""
    scored = kept = 0
    for segment in turn["segments"]:
        if not segment["passages"]:
            continue
        scored += 1
        best = max(
            nli.entailment(passage["text"], segment["text"]) for passage in segment["passages"]
        )
        kept += best >= THRESHOLD
    return scored, kept


def score(
    asked: Asked, nli: Nli, *, set_name: str = "template", reranked: Sequence[str] | None = None
) -> Scored:
    """One turn against its gold. ``reranked`` is what the ranker's retrieval returned."""
    question, turn = asked.question, asked.turn
    verdict = compare(asked)
    raw = kind_of(turn) if asked.status == 200 else None
    answered = raw is ExpectedKind.ANSWER

    numbers = numbers_are_facts(turn) if answered else None
    segments, kept = entailed(turn, nli) if answered else (0, 0)
    produced = raw.value if raw else UNAVAILABLE
    if answered and segments and not kept:
        produced = ExpectedKind.ABSTENTION.value

    relevant = question.expected.documents or []
    recall = rank = again = None
    if relevant:
        ranked = ranked_documents(turn) if asked.status == 200 else []
        recall, rank = recall_at(ranked, relevant), reciprocal_rank(ranked, relevant)
        if reranked is not None:
            again = recall_at(reranked, relevant)

    candidates = None
    if question.expected.kind is ExpectedKind.CLARIFICATION and raw is ExpectedKind.CLARIFICATION:
        candidates = not any(found.startswith("candidates") for found in verdict.mismatches)

    return Scored(
        question=question.id,
        set=set_name,
        pair=question.pair,
        persona=question.persona.value,
        shape=question.shape,
        expected=question.expected.kind.value,
        produced=produced,
        tools=verdict.ok,
        mismatches=verdict.mismatches,
        recall_at_5=recall,
        reciprocal_rank=rank,
        reranked_recall_at_5=again,
        numbers=numbers,
        segments=segments,
        entailed=kept,
        candidates=candidates,
    )
