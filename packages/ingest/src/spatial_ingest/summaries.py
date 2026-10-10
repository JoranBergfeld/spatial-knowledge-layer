"""SAC summaries: one short summary per document, built one clearance class at a time.

A summary is written from a document's own English passages and nothing else.
A document sits in exactly one clearance class, so a summary never mixes
classes, and build walks the classes one by one so that stays true by
construction. Load checks it again from the bundle.

A summary does two things, and both only shape retrieval:

- it goes in front of each of its document's passages when the passage is
  embedded (that is the summary-augmented chunk)
- it goes in the index as a chunk of its own, with role "summary", carrying its
  document's anchors and clearance class

It is never evidence. Nothing quotes it, cites it or checks it for entailment.

A document longer than WINDOW characters is summarised part by part, and the
part summaries are summarised into one. Parts stay in the bundle, not the index.

Summaries are a model's words, so they are kept in documents/summaries.jsonl
with the deployment, model and prompt version that made them. A rebuild reads
them from there and load only ever reads them.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from spatial_core.records import PERSONAS, SUMMARY_ORDINAL, Chunk, ChunkRole, Persona
from spatial_ingest.models import Models

# Bump the version whenever any of the three prompts changes. A stored summary
# made under another version is made again.
PROMPT_VERSION = "summarise-document@1"
_AUDIENCE = (
    "You are indexing documents about a venue: its accessibility pages and the manuals of "
    "equipment installed there."
)
_STYLE = "Use only what the text says. Plain prose, no preamble. Return only the summary."
PROMPTS = {
    "document": (
        f"Summarise the user's document in at most 60 words. {_AUDIENCE} Say what kind of "
        "document it is, which place, equipment or product it covers, and the manufacturer or "
        f"model if named. {_STYLE}"
    ),
    "part": (
        f"Summarise the user's text in at most 60 words. {_AUDIENCE} The text is one part of a "
        "longer document. Say what kind of content it is and which place, equipment or product "
        f"it covers. {_STYLE}"
    ),
    "reduce": (
        f"Summarise the user's text in at most 60 words. {_AUDIENCE} The text is a list of "
        "summaries of consecutive parts of one document. Write one summary of the whole "
        "document: what kind of document it is, which place, equipment or product it covers, "
        f"and the manufacturer or model if named. {_STYLE}"
    ),
}

PART = "documents/summaries.jsonl"
# About 5,000 tokens of English, the window the SAC measurement used. In characters,
# since build has no tokenizer.
WINDOW = 20_000


class SummaryError(Exception):
    pass


class Prompt(StrEnum):
    # The whole document in one call.
    DOCUMENT = "document"
    # One window of a long document.
    PART = "part"
    # The part summaries of a long document into one.
    REDUCE = "reduce"


class Summary(BaseModel):
    """One summary as the model wrote it, and what it was written from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # "<document_id>#summary", or "<document_id>#summary-part-<n>" for a part.
    id: str
    document_id: str
    prompt: Prompt
    # The class of every passage it was made from.
    clearance_class: Persona
    # The passages the model read, or for a reduce the passages its parts read.
    source_chunk_ids: tuple[str, ...]
    # Of the text the model was shown.
    source_sha256: str
    text: str
    deployment: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int

    @property
    def indexed(self) -> bool:
        """Whether it stands for the whole document, which is what the index holds."""
        return self.prompt is not Prompt.PART


def summary_id(document_id: str) -> str:
    return f"{document_id}#summary"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _windows(passages: Sequence[Chunk]) -> list[list[Chunk]]:
    """Consecutive passages, grouped so each group fits the window where it can."""
    groups: list[list[Chunk]] = []
    size = 0
    for passage in passages:
        if groups and size + len(passage.text) <= WINDOW:
            groups[-1].append(passage)
            size += len(passage.text)
        else:
            groups.append([passage])
            size = len(passage.text)
    return groups


def _shown(title: str, passages: Sequence[Chunk]) -> str:
    return f"{title}\n\n" + "\n\n".join(passage.text for passage in passages)


def build(
    chunks: Sequence[Chunk], known: Iterable[Summary], model: Callable[[], Models]
) -> list[Summary]:
    """A summary for every document that has passages, parts included, in id order.

    ``known`` is what the bundle already holds. ``model`` is only called for a
    summary that is not in there.
    """
    have = {
        (made.prompt, made.source_sha256): made
        for made in known
        if made.prompt_version == PROMPT_VERSION
    }
    summaries: list[Summary] = []

    def summarise(
        summary: str, cleared: Persona, prompt: Prompt, sources: Sequence[Chunk], shown: str
    ) -> Summary:
        # The one place a model reads text to summarise, so the one place to refuse
        # anything from another class.
        strays = [source.id for source in sources if source.clearance_class is not cleared]
        if strays:
            raise SummaryError(f"{summary} would be built across clearance classes: {strays}")
        source_hash = _sha256(shown)
        made = have.get((prompt, source_hash))
        fields: dict[str, Any] = {
            "id": summary,
            "document_id": sources[0].document_id,
            "clearance_class": cleared,
            "source_chunk_ids": tuple(source.id for source in sources),
        }
        if made is None:
            answer = model().summarise(PROMPTS[prompt.value], shown)
            made = Summary(
                **fields,
                prompt=prompt,
                source_sha256=source_hash,
                text=answer.text,
                deployment=answer.deployment,
                model=answer.model,
                prompt_version=PROMPT_VERSION,
                input_tokens=answer.input_tokens,
                output_tokens=answer.output_tokens,
            )
        made = made.model_copy(update=fields)
        summaries.append(made)
        return made

    for cleared in PERSONAS:
        documents: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            if chunk.clearance_class is cleared and chunk.role is ChunkRole.PASSAGE:
                documents.setdefault(chunk.document_id, []).append(chunk)
        for document_id, passages in sorted(documents.items()):
            passages.sort(key=lambda passage: passage.ordinal)
            title = passages[0].title
            windows = _windows(passages)
            if len(windows) == 1:
                summarise(
                    summary_id(document_id),
                    cleared,
                    Prompt.DOCUMENT,
                    passages,
                    _shown(title, passages),
                )
                continue
            parts = [
                summarise(
                    f"{summary_id(document_id)}-part-{number}",
                    cleared,
                    Prompt.PART,
                    window,
                    _shown(title, window),
                )
                for number, window in enumerate(windows)
            ]
            listed = "\n".join(f"{number}. {part.text}" for number, part in enumerate(parts, 1))
            summarise(
                summary_id(document_id), cleared, Prompt.REDUCE, passages, f"{title}\n\n{listed}"
            )
    return sorted(summaries, key=lambda made: made.id)


def as_chunk(summary: Summary, passage: Chunk) -> Chunk:
    """The summary as the index holds it: its document's chunk in everything but the text.

    ``passage`` is any passage of its document. The summary takes its anchors,
    so whoever may retrieve the passages may retrieve the summary, and nobody else.
    """
    return passage.model_copy(
        update={
            "id": summary.id,
            "role": ChunkRole.SUMMARY,
            "ordinal": SUMMARY_ORDINAL,
            "text": summary.text,
            "language": "en",
            "original_text_sha256": summary.source_sha256,
            "translated": False,
            "clearance_class": summary.clearance_class,
        }
    )


def chunks_of(summaries: Iterable[Summary], passages: Iterable[Chunk]) -> list[Chunk]:
    """The summary chunks for the index, in id order."""
    first: dict[str, Chunk] = {}
    for passage in passages:
        first.setdefault(passage.document_id, passage)
    made = []
    for summary in summaries:
        if not summary.indexed:
            continue
        if summary.document_id not in first:
            raise SummaryError(f"{summary.id} summarises a document with no passages")
        made.append(as_chunk(summary, first[summary.document_id]))
    return sorted(made, key=lambda chunk: chunk.id)


def by_document(summaries: Iterable[Summary]) -> dict[str, Summary]:
    return {summary.document_id: summary for summary in summaries if summary.indexed}


def manifest(summaries: Sequence[Summary]) -> dict[str, Any]:
    """What the manifest says about the summaries."""
    indexed = [summary for summary in summaries if summary.indexed]
    made_by: dict[tuple[str, str, str], int] = {}
    for summary in summaries:
        key = (summary.deployment, summary.model, summary.prompt_version)
        made_by[key] = made_by.get(key, 0) + 1
    return {
        # One per document. These are the ones the index holds.
        "documents": len(indexed),
        # Windows of documents too long for one call. Bundle only.
        "parts": len(summaries) - len(indexed),
        "by_clearance_class": {
            persona.value: sum(1 for summary in indexed if summary.clearance_class is persona)
            for persona in PERSONAS
        },
        "made_by": [
            {
                "deployment": deployment,
                "model": model,
                "prompt_version": version,
                "summaries": count,
            }
            for (deployment, model, version), count in sorted(made_by.items())
        ],
    }
