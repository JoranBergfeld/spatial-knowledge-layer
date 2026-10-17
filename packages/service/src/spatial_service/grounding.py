"""Grounding: the composer's draft checked, and its placeholders filled from the facts.

Output carrying a digit is rejected, whatever it says: a number the model typed is
a number nobody checked.

A count of assets says what it counts, and the host says it, not the model. Where
a segment does not name the type a count is of, the type's name goes in after the
number, and the type's citation with it: "2 Expansion Tank". A count over several
types, or any, gets "assets". So "{f1} has {f2}." reads as a whole sentence and
the model never has to find a noun. A count placeholder straight before another
placeholder is rejected, unless that one is the type it counts: that is how the
first real run got "has 1 1" and "has 2 Medical-Dental Clinic". No plural for a
type's name, since "Equip-A1155" has none. Reads a bit stiff. It is right, though.

A document answer gets a second list beside the facts: the passages retrieved
this turn, by reference too (p1, p2, ...), with their text. The two lists never
mix. A passage is cited, never a placeholder, so nothing of it is filled in, and
a number can still only come from a fact: the digit rule holds for a document
segment as for any other, quoted or not. What a segment says of a passage is
checked for provenance here, and a segment that fails is dropped on its own, with
a note for the trace, where anything wrong with a fact still rejects the lot:

    it cites a passage that was not retrieved this turn
    it quotes a span ("...") that is not, word for word, in a passage it cites
    it names a fact and cites a passage too. A segment rests on facts or on
        passages, never both: a passage cannot back a count or "X is within Y",
        and the entailment check rightly drops it when asked to
    it says what the documents do not say ("the documents say nothing about X").
        Nothing can entail an absence. With no passage that answers, there is
        no document answer, and the turn abstains (turn.py)
    it writes a digit. A digit anywhere else still rejects the lot

Whether the passage backs what the segment says is not checked here. That is
entailment, the backend's half, which is why each segment carries the full text
of the passages it cites.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_service.facts import ANY_ASSETS, Citation, Fact, FactKind
from spatial_service.retrieval import DroppedSegment, Passage, link

_PLACEHOLDER = re.compile(r"\{(f[0-9]+)\}")
# The model writing the noun anyway, right after the count: not said twice.
_NOUN_WRITTEN = re.compile(r"\s+(?:assets?|items?)\b", re.IGNORECASE)
# A segment about what the documents leave out. Short on purpose: the wordings seen,
# and their near neighbours. "The manual says do not open it" is not one.
_SAYS_NOTHING = re.compile(
    r"\b(?:do|does|did)(?:\s+not|n['’]t)\s+(?:say|mention|cover|contain|describe|include|"
    r"give|provide|address|discuss|specify|state)\b"
    r"|\bsays?\s+nothing\b|\bnothing\s+(?:about|on|regarding)\b|\bnot\s+mentioned\b"
    r"|\bno\s+(?:information|mention|details?)\b"
    r"|\b(?:no|none\s+of\s+the)\s+(?:documents?|passages?|manuals?)\b",
    re.IGNORECASE,
)
_PASSAGE_REF = re.compile(r"p[0-9]+")
# A quoted span: straight or curly double quotes.
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftSegment(_Model):
    text: str = Field(min_length=1, max_length=500)
    cites: list[str]


class Draft(_Model):
    """What the composing model returns."""

    segments: list[DraftSegment]


class CompositionRejected(Exception):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class PassageCitation(_Model):
    """A passage a segment cites, with its whole text, for the entailment check elsewhere."""

    ref: str
    chunk_id: str
    document_id: str
    kind: str
    title: str
    # The ingested English text: the translation, where the original was not English.
    text: str
    # The spans the segment quotes from it, each found in the text word for word.
    quotes: list[str]
    # Where the original lives. None when the source was offline at build: then
    # it is cited unlinked, as of the day build checked.
    url: str | None
    link_status: str
    link_checked_on: str
    translated: bool
    # The original's language.
    language: str


class Segment(_Model):
    text: str
    citations: list[Citation]
    # The passages a document segment cites. Empty for a segment of facts alone.
    passages: list[PassageCitation] = []


def _squash(text: str) -> str:
    return " ".join(text.split())


def _ungrounded(
    segment: DraftSegment, by_passage: dict[str, Passage]
) -> tuple[list[str], dict[str, list[str]]]:
    """Why a segment fails provenance, and the spans it quotes, by the passage each is in."""
    reasons = []
    cited = [ref for ref in segment.cites if _PASSAGE_REF.fullmatch(ref)]
    for ref in cited:
        if ref not in by_passage:
            reasons.append(f"cites {ref}, which is not a passage retrieved this turn")
    known = [ref for ref in dict.fromkeys(cited) if ref in by_passage]
    quotes: dict[str, list[str]] = {ref: [] for ref in known}
    for match in _QUOTED.finditer(segment.text):
        span = _squash(match[1] or match[2])
        home = next((ref for ref in known if span in _squash(by_passage[ref].chunk.text)), None)
        if home is None:
            reasons.append("quotes text that is not in a passage it cites")
        else:
            quotes[home].append(span)
    return reasons, quotes


def _off_the_passages(segment: DraftSegment, by_ref: dict[str, Fact]) -> list[str]:
    """Why a segment of a document answer is not about what the passages say."""
    reasons = []
    names = set(_PLACEHOLDER.findall(segment.text)) | (set(segment.cites) & by_ref.keys())
    if names and any(_PASSAGE_REF.fullmatch(ref) for ref in segment.cites):
        reasons.append("names a fact and cites a passage, and a segment rests on one or the other")
    if _SAYS_NOTHING.search(_PLACEHOLDER.sub("", segment.text)):
        reasons.append("says what the documents do not say")
    return reasons


def _jammed(text: str, by_ref: dict[str, Fact]) -> list[str]:
    """Count placeholders run straight into another placeholder that is not what they count."""
    reasons = []
    found = list(_PLACEHOLDER.finditer(text))
    for one, after in zip(found, found[1:], strict=False):
        fact = by_ref.get(one[1])
        if (
            fact is None
            or fact.kind is not FactKind.COUNT
            or text[one.end() : after.start()].strip()
        ):
            continue
        if after[1] != fact.noun:
            reasons.append(f"runs {{{one[1]}}} into {{{after[1]}}}: a count is followed by words")
    return reasons


def _filled(text: str, by_ref: dict[str, Fact]) -> tuple[str, list[str]]:
    """The segment with its placeholders filled, and the type facts a count put in for itself."""
    used = set(_PLACEHOLDER.findall(text))
    out: list[str] = []
    nouns: list[str] = []
    position = 0
    for match in _PLACEHOLDER.finditer(text):
        fact = by_ref[match[1]]
        out += [text[position : match.start()], fact.text]
        position = match.end()
        if fact.noun is None or fact.noun in used:
            continue
        if fact.noun == ANY_ASSETS:
            if not _NOUN_WRITTEN.match(text, position):
                out.append(" asset" if fact.citation.count == 1 else " assets")
        else:
            out.append(f" {by_ref[fact.noun].text}")
            nouns.append(fact.noun)
    out.append(text[position:])
    return "".join(out), nouns


def ground(
    text: str,
    facts: list[Fact],
    required: Iterable[str] = (),
    together: Iterable[list[str]] = (),
    passages: Iterable[Passage] = (),
    banned: Iterable[str] = (),
) -> tuple[list[Segment], list[DroppedSegment]]:
    """Segments from a model's raw output, and the ones provenance dropped.

    Raises CompositionRejected for anything wrong with a fact. With no passages
    nothing is ever dropped, and a quotation mark is just punctuation.
    """
    by_passage = {passage.ref: passage for passage in passages}
    try:
        draft = Draft.model_validate_json(text)
    except ValidationError as error:
        raise CompositionRejected(
            [f"{'.'.join(map(str, e['loc'])) or 'draft'}: {e['msg']}" for e in error.errors()]
        ) from error

    by_ref = {fact.ref: fact for fact in facts}
    reasons: list[str] = []
    if not draft.segments:
        reasons.append("no segments")
    named: set[str] = set()
    dropped: dict[int, list[str]] = {}
    quoted: dict[int, dict[str, list[str]]] = {}
    for number, segment in enumerate(draft.segments, start=1):
        if by_passage:
            failures, quoted[number] = _ungrounded(segment, by_passage)
            failures += _off_the_passages(segment, by_ref)
            if failures:
                dropped[number] = failures
        reasons += [f"segment {number} {why}" for why in _jammed(segment.text, by_ref)]
        # Placeholders carry digits of their own ({f1}); everything around them may not.
        # A passage segment with one goes alone: it is the model copying the passage's
        # numbers ("pH 700"), and the facts it would take down with it are fine.
        if any(character.isdigit() for character in _PLACEHOLDER.sub("", segment.text)):
            if by_passage and any(_PASSAGE_REF.fullmatch(ref) for ref in segment.cites):
                dropped.setdefault(number, []).append("writes a digit")
            else:
                reasons.append(f"segment {number} contains a digit")
        used = set(_PLACEHOLDER.findall(segment.text))
        named |= used
        for ref in sorted((used | set(segment.cites)) - by_ref.keys()):
            # In a document answer a passage reference is provenance's to judge.
            if not (by_passage and _PASSAGE_REF.fullmatch(ref)):
                reasons.append(f"segment {number} names {ref}, which is not a fact")
        if not segment.cites and not used:
            reasons.append(f"segment {number} cites nothing")
        for word in banned:
            if re.search(rf"\b{re.escape(word)}\b", segment.text, re.IGNORECASE):
                reasons.append(f"segment {number} says {word!r}")
        if "{" in _PLACEHOLDER.sub("", segment.text) or "}" in _PLACEHOLDER.sub("", segment.text):
            reasons.append(f"segment {number} has a malformed placeholder")
    for ref in required:
        if ref not in named:
            reasons.append(f"no segment names {ref}, which the answer has to")
    for halves in together:
        missing = [ref for ref in halves if ref not in named]
        if missing and len(missing) < len(halves):
            reasons.append(
                f"{', '.join(missing)} is left out, but it is one count with "
                f"{', '.join(ref for ref in halves if ref in named)}"
            )
    if reasons:
        raise CompositionRejected(reasons)

    segments = []
    for number, segment in enumerate(draft.segments, start=1):
        if number in dropped:
            continue
        text, nouns = _filled(segment.text, by_ref)
        refs = dict.fromkeys([*_PLACEHOLDER.findall(segment.text), *segment.cites, *nouns])
        segments.append(
            Segment(
                text=text,
                citations=[by_ref[ref].citation for ref in refs if ref in by_ref],
                passages=[
                    _passage_citation(by_passage[ref], quotes)
                    for ref, quotes in quoted.get(number, {}).items()
                ],
            )
        )
    notes = [DroppedSegment(segment=number, reasons=why) for number, why in dropped.items()]
    return segments, notes


def _passage_citation(passage: Passage, quotes: list[str]) -> PassageCitation:
    chunk = passage.chunk
    return PassageCitation(
        ref=passage.ref,
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        kind=chunk.kind.value,
        title=chunk.title,
        text=chunk.text,
        quotes=quotes,
        url=link(chunk),
        link_status=chunk.link_status.value,
        link_checked_on=chunk.link_checked_on,
        translated=chunk.translated,
        language=chunk.language,
    )
