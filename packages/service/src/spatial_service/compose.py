"""Composition: the answer in words, with every number and room label filled in from facts.

The composing model never sees a label, a name or a number. It is given facts by
reference ({f1}, {f2}, ...) with what kind each is, and writes segments that name
things only through those placeholders, each citing the facts it rests on. The host
checks what comes back and fills the placeholders from the facts. Output carrying a
digit is rejected, whatever it says: a number the model typed is a number nobody
checked.

A fact is a place (filled with its label), an asset type (its name) or a count. A
count is always of one record provenance. Where sourced and generated assets mix,
there are two count facts and no fact for their sum, so the model has no way to
print one number over both. A generated count fills in as "3 generated", and its
sourced partner as "1 sourced", so the flag is the host's and cannot be left out.
The two go together: an answer naming one without the other is rejected.

Each answer builder also says which facts the answer has to name. An answer that
leaves one out is rejected too.

A document answer gets a second list beside the facts: the passages retrieved
this turn, by reference too (p1, p2, ...), with their text. The two lists never
mix. A passage is cited, never a placeholder, so nothing of it is filled in, and
a number can still only come from a fact: the digit rule holds for a document
segment as for any other, quoted or not. What a segment says of a passage is
checked for provenance here, and a segment that fails is dropped on its own, with
a note for the trace, where anything wrong with a fact still rejects the lot:

    it cites a passage that was not retrieved this turn
    it quotes a span ("...") that is not, word for word, in a passage it cites

Whether the passage backs what the segment says is not checked here. That is
entailment, the backend's half, which is why each segment carries the full text
of the passages it cites.

To add an answer: build its facts with Facts, return an Answer with the payload
the composer reads, and give the turn an instruction for the shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_core.records import Asset, AssetType, RecordProvenance, Space
from spatial_service.assets import Aggregation, ScopeResult, Tally
from spatial_service.places import Venue, tier
from spatial_service.retrieval import DroppedSegment, Passage, Retrieval, link

_PLACEHOLDER = re.compile(r"\{(f[0-9]+)\}")
_PASSAGE_REF = re.compile(r"p[0-9]+")
# A quoted span: straight or curly double quotes.
_QUOTED = re.compile(r"\"([^\"]+)\"|“([^”]+)”")

# How many rows an answer spells out. The rest are counted, not named.
ROW_CAP = 12

GENERATED = "generated"
SOURCED = "sourced"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftSegment(_Model):
    text: str = Field(min_length=1, max_length=500)
    cites: list[str]


class Draft(_Model):
    """What the composing model returns."""

    segments: list[DraftSegment]


class FactKind(StrEnum):
    PLACE = "place"
    ASSET_TYPE = "asset-type"
    COUNT = "count"


class Citation(_Model):
    kind: FactKind = FactKind.PLACE
    space_id: str | None = None
    type_id: str | None = None
    # What the placeholder was filled with.
    label: str
    # Value provenance of a label or a type's name, and the rule or basis behind it.
    provenance: str | None = None
    rule: str | None = None
    basis: str | None = None
    # Record provenance: of the place or type, or of the assets a count covers.
    record_provenance: str | None = None
    # The number a count stands for.
    count: int | None = None


@dataclass(frozen=True)
class Fact:
    ref: str
    kind: FactKind
    # What its placeholder becomes.
    text: str
    citation: Citation
    # What the composer is told about it, besides its ref and kind.
    told: dict[str, str] = field(default_factory=dict)
    space: Space | None = None


class Facts:
    """The facts of one answer, numbered as they are added. A place or type is added once."""

    def __init__(self) -> None:
        self.items: list[Fact] = []
        self._seen: dict[tuple[str, str], str] = {}

    def _add(self, kind: FactKind, text: str, citation: Citation, **more: Any) -> str:
        ref = f"f{len(self.items) + 1}"
        self.items.append(Fact(ref, kind, text, citation, **more))
        return ref

    def place(self, space: Space) -> str:
        key = ("place", space.id)
        if key not in self._seen:
            assert space.label is not None
            self._seen[key] = self._add(
                FactKind.PLACE,
                space.label.value,
                Citation(
                    space_id=space.id,
                    label=space.label.value,
                    provenance=space.label.provenance.value,
                    rule=space.label.rule,
                    basis=space.label.basis,
                    record_provenance=space.record_provenance.value,
                ),
                told={"tier": tier(space)},
                space=space,
            )
        return self._seen[key]

    def asset_type(self, kind: AssetType) -> str:
        key = ("type", kind.id)
        if key not in self._seen:
            self._seen[key] = self._add(
                FactKind.ASSET_TYPE,
                kind.name.value,
                Citation(
                    kind=FactKind.ASSET_TYPE,
                    type_id=kind.id,
                    label=kind.name.value,
                    provenance=kind.name.provenance.value,
                    rule=kind.name.rule,
                    basis=kind.name.basis,
                    record_provenance=kind.record_provenance.value,
                ),
            )
        return self._seen[key]

    def number(self, count: int, *, about: str, space_id: str | None = None) -> str:
        """A plain number that counts something other than assets of one provenance."""
        return self._add(
            FactKind.COUNT,
            str(count),
            Citation(kind=FactKind.COUNT, space_id=space_id, label=str(count), count=count),
            told={"about": about},
        )

    def counts(
        self, tally: Tally, *, space_id: str | None = None, type_id: str | None = None
    ) -> list[str]:
        """A tally as count facts: one per provenance present, generated first. Zero is one fact."""
        parts: list[tuple[int, RecordProvenance | None, str]] = []
        if tally.generated:
            parts.append((tally.generated, RecordProvenance.SEED, f"{tally.generated} {GENERATED}"))
        if tally.sourced:
            # Said to be sourced only next to a generated count. Alone, it is just the count.
            text = f"{tally.sourced} {SOURCED}" if tally.generated else str(tally.sourced)
            parts.append((tally.sourced, RecordProvenance.SOURCED, text))
        if not parts:
            parts.append((0, None, "0"))
        refs = []
        for count, record, text in parts:
            told = {} if record is None else {"assets": _FLAG[record]}
            refs.append(
                self._add(
                    FactKind.COUNT,
                    text,
                    Citation(
                        kind=FactKind.COUNT,
                        space_id=space_id,
                        type_id=type_id,
                        label=text,
                        record_provenance=record.value if record else None,
                        count=count,
                    ),
                    told=told,
                )
            )
        return refs

    def told(self) -> list[dict[str, str]]:
        return [{"ref": f.ref, "kind": f.kind.value, **f.told} for f in self.items]


_FLAG = {RecordProvenance.SEED: GENERATED, RecordProvenance.SOURCED: SOURCED}


@dataclass(frozen=True)
class Row:
    """One line of an answer's breakdown, as numbers, for the trace."""

    tally: Tally
    space_id: str | None = None
    type_id: str | None = None
    label: str | None = None
    # False for a space with no recorded installations, which is never ranked.
    recorded: bool = True
    rank: int | None = None


@dataclass(frozen=True)
class Answer:
    facts: list[Fact]
    # Everything the composing model is told. References and kinds, never values.
    payload: dict[str, Any]
    # Facts the answer has to name.
    required: list[str]
    # Count facts that are two halves of one tally: named together or not at all.
    together: list[list[str]] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)
    # The passages retrieved this turn. The only ones a segment may cite.
    passages: tuple[Passage, ...] = ()


def location(venue: Venue, question: str, subject: Space) -> Answer:
    """The place asked about, then each labelled place enclosing it, innermost first."""
    facts = Facts()
    for space in [subject, *venue.ancestors(subject)]:
        if space.label:
            facts.place(space)
    payload = {
        "question": question,
        "subject": facts.items[0].ref,
        "facts": [{"ref": f.ref, "tier": f.told["tier"]} for f in facts.items],
    }
    return Answer(facts.items, payload, required=[facts.items[0].ref])


def containment(venue: Venue, question: str, found: ScopeResult) -> Answer:
    """What is installed under a scope: a total, then rows by type, or by space for given types."""
    facts = Facts()
    assets = [venue.assets[i] for i in found.ids]
    scope = _scope(venue, facts, found)
    total = facts.counts(Tally.of(assets), space_id=found.scope)

    by_space = found.types is not None
    grouped: dict[str, list[Asset]] = {}
    for asset in assets:
        grouped.setdefault(asset.space_id if by_space else asset.type_id, []).append(asset)

    rows: list[Row] = []
    for key, members in grouped.items():
        tally = Tally.of(members)
        if by_space:
            label = venue.spaces[key].label
            rows.append(Row(tally, space_id=key, label=label.value if label else None))
        else:
            rows.append(Row(tally, type_id=key, label=venue.asset_types[key].name.value))
    rows.sort(key=lambda r: (-r.tally.total, r.label or "", r.space_id or r.type_id or ""))

    payload: dict[str, Any] = {
        "question": question,
        "shape": "asset-containment",
        "scope": scope,
        "total": total,
        "rows_by": "space" if by_space else "asset-type",
    }
    together = [total]
    told_rows = []
    # A space no rule can address still holds its assets. It has no label to print,
    # so those go in one row of their own.
    nameable = [r for r in rows if r.label is not None]
    for row in nameable[:ROW_CAP]:
        if row.space_id is not None:
            of = facts.place(venue.spaces[row.space_id])
        else:
            assert row.type_id is not None
            of = facts.asset_type(venue.asset_types[row.type_id])
        counts = facts.counts(row.tally, space_id=row.space_id or found.scope, type_id=row.type_id)
        told_rows.append({"of": of, "counts": counts})
        together.append(counts)
    payload["rows"] = told_rows
    if len(nameable) > ROW_CAP:
        payload["more_rows"] = facts.number(len(nameable) - ROW_CAP, about="rows left out")
    unlabelled = Tally.of(a for r in rows if r.label is None for a in grouped[r.space_id or ""])
    if unlabelled.total:
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled)
        together.append(payload["in_spaces_with_no_label"])
    unplaced = sum(1 for asset in assets if asset.position is None)
    if unplaced:
        payload["position_not_recorded"] = facts.number(
            unplaced, about="assets whose position is not recorded", space_id=found.scope
        )

    required = [*([scope] if scope else []), *total]
    payload["required"] = required
    payload["facts"] = facts.told()
    return Answer(facts.items, payload, required, together, rows)


def aggregation(venue: Venue, question: str, found: ScopeResult, counted: Aggregation) -> Answer:
    """Counts per space of a scale level: ranked, leaders named, gaps apart from zeros."""
    facts = Facts()
    scope = _scope(venue, facts, found)
    total = facts.counts(Tally.of(venue.assets[i] for i in found.ids), space_id=found.scope)

    rows = []
    for group in counted.ranked:
        label = venue.spaces[group.space_id].label
        rows.append(
            Row(
                group.tally,
                space_id=group.space_id,
                label=label.value if label else None,
                rank=group.rank,
            )
        )
    for space_id in counted.unrecorded:
        label = venue.spaces[space_id].label
        rows.append(
            Row(Tally(), space_id=space_id, label=label.value if label else None, recorded=False)
        )

    together = [total]
    told_rows = []
    by_space: dict[str, dict[str, Any]] = {}
    nameable = [r for r in rows if r.recorded and r.label is not None]
    for row in nameable[:ROW_CAP]:
        assert row.space_id is not None
        of = facts.place(venue.spaces[row.space_id])
        counts = facts.counts(row.tally, space_id=row.space_id)
        by_space[row.space_id] = {"of": of, "counts": counts}
        told_rows.append(by_space[row.space_id])
        together.append(counts)

    # The most, among the spaces that can be named. Ties share it.
    top = nameable[0].tally.total if nameable else 0
    leaders = [
        by_space[r.space_id or ""] for r in nameable[:ROW_CAP] if top and r.tally.total == top
    ]
    gaps = [r for r in rows if not r.recorded and r.label is not None]
    gap_refs = [facts.place(venue.spaces[r.space_id or ""]) for r in gaps[:ROW_CAP]]

    payload: dict[str, Any] = {
        "question": question,
        "shape": "asset-aggregation",
        "scope": scope,
        "grouped_by": counted.by,
        "total": total,
        "rows": told_rows,
        "leaders": [leader["of"] for leader in leaders],
        "no_recorded_installations": gap_refs,
    }
    required = [ref for leader in leaders for ref in (leader["of"], *leader["counts"])]
    if not leaders:
        required = list(total)
    required += gap_refs
    if len(nameable) > ROW_CAP:
        payload["more_rows"] = facts.number(len(nameable) - ROW_CAP, about="rows left out")
    if len(gaps) > ROW_CAP:
        payload["more_no_recorded_installations"] = facts.number(
            len(gaps) - ROW_CAP, about="more spaces with no recorded installations"
        )
        required.append(payload["more_no_recorded_installations"])
    unlabelled = Tally(
        sum(r.tally.sourced for r in rows if r.label is None),
        sum(r.tally.generated for r in rows if r.label is None),
    )
    if unlabelled.total:
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled)
        together.append(payload["in_spaces_with_no_label"])

    payload["required"] = required
    payload["facts"] = facts.told()
    return Answer(facts.items, payload, required, together, rows)


def asset_documents(
    venue: Venue, question: str, found: ScopeResult, retrieved: Retrieval
) -> Answer:
    """The assets a question is about, as containment gives them, and what was retrieved."""
    base = containment(venue, question, found)
    payload = {
        **base.payload,
        "shape": "asset-document",
        "passages": _told_passages(retrieved),
    }
    return Answer(base.facts, payload, base.required, base.together, base.rows, retrieved.passages)


def place_documents(venue: Venue, question: str, subject: Space, retrieved: Retrieval) -> Answer:
    """The place a question is about, and what was retrieved for it."""
    base = location(venue, question, subject)
    payload = {
        **base.payload,
        "shape": "place-document",
        "required": base.required,
        "passages": _told_passages(retrieved),
    }
    return Answer(base.facts, payload, base.required, passages=retrieved.passages)


def _told_passages(retrieved: Retrieval) -> list[dict[str, str]]:
    """What the composer reads of each passage: its words. No id, no link, no date."""
    return [
        {"ref": p.ref, "kind": p.chunk.kind.value, "title": p.chunk.title, "text": p.chunk.text}
        for p in retrieved.passages
    ]


def _scope(venue: Venue, facts: Facts, found: ScopeResult) -> str | None:
    """The scope as a fact, where it has a label to print. The whole venue is its root."""
    space = venue.spaces[found.scope] if found.scope is not None else venue.root
    return facts.place(space) if space is not None and space.label else None


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


def fill(
    text: str,
    facts: list[Fact],
    required: Iterable[str] = (),
    together: Iterable[list[str]] = (),
) -> list[Segment]:
    """Turn a model's raw output into segments, or raise CompositionRejected."""
    segments, _ = ground(text, facts, required, together)
    return segments


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


def ground(
    text: str,
    facts: list[Fact],
    required: Iterable[str] = (),
    together: Iterable[list[str]] = (),
    passages: Iterable[Passage] = (),
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
            if failures:
                dropped[number] = failures
        # Placeholders carry digits of their own ({f1}); everything around them may not.
        if any(character.isdigit() for character in _PLACEHOLDER.sub("", segment.text)):
            reasons.append(f"segment {number} contains a digit")
        used = set(_PLACEHOLDER.findall(segment.text))
        named |= used
        for ref in sorted((used | set(segment.cites)) - by_ref.keys()):
            # In a document answer a passage reference is provenance's to judge.
            if not (by_passage and _PASSAGE_REF.fullmatch(ref)):
                reasons.append(f"segment {number} names {ref}, which is not a fact")
        if not segment.cites and not used:
            reasons.append(f"segment {number} cites nothing")
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
        refs = dict.fromkeys([*_PLACEHOLDER.findall(segment.text), *segment.cites])
        segments.append(
            Segment(
                text=_PLACEHOLDER.sub(lambda m: by_ref[m[1]].text, segment.text),
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
