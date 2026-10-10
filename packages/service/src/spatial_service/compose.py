"""Composition: the answer in words, with every number and room label filled in from facts.

The composing model never sees a label, a name or a number. It is given facts by
reference ({f1}, {f2}, ...) with what kind each is, and writes segments that name
things only through those placeholders, each citing the facts it rests on. The host
checks what comes back and fills the placeholders from the facts. Output carrying a
digit is rejected, whatever it says: a number the model typed is a number nobody
checked.

A fact is a place (filled with its label), an asset type (its name), one asset (its
name, followed by "(generated)" when it is) or a count. A
count is always of one record provenance. Where sourced and generated assets mix,
there are two count facts and no fact for their sum, so the model has no way to
print one number over both. A generated count fills in as "3 generated", and its
sourced partner as "1 sourced", so the flag is the host's and cannot be left out.
The two go together: an answer naming one without the other is rejected.

Each answer builder also says which facts the answer has to name. An answer that
leaves one out is rejected too.

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

_PLACEHOLDER = re.compile(r"\{(f[0-9]+)\}")

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
    ASSET = "asset"
    COUNT = "count"
    # What a filter tested, in the host's words: "no serial number recorded".
    CONDITION = "condition"


class Citation(_Model):
    kind: FactKind = FactKind.PLACE
    space_id: str | None = None
    type_id: str | None = None
    asset_id: str | None = None
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
    # The attribute a condition tests.
    attribute: str | None = None


POSITION_RECORDED = "recorded"
POSITION_NOT_RECORDED = "position not recorded"


def asset_citation(asset: Asset) -> Citation:
    """One asset, as a citation. A generated one says so in its label."""
    generated = asset.record_provenance is not RecordProvenance.SOURCED
    return Citation(
        kind=FactKind.ASSET,
        space_id=asset.space_id,
        type_id=asset.type_id,
        asset_id=asset.id,
        label=f"{asset.name.value} ({GENERATED})" if generated else asset.name.value,
        provenance=asset.name.provenance.value,
        rule=asset.name.rule,
        basis=asset.name.basis,
        record_provenance=asset.record_provenance.value,
    )


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

    def __init__(self, items: Iterable[Fact] = ()) -> None:
        # Given an answer's facts, more are numbered on from them.
        self.items: list[Fact] = list(items)
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

    def asset(self, asset: Asset) -> str:
        citation = asset_citation(asset)
        position = POSITION_RECORDED if asset.position else POSITION_NOT_RECORDED
        return self._add(FactKind.ASSET, citation.label, citation, told={"position": position})

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

    def condition(self, text: str, citation: Citation, **told: str) -> str:
        """What a filter tested, worded by the host so the model cannot reword it."""
        return self._add(FactKind.CONDITION, text, citation, told=told)

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
    # Words the answer may not use, whole and whatever their case.
    banned: tuple[str, ...] = ()


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


def asset_location(venue: Venue, question: str, asset: Asset) -> Answer:
    """One asset, then the space it is in and each labelled place enclosing that."""
    facts = Facts()
    subject = facts.asset(asset)
    home = venue.spaces[asset.space_id]
    for space in [home, *venue.ancestors(home)]:
        if space.label:
            facts.place(space)
    payload = {
        "question": question,
        "shape": "asset-location",
        "subject": subject,
        "required": [subject],
        "facts": facts.told(),
    }
    return Answer(facts.items, payload, required=[subject])


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


def _scope(venue: Venue, facts: Facts, found: ScopeResult) -> str | None:
    """The scope as a fact, where it has a label to print. The whole venue is its root."""
    space = venue.spaces[found.scope] if found.scope is not None else venue.root
    return facts.place(space) if space is not None and space.label else None


class CompositionRejected(Exception):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class Segment(_Model):
    text: str
    citations: list[Citation]


def fill(
    text: str,
    facts: list[Fact],
    required: Iterable[str] = (),
    together: Iterable[list[str]] = (),
    banned: Iterable[str] = (),
) -> list[Segment]:
    """Turn a model's raw output into segments, or raise CompositionRejected."""
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
    for number, segment in enumerate(draft.segments, start=1):
        # Placeholders carry digits of their own ({f1}); everything around them may not.
        if any(character.isdigit() for character in _PLACEHOLDER.sub("", segment.text)):
            reasons.append(f"segment {number} contains a digit")
        used = set(_PLACEHOLDER.findall(segment.text))
        named |= used
        for ref in sorted((used | set(segment.cites)) - by_ref.keys()):
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
    for segment in draft.segments:
        refs = dict.fromkeys([*_PLACEHOLDER.findall(segment.text), *segment.cites])
        segments.append(
            Segment(
                text=_PLACEHOLDER.sub(lambda m: by_ref[m[1]].text, segment.text),
                citations=[by_ref[ref].citation for ref in refs],
            )
        )
    return segments
