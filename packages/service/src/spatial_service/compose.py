"""Composition: the answer in words, with every number and room label filled in from facts.

The composing model never sees a label, a name or a number. It is given facts by
reference ({f1}, {f2}, ...) with what kind each is, and writes segments that name
things only through those placeholders, each citing the facts it rests on. The host
checks what comes back and fills the placeholders from the facts. Output carrying a
digit is rejected, whatever it says: a number the model typed is a number nobody
checked.

A route's modelled duration and its distance are facts too. They fill in as
"41 seconds (modelled)" and "39 metres", so the word modelled is the host's as well.

A fact is a place (filled with its label), an asset type (its name), one asset (its
name, followed by "(generated)" when it is) or a count. A
count is always of one record provenance. Where sourced and generated assets mix,
there are two count facts and no fact for their sum, so the model has no way to
print one number over both. A generated count fills in as "3 generated", and its
sourced partner as "1 sourced", so the flag is the host's and cannot be left out.
The two go together: an answer naming one without the other is rejected.

A count of assets says what it counts, and the host says it, not the model. Where
a segment does not name the type a count is of, the type's name goes in after the
number, and the type's citation with it: "2 Expansion Tank". A count over several
types, or any, gets "assets". So "{f1} has {f2}." reads as a whole sentence and
the model never has to find a noun. A count placeholder straight before another
placeholder is rejected, unless that one is the type it counts: that is how the
first real run got "has 1 1" and "has 2 Medical-Dental Clinic". No plural for a
type's name, since "Equip-A1155" has none. Reads a bit stiff. It is right, though.

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
    it names a fact and cites a passage too. A segment rests on facts or on
        passages, never both: a passage cannot back a count or "X is within Y",
        and the entailment check rightly drops it when asked to
    it says what the documents do not say ("the documents say nothing about X").
        Nothing can entail an absence. With no passage that answers, there is
        no document answer, and the turn abstains (turn.py)

Whether the passage backs what the segment says is not checked here. That is
entailment, the backend's half, which is why each segment carries the full text
of the passages it cites.

To add an answer: build its facts with Facts, return an Answer with the payload
the composer reads, and give the turn an instruction for the shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_core.records import Asset, AssetType, RecordProvenance, Space
from spatial_service.assets import Aggregation, ScopeResult, Tally
from spatial_service.places import Venue, label, tier
from spatial_service.retrieval import DroppedSegment, Passage, Retrieval, link
from spatial_service.routes import Found, Nearest, Routed

_PLACEHOLDER = re.compile(r"\{(f[0-9]+)\}")
# What a count of assets of several types, or of any, is a count of.
ANY_ASSETS = "assets"
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
    # A modelled duration in seconds and a distance in metres, of one route.
    DURATION = "modelled-duration"
    DISTANCE = "distance"


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
    # What a modelled duration or a distance stands for.
    seconds: int | None = None
    metres: int | None = None


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
    # A count of assets: the ref of the type fact it counts, or ANY_ASSETS. What
    # follows its number where the segment does not name that type itself.
    noun: str | None = None


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
        self,
        tally: Tally,
        *,
        space_id: str | None = None,
        type_id: str | None = None,
        noun: str | None = None,
    ) -> list[str]:
        """A tally as count facts: one per provenance present, generated first. Zero is one fact.

        ``noun`` is what the assets are: a type fact's ref, or ANY_ASSETS.
        """
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
                    noun=noun,
                )
            )
        return refs

    def noun(self, venue: Venue, types: Iterable[str] | None) -> str:
        """What a count over ``types`` is of: the type, where there is one, else ANY_ASSETS."""
        kinds = sorted(set(types or ()))
        return self.asset_type(venue.asset_types[kinds[0]]) if len(kinds) == 1 else ANY_ASSETS

    def condition(self, text: str, citation: Citation, **told: str) -> str:
        """What a filter tested, worded by the host so the model cannot reword it."""
        return self._add(FactKind.CONDITION, text, citation, told=told)

    def duration(self, seconds: int, *, space_id: str) -> str:
        """A route's modelled duration. Says it is modelled, so nobody has to remember to."""
        text = f"{seconds} seconds (modelled)"
        return self._add(
            FactKind.DURATION,
            text,
            Citation(kind=FactKind.DURATION, space_id=space_id, label=text, seconds=seconds),
        )

    def distance(self, metres: int, *, space_id: str) -> str:
        text = f"{metres} metres"
        return self._add(
            FactKind.DISTANCE,
            text,
            Citation(kind=FactKind.DISTANCE, space_id=space_id, label=text, metres=metres),
        )

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
    noun = facts.noun(venue, found.types)
    total = facts.counts(Tally.of(assets), space_id=found.scope, noun=noun)

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
        counts = facts.counts(
            row.tally,
            space_id=row.space_id or found.scope,
            type_id=row.type_id,
            noun=noun if row.space_id is not None else of,
        )
        told_rows.append({"of": of, "counts": counts})
        together.append(counts)
    payload["rows"] = told_rows
    if len(nameable) > ROW_CAP:
        payload["more_rows"] = facts.number(len(nameable) - ROW_CAP, about="rows left out")
    unlabelled = Tally.of(a for r in rows if r.label is None for a in grouped[r.space_id or ""])
    if unlabelled.total:
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled, noun=noun)
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
    noun = facts.noun(venue, found.types)
    total = facts.counts(
        Tally.of(venue.assets[i] for i in found.ids), space_id=found.scope, noun=noun
    )

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
        counts = facts.counts(row.tally, space_id=row.space_id, noun=noun)
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
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled, noun=noun)
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
    return replace(base, payload=payload, passages=retrieved.passages)


def place_documents(venue: Venue, question: str, subject: Space, retrieved: Retrieval) -> Answer:
    """The place a question is about, and what was retrieved for it."""
    base = location(venue, question, subject)
    payload = {
        **base.payload,
        "shape": "place-document",
        "required": base.required,
        "passages": _told_passages(retrieved),
    }
    return replace(base, payload=payload, passages=retrieved.passages)


def _told_passages(retrieved: Retrieval) -> list[dict[str, str]]:
    """What the composer reads of each passage: its words. No id, no link, no date."""
    return [
        {"ref": p.ref, "kind": p.chunk.kind.value, "title": p.chunk.title, "text": p.chunk.text}
        for p in retrieved.passages
    ]


def _place(venue: Venue, facts: Facts, space_id: str) -> str | None:
    space = venue.spaces[space_id]
    return facts.place(space) if space.label else None


def _told_route(venue: Venue, facts: Facts, found: Found) -> dict[str, Any]:
    """One route by reference: its ends, what it passes through, how long and how far."""
    via: list[str] = []
    for leg in found.legs[1:-1]:
        ref = _place(venue, facts, leg.space_id)
        if ref is not None and ref not in via:
            via.append(ref)
    return {
        "destination": _place(venue, facts, found.destination),
        "duration": facts.duration(round(found.seconds), space_id=found.destination),
        "distance": facts.distance(round(found.metres), space_id=found.destination),
        "via": via[:ROW_CAP],
        "changes_level_by": sorted({leg.mode for leg in found.legs if leg.levels}),
    }


def route(venue: Venue, question: str, runs: Sequence[Routed]) -> Answer:
    """Each route found, in order: where from, where to, how long, how far."""
    facts = Facts()
    told = []
    required: list[str] = []
    for run in runs:
        best = run.found[0]
        origin = _place(venue, facts, best.origin)
        one = {
            "origin": origin,
            "origin_is_default": run.default_origin,
            "step_free": run.step_free,
            **_told_route(venue, facts, best),
            "alternatives": [_told_route(venue, facts, other) for other in run.found[1:]],
        }
        told.append(one)
        required += [ref for ref in (one["destination"], one["duration"], one["distance"]) if ref]
    payload = {
        "question": question,
        "shape": "route",
        "routes": told,
        "required": required,
        "facts": facts.told(),
    }
    return Answer(facts.items, payload, required)


def proximity(venue: Venue, question: str, near: Nearest) -> Answer:
    """The nearest assets: what they are, where, and what nearest was measured by."""
    facts = Facts()
    assets = [venue.assets[i] for i in near.asset_ids]
    by_space: dict[str, list[Asset]] = {}
    for asset in assets:
        by_space.setdefault(asset.space_id, []).append(asset)
    rows = [
        Row(Tally.of(members), space_id=space_id, label=label(venue.spaces[space_id]))
        for space_id, members in sorted(by_space.items())
    ]
    payload: dict[str, Any] = {
        "question": question,
        "shape": "proximity",
        "basis": near.basis,
        "anchor": _place(venue, facts, near.anchor),
        "anchor_is_default": near.default_origin,
        "types": [
            facts.asset_type(venue.asset_types[type_id])
            for type_id in sorted({asset.type_id for asset in assets})
        ],
    }
    noun = payload["types"][0] if len(payload["types"]) == 1 else ANY_ASSETS
    together = []
    told_rows: list[dict[str, Any]] = []
    for row in rows[:ROW_CAP]:
        assert row.space_id is not None
        counts = facts.counts(row.tally, space_id=row.space_id, noun=noun)
        told_rows.append({"in": _place(venue, facts, row.space_id), "counts": counts})
        together.append(counts)
    payload["nearest"] = told_rows
    required = [ref for row in told_rows[:1] for ref in (row["in"], *row["counts"]) if ref]
    required += payload["types"]
    if near.route is not None:
        payload["duration"] = facts.duration(
            round(near.route.seconds), space_id=near.route.destination
        )
        payload["distance"] = facts.distance(
            round(near.route.metres), space_id=near.route.destination
        )
        required += [payload["duration"], payload["distance"]]
    elif near.shared is not None:
        payload["shared"] = _place(venue, facts, near.shared)
        if payload["shared"]:
            required.append(payload["shared"])
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
