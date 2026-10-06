"""Composition: the answer in words, with every number and room label filled in from facts.

The composing model never sees a label. It is given facts by reference ({f1},
{f2}, ...) with their tiers, and writes segments that name places only through
those placeholders, each citing the facts it rests on. The host checks what comes
back and fills the placeholders from the facts. Output carrying a digit is
rejected, whatever it says: a number the model typed is a number nobody checked.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from spatial_core.records import Space, Value
from spatial_service.places import Venue, tier

_PLACEHOLDER = re.compile(r"\{(f[0-9]+)\}")


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DraftSegment(_Model):
    text: str = Field(min_length=1, max_length=500)
    cites: list[str]


class Draft(_Model):
    """What the composing model returns."""

    segments: list[DraftSegment]


@dataclass(frozen=True)
class Fact:
    ref: str
    space: Space

    @property
    def label(self) -> Value:
        # Every fact is a space the persona can see and the answer names, so it has a
        # label: resolution only returns labelled places and their labelled ancestors.
        assert self.space.label is not None
        return self.space.label


def location_facts(venue: Venue, subject: Space) -> list[Fact]:
    """The place asked about, then each labelled place enclosing it, innermost first."""
    chain = [subject, *venue.ancestors(subject)]
    return [Fact(f"f{n}", space) for n, space in enumerate(chain, start=1) if space.label]


def composer_input(question: str, facts: list[Fact]) -> dict[str, object]:
    """Everything the composing model is told. References and tiers, never labels or ids."""
    return {
        "question": question,
        "subject": facts[0].ref,
        "facts": [{"ref": f.ref, "tier": tier(f.space)} for f in facts],
    }


class CompositionRejected(Exception):
    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class Citation(_Model):
    space_id: str
    label: str
    provenance: str
    rule: str | None


class Segment(_Model):
    text: str
    citations: list[Citation]


def fill(text: str, facts: list[Fact]) -> list[Segment]:
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
        if "{" in _PLACEHOLDER.sub("", segment.text) or "}" in _PLACEHOLDER.sub("", segment.text):
            reasons.append(f"segment {number} has a malformed placeholder")
    if facts and facts[0].ref not in named:
        reasons.append(f"no segment names {facts[0].ref}, the place asked about")
    if reasons:
        raise CompositionRejected(reasons)

    segments = []
    for segment in draft.segments:
        refs = dict.fromkeys([*_PLACEHOLDER.findall(segment.text), *segment.cites])
        segments.append(
            Segment(
                text=_PLACEHOLDER.sub(lambda m: by_ref[m[1]].label.value, segment.text),
                citations=[_citation(by_ref[ref]) for ref in refs],
            )
        )
    return segments


def _citation(fact: Fact) -> Citation:
    return Citation(
        space_id=fact.space.id,
        label=fact.label.value,
        provenance=fact.label.provenance.value,
        rule=fact.label.rule,
    )
