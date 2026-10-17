"""Facts: what an answer rests on, each behind a placeholder the composer writes.

A route's modelled duration and its distance are facts too. They fill in as
"41 seconds (modelled)" and "39 metres", so the word modelled is the host's as well.

A fact is a place (filled with its label), an asset type (its name), one asset (its
name, followed by "(generated)" when it is) or a count. A
count is always of one record provenance. Where sourced and generated assets mix,
there are two count facts and no fact for their sum, so the model has no way to
print one number over both. A generated count fills in as "3 generated", and its
sourced partner as "1 sourced", so the flag is the host's and cannot be left out.
The two go together: an answer naming one without the other is rejected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Asset, AssetType, RecordProvenance, Space
from spatial_service.assets import Tally
from spatial_service.places import Venue, tier
from spatial_service.retrieval import Passage

# What a count of assets of several types, or of any, is a count of.
ANY_ASSETS = "assets"

GENERATED = "generated"
SOURCED = "sourced"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


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
