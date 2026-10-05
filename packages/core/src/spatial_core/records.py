"""Record models for a snapshot. Ingestion writes them, the service will read them.

Terms are the ones in CONTEXT.md of the site's repository. Provenance is two
independent fields: record provenance on the record, value provenance on each value.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, model_validator


class RecordProvenance(StrEnum):
    SOURCED = "sourced"
    SEED = "seed"


class ValueProvenance(StrEnum):
    SOURCED = "sourced"
    DERIVED = "derived"
    INFERRED = "inferred"


class _Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Value(_Record):
    """One attribute value and how it was reached."""

    value: str
    provenance: ValueProvenance
    # The published rule behind a derived value, e.g. "strip-person-names@1".
    rule: str | None = None
    # What an inferred value rests on.
    basis: str | None = None

    @model_validator(mode="after")
    def _names_its_rule(self) -> Self:
        if self.provenance is ValueProvenance.DERIVED and not self.rule:
            raise ValueError("a derived value names its rule")
        if self.provenance is ValueProvenance.INFERRED and not self.basis:
            raise ValueError("an inferred value records its basis")
        return self

    @classmethod
    def sourced(cls, value: str) -> Value:
        return cls(value=value, provenance=ValueProvenance.SOURCED)

    @classmethod
    def derived(cls, value: str, rule: str) -> Value:
        return cls(value=value, provenance=ValueProvenance.DERIVED, rule=rule)


class Ancestor(_Record):
    tier: str
    id: str


class Space(_Record):
    """A node in the containment tree. Its scale level is the tier label it carries."""

    id: str
    record_provenance: RecordProvenance
    # The OSM element this came from ("way/123"). None for spaces a rule created,
    # such as a floor derived from level tags.
    source: str | None
    tier: Value
    parent_id: str | None
    # Every enclosing space, root first. Never includes the space itself.
    ancestor_path: tuple[Ancestor, ...]
    name: Value | None = None
    ref: Value | None = None
    # The room label. None means no rule can address this space.
    label: Value | None = None
    addressable: bool
    # Every level the space occupies, lowest first.
    levels: Value | None = None
    attributes: dict[str, Value] = {}
    # GeoJSON geometry in WGS84. Missing where the source only has a point or nothing.
    outline: dict[str, Any] | None = None


class Span(_Record):
    """A space covering more floors than the one it hangs from. Location answers only."""

    space_id: str
    floor_id: str
    provenance: ValueProvenance
    rule: str
