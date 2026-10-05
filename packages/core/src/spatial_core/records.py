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


class Persona(StrEnum):
    """Who a conversation is held as. Strictly nested: each reaches all the one before it does."""

    VISITOR = "visitor"
    STAFF = "staff"
    MAINTENANCE = "maintenance"

    @property
    def rank(self) -> int:
        return PERSONAS.index(self)

    def clears(self, clearance: Clearance) -> bool:
        return self.rank >= clearance.persona.rank


# Least cleared first.
PERSONAS = (Persona.VISITOR, Persona.STAFF, Persona.MAINTENANCE)


class Clearance(_Record):
    """The lowest persona that may reach something, and what that was read from."""

    persona: Persona
    # The raw `access` tag it was mapped from, kept beside the persona it mapped to.
    # None for an authored default.
    access: str | None
    # The published rule that gave it.
    rule: str
    # True when the element carried no tag and a rule supplied the clearance.
    authored_default: bool

    @model_validator(mode="after")
    def _default_has_no_tag(self) -> Self:
        if self.authored_default != (self.access is None):
            raise ValueError("a clearance is an authored default exactly when it has no tag")
        return self


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
    # None for a space with no access tag above the room tier. Such a space is
    # visible only through the spaces it encloses.
    clearance: Clearance | None = None


class Span(_Record):
    """A space covering more floors than the one it hangs from. Location answers only."""

    space_id: str
    floor_id: str
    provenance: ValueProvenance
    rule: str


class Edge(_Record):
    """One way through a door: from one space into another. Circulation only.

    A door joining two spaces is two edges, one each way, because an edge takes
    the clearance of the space it enters.
    """

    id: str
    # The OSM node of the door ("node/123").
    door: str
    # None when the door opens onto no mapped space, such as an outside entrance.
    from_id: str | None
    to_id: str
    # The door's own tag. None for an untagged door, which constrains nothing.
    door_clearance: Clearance | None
    # The stricter of the door's and the entered space's.
    clearance: Clearance
    provenance: ValueProvenance
    rule: str


class VisibleSpace(_Record):
    """One space that exists for a persona. One set per persona per snapshot."""

    persona: Persona
    space_id: str
