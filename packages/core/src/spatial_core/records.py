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
    # The raw `access` tag it was mapped from, kept beside the persona it mapped
    # to. None for an authored default.
    access: str | None
    # The published rule that gave it.
    rule: str
    # True when the element carried no tag and a rule supplied the clearance. Every
    # asset type's clearance is one: its Table 23 code is sourced, the tier is not.
    authored_default: bool
    # True when an asset type's code matched nothing and it failed closed.
    unrecognised_category: bool = False

    @model_validator(mode="after")
    def _default_has_no_tag(self) -> Self:
        if self.authored_default != (self.access is None):
            raise ValueError("a clearance is an authored default exactly when it has no tag")
        if self.unrecognised_category and not self.authored_default:
            raise ValueError("an unrecognised category is an authored default")
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


class Position(_Record):
    """Where an asset is, as a WGS84 point, and how that was reached."""

    lon: float
    lat: float
    provenance: ValueProvenance
    rule: str | None = None
    basis: str | None = None

    @model_validator(mode="after")
    def _names_its_rule(self) -> Self:
        if self.provenance is ValueProvenance.DERIVED and not self.rule:
            raise ValueError("a derived position names its rule")
        if self.provenance is ValueProvenance.INFERRED and not self.basis:
            raise ValueError("an inferred position records its basis")
        return self


class AssetType(_Record):
    """A kind of equipment, after COBie Type. Its assets share its clearance.

    A generated type carries all of manufacturer, model, warranties and expected
    life. A type a rule read from OSM has only what the tag says, so those are None.
    """

    id: str
    record_provenance: RecordProvenance
    name: Value
    # The OmniClass Table 23 (Products) code, e.g. "23-21 19 13".
    omniclass: Value
    manufacturer: Value | None = None
    model: Value | None = None
    warranty_parts_years: Value | None = None
    warranty_labor_years: Value | None = None
    expected_life_years: Value | None = None
    # From the Table 23 code by clearance-from-omniclass, an authored default.
    # Unrecognised codes fail closed.
    clearance: Clearance


class Asset(_Record):
    """One installed thing, after COBie Component. It sits in exactly one space.

    An asset is never an edge. A door asset is the thing in the doorway, bound to
    the sourced door whose edges carry the circulation.
    """

    id: str
    record_provenance: RecordProvenance
    # The OSM element it was read from ("node/123"). None for a generated asset.
    source: str | None
    type_id: str
    space_id: str
    name: Value
    # The OSM door node ("node/123") a door asset is bound to.
    door: str | None = None
    # None where the space has no outline to lay it out in.
    position: Position | None = None
    # serial_number, installation_date and the like. Absent where the data has a gap.
    attributes: dict[str, Value] = {}


class MaintenanceTask(_Record):
    """A planned job on every asset of a type, after COBie 2.4 Job."""

    id: str
    record_provenance: RecordProvenance
    type_id: str
    name: Value
    category: Value
    status: Value
    description: Value
    duration: Value
    duration_unit: Value
    start: Value
    task_start_unit: Value
    frequency: Value
    frequency_unit: Value
    task_number: Value
    priors: Value | None = None
    resource_names: Value | None = None


class ServiceEvent(_Record):
    """Something done to one asset on a date, after COBie v3 Event."""

    id: str
    record_provenance: RecordProvenance
    asset_id: str
    # The task it carried out, where it was a planned one.
    task_id: str | None
    name: Value
    category: Value
    date: Value
    description: Value


class VisibleAsset(_Record):
    persona: Persona
    asset_id: str


class VisibleAssetType(_Record):
    persona: Persona
    type_id: str
