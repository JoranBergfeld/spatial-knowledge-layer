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

    A generated type carries what the test bench records for it, so a gap there is
    None here. A type a rule read from OSM has only what the tag says, so
    manufacturer, model, warranties and expected life are all None.
    """

    id: str
    record_provenance: RecordProvenance
    name: Value
    # The OmniClass Table 23 (Products) code, e.g. "23-30 10".
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
    # Its space's path and then its space, root first, so scope is a filter here too.
    ancestor_path: tuple[Ancestor, ...]
    name: Value
    # The OSM door node ("node/123") a door asset is bound to.
    door: str | None = None
    # None where the space has no outline to lay it out in.
    position: Position | None = None
    # serial_number, installation_date, warranty_start_date and the like. Absent
    # where the data has a gap.
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
    """Something done to one asset, after COBie v3 Event. It says nothing of the outcome."""

    id: str
    record_provenance: RecordProvenance
    asset_id: str
    # The task it carried out, where it was a planned one.
    task_id: str | None
    name: Value
    start: Value
    # None where the event records no end.
    end: Value | None = None
    performer: Value


class VisibleAsset(_Record):
    persona: Persona
    asset_id: str


class VisibleAssetType(_Record):
    persona: Persona
    type_id: str


class DocumentKind(StrEnum):
    # A page the venue publishes about one of its places.
    ACCESSIBILITY_PAGE = "accessibility-page"
    # A maker's manual for an asset type. Quoted and linked, never hosted.
    MANUAL = "manual"


class AnchorKind(StrEnum):
    SPACE = "space"
    ASSET_TYPE = "asset_type"
    ASSET = "asset"


class Anchor(_Record):
    """What ties a document to the venue: a space, an asset type or one asset."""

    kind: AnchorKind
    id: str
    # Sourced when the corpus names the id, derived when a rule found it.
    provenance: ValueProvenance
    rule: str | None = None

    @model_validator(mode="after")
    def _names_its_rule(self) -> Self:
        if self.provenance is ValueProvenance.DERIVED and not self.rule:
            raise ValueError("a derived anchor names its rule")
        return self


class LinkStatus(StrEnum):
    ONLINE = "online"
    OFFLINE = "offline"


class LinkCheck(_Record):
    """Whether a document's source answered, and the day build asked."""

    status: LinkStatus
    checked_on: str


class TranslationSource(_Record):
    """What turned a document's text into English."""

    deployment: str
    # The model the deployment said it ran.
    model: str
    prompt_version: str


class Document(_Record):
    """A piece of prose anchored to the venue. Its chunks are what gets retrieved."""

    id: str
    record_provenance: RecordProvenance
    kind: DocumentKind
    title: str
    # Where the original lives. The layer links to it and never hosts it.
    url: str
    # The original's language, as a BCP 47 tag ("de").
    language: str
    # The sha256 of the original as fetched, which stays in the private bundle.
    original_sha256: str
    anchors: tuple[Anchor, ...]
    # The lowest persona that may read it, by document-clearance.
    clearance_class: Persona
    # The day the source says it last changed, read by a published rule. None is undated.
    date: Value | None = None
    link: LinkCheck
    # None for a document that was English already.
    translation: TranslationSource | None = None
    chunks: int

    @model_validator(mode="after")
    def _is_anchored(self) -> Self:
        if not self.anchors:
            raise ValueError("a document has at least one anchor")
        return self


class ChunkRole(StrEnum):
    # A passage of the document itself. The only thing that is ever evidence.
    PASSAGE = "passage"
    # A model's summary of the document. It helps find the document's passages and
    # is never quoted, cited, checked for entailment or shown to anyone.
    SUMMARY = "summary"


# What a summary chunk says for an ordinal: it has no place in the document.
SUMMARY_ORDINAL = -1


class Chunk(_Record):
    """One passage of a document, as the search index holds it.

    The text is English with numbers-in-words applied. The original text is not
    here: it stays in the private bundle, and the chunk keeps its hash.

    A summary chunk is the one exception: its text is a model's summary of the
    document, and its hash is of what the model read. Everything else on it is
    its document's, so the filter that admits a passage admits its summary too.
    """

    id: str
    document_id: str
    role: ChunkRole = ChunkRole.PASSAGE
    # Its place in the document, from 0. SUMMARY_ORDINAL for a summary.
    ordinal: int
    kind: DocumentKind
    title: str
    text: str
    # The original's language, URL and the sha256 of this passage's original text.
    language: str
    url: str
    original_text_sha256: str
    translated: bool
    # Its document's anchors, by kind.
    space_ids: tuple[str, ...]
    asset_type_ids: tuple[str, ...]
    asset_ids: tuple[str, ...]
    # Every space on the ancestor path of a space or asset anchor, and the anchored
    # spaces themselves, flattened to one set so scope is a filter.
    ancestor_ids: tuple[str, ...]
    clearance_class: Persona
    # ISO date, or None for undated. Trace only: it never reaches a model.
    document_date: str | None = None
    link_status: LinkStatus
    link_checked_on: str

    @property
    def is_evidence(self) -> bool:
        """Whether it may be quoted and cited. A summary never is."""
        return self.role is ChunkRole.PASSAGE
