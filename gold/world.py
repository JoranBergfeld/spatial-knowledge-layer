"""What gold is computed over: the building as plain data, and one persona's cut of it.

Nothing here knows where the data came from. from_cobie.py fills a World from a
handover workbook, from_bundle.py from a snapshot bundle.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Collection
from dataclasses import dataclass, field
from functools import cached_property

PERSONAS = ("visitor", "staff", "maintenance")


def clears(persona: str, lowest: str | None) -> bool:
    """Whether ``persona`` may reach something whose lowest persona is ``lowest``."""
    return lowest is None or PERSONAS.index(persona) >= PERSONAS.index(lowest)


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", fold(text)))


@dataclass(frozen=True)
class Space:
    id: str
    tier: str
    parent: str | None
    label: str | None
    # The room or building number, where it has one.
    ref: str | None
    name: str | None
    # The lowest persona that may enter. None for a space that has no clearance of
    # its own: a floor or a building is there for whoever sees something inside it.
    lowest: str | None


@dataclass(frozen=True)
class AssetType:
    id: str
    name: str
    lowest: str


@dataclass(frozen=True)
class Asset:
    id: str
    type: str
    space: str
    # Only what is recorded. A gap is a missing key.
    attributes: dict[str, str]
    # The day each service event started, as written.
    serviced: tuple[str, ...] = ()


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    kind: str
    # (kind, id) with kind one of space, asset_type, asset.
    anchors: tuple[tuple[str, str], ...]
    lowest: str


@dataclass
class World:
    spaces: dict[str, Space]
    types: dict[str, AssetType]
    assets: dict[str, Asset]
    documents: dict[str, Document]
    # Per persona, the ids that exist for them: {"spaces": ..., "assets": ..., "types": ...}.
    visible: dict[str, dict[str, frozenset[str]]] = field(default_factory=dict)

    def path(self, space_id: str) -> list[str]:
        """The spaces enclosing ``space_id``, outermost first, without itself."""
        found: list[str] = []
        at = self.spaces[space_id].parent
        while at is not None:
            found.append(at)
            at = self.spaces[at].parent
        return found[::-1]

    @cached_property
    def tiers(self) -> list[str]:
        """The tiers there are, outermost first, by how deep each first shows up."""
        depth: dict[str, int] = {}
        for space in self.spaces.values():
            deep = len(self.path(space.id))
            depth[space.tier] = min(depth.get(space.tier, deep), deep)
        return sorted(depth, key=lambda tier: (depth[tier], tier))

    def view(self, persona: str) -> View:
        return View(self, persona)


def parse_day(text: str | None) -> datetime.date | None:
    """The day a recorded value names: its first ten characters as an ISO date."""
    try:
        return datetime.date.fromisoformat((text or "")[:10])
    except ValueError:
        return None


def a_year_after(day: datetime.date) -> datetime.date:
    try:
        return day.replace(year=day.year + 1)
    except ValueError:
        # 29 February. The year after has none, so the day before.
        return day.replace(year=day.year + 1, day=28)


SERVICE_EVENT = "service_event"
SERVICE_DUE = "service_due"


class View:
    """One persona's cut: only what is visible to them, and what follows from that."""

    def __init__(self, world: World, persona: str) -> None:
        self.world = world
        self.persona = persona
        seen = world.visible[persona]
        self.spaces = {i: s for i, s in world.spaces.items() if i in seen["spaces"]}
        self.assets = {i: a for i, a in world.assets.items() if i in seen["assets"]}
        self.types = {i: t for i, t in world.types.items() if i in seen["types"]}

    @cached_property
    def under(self) -> dict[str, list[Asset]]:
        """Each space's visible assets: its own and those of everything inside it."""
        found: dict[str, list[Asset]] = {}
        for asset in self.assets.values():
            for space_id in (*self.world.path(asset.space), asset.space):
                found.setdefault(space_id, []).append(asset)
        return found

    def inside(self, space_id: str) -> list[str]:
        """The visible spaces under ``space_id``, itself not included."""
        return [i for i in self.spaces if space_id in self.world.path(i)]

    def building_of(self, space_id: str) -> str | None:
        for candidate in (space_id, *reversed(self.world.path(space_id))):
            if self.world.spaces[candidate].tier == "building":
                return candidate
        return None

    def latest_service(self, asset: Asset) -> datetime.date | None:
        days = [day for day in map(parse_day, asset.serviced) if day is not None]
        return max(days, default=None)

    def value(self, asset: Asset, attribute: str) -> str | None:
        """What an asset records for an attribute, the two derived ones included."""
        if attribute == SERVICE_EVENT:
            day = self.latest_service(asset)
            return day.isoformat() if day else None
        if attribute == SERVICE_DUE:
            # The latest service event, or the installation date where there is
            # none, plus one year. For every type.
            day = self.latest_service(asset) or parse_day(asset.attributes.get("installation_date"))
            return a_year_after(day).isoformat() if day else None
        return asset.attributes.get(attribute)

    @cached_property
    def catalogue(self) -> frozenset[str]:
        """The attributes at least one visible asset carries."""
        names: set[str] = set()
        for asset in self.assets.values():
            names.update(asset.attributes)
            for derived in (SERVICE_EVENT, SERVICE_DUE):
                if self.value(asset, derived) is not None:
                    names.add(derived)
        return frozenset(names)

    def dated(self, attribute: str) -> bool:
        """Whether every visible value of the attribute is a date."""
        values = [self.value(asset, attribute) for asset in self.assets.values()]
        recorded = [value for value in values if value is not None]
        return bool(recorded) and all(parse_day(value) is not None for value in recorded)

    def reads(self, document: Document) -> bool:
        """Whether the document exists for this persona at all."""
        return clears(self.persona, document.lowest) and any(
            self.sees(kind, anchor) for kind, anchor in document.anchors
        )

    def sees(self, kind: str, anchor: str) -> bool:
        pools: dict[str, Collection[str]] = {
            "space": self.spaces,
            "asset": self.assets,
            "asset_type": self.types,
        }
        return anchor in pools[kind]

    def retrievable(self, anchors: set[tuple[str, str]]) -> list[str]:
        """The documents this persona may read that are tied to any of ``anchors``."""
        return sorted(
            document.id
            for document in self.world.documents.values()
            if clears(self.persona, document.lowest) and anchors.intersection(document.anchors)
        )

    def anchors_of_place(self, space_id: str) -> set[tuple[str, str]]:
        """A place, what is visible under it, the types of the assets there, and up to its building.

        Upwards it is the spaces only, the building included and nothing past it.
        """
        found = {("space", space_id), *(("space", inner) for inner in self.inside(space_id))}
        path = self.world.path(space_id)
        buildings = [i for i, up in enumerate(path) if self.world.spaces[up].tier == "building"]
        if buildings:
            found |= {("space", up) for up in path[buildings[-1] :] if up in self.spaces}
        return found | self.anchors_of_assets(self.under.get(space_id, []))

    def anchors_of_assets(self, assets: list[Asset]) -> set[tuple[str, str]]:
        return {("asset", a.id) for a in assets} | {("asset_type", a.type) for a in assets}
