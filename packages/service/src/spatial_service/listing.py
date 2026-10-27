"""What's on campus, as one persona sees it: GET /places and GET /places/{building_id}.

A listing, so a visitor has something to ask about. It reads the same venue a turn
does, already cut to the persona's visible spaces and assets, so a building, room or
equipment type beyond clearance isn't missing here, it never existed.

    GET /places              every building the persona can see, with its equipment
                             types, how many of each, and their record provenance
    GET /places/{id}         one building's floors with their rooms, and each
                             equipment type mapped to the rooms that hold it

No outlines, positions or coordinates, the same rule as a turn. No individual assets
either, only counts and room ids. Nothing here calls a model.

The snapshot is always named. One that is not served answers like a turn on it would:
409, snapshot-dropped, retired and unknown told apart nowhere.

places_contract/ next to this file holds one body of each kind, read off the test
fixture. Tests pin them to the models and to what the service answers, and the
backend's contract tests copy them from there.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Asset, Persona, RecordProvenance, Space
from spatial_service.places import BUILDING, ROOM, Venue, label, tier
from spatial_service.store import Store, StoreUnreachable
from spatial_service.turn_models import SNAPSHOT_DROPPED

CONTRACT_DIR = Path(__file__).parent / "places_contract"
FLOOR = "floor"
NO_SUCH_BUILDING = "no-such-building"


class PlacesRefused(Exception):
    """No listing to give. The code is what the caller acts on; the message is for reading."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Equipment(_Model):
    """One equipment type and how many of it the persona can see in a building."""

    type_id: str
    name: str
    count: int
    # sourced for the fire kit OSM maps, seed for the generated rest.
    provenance: RecordProvenance


class Building(_Model):
    id: str
    name: str | None
    # None where no rule could give one, so a turn can't address it by label either.
    label: str | None
    equipment: tuple[Equipment, ...]


class Places(_Model):
    snapshot: str
    persona: Persona
    buildings: tuple[Building, ...]


class Room(_Model):
    id: str
    label: str | None
    # As mapped, e.g. "107" or "D165". The building's number is in the label, not here.
    ref: str | None


class Floor(_Model):
    id: str
    label: str | None
    rooms: tuple[Room, ...]


class RoomEquipment(Equipment):
    # The rooms holding at least one. An asset mapped onto the building or a floor
    # itself counts, but sits in no room, so count can be more than these hold.
    rooms: tuple[str, ...]


class BuildingPlaces(_Model):
    snapshot: str
    persona: Persona
    id: str
    name: str | None
    label: str | None
    floors: tuple[Floor, ...]
    # Rooms hanging from the building itself, on no floor. Some buildings map no levels.
    floorless_rooms: tuple[Room, ...]
    equipment: tuple[RoomEquipment, ...]


def served_venue(store: Store, snapshot: str, persona: Persona) -> Venue:
    """The persona's venue in a served snapshot, or a refusal a turn would also give."""
    try:
        current, previous = store.served()
        if snapshot not in (current, previous):
            # Retired, or never there at all. Told apart nowhere, as in a turn.
            raise PlacesRefused(409, SNAPSHOT_DROPPED, f"{snapshot} is not served")
        return store.venue(snapshot, persona)
    except StoreUnreachable as error:
        raise PlacesRefused(503, "store-unreachable", "the store cannot be reached") from error


def places(venue: Venue) -> Places:
    buildings = sorted((s for s in venue.spaces.values() if tier(s) == BUILDING), key=_by_label)
    return Places(
        snapshot=venue.snapshot,
        persona=venue.persona,
        buildings=tuple(
            Building(
                id=building.id,
                name=building.name.value if building.name else None,
                label=label(building),
                equipment=_equipment(venue, venue.assets_under.get(building.id, [])),
            )
            for building in buildings
        ),
    )


def building_places(venue: Venue, building_id: str) -> BuildingPlaces:
    building = venue.spaces.get(building_id)
    if building is None or tier(building) != BUILDING:
        # Beyond clearance, not a building, or not there: the same answer.
        raise PlacesRefused(404, NO_SUCH_BUILDING, f"no building {building_id}")
    floors: dict[str, list[Room]] = {}
    floorless: list[Room] = []
    rooms: set[str] = set()
    for space in sorted(venue.spaces.values(), key=_by_label):
        enclosing = venue.building_of(space)
        if tier(space) != ROOM or enclosing is None or enclosing.id != building.id:
            continue
        rooms.add(space.id)
        room = Room(id=space.id, label=label(space), ref=space.ref.value if space.ref else None)
        floor = next((a for a in venue.ancestors(space) if tier(a) == FLOOR), None)
        if floor is None:
            floorless.append(room)
        else:
            floors.setdefault(floor.id, []).append(room)
    under = venue.assets_under.get(building.id, [])
    holding: dict[str, set[str]] = {}
    for asset in under:
        if asset.space_id in rooms:
            holding.setdefault(asset.type_id, set()).add(asset.space_id)
    return BuildingPlaces(
        snapshot=venue.snapshot,
        persona=venue.persona,
        id=building.id,
        name=building.name.value if building.name else None,
        label=label(building),
        floors=tuple(
            Floor(id=floor.id, label=label(floor), rooms=tuple(floors[floor.id]))
            for floor in sorted((venue.spaces[i] for i in floors), key=_by_level)
        ),
        floorless_rooms=tuple(floorless),
        equipment=tuple(
            RoomEquipment(**kind.model_dump(), rooms=tuple(sorted(holding.get(kind.type_id, ()))))
            for kind in _equipment(venue, under)
        ),
    )


def _equipment(venue: Venue, assets: Iterable[Asset]) -> tuple[Equipment, ...]:
    # A visible asset has a visible type, but only what both say is counted.
    counts = Counter(a.type_id for a in assets if a.type_id in venue.asset_types)
    kinds = sorted(
        (venue.asset_types[type_id] for type_id in counts),
        key=lambda kind: (kind.name.value, kind.id),
    )
    return tuple(
        Equipment(
            type_id=kind.id,
            name=kind.name.value,
            count=counts[kind.id],
            provenance=kind.record_provenance,
        )
        for kind in kinds
    )


def _by_label(space: Space) -> tuple[bool, str, str]:
    # Unlabelled last, then by label, then by id so the order never wobbles.
    return (space.label is None, label(space) or "", space.id)


def _by_level(floor: Space) -> tuple[float, bool, str, str]:
    # Lowest first. A floor with no level that parses goes on top.
    try:
        level = float(floor.levels.value) if floor.levels else float("inf")
    except ValueError:
        level = float("inf")
    return (level, *_by_label(floor))
