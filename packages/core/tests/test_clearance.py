import pytest
from pydantic import ValidationError

from spatial_core.clearance import (
    door_clearance,
    edge_clearance,
    from_access,
    space_clearance,
    stricter,
    type_clearance,
    visible_assets,
    visible_sets,
)
from spatial_core.records import (
    Ancestor,
    Asset,
    AssetType,
    Clearance,
    Edge,
    Persona,
    RecordProvenance,
    Space,
    Value,
    ValueProvenance,
)


@pytest.mark.parametrize(
    ("access", "persona"),
    [
        ("yes", Persona.VISITOR),
        ("permissive", Persona.VISITOR),
        ("customers", Persona.VISITOR),
        ("private", Persona.STAFF),
        ("no", Persona.STAFF),
        ("delivery", Persona.STAFF),
        ("permit", Persona.STAFF),
        ("emergency", Persona.STAFF),
    ],
)
def test_the_published_mapping(access: str, persona: Persona) -> None:
    clearance = from_access(access)

    assert clearance.persona is persona
    assert clearance.access == access
    assert not clearance.authored_default


@pytest.mark.parametrize("access", ["destination", "Yes", "private;no", "", "staff_only"])
def test_an_unrecognised_value_fails_closed_to_maintenance(access: str) -> None:
    clearance = from_access(access)

    assert clearance.persona is Persona.MAINTENANCE
    assert clearance.access == access


@pytest.mark.parametrize(
    ("tags", "persona"),
    [
        ({"indoor": "room"}, Persona.STAFF),
        ({"indoor": "room", "room": "office"}, Persona.STAFF),
        ({"indoor": "corridor"}, Persona.VISITOR),
        ({"indoor": "area"}, Persona.VISITOR),
        ({"indoor": "room", "room": "toilets"}, Persona.VISITOR),
        ({"indoor": "room", "amenity": "toilets"}, Persona.VISITOR),
        ({"indoor": "room", "room": "auditorium"}, Persona.VISITOR),
        ({"indoor": "room", "room": "class"}, Persona.VISITOR),
        ({"indoor": "room", "room": "seminar"}, Persona.VISITOR),
        ({"indoor": "room", "room": "hall"}, Persona.VISITOR),
        ({"indoor": "room", "room": "library"}, Persona.VISITOR),
        ({"indoor": "room", "room": "stairs"}, Persona.VISITOR),
        ({"indoor": "room", "highway": "elevator"}, Persona.VISITOR),
        ({"indoor": "room", "room": "entrance"}, Persona.VISITOR),
    ],
)
def test_untagged_rooms_get_the_authored_defaults_flagged_as_such(
    tags: dict[str, str], persona: Persona
) -> None:
    clearance = space_clearance(tags)

    assert clearance is not None
    assert clearance.persona is persona
    assert clearance.authored_default
    assert clearance.access is None
    assert clearance.rule == "clearance-authored-default@1"


@pytest.mark.parametrize(
    "tags",
    [
        {"indoor": "corridor", "access": "private"},
        {"indoor": "room", "room": "toilets", "access": "no"},
    ],
)
def test_a_default_never_overrules_a_sourced_tag(tags: dict[str, str]) -> None:
    clearance = space_clearance(tags)

    assert clearance == from_access(tags["access"])
    assert clearance.persona is Persona.STAFF


def test_a_sourced_tag_can_also_open_a_room_the_default_would_close() -> None:
    clearance = space_clearance({"indoor": "room", "access": "yes"})

    assert clearance is not None and clearance.persona is Persona.VISITOR


def test_untagged_spaces_above_the_room_tier_get_no_clearance() -> None:
    assert space_clearance({"building": "university"}) is None
    assert space_clearance({}) is None
    assert space_clearance({"building": "yes", "access": "private"}) == from_access("private")


@pytest.mark.parametrize("indoor", ["room", "corridor", "area"])
def test_a_building_that_also_carries_an_indoor_tag_gets_no_room_default(indoor: str) -> None:
    assert space_clearance({"building": "university", "indoor": indoor}) is None
    assert space_clearance({"building": "no", "indoor": indoor}) is not None


def test_doors_get_no_default() -> None:
    assert door_clearance({"door": "hinged"}) is None
    assert door_clearance({"door": "yes", "access": "no"}) == from_access("no")


def test_an_edge_takes_the_stricter_of_door_and_entered_space() -> None:
    private_room = from_access("private")
    yes_door = from_access("yes")

    assert edge_clearance(yes_door, private_room) == private_room
    assert edge_clearance(from_access("no"), space_clearance({"indoor": "corridor"})) == (
        from_access("no")
    )
    assert edge_clearance(None, private_room) == private_room


def test_on_a_tie_the_sourced_clearance_wins_so_it_is_not_flagged() -> None:
    default = space_clearance({"indoor": "corridor"})
    assert default is not None

    assert stricter(default, from_access("yes")) == from_access("yes")
    assert stricter(from_access("yes"), default) == from_access("yes")


def test_a_clearance_is_a_default_exactly_when_it_has_no_tag() -> None:
    with pytest.raises(ValidationError):
        Clearance(persona=Persona.STAFF, access="no", rule="r@1", authored_default=True)
    with pytest.raises(ValidationError):
        Clearance(persona=Persona.STAFF, access=None, rule="r@1", authored_default=False)


def test_personas_nest() -> None:
    staff = from_access("private")

    assert not Persona.VISITOR.clears(staff)
    assert Persona.STAFF.clears(staff)
    assert Persona.MAINTENANCE.clears(staff)


# A floor holding a corridor, a room behind a door from it, and a room with no door.
_SITE = Ancestor(tier="site", id="site")
_FLOOR = Ancestor(tier="floor", id="floor")


def _space(space_id: str, tags: dict[str, str] | None) -> Space:
    room = tags is not None
    return Space(
        id=space_id,
        record_provenance=RecordProvenance.SOURCED,
        source=None,
        tier=Value.derived("room" if room else space_id, "r@1"),
        parent_id=("floor" if room else "site" if space_id == "floor" else None),
        ancestor_path=(_SITE, _FLOOR) if room else (_SITE,) if space_id == "floor" else (),
        addressable=True,
        clearance=space_clearance(tags) if tags is not None else None,
    )


def _edge(from_id: str | None, to: Space, door: Clearance | None) -> Edge:
    assert to.clearance is not None
    return Edge(
        id=f"{from_id}>{to.id}",
        door="node/1",
        from_id=from_id,
        to_id=to.id,
        door_clearance=door,
        clearance=edge_clearance(door, to.clearance),
        provenance=ValueProvenance.DERIVED,
        rule="edges-from-doors@1",
    )


def test_a_space_is_visible_when_a_permitted_edge_enters_it_and_its_ancestors_with_it() -> None:
    site, floor = _space("site", None), _space("floor", None)
    corridor = _space("corridor", {"indoor": "corridor"})
    toilets = _space("toilets", {"indoor": "room", "room": "toilets"})
    office = _space("office", {"indoor": "room"})
    plant = _space("plant", {"indoor": "room", "access": "technicians"})
    edges = [
        _edge(None, corridor, None),
        # A visitor room behind a door that admits staff.
        _edge("corridor", toilets, from_access("private")),
    ]

    sets = visible_sets([site, floor, corridor, toilets, office, plant], edges)

    assert sets[Persona.VISITOR] == {"site", "floor", "corridor"}
    assert sets[Persona.STAFF] == {"site", "floor", "corridor", "toilets", "office"}
    # A room with no edge takes its own clearance.
    assert sets[Persona.MAINTENANCE] == {"site", "floor", "corridor", "toilets", "office", "plant"}


def test_nothing_entered_and_nothing_cleared_is_visible_to_nobody() -> None:
    sets = visible_sets([_space("site", None), _space("floor", None)], [])

    assert all(not visible for visible in sets.values())


@pytest.mark.parametrize(
    ("code", "persona"),
    [
        ("23-17 11 13", Persona.VISITOR),  # a door
        ("23-21 19 13", Persona.VISITOR),  # furniture
        ("23-29 21 11", Persona.VISITOR),  # a fire extinguisher
        ("23-25 31 11", Persona.STAFF),  # laboratory equipment
        ("23-37 21 11", Persona.STAFF),  # a projector
        ("23-33 25 11", Persona.MAINTENANCE),  # an air handling unit
        ("23-35 11 11", Persona.MAINTENANCE),  # a distribution board
    ],
)
def test_a_type_takes_the_clearance_its_table_23_code_maps_to(code: str, persona: Persona) -> None:
    cleared = type_clearance(code)

    assert cleared.persona is persona
    assert cleared.access == code
    assert cleared.rule == "clearance-from-omniclass@1"
    assert not cleared.authored_default


def test_a_type_code_takes_its_longest_listed_prefix() -> None:
    # 23-17 11 is doors. The rest of 23-17 is not listed.
    assert type_clearance("23-17 11 00").persona is Persona.VISITOR
    assert type_clearance("23-17 13 00").persona is Persona.MAINTENANCE


@pytest.mark.parametrize(
    "code", ["23-39 11 00", "23-11", "13-55 11 00", "2321 19 13", "23-21-19-13", "", "camera"]
)
def test_an_unrecognised_type_code_fails_closed_to_maintenance(code: str) -> None:
    assert type_clearance(code).persona is Persona.MAINTENANCE


def _kind(type_id: str, code: str) -> AssetType:
    return AssetType(
        id=type_id,
        record_provenance=RecordProvenance.SEED,
        name=Value.sourced(type_id),
        omniclass=Value.sourced(code),
        clearance=type_clearance(code),
    )


def _asset(asset_id: str, type_id: str, space_id: str) -> Asset:
    return Asset(
        id=asset_id,
        record_provenance=RecordProvenance.SEED,
        source=None,
        type_id=type_id,
        space_id=space_id,
        name=Value.sourced(asset_id),
    )


def test_an_asset_is_visible_when_its_type_is_cleared_and_its_space_is_visible() -> None:
    chair, hood, unit = (
        _kind("chair", "23-21 19 13"),
        _kind("hood", "23-25 31 11"),
        _kind("unit", "23-33 25 11"),
    )
    spaces = {
        Persona.VISITOR: frozenset({"hall"}),
        Persona.STAFF: frozenset({"hall", "lab"}),
        Persona.MAINTENANCE: frozenset({"hall", "lab", "plant"}),
    }
    assets = [
        _asset("chair-in-hall", "chair", "hall"),
        _asset("chair-in-lab", "chair", "lab"),
        _asset("hood-in-hall", "hood", "hall"),
        _asset("unit-in-hall", "unit", "hall"),
        _asset("chair-in-plant", "chair", "plant"),
    ]

    seen, types = visible_assets(assets, [chair, hood, unit], spaces)

    assert seen[Persona.VISITOR] == {"chair-in-hall"}
    assert seen[Persona.STAFF] == {"chair-in-hall", "chair-in-lab", "hood-in-hall"}
    assert seen[Persona.MAINTENANCE] == {asset.id for asset in assets}
    # A type exists for a persona only through an asset they can see.
    assert types[Persona.VISITOR] == {"chair"}
    assert types[Persona.STAFF] == {"chair", "hood"}
    assert types[Persona.MAINTENANCE] == {"chair", "hood", "unit"}


def test_a_type_with_no_visible_asset_is_not_visible() -> None:
    chair = _kind("chair", "23-21 19 13")
    spaces = {persona: frozenset({"hall"}) for persona in Persona}

    seen, types = visible_assets([_asset("a", "chair", "office")], [chair], spaces)

    assert all(not ids for ids in seen.values())
    assert all(not ids for ids in types.values())
