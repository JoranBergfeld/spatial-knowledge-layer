import pytest
from pydantic import ValidationError

from spatial_core.clearance import (
    door_clearance,
    edge_clearance,
    from_access,
    space_clearance,
    stricter,
    visible_sets,
)
from spatial_core.records import (
    Ancestor,
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
