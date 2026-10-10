import pytest
from pydantic import ValidationError

from spatial_core.clearance import (
    document_clearance,
    door_clearance,
    edge_clearance,
    from_access,
    space_category_clearance,
    space_clearance,
    stricter,
    type_clearance,
    visible_assets,
    visible_sets,
)
from spatial_core.records import (
    Ancestor,
    AnchorKind,
    Asset,
    AssetType,
    Clearance,
    DocumentKind,
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
        ("23-30 10 00", Persona.VISITOR),  # a door
        ("23-80 70 11", Persona.VISITOR),  # lighting
        ("23-65 70 11", Persona.VISITOR),  # a sprinkler
        ("23-45 10 00", Persona.VISITOR),  # a plumbing fixture
        ("23-40 35 17", Persona.STAFF),  # kitchen equipment
        ("23-85 10 00", Persona.STAFF),  # AV
        ("23-65 35 11", Persona.MAINTENANCE),  # a valve
        ("23-75 35 00", Persona.MAINTENANCE),  # an AC unit
        ("23-80 30 20", Persona.MAINTENANCE),  # a panelboard
    ],
)
def test_a_type_takes_the_clearance_its_table_23_code_maps_to(code: str, persona: Persona) -> None:
    cleared = type_clearance(code)

    assert cleared.persona is persona
    assert cleared.rule == "clearance-from-omniclass@1"
    # The code is sourced, the tier it maps to is authored.
    assert cleared.authored_default
    assert cleared.access is None
    assert not cleared.unrecognised_category


def test_a_type_code_takes_its_longest_listed_prefix() -> None:
    # Air terminals split: VAV boxes sit above the ceiling, diffusers in the room.
    assert type_clearance("23-75 70 21 24").persona is Persona.MAINTENANCE
    assert type_clearance("23-75 70 21 27").persona is Persona.VISITOR
    # 23-65 is not listed on its own, only what is under it.
    assert type_clearance("23-65 70").persona is Persona.VISITOR
    assert type_clearance("23-65 35").persona is Persona.MAINTENANCE
    assert type_clearance("23-65 10").unrecognised_category


def test_a_type_is_matched_on_its_numeric_code_never_the_title() -> None:
    titled = type_clearance("23-75 70 21 24: Variable Air Volume Terminals")

    assert titled.persona is Persona.MAINTENANCE
    assert not titled.unrecognised_category
    assert type_clearance("Doors").unrecognised_category
    # A prefix is whole pairs of digits: 23-150 is not under 23-15.
    assert type_clearance("23-150").unrecognised_category


@pytest.mark.parametrize(
    "code",
    ["23-39 11 00", "23-11", "13-55 11 00", "2330 10 00", "23-75 70 21", "", None, "camera"],
)
def test_an_unrecognised_type_code_fails_closed_to_maintenance(code: str | None) -> None:
    cleared = type_clearance(code)

    assert cleared.persona is Persona.MAINTENANCE
    assert cleared.authored_default
    assert cleared.unrecognised_category


def test_only_an_authored_default_can_be_an_unrecognised_category() -> None:
    with pytest.raises(ValueError, match="unrecognised category"):
        Clearance(
            persona=Persona.MAINTENANCE,
            access="no",
            rule="r@1",
            authored_default=False,
            unrecognised_category=True,
        )


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
        ancestor_path=(),
        name=Value.sourced(asset_id),
    )


def test_an_asset_is_visible_when_its_type_is_cleared_and_its_space_is_visible() -> None:
    chair, hood, unit = (
        _kind("chair", "23-40 20 00"),
        _kind("hood", "23-40 35 17"),
        _kind("unit", "23-75 35 00"),
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
    chair = _kind("chair", "23-40 20 00")
    spaces = {persona: frozenset({"hall"}) for persona in Persona}

    seen, types = visible_assets([_asset("a", "chair", "office")], [chair], spaces)

    assert all(not ids for ids in seen.values())
    assert all(not ids for ids in types.values())


PAGE, MANUAL = DocumentKind.ACCESSIBILITY_PAGE, DocumentKind.MANUAL
SPACE, TYPE, ASSET = AnchorKind.SPACE, AnchorKind.ASSET_TYPE, AnchorKind.ASSET
VISITOR, STAFF, MAINTENANCE = Persona.VISITOR, Persona.STAFF, Persona.MAINTENANCE


@pytest.mark.parametrize(
    ("kind", "anchors", "expected"),
    [
        # One anchor: its clearance.
        (PAGE, [(SPACE, STAFF)], STAFF),
        (MANUAL, [(TYPE, MAINTENANCE)], MAINTENANCE),
        (MANUAL, [(ASSET, STAFF)], STAFF),
        # Several: the strictest.
        (PAGE, [(SPACE, VISITOR), (SPACE, STAFF)], STAFF),
        (PAGE, [(SPACE, VISITOR), (ASSET, MAINTENANCE)], MAINTENANCE),
        (PAGE, [(SPACE, STAFF), (SPACE, VISITOR), (SPACE, VISITOR)], STAFF),
    ],
)
def test_a_document_takes_its_strictest_anchors_clearance(
    kind: DocumentKind, anchors: list[tuple[AnchorKind, Persona | None]], expected: Persona
) -> None:
    assert document_clearance(kind, anchors) is expected


@pytest.mark.parametrize(
    ("anchors", "expected"),
    [
        ([(TYPE, MAINTENANCE), (TYPE, VISITOR)], VISITOR),
        ([(TYPE, MAINTENANCE), (TYPE, STAFF), (TYPE, MAINTENANCE)], STAFF),
        # The types count as one anchor. Its other anchors still bind.
        ([(TYPE, MAINTENANCE), (TYPE, VISITOR), (SPACE, STAFF)], STAFF),
        ([(TYPE, MAINTENANCE), (TYPE, VISITOR), (ASSET, MAINTENANCE)], MAINTENANCE),
        ([(TYPE, MAINTENANCE), (TYPE, STAFF), (SPACE, VISITOR)], STAFF),
    ],
)
def test_a_manual_shared_by_several_types_takes_the_least_restrictive_of_them(
    anchors: list[tuple[AnchorKind, Persona | None]], expected: Persona
) -> None:
    assert document_clearance(MANUAL, anchors) is expected


def test_only_a_manual_gets_the_shared_type_exception() -> None:
    assert document_clearance(PAGE, [(TYPE, MAINTENANCE), (TYPE, VISITOR)]) is MAINTENANCE


def test_an_anchor_with_no_clearance_of_its_own_limits_nothing() -> None:
    # A building nobody tagged has none. It is visible only through its rooms.
    assert document_clearance(PAGE, [(SPACE, None)]) is VISITOR
    assert document_clearance(PAGE, [(SPACE, None), (SPACE, STAFF)]) is STAFF


def test_a_document_with_no_anchor_has_no_clearance() -> None:
    with pytest.raises(ValueError, match="at least one anchor"):
        document_clearance(PAGE, [])


@pytest.mark.parametrize(
    ("category", "persona"),
    [
        ("13-85 11 11: Corridor", Persona.VISITOR),
        ("13-85 21 11: Stairway", Persona.VISITOR),
        ("13-41 11 14 21: Restroom", Persona.VISITOR),
        ("13-51 31 11: Waiting Room", Persona.VISITOR),
        ("13-11 11 31: Reception Space", Persona.VISITOR),
        ("13-85 31 11: Entry Vestibule", Persona.VISITOR),
        ("13-85 31 99: Other Transitional Circulation Spaces", Persona.VISITOR),
        ("13-81 21 21: Electrical Room", Persona.MAINTENANCE),
        ("13-81 21 11: Equipment Room", Persona.MAINTENANCE),
        ("13-81 21 17: Mechanical Room", Persona.MAINTENANCE),
        ("13-81 11 99: Other General Facility Service Spaces", Persona.MAINTENANCE),
        ("13-75 41 24: Hazardous Material Storage Space", Persona.MAINTENANCE),
        ("13-91 11 11 33: Loading Dock", Persona.MAINTENANCE),
        ("13-81 21 31: Elevator Shaft", Persona.MAINTENANCE),
        ("13-81 31: Service Distribution Spaces", Persona.MAINTENANCE),
        ("13-15 11 34 11: Office", Persona.STAFF),
        ("13-75 11 11: Storage Room", Persona.STAFF),
        ("13-41 41 14: Physical Examination Room", Persona.STAFF),
    ],
)
def test_a_test_bench_space_takes_the_clearance_its_table_13_category_maps_to(
    category: str, persona: Persona
) -> None:
    cleared = space_category_clearance(category)

    assert cleared.persona is persona
    assert cleared.rule == "clearance-from-space-category@1"


@pytest.mark.parametrize("category", [None, "", "n/a", "Corridor", "13-99 99: Something New"])
def test_a_missing_or_unlisted_space_category_is_staff(category: str | None) -> None:
    assert space_category_clearance(category).persona is Persona.STAFF


def test_a_space_category_is_matched_on_its_numeric_code_never_the_title() -> None:
    assert space_category_clearance("13-15 11 34 11: Corridor").persona is Persona.STAFF
    assert space_category_clearance("13-85 11 11").persona is Persona.VISITOR
    assert space_category_clearance("13-85 11 11: Bureau").persona is Persona.VISITOR


def test_every_space_category_clearance_is_an_authored_default() -> None:
    for category in ("13-85 11 11: Corridor", "13-81 21 21: Electrical Room", None):
        cleared = space_category_clearance(category)
        assert cleared.authored_default
        assert cleared.access is None
