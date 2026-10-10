"""crossable-by-tags and crossable-by-door-count: what a route may pass through."""

from __future__ import annotations

import pytest

from spatial_core.circulation import (
    DOOR_COUNT_RULE,
    TAGS_RULE,
    crossable_rule,
    is_circulation,
    is_lift,
    is_room,
    is_stairs,
)


@pytest.mark.parametrize(
    "tags",
    [
        {"indoor": "corridor"},
        {"indoor": "area"},
        {"indoor": "room", "room": "hall"},
        {"indoor": "room", "room": "lobby"},
        {"indoor": "room", "room": "entrance"},
        {"indoor": "room", "room": "foyer"},
        {"indoor": "room", "room": "corridor"},
        {"highway": "footway"},
        {"highway": "pedestrian"},
        {"indoor": "room", "room": "stairs"},
        {"indoor": "room", "room": "elevator"},
    ],
)
def test_circulation_is_crossable_by_its_tags_however_many_doors_it_has(
    tags: dict[str, str],
) -> None:
    assert is_circulation(tags)
    assert crossable_rule(tags, 0) == TAGS_RULE
    assert crossable_rule(tags, 5) == TAGS_RULE


def test_an_untagged_room_is_crossable_from_three_doors_up_by_the_derived_rule() -> None:
    room = {"indoor": "room"}

    assert not is_circulation(room)
    assert crossable_rule(room, 2) is None
    assert crossable_rule(room, 3) == DOOR_COUNT_RULE
    assert crossable_rule(room, 7) == DOOR_COUNT_RULE


@pytest.mark.parametrize(
    "tags",
    [
        {"indoor": "room", "room": "office"},
        {"indoor": "room", "room": "seminar"},
        {"indoor": "room", "amenity": "toilets"},
        {"indoor": "room", "office": "university"},
    ],
)
def test_a_room_with_a_sourced_function_is_never_crossed_whatever_its_doors(
    tags: dict[str, str],
) -> None:
    assert crossable_rule(tags, 3) is None
    assert crossable_rule(tags, 12) is None


def test_a_name_or_a_number_is_not_a_function() -> None:
    # Only room, amenity and office say what a room is for.
    assert crossable_rule({"indoor": "room", "name": "Kopierraum", "ref": "107"}, 3) == (
        DOOR_COUNT_RULE
    )


def test_door_count_only_ever_applies_to_something_mapped_as_a_room() -> None:
    assert crossable_rule({}, 9) is None
    assert crossable_rule({"building": "yes"}, 9) is None
    assert crossable_rule({"indoor": "wall"}, 9) is None


def test_stairs_are_told_by_room_by_a_stairs_tag_or_by_steps() -> None:
    assert is_stairs({"room": "stairs"})
    assert is_stairs({"stairs": "yes"})
    assert is_stairs({"highway": "steps"})
    assert not is_stairs({"stairs": "no"})
    assert not is_stairs({"room": "elevator"})
    assert not is_stairs({"indoor": "corridor"})


def test_a_lift_is_told_by_room_or_by_highway() -> None:
    assert is_lift({"room": "elevator"})
    assert is_lift({"highway": "elevator"})
    assert not is_lift({"room": "stairs"})
    assert not is_lift({"indoor": "room"})


def test_a_room_is_what_is_mapped_indoor_room_and_a_corridor_or_area_is_not() -> None:
    assert is_room({"indoor": "room"})
    assert is_room({"indoor": "room", "room": "stairs"})
    assert not is_room({"indoor": "corridor"})
    assert not is_room({"indoor": "area"})
    assert not is_room({"building": "yes"})
