"""snap-doorless-room: a room nobody mapped a door on, joined to the circulation beside it.

Drawn in metres, one unit a metre, x east and y north. The corridor is always
(0,0)-(20,2), so its north wall is y=2.
"""

from __future__ import annotations

import pytest
from shapely.geometry import Polygon

from spatial_core.metres import Local
from spatial_ingest.rules.snap import SNAP_M, Outline, Snap, snaps

LOCAL = Local(8.41, 49.01)
CORRIDOR = "corridor"
ROOM = "room"


def box(x0: float, y0: float, x1: float, y1: float, *levels: str) -> Outline:
    corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    return Outline(Polygon([LOCAL.degrees(x, y) for x, y in corners]), levels or ("0",))


def at(snap: Snap) -> tuple[float, float]:
    """Where a snap is, back in metres, to the centimetre."""
    x, y = LOCAL.metres(snap.lon, snap.lat)
    return round(x, 2), round(y, 2)


def test_a_doorless_room_is_joined_to_circulation_it_sits_within_two_metres_of() -> None:
    outlines = {CORRIDOR: box(0, 0, 20, 2), ROOM: box(5, 3.5, 9, 7)}

    [found] = snaps(outlines, [ROOM], [CORRIDOR])

    assert (found.room, found.circulation, found.level) == (ROOM, CORRIDOR, "0")
    # On the corridor's wall, straight below the room. Never inside the room.
    x, y = at(found)
    assert y == 2.0
    assert 5.0 <= x <= 9.0


def test_the_snap_distance_is_two_metres_and_not_a_step_further() -> None:
    assert SNAP_M == 2.0
    corridor = box(0, 0, 20, 2)

    near = snaps({CORRIDOR: corridor, ROOM: box(5, 3.9, 9, 7)}, [ROOM], [CORRIDOR])
    far = snaps({CORRIDOR: corridor, ROOM: box(5, 4.1, 9, 7)}, [ROOM], [CORRIDOR])

    assert [found.room for found in near] == [ROOM]
    assert far == []


def test_the_nearest_circulation_wins_and_a_tie_goes_by_id() -> None:
    outlines = {
        "corridor-far": box(0, 0, 20, 2),
        "corridor-near": box(0, 8, 20, 10),
        ROOM: box(5, 3.5, 9, 7),
    }
    # 1.5 m to the one below, 1 m to the one above.
    [found] = snaps(outlines, [ROOM], ["corridor-far", "corridor-near"])
    assert found.circulation == "corridor-near"

    level = {"b": box(0, 0, 20, 2), "a": box(0, 8, 20, 10), ROOM: box(5, 3, 9, 7)}
    [tied] = snaps(level, [ROOM], ["b", "a"])
    assert tied.circulation == "a"


def test_a_room_is_only_joined_to_circulation_on_a_level_they_share() -> None:
    upstairs = {CORRIDOR: box(0, 0, 20, 2, "0"), ROOM: box(5, 3, 9, 7, "1")}
    stairwell = {CORRIDOR: box(0, 0, 20, 2, "1", "2"), ROOM: box(5, 3, 9, 7, "0", "1", "2")}

    assert snaps(upstairs, [ROOM], [CORRIDOR]) == []
    # The first level of the circulation's that the room is on too.
    [found] = snaps(stairwell, [ROOM], [CORRIDOR])
    assert found.level == "1"


def test_two_rooms_are_never_joined_through_a_wall() -> None:
    # Wall to wall, and neither is circulation.
    outlines = {"room-a": box(0, 0, 4, 4), "room-b": box(4, 0, 8, 4), CORRIDOR: box(0, 20, 20, 22)}

    assert snaps(outlines, ["room-a", "room-b"], [CORRIDOR]) == []


def test_a_space_is_never_snapped_to_itself() -> None:
    # A doorless stairwell is a room and circulation both.
    outlines = {"stairs": box(5, 3, 9, 7), CORRIDOR: box(0, 0, 20, 2)}

    [found] = snaps(outlines, ["stairs"], ["stairs", CORRIDOR])

    assert (found.room, found.circulation) == ("stairs", CORRIDOR)


def test_two_doorless_shafts_side_by_side_are_joined_once() -> None:
    # Two lifts wall to wall, each a room and circulation both, with no door mapped.
    # Each is the other's nearest, which used to make the same pair twice.
    outlines = {"lift-a": box(0, 0, 3, 3), "lift-b": box(3, 0, 6, 3)}

    found = snaps(outlines, ["lift-a", "lift-b"], ["lift-a", "lift-b"])

    assert [(one.room, one.circulation) for one in found] == [("lift-a", "lift-b")]


def test_each_room_gets_one_snap_in_a_stable_order() -> None:
    outlines = {
        CORRIDOR: box(0, 0, 20, 2),
        "room-b": box(11, 3, 15, 7),
        "room-a": box(5, 3, 9, 7),
        "room-far": box(5, 30, 9, 34),
    }

    found = snaps(outlines, ["room-far", "room-b", "room-a"], [CORRIDOR])

    assert [(one.room, one.circulation) for one in found] == [
        ("room-a", CORRIDOR),
        ("room-b", CORRIDOR),
    ]


@pytest.mark.parametrize(
    ("doorless", "circulation"),
    [
        ([], [CORRIDOR]),
        ([ROOM], []),
        # A room mapped as a point has no outline to measure from.
        (["room-with-no-outline"], [CORRIDOR]),
        ([ROOM], ["corridor-with-no-outline"]),
    ],
)
def test_nothing_to_join_is_no_snaps(doorless: list[str], circulation: list[str]) -> None:
    outlines = {CORRIDOR: box(0, 0, 20, 2), ROOM: box(5, 3, 9, 7)}

    assert snaps(outlines, doorless, circulation) == []
