from typing import Any

import pytest
from shapely import Point
from shapely.geometry import shape

from spatial_ingest.rules import layout

# A room of about 22 by 11 metres near Karlsruhe.
ROOM: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [[8.41, 49.01], [8.4103, 49.01], [8.4103, 49.0101], [8.41, 49.0101], [8.41, 49.01]]
    ],
}
# An L-shaped room, so a grid over its bounding box has cells outside it.
L_ROOM: dict[str, Any] = {
    "type": "Polygon",
    "coordinates": [
        [
            [8.41, 49.01],
            [8.4104, 49.01],
            [8.4104, 49.01005],
            [8.41005, 49.01005],
            [8.41005, 49.0104],
            [8.41, 49.0104],
            [8.41, 49.01],
        ]
    ],
}


def test_area_is_in_square_metres() -> None:
    area = layout.area_m2(ROOM)

    assert area is not None
    assert 230 < area < 250


def test_no_outline_no_area_and_no_placement() -> None:
    assert layout.area_m2(None) is None
    assert layout.place(None, 3, layout.GRID) is None
    assert layout.place({"type": "Point", "coordinates": [8.41, 49.01]}, 3, layout.GRID) is None


@pytest.mark.parametrize("outline", [ROOM, L_ROOM])
@pytest.mark.parametrize("count", [1, 7, 40])
@pytest.mark.parametrize("kind", layout.LAYOUTS)
def test_every_point_lies_in_the_outline(outline: dict[str, Any], count: int, kind: str) -> None:
    points = layout.place(outline, count, kind)

    assert points is not None and len(points) == count
    room = shape(outline)
    assert all(room.covers(Point(lon, lat)) for lon, lat in points)


def test_a_grid_spreads_its_points_and_never_stacks_them() -> None:
    points = layout.place(L_ROOM, 20, layout.GRID)

    assert points is not None
    assert len(set(points)) == 20


def test_wall_mounted_things_stand_along_the_walls() -> None:
    points = layout.place(ROOM, 6, layout.PERIMETER)

    assert points is not None
    walls = shape(ROOM).exterior
    room = shape(ROOM)
    # Within a metre or so of a wall, which is about 0.00001 degrees.
    assert all(walls.distance(Point(p)) < 0.00002 for p in points)
    assert all(room.distance(Point(p)) == 0 for p in points)


def test_the_same_outline_and_count_give_the_same_points() -> None:
    assert layout.place(L_ROOM, 12, layout.GRID) == layout.place(L_ROOM, 12, layout.GRID)


def test_an_unknown_layout_is_refused() -> None:
    with pytest.raises(ValueError):
        layout.place(ROOM, 1, "ceiling")
