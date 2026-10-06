"""layout-in-outline: where generated assets stand in a room, from its real outline.

The outline is projected to local metres around its own centroid, which is close
enough at room scale. It is inset from the walls; a room too small to inset keeps
its outline.

    grid       floor-standing things: the centres of a square grid whose cell is
               the room's area over the count, kept where they fall inside the
               inset outline. The grid shrinks until all of them fit.
    perimeter  wall-mounted things: evenly along the inset outline, starting at
               its first vertex.

A room without an outline gets no placement. The rule needs no randomness: the
same outline and count give the same points.
"""

from __future__ import annotations

import math
from typing import Any

from shapely import Point, Polygon
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

RULE = "layout-in-outline@1"

GRID = "grid"
PERIMETER = "perimeter"
LAYOUTS = (GRID, PERIMETER)

# Metres per degree of latitude, and of longitude at the equator.
_M_PER_DEG_LAT = 110_574.0
_M_PER_DEG_LON = 111_320.0
# How far from the walls things stand.
INSET_M = 0.5
# Coordinates are rounded to this many decimals: about a centimetre.
_DECIMALS = 7


class _Local:
    """A projection to metres east and north of a fixed point."""

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0, self.lat0 = lon0, lat0
        self.kx = _M_PER_DEG_LON * math.cos(math.radians(lat0))

    def to_metres(self, geometry: BaseGeometry) -> BaseGeometry:
        return transform(self._metres, geometry)

    def _metres(self, x: float, y: float, z: float | None = None) -> tuple[float, float]:
        return (x - self.lon0) * self.kx, (y - self.lat0) * _M_PER_DEG_LAT

    def to_degrees(self, x: float, y: float) -> tuple[float, float]:
        return (
            round(self.lon0 + x / self.kx, _DECIMALS),
            round(self.lat0 + y / _M_PER_DEG_LAT, _DECIMALS),
        )


def _polygon(outline: dict[str, Any] | None) -> BaseGeometry | None:
    if outline is None or outline.get("type") not in ("Polygon", "MultiPolygon"):
        return None
    geometry = shape(outline)
    return geometry if geometry.is_valid and not geometry.is_empty else None


def area_m2(outline: dict[str, Any] | None) -> float | None:
    """The outline's area in square metres. None for a space without one."""
    geometry = _polygon(outline)
    if geometry is None:
        return None
    centre = geometry.centroid
    return float(_Local(centre.x, centre.y).to_metres(geometry).area)


def place(
    outline: dict[str, Any] | None, count: int, layout: str
) -> list[tuple[float, float]] | None:
    """``count`` (lon, lat) points in the outline. None when there is no outline."""
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}")
    geometry = _polygon(outline)
    if geometry is None:
        return None
    if count <= 0:
        return []
    centre = geometry.centroid
    local = _Local(centre.x, centre.y)
    room = local.to_metres(geometry)
    inset = room.buffer(-INSET_M)
    area = inset if not inset.is_empty and inset.area > 0 else room
    points = _perimeter(area, count) if layout == PERIMETER else _grid(area, count)
    return [local.to_degrees(point.x, point.y) for point in points]


def _rings(area: BaseGeometry) -> list[Polygon]:
    polygons = list(getattr(area, "geoms", [area]))
    return sorted(
        (p for p in polygons if isinstance(p, Polygon)), key=lambda p: (-p.area, p.bounds)
    )


def _perimeter(area: BaseGeometry, count: int) -> list[Point]:
    # Along the largest part's outer wall.
    ring = _rings(area)[0].exterior
    step = ring.length / count
    return [ring.interpolate(step * (index + 0.5)) for index in range(count)]


def _grid(area: BaseGeometry, count: int) -> list[Point]:
    min_x, min_y, max_x, max_y = area.bounds
    cell = math.sqrt(area.area / count)
    while cell > 0.05:
        columns = max(1, math.ceil((max_x - min_x) / cell))
        rows = max(1, math.ceil((max_y - min_y) / cell))
        # Centre the grid on the bounding box so it does not hug one wall.
        x0 = min_x + ((max_x - min_x) - (columns - 1) * cell) / 2
        y0 = min_y + ((max_y - min_y) - (rows - 1) * cell) / 2
        inside = [
            point
            for row in range(rows)
            for column in range(columns)
            if area.covers(point := Point(x0 + column * cell, y0 + row * cell))
        ]
        if len(inside) >= count:
            return inside[:count]
        cell *= 0.9
    # Nothing left to shrink: stack them on the one point surely inside.
    return [area.representative_point()] * count
