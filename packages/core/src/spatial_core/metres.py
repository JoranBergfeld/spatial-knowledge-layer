"""Metres from WGS84, by a flat projection around a fixed point.

Good to well under a percent across a campus, which is all a snap of two metres or
a modelled walk needs. The same two constants as the seed layout rule uses.
"""

from __future__ import annotations

import math

# Metres per degree of latitude, and of longitude at the equator.
M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON = 111_320.0
# Degrees are rounded to this many decimals on the way back: about a centimetre.
_DECIMALS = 7


class Local:
    """A projection to metres east and north of a fixed point."""

    def __init__(self, lon0: float, lat0: float) -> None:
        self.lon0, self.lat0 = lon0, lat0
        self.kx = M_PER_DEG_LON * math.cos(math.radians(lat0))

    def metres(self, lon: float, lat: float, _z: float | None = None) -> tuple[float, float]:
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * M_PER_DEG_LAT

    def degrees(self, x: float, y: float) -> tuple[float, float]:
        return (
            round(self.lon0 + x / self.kx, _DECIMALS),
            round(self.lat0 + y / M_PER_DEG_LAT, _DECIMALS),
        )
