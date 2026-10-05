"""level-parse: reads the levels a space occupies from its OSM `level` tag.

Simple Indoor Tagging: values are separated by `;`, a piece is a number or an
`a-b` range, half levels are decimals. A range fills in the whole levels between
its ends. Pieces the rule cannot read are skipped.

`repeat_on` is left alone. It says the same shape is copied onto other levels,
which is several spaces and not one space spanning floors.
"""

from __future__ import annotations

import math
import re
from decimal import Decimal

RULE = "level-parse@1"

_NUMBER = r"-?\d+(?:\.\d+)?"
_RANGE = re.compile(rf"^({_NUMBER})-({_NUMBER})$")


def _number(text: str) -> Decimal | None:
    return Decimal(text) if re.fullmatch(_NUMBER, text) else None


def _piece(text: str) -> list[Decimal]:
    single = _number(text)
    if single is not None:
        return [single]
    span = _RANGE.match(text)
    if not span:
        return []
    low, high = sorted((Decimal(span.group(1)), Decimal(span.group(2))))
    between = [Decimal(n) for n in range(math.ceil(low), math.floor(high) + 1)]
    return [low, *between, high]


def _canonical(level: Decimal) -> str:
    if level == level.to_integral_value():
        return str(int(level))
    return format(level.normalize(), "f")


def parse_levels(level: str | None) -> tuple[str, ...]:
    """Every level named by the tag, lowest first, in canonical spelling."""
    found: set[Decimal] = set()
    for piece in (level or "").split(";"):
        found.update(_piece(piece.strip()))
    return tuple(_canonical(value) for value in sorted(found))
