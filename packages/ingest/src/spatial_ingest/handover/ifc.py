"""Where each element of an IFC model is, read straight from the STEP file.

Only placements are read: for every product with a GlobalId and a local
placement, its origin in the model's own coordinates, in metres. A local
placement sits relative to another, each with its own axes, so the chain is
followed and turned on the way. No geometry, no properties, no relations.

This is a small reader of its own and not ifcopenshell, which is a large
dependency for one number per element. It knows IFC2x3 and IFC4 placements and
nothing else, and it has only ever read the hand-made fixture.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

Point = tuple[float, float, float]
_Axes = tuple[Point, Point, Point]

_ENTITY = re.compile(r"#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)\s*$", re.DOTALL)
_KEPT = ("IFCCARTESIANPOINT", "IFCDIRECTION", "IFCAXIS2PLACEMENT3D", "IFCLOCALPLACEMENT")
_PREFIXES = {"MILLI": 0.001, "CENTI": 0.01, "DECI": 0.1, "KILO": 1000.0}
_IDENTITY: _Axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
# A GlobalId is 22 characters of IFC's own base 64.
_GLOBAL_ID = re.compile(r"'([0-9A-Za-z_$]{22})'")


class IfcError(Exception):
    pass


def _statements(text: str) -> list[str]:
    """The file's statements. A statement ends at a semicolon outside a string."""
    found, start, quoted = [], 0, False
    for position, character in enumerate(text):
        if character == "'":
            quoted = not quoted
        elif character == ";" and not quoted:
            found.append(text[start:position].strip())
            start = position + 1
    return found


def _arguments(text: str) -> list[str]:
    """An entity's arguments, split at the commas outside strings and brackets."""
    found, depth, quoted, start = [], 0, False, 0
    for position, character in enumerate(text):
        if character == "'":
            quoted = not quoted
        elif quoted:
            continue
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
        elif character == "," and depth == 0:
            found.append(text[start:position].strip())
            start = position + 1
    found.append(text[start:].strip())
    return found


def _numbers(text: str) -> Point:
    values = [float(part) for part in text.strip("() \n\t").split(",") if part.strip()]
    values += [0.0] * (3 - len(values))
    return (values[0], values[1], values[2])


def _ref(text: str) -> int | None:
    return int(text[1:]) if text.startswith("#") else None


def _unit(vector: Point) -> Point:
    length = math.sqrt(sum(part * part for part in vector)) or 1.0
    return (vector[0] / length, vector[1] / length, vector[2] / length)


def _cross(a: Point, b: Point) -> Point:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _turned(axes: _Axes, point: Point) -> Point:
    """``point``, given along ``axes``, in the coordinates the axes are written in."""
    x, y, z = axes
    return (
        x[0] * point[0] + y[0] * point[1] + z[0] * point[2],
        x[1] * point[0] + y[1] * point[1] + z[1] * point[2],
        x[2] * point[0] + y[2] * point[1] + z[2] * point[2],
    )


class _Model:
    def __init__(self, text: str) -> None:
        self.entities: dict[int, tuple[str, list[str]]] = {}
        self.products: list[tuple[str, int]] = []
        self.scale = 1.0
        for statement in _statements(text):
            parsed = _ENTITY.match(statement)
            if parsed is None:
                continue
            number, kind, body = int(parsed[1]), parsed[2], parsed[3]
            if kind in _KEPT:
                self.entities[number] = (kind, _arguments(body))
            elif kind == "IFCSIUNIT" and ".LENGTHUNIT." in body:
                prefix = _arguments(body)[2].strip(".")
                self.scale = _PREFIXES.get(prefix, 1.0)
            elif _GLOBAL_ID.match(body):
                # Every IfcProduct has its placement sixth, after the five an IfcObject has.
                arguments = _arguments(body)
                placed = _ref(arguments[5]) if len(arguments) > 5 else None
                if placed is not None:
                    self.products.append((arguments[0].strip("'"), placed))
        self._placed: dict[int, tuple[Point, _Axes]] = {}

    def _point(self, number: int | None) -> Point:
        entity = self.entities.get(number) if number is not None else None
        if entity is None or entity[0] not in ("IFCCARTESIANPOINT", "IFCDIRECTION"):
            raise IfcError(f"#{number} is not a point or a direction")
        return _numbers(entity[1][0])

    def _axes(self, arguments: list[str]) -> _Axes:
        axis, direction = _ref(arguments[1]), _ref(arguments[2])
        z = _unit(self._point(axis)) if axis is not None else _IDENTITY[2]
        toward = self._point(direction) if direction is not None else _IDENTITY[0]
        # The reference direction need not be square to the axis. Its part along it is dropped.
        along = sum(a * b for a, b in zip(toward, z, strict=True))
        x = _unit((toward[0] - along * z[0], toward[1] - along * z[1], toward[2] - along * z[2]))
        return (x, _cross(z, x), z)

    def placement(self, number: int) -> tuple[Point, _Axes]:
        """A local placement's origin and axes in the model's coordinates."""
        if number in self._placed:
            return self._placed[number]
        chain: list[int] = []
        at: int | None = number
        while at is not None and at not in self._placed:
            entity = self.entities.get(at)
            if entity is None or entity[0] != "IFCLOCALPLACEMENT" or at in chain:
                raise IfcError(f"#{at} is not a local placement")
            chain.append(at)
            at = _ref(entity[1][0])
        origin, axes = self._placed[at] if at is not None else ((0.0, 0.0, 0.0), _IDENTITY)
        for link in reversed(chain):
            relative = self.entities.get(_ref(self.entities[link][1][1]) or -1)
            if relative is None or relative[0] != "IFCAXIS2PLACEMENT3D":
                raise IfcError(f"#{link} has no 3D placement")
            moved = _turned(axes, self._point(_ref(relative[1][0])))
            origin = (origin[0] + moved[0], origin[1] + moved[1], origin[2] + moved[2])
            own = self._axes(relative[1])
            axes = (_turned(axes, own[0]), _turned(axes, own[1]), _turned(axes, own[2]))
            self._placed[link] = (origin, axes)
        return origin, axes


def placements(path: Path) -> dict[str, Point]:
    """GlobalId to where the element is, in metres, for every placed product in the file."""
    try:
        model = _Model(path.read_text(encoding="utf-8", errors="replace"))
        found = {}
        for global_id, placed in model.products:
            # A relation has a GlobalId first and may have a reference sixth too.
            if model.entities.get(placed, ("",))[0] != "IFCLOCALPLACEMENT":
                continue
            origin, _ = model.placement(placed)
            found[global_id] = (
                origin[0] * model.scale,
                origin[1] * model.scale,
                origin[2] * model.scale,
            )
        return found
    except (OSError, ValueError, IndexError, IfcError) as error:
        raise IfcError(f"cannot read placements from {path}: {error}") from error
