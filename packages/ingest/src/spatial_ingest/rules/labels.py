"""room-label: how a space is named wherever the layer prints it.

The sourced name where there is one. Otherwise the enclosing building's number
(or its name, where it has no number) plus the space's own number. A bare room
number is never a label, and a label is never inferred. No label means no rule
can address the space. It is still kept.
"""

from __future__ import annotations

from dataclasses import dataclass

RULE = "room-label@1"


@dataclass(frozen=True)
class Label:
    text: str
    # True when the label is the sourced name as is, False when the rule composed it.
    from_name: bool


def _clean(text: str | None) -> str | None:
    return (text or "").strip() or None


def first_ref(ref: str | None) -> str | None:
    """A room with several refs prints the first. The ref itself prints verbatim."""
    return _clean((ref or "").split(";")[0])


def _building_part(number: str | None, name: str | None) -> str | None:
    return _clean(number) or _clean(name)


def building_label(*, name: str | None, number: str | None) -> Label | None:
    if cleaned := _clean(name):
        return Label(cleaned, from_name=True)
    if cleaned := _clean(number):
        return Label(cleaned, from_name=False)
    return None


def floor_label(
    *, building_number: str | None, building_name: str | None, level: str
) -> Label | None:
    building = _building_part(building_number, building_name)
    if building is None:
        return None
    return Label(f"{building} level {level}", from_name=False)


def room_label(
    *,
    name: str | None,
    ref: str | None,
    building_number: str | None,
    building_name: str | None,
) -> Label | None:
    if cleaned := _clean(name):
        return Label(cleaned, from_name=True)
    building = _building_part(building_number, building_name)
    number = first_ref(ref)
    if building is None or number is None:
        return None
    return Label(f"{building} room {number}", from_name=False)
