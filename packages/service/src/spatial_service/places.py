"""Place resolution: which space a mention names, among the spaces a persona can see.

The tool is handed a venue already cut down to one persona's visible set in one
snapshot. It never sees the rest, so a place beyond clearance and a place that
does not exist come out the same: nothing found.

In order, the first that finds anything decides:

    label             the text is a space's label, exactly, after normalising
    building + room   the text parses as a building (number or name) and a room
                      number, the way room-label@1 composes labels: "50.34 room 107",
                      "50.34/107", "room 107 in 50.34"
    bare number       the text is a number alone. A room number is never a label,
                      so any room carrying it means a clarification, even if only
                      one does. A building's number alone resolves the building.
    name              every word of the text is a word of a space's name

One match resolves. More than one asks which, naming at most CLARIFY_CAP of them.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property

from spatial_core.records import Persona, Space

# How many candidates a clarification names.
CLARIFY_CAP = 6

ROOM = "room"
BUILDING = "building"

_NOISE = re.compile(r"^[\s\"'„“”‚‘’«»?!.,;:]+|[\s\"'„“”‚‘’«»?!.,;:]+$")
_ARTICLE = re.compile(r"^(?:the|der|die|das|den|dem) ")
_ROOM_WORD = r"(?:room|raum|zimmer|nr\.?)"
_BUILDING_WORD = r"(?:building|gebäude|geb\.?)"
_ROOM_IN_BUILDING = re.compile(
    rf"(?:{_ROOM_WORD} )?(?P<room>\S+) (?:in|im|at|of) (?:{_BUILDING_WORD} )?(?P<building>.+)"
)
_BUILDING_THEN_ROOM = re.compile(
    rf"(?:{_BUILDING_WORD} )?(?P<building>.+?)(?: {_ROOM_WORD} | ?/ ?| )(?P<room>[^\s/]+)"
)
_BARE_NUMBER = re.compile(rf"(?:(?:{_ROOM_WORD}|{_BUILDING_WORD}) )?(?P<number>[0-9][^\s/]*)")


def normalise(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    folded = re.sub(r"\s+", " ", folded).strip()
    return _ARTICLE.sub("", _NOISE.sub("", folded))


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", normalise(text)))


def first_ref(space: Space) -> str | None:
    """A space's number as room-label@1 prints it: the first of its refs."""
    if space.ref is None:
        return None
    return normalise(space.ref.value.split(";")[0]) or None


@dataclass(frozen=True)
class Venue:
    """One snapshot as one persona sees it: their visible spaces and nothing else."""

    snapshot: str
    persona: Persona
    spaces: Mapping[str, Space]

    def __contains__(self, space_id: object) -> bool:
        return space_id in self.spaces

    def ancestors(self, space: Space) -> list[Space]:
        """Enclosing spaces, innermost first. All visible, since visibility closes upward."""
        return [self.spaces[a.id] for a in reversed(space.ancestor_path) if a.id in self.spaces]

    def building_of(self, space: Space) -> Space | None:
        return next((a for a in self.ancestors(space) if tier(a) == BUILDING), None)

    @cached_property
    def by_label(self) -> dict[str, list[Space]]:
        index: dict[str, list[Space]] = {}
        for space in self.spaces.values():
            if space.label is not None:
                index.setdefault(normalise(space.label.value), []).append(space)
        return index


def tier(space: Space) -> str:
    return space.tier.value


class Outcome(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class How(StrEnum):
    LABEL = "label"
    BUILDING_AND_ROOM = "building-and-room"
    BARE_NUMBER = "bare-number"
    NAME = "name"


@dataclass(frozen=True)
class PlaceResult:
    outcome: Outcome
    # The one resolved, or the candidates to ask about. Empty for none.
    ids: tuple[str, ...]
    how: How | None
    # How many matched in all, when a clarification names fewer.
    matched: int


def resolve_place(venue: Venue, text: str, within: Collection[str] | None = None) -> PlaceResult:
    """The place ``text`` names in ``venue``, inside one of ``within`` if given."""
    query = normalise(text)
    # A space with no label cannot be addressed, so it is never an answer or a candidate.
    pool = [s for s in venue.spaces.values() if s.label is not None and _inside(s, within)]
    if not query or not pool:
        return PlaceResult(Outcome.NONE, (), None, 0)

    for how, find in (
        (How.LABEL, _by_label),
        (How.BUILDING_AND_ROOM, _by_building_and_room),
        (How.BARE_NUMBER, _by_bare_number),
        (How.NAME, _by_name),
    ):
        found = find(venue, pool, query)
        if found is None:
            continue
        matches, must_ask = found
        if not matches:
            continue
        ordered = sorted(matches, key=lambda s: (s.label.value if s.label else "", s.id))
        # A number alone asks which, unless an earlier step already said where to look.
        if len(ordered) == 1 and not (must_ask and within is None):
            return PlaceResult(Outcome.RESOLVED, (ordered[0].id,), how, 1)
        return PlaceResult(
            Outcome.AMBIGUOUS,
            tuple(s.id for s in ordered[:CLARIFY_CAP]),
            how,
            len(ordered),
        )
    return PlaceResult(Outcome.NONE, (), None, 0)


def _inside(space: Space, within: Collection[str] | None) -> bool:
    return within is None or any(a.id in within for a in space.ancestor_path)


_Found = tuple[list[Space], bool] | None


def _by_label(venue: Venue, pool: list[Space], query: str) -> _Found:
    ids = {s.id for s in pool}
    return [s for s in venue.by_label.get(query, []) if s.id in ids], False


def _by_building_and_room(venue: Venue, pool: list[Space], query: str) -> _Found:
    parsed = _ROOM_IN_BUILDING.fullmatch(query) or _BUILDING_THEN_ROOM.fullmatch(query)
    if parsed is None:
        return None
    building, room = normalise(parsed["building"]), normalise(parsed["room"])
    matches = []
    for space in pool:
        if tier(space) != ROOM or first_ref(space) != room:
            continue
        enclosing = venue.building_of(space)
        if enclosing is not None and building in _building_names(enclosing):
            matches.append(space)
    return matches, False


def _building_names(building: Space) -> set[str]:
    return {normalise(v.value) for v in (building.ref, building.name, building.label) if v}


def _by_bare_number(venue: Venue, pool: list[Space], query: str) -> _Found:
    parsed = _BARE_NUMBER.fullmatch(query)
    if parsed is None:
        return None
    number = normalise(parsed["number"])
    rooms = [s for s in pool if tier(s) == ROOM and first_ref(s) == number]
    buildings = [s for s in pool if tier(s) == BUILDING and first_ref(s) == number]
    return rooms + buildings, bool(rooms)


def _by_name(venue: Venue, pool: list[Space], query: str) -> _Found:
    wanted = _words(query)
    if not wanted:
        return None
    exact = [s for s in pool if s.name is not None and normalise(s.name.value) == query]
    if exact:
        return exact, False
    return [s for s in pool if s.name is not None and wanted <= _words(s.name.value)], False
