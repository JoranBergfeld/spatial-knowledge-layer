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

Text with nothing to look up once normalised (punctuation, an article alone) is a
malformed argument, rejected before the venue is read, so a rejection never
depends on what the persona can see.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cached_property

from spatial_core.records import Asset, AssetType, Edge, Persona, ServiceEvent, Space
from spatial_core.search import IndexRef

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
    """One snapshot as one persona sees it: what is visible to them and nothing else."""

    snapshot: str
    persona: Persona
    spaces: Mapping[str, Space]
    assets: Mapping[str, Asset] = field(default_factory=dict)
    # Only the types of visible assets: a type is visible through one or not at all.
    asset_types: Mapping[str, AssetType] = field(default_factory=dict)
    # Every tier the snapshot has, outermost first, whoever is asking. A tier is a
    # word, not a place, so it is the one thing here that is not cut to the persona:
    # whether "floor" is a scale level must not depend on who asks.
    tiers: tuple[str, ...] = ()
    # The snapshot's search index, where it has one. Which chunks of it the
    # persona may read is the retrieval filter's business, not the venue's.
    index: IndexRef | None = None
    # The service events of visible assets, by asset. An event is hidden with its asset.
    service_events: Mapping[str, tuple[ServiceEvent, ...]] = field(default_factory=dict)
    # The circulation edges the persona may take, between spaces they can see.
    edges: tuple[Edge, ...] = ()
    # How many doors the persona may take each space has (routes.door_counts).
    doors: Mapping[str, int] = field(default_factory=dict)
    # The edge a route starts by when the question names no origin: one entrance,
    # the same for every persona. None when the venue names none, or this persona
    # may not take it.
    default_origin: str | None = None

    def __contains__(self, space_id: object) -> bool:
        return space_id in self.spaces

    @cached_property
    def root(self) -> Space | None:
        return next((s for s in self.spaces.values() if s.parent_id is None), None)

    @cached_property
    def assets_under(self) -> dict[str, list[Asset]]:
        """Each space's assets, its own and those of every space inside it.

        By ancestor path, which is containment and nothing else, so an asset in a
        space spanning several floors sits under the one floor that space hangs from.
        """
        index: dict[str, list[Asset]] = {}
        for asset in self.assets.values():
            for ancestor in asset.ancestor_path:
                index.setdefault(ancestor.id, []).append(asset)
        return index

    @cached_property
    def memo(self) -> dict[object, object]:
        """Scratch space for whatever is worked out once per venue, such as the collapsed tree."""
        return {}

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


def label(space: Space) -> str | None:
    """A space's label, or None where no rule can address it."""
    return space.label.value if space.label else None


def labelled(space: Space) -> str:
    """The label of a space that is known to have one."""
    assert space.label is not None
    return space.label.value


class Outcome(StrEnum):
    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    NONE = "none"
    # The tool rejected its arguments and looked nothing up.
    REJECTED = "rejected"


class ArgumentsRejected(Exception):
    """A tool's arguments are malformed: it looked nothing up."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class How(StrEnum):
    LABEL = "label"
    BUILDING_AND_ROOM = "building-and-room"
    BARE_NUMBER = "bare-number"
    NAME = "name"
    # A description fitting a set of places. Only places_matching gives it.
    DESCRIPTION = "description"


@dataclass(frozen=True)
class PlaceResult:
    outcome: Outcome
    # The one resolved, or the candidates to ask about. Empty for none.
    ids: tuple[str, ...]
    how: How | None
    # How many matched in all, when a clarification names fewer.
    matched: int


def check_arguments(text: str) -> list[str]:
    """Why ``text`` is no argument for resolve_place. Reads no venue, so no persona."""
    if not re.search(r"\w", normalise(text)):
        return [f"{text!r} has nothing to look up"]
    return []


def resolve_place(venue: Venue, text: str, within: Collection[str] | None = None) -> PlaceResult:
    """The place ``text`` names in ``venue``, inside one of ``within`` if given.

    Raises ArgumentsRejected when ``text`` fails check_arguments.
    """
    problems = check_arguments(text)
    if problems:
        raise ArgumentsRejected(problems)
    query = normalise(text)
    # A space with no label cannot be addressed, so it is never an answer or a candidate.
    pool = [s for s in venue.spaces.values() if s.label is not None and _inside(s, within)]
    if not pool:
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


# --- a description, which names a set -----------------------------------------------------

# Words a description uses for what a room is for, and the tag values that say so.
# Short on purpose. Anything not here is looked for in names.
_FUNCTIONS: dict[str, frozenset[str]] = {
    "toilet": frozenset({"toilets", "toilet"}),
    "wc": frozenset({"toilets", "toilet"}),
    "restroom": frozenset({"toilets", "toilet"}),
    "bathroom": frozenset({"toilets", "toilet"}),
    "lift": frozenset({"elevator"}),
    "elevator": frozenset({"elevator"}),
    "stair": frozenset({"stairs"}),
    "staircase": frozenset({"stairs"}),
    "stairwell": frozenset({"stairs"}),
    "auditorium": frozenset({"auditorium", "lecture"}),
    "lecture": frozenset({"auditorium", "lecture"}),
    "library": frozenset({"library"}),
    "seminar": frozenset({"seminar", "seminar_room"}),
    "classroom": frozenset({"class", "classroom"}),
    "entrance": frozenset({"entrance"}),
    "kitchen": frozenset({"kitchen"}),
    "office": frozenset({"office"}),
}
# Words that describe the asking, not the place.
_FILLER = frozenset({"nearest", "closest", "next", "a", "an", "any", "room", "hall"})
_FUNCTION_KEYS = ("room", "amenity")


def stem(word: str) -> str:
    # Enough to fold "lights" onto "light". Not a stemmer.
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def places_matching(venue: Venue, text: str) -> PlaceResult:
    """Every visible place a description fits: by what the room is for, else by name.

    A set, and nobody is asked which: "the nearest toilet" is all the toilets, and
    whoever reads the set picks. Rooms with no label count too. Nothing found is
    nothing found, whether none exists or none the persona can see.
    """
    problems = check_arguments(text)
    if problems:
        raise ArgumentsRejected(problems)
    words = {stem(word) for word in _words(text)} - _FILLER or {stem(word) for word in _words(text)}
    wanted = [_FUNCTIONS[word] for word in words if word in _FUNCTIONS]
    found: list[Space] = []
    if wanted:
        for space in venue.spaces.values():
            functions = {
                space.attributes[key].value for key in _FUNCTION_KEYS if key in space.attributes
            }
            if all(functions & values for values in wanted):
                found.append(space)
    if not found:
        for space in venue.spaces.values():
            named = {stem(word) for word in _words(space.name.value)} if space.name else set()
            if words and words <= named:
                found.append(space)
    if not found:
        return PlaceResult(Outcome.NONE, (), None, 0)
    ids = tuple(sorted(space.id for space in found))
    return PlaceResult(Outcome.RESOLVED, ids, How.DESCRIPTION, len(ids))
