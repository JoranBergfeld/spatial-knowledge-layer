"""Template questions per question shape, each with what a correct turn would hold.

Every question is asked twice, as a visitor and as a maintenance technician, and
each of the two gets gold worked out over what that persona can see. The pair
shares a "pair" id.

A template only uses words that name one thing: a label no other space shares, a
building and room number, a type's full name. Whether the layer copes with loose
wording is what the hand-written questions are for, not these.

Routes and proximity have no templates. The test bench has no circulation graph,
so they are listed as unmeasured and left at that.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

from world import (
    SERVICE_DUE,
    Asset,
    Document,
    Space,
    View,
    World,
    fold,
    parse_day,
    words,
)

ASKED_AS = ("visitor", "maintenance")
UNMEASURED = ("proximity", "route")

ANSWER = "answer"
CLARIFICATION = "clarification"
ABSTENTION = "abstention"
COVERAGE_GAP = "coverage-gap"

Expected = dict[str, Any]
Evaluate = Callable[[View], Expected]


@dataclass(frozen=True)
class Template:
    shape: str
    name: str
    question: str
    slots: dict[str, Any]
    evaluate: Evaluate


def abstain(reason: str) -> Expected:
    return {"kind": ABSTENTION, "reason": reason}


def spread(items: list[Any], count: int) -> list[Any]:
    """``count`` of ``items``, evenly spaced, the first always among them."""
    if len(items) <= count:
        return list(items)
    return [items[(index * len(items)) // count] for index in range(count)]


def stratified(items: list[Any], key: Callable[[Any], Any], count: int) -> list[Any]:
    """``count`` of ``items``, taken in turn from each group ``key`` makes, evenly within one."""
    groups: dict[Any, list[Any]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    picked = [spread(groups[name], count) for name in sorted(groups, key=str)]
    found: list[Any] = []
    for index in range(count):
        for group in picked:
            if index < len(group) and len(found) < count:
                found.append(group[index])
    return found


class Naming:
    """The words a template may use for a space or a type: ones that name it and nothing else."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.labels = Counter(fold(s.label) for s in world.spaces.values() if s.label)
        self.names = Counter(fold(t.name) for t in world.types.values())

    def _confused(self, text: str, own: Space) -> bool:
        """Whether another space's name holds every word of ``text``."""
        wanted = words(text)
        return any(
            other.id != own.id and other.name and wanted <= words(other.name)
            for other in self.world.spaces.values()
        )

    def by_number(self, space: Space) -> str | None:
        """Building number and room number, where both exist and no other room shares them."""
        if space.tier != "room" or not space.ref or " " in space.ref:
            return None
        enclosing = [self.world.spaces[i] for i in self.world.path(space.id)]
        building = next((s for s in enclosing if s.tier == "building" and s.ref), None)
        if building is None:
            return None
        twins = [
            other
            for other in self.world.spaces.values()
            if other.tier == "room"
            and other.ref
            and fold(other.ref) == fold(space.ref)
            and building.id in self.world.path(other.id)
        ]
        return f"{building.ref} room {space.ref}" if len(twins) == 1 else None

    def place(self, space: Space) -> str | None:
        if space.label and self.labels[fold(space.label)] == 1:
            plain = all(c.isalnum() or c in " .-/" for c in space.label)
            if plain and not self._confused(space.label, space):
                return space.label
        return self.by_number(space)

    def kind(self, type_id: str) -> str | None:
        """A type's full name, where no other type's name holds all of its words."""
        name = self.world.types[type_id].name
        wanted = {word.rstrip("s") for word in words(name)}
        for other in self.world.types.values():
            if fold(other.name) == fold(name):
                continue
            if wanted <= {word.rstrip("s") for word in words(other.name)}:
                return None
        return name

    def same_name(self, type_id: str) -> list[str]:
        """Every type carrying this type's name. Asking by name asks for all of them."""
        name = fold(self.world.types[type_id].name)
        return sorted(t.id for t in self.world.types.values() if fold(t.name) == name)


def _gap_tiers(world: World) -> set[str]:
    """The tiers a coverage gap is stated at: a building and anything outside it."""
    if "building" in world.tiers:
        return set(world.tiers[: world.tiers.index("building") + 1])
    return set(world.tiers[:1])


def nothing_installed(view: View, place: str | None) -> Expected:
    """What a turn is when nothing the persona can see is installed in the scope.

    A gap in the venue's data is only ever stated for a building or anything
    bigger. A room or floor with nothing in it, inside a building that has
    installations, is an abstention that gives no reason.
    """
    world = view.world
    if place is None or world.spaces[place].tier in _gap_tiers(world):
        return {"kind": COVERAGE_GAP, "reason": "no-recorded-installations"}
    building = view.building_of(place)
    if building is not None and not view.under.get(building):
        return {"kind": COVERAGE_GAP, "reason": "no-recorded-installations"}
    return abstain("nothing-recorded")


def _scope(view: View, place: str | None, kinds: list[str] | None) -> Expected | list[Asset]:
    """The visible assets a question is about, or the turn it ends in before getting that far."""
    if place is not None and place not in view.spaces:
        return abstain("no-such-place")
    if kinds is not None and not any(kind in view.types for kind in kinds):
        return abstain("no-such-asset-type")
    installed = view.under.get(place, []) if place is not None else list(view.assets.values())
    if not installed:
        return nothing_installed(view, place)
    return [asset for asset in installed if kinds is None or asset.type in kinds]


def _ids(assets: list[Asset]) -> list[str]:
    return sorted(asset.id for asset in assets)


def located(space: Space) -> Evaluate:
    def evaluate(view: View) -> Expected:
        if space.id not in view.spaces:
            return abstain("no-such-place")
        return {"kind": ANSWER, "ids": [space.id], "within": view.world.path(space.id)}

    return evaluate


def among(matches: list[str]) -> Evaluate:
    """A mention several spaces answer to: whichever of them the persona can see."""

    def evaluate(view: View) -> Expected:
        seen = sorted(space_id for space_id in matches if space_id in view.spaces)
        if not seen:
            return abstain("no-such-place")
        if len(seen) == 1:
            return {"kind": ANSWER, "ids": seen, "within": view.world.path(seen[0])}
        return {"kind": CLARIFICATION, "candidates": seen, "matched": len(seen)}

    return evaluate


def bare_number(rooms: list[str]) -> Evaluate:
    """A room number alone is never a label, so it is asked about even when one room has it."""

    def evaluate(view: View) -> Expected:
        seen = sorted(room for room in rooms if room in view.spaces)
        if not seen:
            return abstain("no-such-place")
        return {"kind": CLARIFICATION, "candidates": seen, "matched": len(seen)}

    return evaluate


def contained(place: str | None, kinds: list[str] | None) -> Evaluate:
    def evaluate(view: View) -> Expected:
        found = _scope(view, place, kinds)
        if isinstance(found, dict):
            return found
        expected: Expected = {"kind": ANSWER, "ids": _ids(found), "total": len(found)}
        if kinds is None:
            expected["counts"] = dict(sorted(Counter(a.type for a in found).items()))
        return expected

    return evaluate


def counted(place: str | None, kinds: list[str] | None, tier: str) -> Evaluate:
    def evaluate(view: View) -> Expected:
        found = _scope(view, place, kinds)
        if isinstance(found, dict):
            return found
        world = view.world
        groups = sorted(
            s.id
            for s in view.spaces.values()
            if s.tier == tier and (place is None or place in world.path(s.id))
        )
        if not groups:
            return abstain("nothing-at-scale-level")
        counts: dict[str, int] = {}
        grouped: set[str] = set()
        for group in groups:
            # Nothing recorded there at all is not a zero, and is never ranked.
            if not view.under.get(group):
                continue
            inside = [a for a in found if group in (*world.path(a.space), a.space)]
            counts[group] = len(inside)
            grouped.update(a.id for a in inside)
        most = max(counts.values(), default=0)
        return {
            "kind": ANSWER,
            "ids": _ids(found),
            "total": len(found),
            "counts": counts,
            "not_recorded": [group for group in groups if group not in counts],
            "most": sorted(g for g, n in counts.items() if n == most and most > 0),
            "ungrouped": len(found) - len(grouped),
        }

    return evaluate


def filtered(
    place: str | None, kinds: list[str] | None, attribute: str, test: str, date: str | None
) -> Evaluate:
    def evaluate(view: View) -> Expected:
        if place is not None and place not in view.spaces:
            return abstain("no-such-place")
        if kinds is not None and not any(kind in view.types for kind in kinds):
            return abstain("no-such-asset-type")
        # An attribute only hidden assets carry is one that does not exist.
        if attribute not in view.catalogue:
            return abstain("attribute-unknown")
        if date is not None and not view.dated(attribute):
            return abstain("attribute-not-a-date")
        found = _scope(view, place, kinds)
        if isinstance(found, dict):
            return found
        carrying = [asset for asset in found if view.value(asset, attribute) is not None]
        fill = {
            "attribute": attribute,
            "carried": len(carrying),
            "of": len(found),
            "not_recorded": len(found) - len(carrying),
        }
        if not found:
            return {"kind": ANSWER, "ids": [], "total": 0, "fill": fill}
        if not carrying:
            return {"kind": COVERAGE_GAP, "reason": "attribute-not-recorded", "fill": fill}
        if test == "recorded":
            passing = carrying
        elif test == "not-recorded":
            passing = [asset for asset in found if asset not in carrying]
        else:
            day = parse_day(date)
            assert day is not None
            passing = []
            for asset in carrying:
                recorded = parse_day(view.value(asset, attribute))
                # Strictly before, strictly after.
                if recorded is not None and (
                    recorded < day if test == "before" else recorded > day
                ):
                    passing.append(asset)
        return {"kind": ANSWER, "ids": _ids(passing), "total": len(passing), "fill": fill}

    return evaluate


def documented(
    document: Document, place: str | None, kinds: list[str] | None, by_assets: bool
) -> Evaluate:
    """A question after one document. The answer may cite any document the anchors reach."""

    def evaluate(view: View) -> Expected:
        if by_assets:
            found = _scope(view, place, kinds)
            if isinstance(found, dict):
                # No assets to anchor to, so no document either way.
                return abstain(found.get("reason", "no-document"))
            anchors = view.anchors_of_assets(found)
        else:
            if place is None or place not in view.spaces:
                return abstain("no-such-place")
            anchors = view.anchors_of_place(place)
        pool = view.retrievable(anchors)
        if not pool:
            return abstain("no-document")
        return {
            "kind": ANSWER,
            # What retrieval may return, and what it should rank first.
            "retrievable": pool,
            "documents": [document.id] if document.id in pool else [],
        }

    return evaluate


def _plain(attribute: str) -> bool:
    """Whether an attribute's name is plain words: lower case letters, digits, underscores.

    A sourced tag like "fire_extinguisher:type" is not, and no filter can name it.
    """
    return attribute.replace("_", "").isalnum() and attribute == attribute.lower()


def _spoken(attribute: str) -> str:
    return attribute.replace("_", " ")


def _median_day(days: list[str]) -> str | None:
    parsed = sorted(day for day in map(parse_day, days) if day is not None)
    if not parsed:
        return None
    return parsed[statistics.median_low(range(len(parsed)))].isoformat()


def templates(world: World, per_template: int) -> Iterator[Template]:
    naming = Naming(world)
    everything = world.view("maintenance")
    rooms = sorted((s for s in world.spaces.values() if s.tier == "room"), key=lambda s: s.id)
    floors = sorted((s for s in world.spaces.values() if s.tier == "floor"), key=lambda s: s.id)
    buildings = sorted(
        (s for s in world.spaces.values() if s.tier == "building"), key=lambda s: s.id
    )
    named_rooms = [room for room in rooms if naming.place(room)]
    named_floors = [floor for floor in floors if naming.place(floor)]
    named_buildings = [building for building in buildings if naming.place(building)]
    kinds = sorted(t for t in world.types if naming.kind(t))

    def by_clearance(space: Space) -> str:
        return space.lowest or ""

    def type_clearance(type_id: str) -> str:
        return world.types[type_id].lowest

    # --- place location ---------------------------------------------------------------
    for room in stratified(named_rooms, by_clearance, per_template):
        place = naming.place(room)
        yield Template(
            "place-location", "named", f"Where is {place}?", {"place": place}, located(room)
        )
    numbered = [room for room in rooms if naming.by_number(room)]
    for room in stratified(numbered, by_clearance, per_template):
        place = naming.by_number(room)
        yield Template(
            "place-location", "by-number", f"Where is {place}?", {"place": place}, located(room)
        )
    # A number that starts with a digit reads as a number. Anything else reads as a name.
    digits = [room for room in numbered if room.ref and room.ref[0].isdigit()]
    for room in stratified(digits, by_clearance, per_template):
        number = fold(room.ref or "")
        # A building with that number would be meant too. Leave those out.
        if any(s.tier != "room" and s.ref and fold(s.ref) == number for s in world.spaces.values()):
            continue
        yield Template(
            "place-location",
            "bare-number",
            f"Where is room {room.ref}?",
            {"place": f"room {room.ref}"},
            # Every room with that number, whichever building it is in.
            bare_number(sorted(r.id for r in rooms if r.ref and fold(r.ref) == number)),
        )
    shared: dict[str, list[Space]] = {}
    for space in world.spaces.values():
        if space.label and naming.labels[fold(space.label)] > 1:
            shared.setdefault(fold(space.label), []).append(space)
    for label in spread(sorted(shared), per_template):
        spaces = shared[label]
        text = spaces[0].label or ""
        # Skip a label some space outside the group also answers to by name: then
        # who is meant depends on more than the label.
        sharing = {space.id for space in spaces}
        if any(
            other.id not in sharing and other.name and words(text) <= words(other.name)
            for other in world.spaces.values()
        ):
            continue
        yield Template(
            "place-location",
            "shared-label",
            f"Where is {text}?",
            {"place": text},
            among(sorted(space.id for space in spaces)),
        )
    for building in named_buildings[:1]:
        if building.ref:
            place = f"{building.ref} room 9Z99X"
            yield Template(
                "place-location",
                "no-such-room",
                f"Where is {place}?",
                {"place": place},
                lambda view: abstain("no-such-place"),
            )

    # --- asset containment ------------------------------------------------------------
    for room in stratified(named_rooms, by_clearance, per_template):
        place = naming.place(room)
        yield Template(
            "asset-containment",
            "in-room",
            f"What is installed in {place}?",
            {"place": place},
            contained(room.id, None),
        )
    for scope in spread(named_floors, per_template) + named_buildings[:1]:
        place = naming.place(scope)
        yield Template(
            "asset-containment",
            f"in-{scope.tier}",
            f"What is installed in {place}?",
            {"place": place},
            contained(scope.id, None),
        )
    wide = named_floors or named_buildings
    for index, kind in enumerate(stratified(kinds, type_clearance, per_template)):
        if not wide:
            break
        scope = wide[index % len(wide)]
        place, name = naming.place(scope), naming.kind(kind)
        yield Template(
            "asset-containment",
            "type-in-place",
            f"Where are the {name} in {place}?",
            {"place": place, "asset_type": name},
            contained(scope.id, naming.same_name(kind)),
        )

    # --- aggregation ------------------------------------------------------------------
    for tier in ("building", "floor"):
        if sum(1 for space in world.spaces.values() if space.tier == tier) < 2:
            continue
        for kind in stratified(kinds, type_clearance, per_template):
            name = naming.kind(kind)
            yield Template(
                "asset-aggregation",
                f"most-by-{tier}",
                f"Which {tier} has the most {name}?",
                {"asset_type": name, "by": tier},
                counted(None, naming.same_name(kind), tier),
            )
    for index, kind in enumerate(stratified(kinds, type_clearance, per_template)):
        if not named_floors:
            break
        floor = named_floors[index % len(named_floors)]
        place, name = naming.place(floor), naming.kind(kind)
        yield Template(
            "asset-aggregation",
            "most-by-room-on-floor",
            f"Which room on {place} has the most {name}?",
            {"place": place, "asset_type": name, "by": "room"},
            counted(floor.id, naming.same_name(kind), "room"),
        )

    # --- attribute filters, service due included --------------------------------------
    # Asked of the building with the most in it, so the filters have something to chew on.
    scope = max(
        named_buildings or named_floors,
        key=lambda space: (len(everything.under.get(space.id, [])), space.id),
        default=None,
    )
    place = naming.place(scope) if scope else None
    scope_id = scope.id if scope else None
    within = f" in {place}" if place else ""
    for kind in stratified(kinds, type_clearance, per_template):
        name, same = naming.kind(kind), naming.same_name(kind)
        of_kind = [a for a in everything.assets.values() if a.type in same]
        carried = sorted({key for a in of_kind for key in a.attributes if _plain(key)})
        for attribute in carried[:2]:
            yield Template(
                "attribute-filter",
                "not-recorded",
                f"Which {name}{within} have no {_spoken(attribute)} recorded?",
                {
                    "place": place,
                    "asset_type": name,
                    "filter": {"attribute": attribute, "test": "not-recorded", "date": None},
                },
                filtered(scope_id, same, attribute, "not-recorded", None),
            )
        # An attribute other types carry and this one never does: a fill rate of zero.
        stored = {key for a in everything.assets.values() for key in a.attributes if _plain(key)}
        for attribute in sorted(stored - set(carried))[:1]:
            yield Template(
                "attribute-filter",
                "never-recorded",
                f"Which {name}{within} have a {_spoken(attribute)} recorded?",
                {
                    "place": place,
                    "asset_type": name,
                    "filter": {"attribute": attribute, "test": "recorded", "date": None},
                },
                filtered(scope_id, same, attribute, "recorded", None),
            )
        installed = _median_day([a.attributes.get("installation_date", "") for a in of_kind])
        if installed:
            yield Template(
                "attribute-filter",
                "installed-before",
                f"Which {name}{within} were installed before {installed}?",
                {
                    "place": place,
                    "asset_type": name,
                    "filter": {
                        "attribute": "installation_date",
                        "test": "before",
                        "date": installed,
                    },
                },
                filtered(scope_id, same, "installation_date", "before", installed),
            )
        due = _median_day([everything.value(a, SERVICE_DUE) or "" for a in of_kind])
        if due:
            yield Template(
                "attribute-filter",
                "service-due-before",
                f"Which {name}{within} fall due for service before {due}?",
                {
                    "place": place,
                    "asset_type": name,
                    "filter": {"attribute": SERVICE_DUE, "test": "before", "date": due},
                },
                filtered(scope_id, same, SERVICE_DUE, "before", due),
            )

    # --- documents --------------------------------------------------------------------
    documents = sorted(world.documents.values(), key=lambda d: d.id)
    by_type = [d for d in documents if any(kind == "asset_type" for kind, _ in d.anchors)]
    for document in stratified(by_type, lambda d: d.lowest, per_template):
        anchored = [i for kind, i in document.anchors if kind == "asset_type" and i in kinds]
        if not anchored or scope is None:
            continue
        name = naming.kind(anchored[0])
        yield Template(
            "asset-document",
            "document-for-type",
            f"What does {document.title} say about the {name}{within}?",
            {"place": place, "asset_type": name, "query": document.title},
            documented(document, scope_id, naming.same_name(anchored[0]), by_assets=True),
        )
    by_place = [d for d in documents if any(kind == "space" for kind, _ in d.anchors)]
    for document in stratified(by_place, lambda d: d.lowest, per_template):
        spaces = [world.spaces[i] for kind, i in document.anchors if kind == "space"]
        named = [(space, naming.place(space)) for space in spaces if naming.place(space)]
        if not named:
            continue
        space, where = named[0]
        yield Template(
            "place-document",
            "document-for-place",
            f"What does {document.title} say about {where}?",
            {"place": where, "query": document.title},
            documented(document, space.id, None, by_assets=False),
        )

    # --- declined by shape ------------------------------------------------------------
    for kind in kinds[:1]:
        yield Template(
            "step-dependent",
            "room-with-the-most",
            f"What is installed in the room with the most {naming.kind(kind)}?",
            {},
            lambda view: abstain("step-dependent"),
        )


def generate(world: World, per_template: int) -> list[dict[str, Any]]:
    """Every template question, once per persona, with its gold."""
    views = {persona: world.view(persona) for persona in ASKED_AS}
    numbered: Counter[tuple[str, str]] = Counter()
    seen: set[tuple[str, str]] = set()
    questions: list[dict[str, Any]] = []
    for template in templates(world, per_template):
        # Two templates may land on the same words. Asked once is enough.
        if (template.shape, template.question) in seen:
            continue
        seen.add((template.shape, template.question))
        numbered[(template.shape, template.name)] += 1
        pair = f"{template.shape}/{template.name}/{numbered[(template.shape, template.name)]:03d}"
        for persona in ASKED_AS:
            questions.append(
                {
                    "id": f"{pair}/{persona}",
                    "pair": pair,
                    "shape": template.shape,
                    "template": template.name,
                    "persona": persona,
                    "question": template.question,
                    "slots": template.slots,
                    "expected": template.evaluate(views[persona]),
                }
            )
    return questions
