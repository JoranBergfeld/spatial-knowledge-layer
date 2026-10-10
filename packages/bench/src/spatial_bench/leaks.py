"""The two leak checks. Each passes or fails. Neither is a rate.

Input leak check: nothing a model was sent holds an id outside the persona's
visible set. "A model" is the planner, the composer and the embedding deployment.
The filter the index is given is checked too, since an id in it is an id the
host was willing to ask about.

    It also looks for names. No model input in this layer carries an id at all
    today, so an id-only check would pass on a service that showed a visitor the
    whole building by label. A name counts when only hidden things carry it: no
    visible space, asset, type, document or passage says it anywhere. What the
    asker typed themselves is not a leak, so the question and the words of the
    plan are taken out before looking.

Counterfactual leak check: a visitor's turns are the same, whole, whether the
restricted assets and documents are there, gone, or doubled. The turn is compared
as it came back from the endpoint, so turn kind, ids and numbers are all in it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from spatial_bench.bundle import Bundle
from spatial_bench.run import Asked
from spatial_core.records import Persona

# What splits a payload into things that could be an id.
_BREAK = re.compile(r"[\s\"',\[\]{}()<>|\\]+")
_TRAILING = ".;:!?"
# A name shorter than this is too likely to be a word of its own.
MIN_NAME = 4


@dataclass(frozen=True)
class Leak:
    question: str
    persona: Persona
    # "id", "name", "attribute" or "filter".
    kind: str
    value: str
    # Which input it was found in, counted from 0, or "index" or "embedding".
    where: str


@dataclass(frozen=True)
class Hidden:
    """What does not exist for one persona, and so may not reach a model for them."""

    persona: Persona
    ids: frozenset[str]
    # Names only hidden things carry.
    names: frozenset[str]
    # Attributes only hidden assets carry.
    attributes: frozenset[str]

    @classmethod
    def of(cls, bundle: Bundle, persona: Persona) -> Hidden:
        seen = bundle.visible(persona)
        read = {d.id for d in bundle.documents if seen.reads(d)}
        events = {e.id: e.asset_id for e in bundle.events}
        ids: set[str] = set()
        ids.update(s.id for s in bundle.spaces if s.id not in seen.spaces)
        ids.update(a.id for a in bundle.assets if a.id not in seen.assets)
        ids.update(t.id for t in bundle.types if t.id not in seen.types)
        ids.update(d.id for d in bundle.documents if d.id not in read)
        ids.update(c.id for c in bundle.chunks if c.document_id not in read)
        ids.update(s.id for s in bundle.summaries if s.document_id not in read)
        ids.update(e for e, asset in events.items() if asset not in seen.assets)

        shown: list[str] = []
        unseen: set[str] = set()

        def sort(visible: bool, *texts: str | None) -> None:
            for text in texts:
                if text and visible:
                    shown.append(text)
                elif text:
                    unseen.add(text)

        for space in bundle.spaces:
            sort(
                space.id in seen.spaces,
                space.label.value if space.label else None,
                space.name.value if space.name else None,
            )
        for asset in bundle.assets:
            sort(asset.id in seen.assets, asset.name.value)
        for kind in bundle.types:
            sort(kind.id in seen.types, kind.name.value)
        for document in bundle.documents:
            sort(document.id in read, document.title)
        for chunk in bundle.chunks:
            sort(chunk.document_id in read, chunk.text if chunk.document_id in read else None)
        everything = "\n".join(shown)
        names = {name for name in unseen if len(name) >= MIN_NAME and not _says(everything, name)}

        carried: set[str] = set()
        lacking: set[str] = set()
        for asset in bundle.assets:
            (carried if asset.id in seen.assets else lacking).update(asset.attributes)
        return cls(persona, frozenset(ids), frozenset(names), frozenset(lacking - carried))


def _says(text: str, name: str) -> bool:
    """Whether ``text`` holds ``name`` as words of its own, not inside longer ones."""
    if name not in text:
        return False
    return re.search(rf"(?<!\w){re.escape(name)}(?!\w)", text) is not None


def _tokens(text: str) -> set[str]:
    found = set()
    for piece in _BREAK.split(text):
        found.add(piece)
        found.add(piece.rstrip(_TRAILING))
    return found


def _own_words(asked: Asked) -> list[str]:
    """What the asker and the scripted plan said themselves. None of it is a leak."""
    slots = asked.question.slots
    said = [asked.question.question, slots.place, slots.asset_type, slots.query, slots.by]
    return sorted((text for text in said if text), key=len, reverse=True)


def input_leaks(asked: Sequence[Asked], hidden: Hidden) -> list[Leak]:
    """Every hidden id, name or attribute that reached a model or the index for these turns."""
    awkward = [i for i in hidden.ids if _BREAK.search(i)]
    leaks: list[Leak] = []
    for one in asked:
        if one.question.persona is not hidden.persona:
            continue

        def leak(kind: str, value: str, where: str, one: Asked = one) -> None:
            leaks.append(Leak(one.question.id, hidden.persona, kind, value, where))

        own = _own_words(one)
        named = one.question.slots.filter.attribute if one.question.slots.filter else None
        sent = [(str(n), text) for n, text in enumerate(one.inputs)]
        sent += [("embedding", text) for text in one.embedded]
        for where, text in sent:
            tokens = _tokens(text)
            for found in sorted(tokens & hidden.ids):
                leak("id", found, where)
            for found in awkward:
                if found in text:
                    leak("id", found, where)
            for found in sorted(tokens & hidden.attributes - {named}):
                leak("attribute", found, where)
            # Names are looked for in payloads only. The instructions are the same
            # words for everyone and say nothing of any venue.
            if not text.lstrip().startswith("{") and where != "embedding":
                continue
            for said in own:
                text = text.replace(said, " ").replace(json.dumps(said)[1:-1], " ")
            for name in sorted(hidden.names):
                if _says(text, name):
                    leak("name", name, where)
        for query in one.queries:
            chosen = query.filter
            for found in sorted(
                {*chosen.space_ids, *chosen.asset_type_ids, *chosen.asset_ids, *chosen.document_ids}
                & hidden.ids
            ):
                leak("filter", found, "index")
    return leaks


@dataclass(frozen=True)
class Difference:
    question: str
    # Which copy of the bench the turn differs in.
    variant: str
    # Where in the turn, as JSON Pointer paths. The first few.
    paths: tuple[str, ...]


def _differences(first: Any, second: Any, at: str = "") -> Iterable[str]:
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted({*first, *second}):
            if key not in first or key not in second:
                yield f"{at}/{key}"
            else:
                yield from _differences(first[key], second[key], f"{at}/{key}")
    elif isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            yield at or "/"
        else:
            for index, (a, b) in enumerate(zip(first, second, strict=True)):
                yield from _differences(a, b, f"{at}/{index}")
    elif first != second:
        yield at or "/"


def turn_differences(
    base: Sequence[Asked], other: Sequence[Asked], variant: str
) -> list[Difference]:
    """The questions whose turn is not the same in ``other`` as in ``base``."""
    theirs = {asked.question.id: asked for asked in other}
    found = []
    for asked in base:
        twin = theirs.get(asked.question.id)
        if twin is None:
            found.append(Difference(asked.question.id, variant, ("/",)))
            continue
        paths = list(_differences(asked.turn, twin.turn))
        if asked.status != twin.status:
            paths.insert(0, "/status")
        if paths:
            found.append(Difference(asked.question.id, variant, tuple(paths[:5])))
    return found


@dataclass(frozen=True)
class LeakReport:
    # Hidden things that reached a model or the index.
    inputs: tuple[Leak, ...] = ()
    # Visitor turns that changed when restricted things were added or removed.
    counterfactual: tuple[Difference, ...] = ()
    # How many turns each check looked at.
    turns: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.inputs and not self.counterfactual
