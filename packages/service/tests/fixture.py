"""A snapshot small enough to read, and a model that answers from a script.

    site:mini                     Mini Campus
      osm:way/100                 50.34 "Informatik"
        floor:way/100:0           50.34 level 0
          osm:way/1002            "Audimax"            auditorium, untagged: visitor
          osm:way/1009            "Serverraum"         access=staff_only: maintenance
        floor:way/100:1           50.34 level 1
          osm:way/1003            50.34 room 107       corridor, untagged: visitor
          osm:way/1004            50.34 room 108       access=private: staff
      osm:way/200                 30.10, no name
        floor:way/200:0           30.10 level 0
          osm:way/2001            30.10 room 107       access=yes: visitor
          osm:way/2002            "Seminarraum Nord"   seminar room, untagged: visitor
          osm:way/2003            "Seminarraum Süd"    seminar room, untagged: visitor
      osm:way/300                 10.50 "Mensa"
        osm:way/3001              10.50 room 107       access=private: staff

So a visitor asking for "107" has two candidates and staff three, nobody but
maintenance sees the Serverraum, and there is no Bibliothek at all. No edges: each
space is visible by its own clearance.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any

from agent_framework import BaseChatClient, ChatResponse, Message, UsageDetails

from spatial_core import clearance
from spatial_core.records import (
    PERSONAS,
    Ancestor,
    Clearance,
    RecordProvenance,
    Space,
    Value,
    VisibleSpace,
)
from spatial_service.llm import Deployment, Llm
from spatial_service.store import MemoryStore

SNAPSHOT = "2026-09-15"
PREVIOUS = "2026-09-08"
LABEL_RULE = "room-label@1"


def _space(
    id: str,
    tier: str,
    parent: Space | None,
    *,
    name: str | None = None,
    ref: str | None = None,
    label: Value | None = None,
    cleared: Clearance | None = None,
) -> Space:
    path = (*parent.ancestor_path, Ancestor(tier=parent.tier.value, id=parent.id)) if parent else ()
    if label is None and name is not None:
        label = Value.sourced(name)
    return Space(
        id=id,
        record_provenance=RecordProvenance.SOURCED,
        source=id.removeprefix("osm:") if id.startswith("osm:") else None,
        tier=Value.derived(tier, "tier-from-osm-tags@1"),
        parent_id=parent.id if parent else None,
        ancestor_path=path,
        name=Value.sourced(name) if name else None,
        ref=Value.sourced(ref) if ref else None,
        label=label,
        addressable=label is not None,
        clearance=cleared,
    )


def _composed(text: str) -> Value:
    return Value.derived(text, LABEL_RULE)


def spaces() -> list[Space]:
    site = _space("site:mini", "site", None, label=Value.derived("Mini Campus", "site@1"))
    a = _space("osm:way/100", "building", site, name="Informatik", ref="50.34")
    a0 = _space("floor:way/100:0", "floor", a, label=_composed("50.34 level 0"))
    a1 = _space("floor:way/100:1", "floor", a, label=_composed("50.34 level 1"))
    b = _space("osm:way/200", "building", site, ref="30.10", label=_composed("30.10"))
    b0 = _space("floor:way/200:0", "floor", b, label=_composed("30.10 level 0"))
    c = _space("osm:way/300", "building", site, name="Mensa", ref="10.50")
    return [
        site,
        a,
        a0,
        a1,
        b,
        b0,
        c,
        _space(
            "osm:way/1002",
            "room",
            a0,
            name="Audimax",
            cleared=clearance.space_clearance({"indoor": "room", "room": "auditorium"}),
        ),
        _space(
            "osm:way/1009",
            "room",
            a0,
            name="Serverraum",
            cleared=clearance.from_access("staff_only"),
        ),
        _space(
            "osm:way/1003",
            "room",
            a1,
            ref="107",
            label=_composed("50.34 room 107"),
            cleared=clearance.space_clearance({"indoor": "corridor"}),
        ),
        _space(
            "osm:way/1004",
            "room",
            a1,
            ref="108",
            label=_composed("50.34 room 108"),
            cleared=clearance.from_access("private"),
        ),
        _space(
            "osm:way/2001",
            "room",
            b0,
            ref="107",
            label=_composed("30.10 room 107"),
            cleared=clearance.from_access("yes"),
        ),
        _space(
            "osm:way/2002",
            "room",
            b0,
            name="Seminarraum Nord",
            cleared=clearance.space_clearance({"indoor": "room", "room": "seminar"}),
        ),
        _space(
            "osm:way/2003",
            "room",
            b0,
            name="Seminarraum Süd",
            cleared=clearance.space_clearance({"indoor": "room", "room": "seminar"}),
        ),
        _space(
            "osm:way/3001",
            "room",
            c,
            ref="107",
            label=_composed("10.50 room 107"),
            cleared=clearance.from_access("private"),
        ),
    ]


def visible() -> list[VisibleSpace]:
    sets = clearance.visible_sets(spaces(), [])
    return [
        VisibleSpace(persona=persona, space_id=space_id)
        for persona in PERSONAS
        for space_id in sorted(sets[persona])
    ]


def store() -> MemoryStore:
    return MemoryStore(SNAPSHOT, spaces(), visible(), previous=PREVIOUS)


# --- the model ---------------------------------------------------------------------------

Reply = Callable[[dict[str, Any]], str]


class Unreachable(ConnectionError):
    pass


class ScriptedClient(BaseChatClient):
    """A chat client that answers from a function of the payload, and keeps every input."""

    def __init__(self, reply: Reply | None, inputs: list[str], usage: tuple[int, int]) -> None:
        super().__init__()
        self.reply = reply
        self.inputs = inputs
        self.usage = usage

    async def _inner_get_response(  # type: ignore[override]
        self, *, messages: Sequence[Message], stream: bool, options: dict[str, Any], **_: Any
    ) -> ChatResponse:
        self.inputs.append(str(options.get("instructions", "")))
        self.inputs.extend(message.text for message in messages)
        if self.reply is None:
            raise Unreachable("connection refused")
        payload = json.loads(messages[-1].text)
        prompt, completion = self.usage
        return ChatResponse(
            messages=[Message(role="assistant", contents=[self.reply(payload)])],
            usage_details=UsageDetails(
                input_token_count=prompt,
                output_token_count=completion,
                total_token_count=prompt + completion,
            ),
        )


PLANNER = "plan-deployment"
COMPOSER = "compose-deployment"


def llm(planner: Reply | None, composer: Reply | None, inputs: list[str]) -> Llm:
    return Llm(
        planning=Deployment(PLANNER, ScriptedClient(planner, inputs, (120, 40))),
        composition=Deployment(COMPOSER, ScriptedClient(composer, inputs, (80, 20))),
    )


def where(*texts: str) -> str:
    """A place-location plan resolving each text inside the one before it."""
    mentions = [{"id": f"m{n}", "text": t, "kind": "place"} for n, t in enumerate(texts, 1)]
    steps: list[dict[str, Any]] = []
    for n in range(1, len(texts) + 1):
        step: dict[str, Any] = {"id": f"s{n}", "tool": "resolve_place", "mention": f"m{n}"}
        if n > 1:
            step["within"] = {"step": f"s{n - 1}"}
        steps.append(step)
    constraints = [
        {"kind": "within", "mention": f"m{n}", "container": f"m{n - 1}"}
        for n in range(2, len(texts) + 1)
    ]
    return json.dumps(
        {
            "shape": "place-location",
            "mentions": mentions,
            "constraints": constraints,
            "steps": steps,
        }
    )


def located(payload: dict[str, Any]) -> str:
    """A composer that says the subject is in what encloses it, by placeholder only."""
    facts = payload["facts"]
    subject, rest = facts[0]["ref"], [f["ref"] for f in facts[1:]]
    text = f"{{{subject}}} is in " + ", ".join(f"{{{ref}}}" for ref in rest) + "."
    return json.dumps({"segments": [{"text": text, "cites": [f["ref"] for f in facts]}]})
