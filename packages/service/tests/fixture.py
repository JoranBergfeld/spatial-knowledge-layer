"""A snapshot small enough to read, and a model that answers from a script.

    site:mini                     Mini Campus
      osm:way/100                 50.34 "Informatik"
        floor:way/100:0           50.34 level 0
          osm:way/1002            "Audimax"            auditorium, untagged: visitor
          osm:way/1009            "Serverraum"         access=staff_only: maintenance
          osm:way/1010            "Treppenhaus"        stairs on levels 0 and 1: visitor
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

What is installed (lights and the extinguisher are visitor types, panelboards
maintenance; all generated but the extinguisher, which OSM maps):

    Audimax            3 Light Fixture B1, 1 Fire extinguisher (sourced), 1 Panelboard P1
    Serverraum         2 Panelboard P1, 1 Light Fixture B1
    Treppenhaus        1 Light Fixture Q4
    50.34 room 107     2 Light Fixture B1, one of them with no placement
    50.34 room 108     1 Light Fixture B1
    Seminarraum Nord   1 Light Fixture B1

Nothing in 30.10 room 107, Seminarraum Süd or anywhere in the Mensa. So a visitor
counts 8 assets, staff 9 and maintenance 13.
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
    Asset,
    AssetType,
    Clearance,
    Persona,
    Position,
    RecordProvenance,
    Space,
    Value,
    ValueProvenance,
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
    levels: str | None = None,
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
        levels=Value.derived(levels, "level-parse@1") if levels else None,
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
        # Covers two floors, hangs from the lower one.
        _space(
            "osm:way/1010",
            "room",
            a0,
            name="Treppenhaus",
            levels="0;1",
            cleared=clearance.space_clearance({"indoor": "room", "room": "stairs"}),
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


LIGHT_B1 = "seed:type:light-fixture-b1"
LIGHT_Q4 = "seed:type:light-fixture-q4"
PANEL = "seed:type:panelboard-p1"
EXTINGUISHER = "osm:type:fire_extinguisher"
FIRE_RULE = "fire-assets-from-emergency@1"
BASIS = "seed-inventory@1, seed 7"


def _generated_type(id: str, name: str, code: str) -> AssetType:
    return AssetType(
        id=id,
        record_provenance=RecordProvenance.SEED,
        name=Value(value=name, provenance=ValueProvenance.INFERRED, basis=BASIS),
        omniclass=Value(value=code, provenance=ValueProvenance.INFERRED, basis=BASIS),
        clearance=clearance.type_clearance(code),
    )


def asset_types() -> list[AssetType]:
    return [
        _generated_type(LIGHT_B1, "Light Fixture B1", "23-80 70 11 14 11"),
        _generated_type(LIGHT_Q4, "Light Fixture Q4", "23-80 70 11 11"),
        _generated_type(PANEL, "Panelboard P1", "23-80 30 20 11"),
        AssetType(
            id=EXTINGUISHER,
            record_provenance=RecordProvenance.SOURCED,
            name=Value.derived("Fire extinguisher", FIRE_RULE),
            omniclass=Value.derived("23-65 70 17 11", FIRE_RULE),
            clearance=clearance.type_clearance("23-65 70 17 11"),
        ),
    ]


# (asset id, type, space, placed)
_INSTALLED = [
    ("seed:asset:L1", LIGHT_B1, "osm:way/1002", True),
    ("seed:asset:L2", LIGHT_B1, "osm:way/1002", True),
    ("seed:asset:L3", LIGHT_B1, "osm:way/1002", True),
    ("osm:asset:node/9001", EXTINGUISHER, "osm:way/1002", True),
    ("seed:asset:P1", PANEL, "osm:way/1002", True),
    ("seed:asset:P2", PANEL, "osm:way/1009", True),
    ("seed:asset:P3", PANEL, "osm:way/1009", True),
    ("seed:asset:L4", LIGHT_B1, "osm:way/1009", True),
    ("seed:asset:L5", LIGHT_Q4, "osm:way/1010", True),
    ("seed:asset:L6", LIGHT_B1, "osm:way/1003", True),
    ("seed:asset:L7", LIGHT_B1, "osm:way/1003", False),
    ("seed:asset:L8", LIGHT_B1, "osm:way/1004", True),
    ("seed:asset:L9", LIGHT_B1, "osm:way/2002", True),
]


def assets() -> list[Asset]:
    by_id = {space.id: space for space in spaces()}
    names = {kind.id: kind.name.value for kind in asset_types()}
    listed = []
    for number, (id, type_id, space_id, placed) in enumerate(_INSTALLED, start=1):
        space = by_id[space_id]
        sourced = id.startswith("osm:")
        position = None
        if placed and sourced:
            position = Position(lon=8.41, lat=49.01, provenance=ValueProvenance.SOURCED)
        elif placed:
            position = Position(
                lon=8.41, lat=49.01, provenance=ValueProvenance.DERIVED, rule="layout-in-outline@1"
            )
        listed.append(
            Asset(
                id=id,
                record_provenance=RecordProvenance.SOURCED if sourced else RecordProvenance.SEED,
                source=id.removeprefix("osm:asset:") if sourced else None,
                type_id=type_id,
                space_id=space_id,
                ancestor_path=(
                    *space.ancestor_path,
                    Ancestor(tier=space.tier.value, id=space.id),
                ),
                name=Value.derived(names[type_id], FIRE_RULE)
                if sourced
                else Value(
                    value=f"{names[type_id]} {number}",
                    provenance=ValueProvenance.INFERRED,
                    basis=BASIS,
                ),
                position=position,
            )
        )
    return listed


def visible_assets() -> tuple[dict[Persona, frozenset[str]], dict[Persona, frozenset[str]]]:
    """The asset ids, then the type ids, each persona can see."""
    return clearance.visible_assets(assets(), asset_types(), clearance.visible_sets(spaces(), []))


def store() -> MemoryStore:
    seen_assets, seen_types = visible_assets()
    return MemoryStore(
        SNAPSHOT,
        spaces(),
        visible(),
        previous=PREVIOUS,
        assets=assets(),
        asset_types=asset_types(),
        visible_assets=seen_assets,
        visible_asset_types=seen_types,
    )


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


def installed(place: str | None = None, kind: str | None = None) -> str:
    """An asset-containment plan: what is in ``place``, of type ``kind`` if given."""
    return _asset_plan("asset-containment", place, kind, None)


def most(by: str, kind: str | None = None, place: str | None = None) -> str:
    """An asset-aggregation plan: counts per ``by`` under ``place``, or the whole venue."""
    return _asset_plan("asset-aggregation", place, kind, by)


def _asset_plan(shape: str, place: str | None, kind: str | None, by: str | None) -> str:
    mentions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    scope: dict[str, Any] = {"id": "", "tool": "assets_in_scope"}
    for text, mention_kind, tool, reads in (
        (place, "place", "resolve_place", "scope"),
        (kind, "asset-type", "resolve_asset_type", "types"),
    ):
        if text is None:
            continue
        mentions.append({"id": f"m{len(mentions) + 1}", "text": text, "kind": mention_kind})
        steps.append({"id": f"s{len(steps) + 1}", "tool": tool, "mention": mentions[-1]["id"]})
        scope[reads] = {"step": steps[-1]["id"]}
    scope["id"] = f"s{len(steps) + 1}"
    steps.append(scope)
    if by is not None:
        steps.append(
            {
                "id": f"s{len(steps) + 1}",
                "tool": "aggregate",
                "assets": {"step": scope["id"]},
                "by": by,
            }
        )
    return json.dumps({"shape": shape, "mentions": mentions, "steps": steps})


def _named(refs: list[str]) -> str:
    return " and ".join(f"{{{ref}}}" for ref in refs)


def counted(payload: dict[str, Any]) -> str:
    """A composer for the asset shapes: the scope, its total, then every row, by placeholder."""
    scope = f"{{{payload['scope']}}}" if payload.get("scope") else "The venue"
    text = f"{scope} holds {_named(payload['total'])}."
    cites = [*([payload["scope"]] if payload.get("scope") else []), *payload["total"]]
    segments = [{"text": text, "cites": cites}]
    for row in payload["rows"]:
        segments.append(
            {
                "text": f"{{{row['of']}}}: {_named(row['counts'])}.",
                "cites": [row["of"], *row["counts"]],
            }
        )
    if payload.get("no_recorded_installations"):
        refs = payload["no_recorded_installations"]
        segments.append({"text": f"No recorded installations: {_named(refs)}.", "cites": refs})
    return json.dumps({"segments": segments})
