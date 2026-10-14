"""One building with a circulation graph, drawn in metres so the sums can be done by hand.

Building 20.20 "Physik", two levels. x runs east, y north, one unit a metre.

Level 0
    Foyer         (0,0)-(10,10)    room=entrance. Entrance E at (0,5), from outside
    corridor 0    (10,4)-(40,6)    indoor=corridor, no label. D1 (10,5) from the Foyer
    WC            (14,6)-(18,10)   room=toilets. D2 (16,6)
    Seminarraum   (26,0)-(32,4)    room=seminar. D7 (29,4)
Both levels
    Treppenhaus   (20,6)-(24,10)   room=stairs. D3 (22,6) on level 0, D4 (22,10) on level 1
    Aufzug        (34,6)-(38,10)   room=elevator. D5 (36,6) on level 0, D6 (36,10) on level 1
Level 1
    corridor 1    (10,10)-(40,12)  indoor=corridor, no label
    WC            (12,12)-(16,16)  room=toilets. D8 (14,12)
    room 201      (24,12)-(30,18)  untagged, access=yes. Three doors: D9 (27,12) to the
                                   corridor, D10 (30,15) to 202, D11 (24,15) to 203. So
                                   crossable, by the derived rule.
    room 202      (30,12)-(36,18)  access=yes. Only D10, so only through 201
    room 203      (18,12)-(24,18)  room=office, access=yes. Three doors too: D11, D13
                                   (21,12) to the corridor, D12 (18,15) to 205. An
                                   office, so never crossed.
    room 205      (16,13)-(18,17)  access=yes. Only D12, so only through the office:
                                   everyone sees it and nobody gets there
    staff corr.   (40,10)-(50,12)  indoor=corridor, access=private: staff. D14 (40,11)
    room 210      (44,12)-(48,16)  access=yes. Only D15 (46,12), off the staff corridor:
                                   a visitor sees it and cannot get there
    Teeküche      (36,12.5)-(39,15) room=kitchen, access=yes. No door, half a metre off
                                   corridor 1: snapped to it at (37.5,12)
    Lager         (60,0)-(64,4)    access=yes. No door and nothing near: no way in at all

Installed: a fire extinguisher (sourced, a visitor type) in corridor 0 and one in the
staff corridor. A hydrant on the site three metres west of the Foyer wall, and a
standpipe on the site thirty metres from anything. A defibrillator hung on level 0
with no room, at a point that lies in the Seminarraum. A panelboard (maintenance) in 202.

The default origin is the entrance E.

Every builder takes ``private``: rooms to tag access=private on top of the drawing. With
room 203 private a visitor's room 201 has two doors they can use and one they cannot.

Sums the tests lean on, all door to door in straight lines at 1.34 m/s:

    E to the Seminarraum    10 + 19.026 + 2                         = 31.03 m, 23.2 s
    E to room 202, stairs   10 + 12.042 + 4 + 5.385 + 4.243 + 3     = 38.67 m, 28.9 + 12 s
    E to room 202, lift     10 + 26.019 + 4 + 9.220 + 4.243 + 3     = 56.48 m, 42.2 + 44 s
    E to the WC on level 0  10 + 6.083 + 2                          = 18.08 m, 13.5 s
    E to the WC on level 1  10 + 12.042 + 4 + 8.246 + 2             = 36.29 m, 27.1 + 12 s
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from spatial_core import clearance
from spatial_core.metres import Local
from spatial_core.records import (
    PERSONAS,
    Ancestor,
    Asset,
    AssetType,
    Edge,
    Persona,
    Position,
    RecordProvenance,
    Space,
    Value,
    ValueProvenance,
    VisibleAsset,
    VisibleAssetType,
    VisibleSpace,
)
from spatial_service.store import MemoryStore

SNAPSHOT = "2026-09-15"
LOCAL = Local(8.41, 49.01)

SITE = "site:routes"
BUILDING = "osm:way/20"
LEVEL_0 = "floor:way/20:0"
LEVEL_1 = "floor:way/20:1"
FOYER = "osm:way/2000"
CORRIDOR_0 = "osm:way/2001"
WC_0 = "osm:way/2002"
SEMINAR = "osm:way/2003"
STAIRS = "osm:way/2004"
LIFT = "osm:way/2005"
CORRIDOR_1 = "osm:way/2010"
WC_1 = "osm:way/2011"
ROOM_201 = "osm:way/2012"
ROOM_202 = "osm:way/2013"
ROOM_203 = "osm:way/2014"
ROOM_205 = "osm:way/2015"
STAFF_CORRIDOR = "osm:way/2016"
ROOM_210 = "osm:way/2017"
KITCHEN = "osm:way/2018"
STORE = "osm:way/2019"

ENTRANCE = "node/1:outside>" + FOYER
SNAP_RULE = "snap-doorless-room@1"

EXTINGUISHER = "osm:type:fire_extinguisher"
HYDRANT = "osm:type:fire_hydrant"
STANDPIPE = "osm:type:standpipe"
DEFIBRILLATOR = "osm:type:defibrillator"
PANEL = "seed:type:panelboard-p1"
EXTINGUISHER_0 = "osm:asset:node/9101"
EXTINGUISHER_STAFF = "osm:asset:node/9102"


def _outline(x0: float, y0: float, x1: float, y1: float) -> dict[str, Any]:
    ring = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    return {"type": "Polygon", "coordinates": [[list(LOCAL.degrees(x, y)) for x, y in ring]]}


def _space(
    id: str,
    tier: str,
    parent: Space | None,
    *,
    label: str | None = None,
    name: str | None = None,
    box: tuple[float, float, float, float] | None = None,
    levels: str | None = None,
    **tags: str,
) -> Space:
    path = (*parent.ancestor_path, Ancestor(tier=parent.tier.value, id=parent.id)) if parent else ()
    shown = Value.sourced(name) if name else Value.derived(label, "room-label@1") if label else None
    return Space(
        id=id,
        record_provenance=RecordProvenance.SOURCED,
        source=id.removeprefix("osm:") if id.startswith("osm:") else None,
        tier=Value.derived(tier, "tier-from-osm-tags@1"),
        parent_id=parent.id if parent else None,
        ancestor_path=path,
        name=Value.sourced(name) if name else None,
        label=shown,
        addressable=shown is not None,
        levels=Value.derived(levels, "level-parse@1") if levels else None,
        attributes={key: Value.sourced(value) for key, value in tags.items()},
        outline=_outline(*box) if box else None,
        clearance=clearance.space_clearance(tags) if tier == "room" else None,
    )


def spaces(private: Collection[str] = ()) -> list[Space]:
    site = _space(SITE, "site", None, label="Routes Campus")
    building = _space(BUILDING, "building", site, name="Physik", box=(0, 0, 64, 18))
    low = _space(LEVEL_0, "floor", building, label="20.20 level 0", levels="0")
    high = _space(LEVEL_1, "floor", building, label="20.20 level 1", levels="1")

    def room(id: str, floor: Space, box: tuple[float, float, float, float], **more: Any) -> Space:
        levels = more.pop("levels", floor.levels.value if floor.levels else None)
        if id in private:
            more["access"] = "private"
        return _space(id, "room", floor, box=box, levels=levels, **more)

    return [
        site,
        building,
        low,
        high,
        room(FOYER, low, (0, 0, 10, 10), name="Foyer", indoor="room", room="entrance"),
        room(CORRIDOR_0, low, (10, 4, 40, 6), indoor="corridor"),
        room(WC_0, low, (14, 6, 18, 10), name="WC", indoor="room", room="toilets"),
        room(SEMINAR, low, (26, 0, 32, 4), name="Seminarraum", indoor="room", room="seminar"),
        room(
            STAIRS,
            low,
            (20, 6, 24, 10),
            name="Treppenhaus",
            levels="0;1",
            indoor="room",
            room="stairs",
        ),
        room(
            LIFT, low, (34, 6, 38, 10), name="Aufzug", levels="0;1", indoor="room", room="elevator"
        ),
        room(CORRIDOR_1, high, (10, 10, 40, 12), indoor="corridor"),
        room(WC_1, high, (12, 12, 16, 16), label="20.20 WC level 1", indoor="room", room="toilets"),
        room(ROOM_201, high, (24, 12, 30, 18), label="20.20 room 201", indoor="room", access="yes"),
        room(ROOM_202, high, (30, 12, 36, 18), label="20.20 room 202", indoor="room", access="yes"),
        room(
            ROOM_203,
            high,
            (18, 12, 24, 18),
            label="20.20 room 203",
            indoor="room",
            room="office",
            access="yes",
        ),
        room(ROOM_205, high, (16, 13, 18, 17), label="20.20 room 205", indoor="room", access="yes"),
        room(STAFF_CORRIDOR, high, (40, 10, 50, 12), indoor="corridor", access="private"),
        room(ROOM_210, high, (44, 12, 48, 16), label="20.20 room 210", indoor="room", access="yes"),
        room(
            KITCHEN,
            high,
            (36, 12.5, 39, 15),
            name="Teeküche",
            indoor="room",
            room="kitchen",
            access="yes",
        ),
        room(STORE, high, (60, 0, 64, 4), name="Lager", indoor="room", access="yes"),
    ]


# (door node, x, y, level, the spaces it joins). One space is a door from outside.
_DOORS: list[tuple[int, float, float, str, tuple[str, ...]]] = [
    (1, 0, 5, "0", (FOYER,)),
    (2, 10, 5, "0", (FOYER, CORRIDOR_0)),
    (3, 16, 6, "0", (CORRIDOR_0, WC_0)),
    (4, 22, 6, "0", (CORRIDOR_0, STAIRS)),
    (5, 22, 10, "1", (STAIRS, CORRIDOR_1)),
    (6, 36, 6, "0", (CORRIDOR_0, LIFT)),
    (7, 36, 10, "1", (LIFT, CORRIDOR_1)),
    (8, 29, 4, "0", (CORRIDOR_0, SEMINAR)),
    (9, 14, 12, "1", (CORRIDOR_1, WC_1)),
    (10, 27, 12, "1", (CORRIDOR_1, ROOM_201)),
    (11, 30, 15, "1", (ROOM_201, ROOM_202)),
    (12, 24, 15, "1", (ROOM_201, ROOM_203)),
    (13, 21, 12, "1", (CORRIDOR_1, ROOM_203)),
    (14, 18, 15, "1", (ROOM_203, ROOM_205)),
    (15, 40, 11, "1", (CORRIDOR_1, STAFF_CORRIDOR)),
    (16, 46, 12, "1", (STAFF_CORRIDOR, ROOM_210)),
]


def edges(private: Collection[str] = ()) -> list[Edge]:
    by_id = {space.id: space for space in spaces(private)}

    def edge(
        id: str, door: str | None, a: str | None, b: str, x: float, y: float, level: str, rule: str
    ) -> Edge:
        entered = by_id[b].clearance
        assert entered is not None
        lon, lat = LOCAL.degrees(x, y)
        return Edge(
            id=id,
            door=door,
            from_id=a,
            to_id=b,
            door_clearance=None,
            clearance=clearance.edge_clearance(None, entered),
            provenance=ValueProvenance.DERIVED,
            rule=rule,
            lon=lon,
            lat=lat,
            level=level,
        )

    built = []
    for node, x, y, level, joined in _DOORS:
        door = f"node/{node}"
        pairs: list[tuple[str | None, str]] = [(None, joined[0])]
        if len(joined) == 2:
            pairs = [(joined[0], joined[1]), (joined[1], joined[0])]
        for a, b in pairs:
            built.append(
                edge(f"{door}:{a or 'outside'}>{b}", door, a, b, x, y, level, "edges-from-doors@1")
            )
    # The Teeküche has no door. Ingestion snaps it to the corridor it sits against.
    for a, b in ((KITCHEN, CORRIDOR_1), (CORRIDOR_1, KITCHEN)):
        built.append(edge(f"snap:{a}>{b}", None, a, b, 37.5, 12, "1", SNAP_RULE))
    return built


def _sets(private: Collection[str] = ()) -> dict[Persona, frozenset[str]]:
    return clearance.visible_sets(spaces(private), edges(private))


def visible(private: Collection[str] = ()) -> list[VisibleSpace]:
    sets = _sets(private)
    return [
        VisibleSpace(persona=persona, space_id=space_id)
        for persona in PERSONAS
        for space_id in sorted(sets[persona])
    ]


def _type(id: str, name: str, code: str) -> AssetType:
    sourced = id.startswith("osm:")
    return AssetType(
        id=id,
        record_provenance=RecordProvenance.SOURCED if sourced else RecordProvenance.SEED,
        name=Value.derived(name, "fire-assets-from-emergency@1")
        if sourced
        else Value(value=name, provenance=ValueProvenance.INFERRED, basis="seed-inventory@1"),
        omniclass=Value.derived(code, "fire-assets-from-emergency@1"),
        clearance=clearance.type_clearance(code),
    )


def asset_types() -> list[AssetType]:
    return [
        _type(EXTINGUISHER, "Fire extinguisher", "23-65 70 17 11"),
        _type(HYDRANT, "Fire hydrant", "23-65 70 17 11"),
        _type(STANDPIPE, "Standpipe", "23-65 70 17 11"),
        _type(DEFIBRILLATOR, "Defibrillator", "23-65 70 17 11"),
        _type(PANEL, "Panelboard P1", "23-80 30 20 11"),
    ]


# (asset id, type, space, x, y)
_INSTALLED = [
    (EXTINGUISHER_0, EXTINGUISHER, CORRIDOR_0, 12.0, 5.0),
    (EXTINGUISHER_STAFF, EXTINGUISHER, STAFF_CORRIDOR, 45.0, 11.0),
    ("osm:asset:node/9103", HYDRANT, SITE, -3.0, 5.0),
    ("osm:asset:node/9104", STANDPIPE, SITE, -30.0, 40.0),
    ("osm:asset:node/9105", DEFIBRILLATOR, LEVEL_0, 29.0, 2.0),
    ("seed:asset:P1", PANEL, ROOM_202, 33.0, 15.0),
]


def assets() -> list[Asset]:
    by_id = {space.id: space for space in spaces()}
    names = {kind.id: kind.name for kind in asset_types()}
    listed = []
    for id, type_id, space_id, x, y in _INSTALLED:
        space = by_id[space_id]
        lon, lat = LOCAL.degrees(x, y)
        sourced = id.startswith("osm:")
        listed.append(
            Asset(
                id=id,
                record_provenance=RecordProvenance.SOURCED if sourced else RecordProvenance.SEED,
                source=id.removeprefix("osm:asset:") if sourced else None,
                type_id=type_id,
                space_id=space_id,
                ancestor_path=(*space.ancestor_path, Ancestor(tier=space.tier.value, id=space.id)),
                name=names[type_id],
                position=Position(lon=lon, lat=lat, provenance=ValueProvenance.SOURCED),
            )
        )
    return listed


def visible_assets(
    private: Collection[str] = (),
) -> tuple[dict[Persona, frozenset[str]], dict[Persona, frozenset[str]]]:
    return clearance.visible_assets(assets(), asset_types(), _sets(private))


def store(default_origin: str | None = ENTRANCE, *, private: Collection[str] = ()) -> MemoryStore:
    seen_assets, seen_types = visible_assets(private)
    return MemoryStore(
        SNAPSHOT,
        spaces(private),
        visible(private),
        assets=assets(),
        asset_types=asset_types(),
        visible_assets=seen_assets,
        visible_asset_types=seen_types,
        edges=edges(private),
        default_origin=default_origin,
    )


def bundle(root: Path, private: Collection[str] = ()) -> Path:
    """The drawing as a bundle on disk, for load to read."""
    out = root / SNAPSHOT
    seen_assets, seen_types = visible_assets(private)
    records: dict[str, list[BaseModel]] = {
        "osm/spaces.jsonl": list(spaces(private)),
        "osm/edges.jsonl": list(edges(private)),
        "osm/visible_spaces.jsonl": list(visible(private)),
    }
    for half in ("osm", "seed"):
        records |= {
            f"{half}/asset_types.jsonl": [t for t in asset_types() if t.id.startswith(half)],
            f"{half}/assets.jsonl": [a for a in assets() if a.id.startswith(half)],
            f"{half}/visible_assets.jsonl": [
                VisibleAsset(persona=persona, asset_id=asset_id)
                for persona in PERSONAS
                for asset_id in sorted(seen_assets[persona])
                if asset_id.startswith(half)
            ],
            f"{half}/visible_asset_types.jsonl": [
                VisibleAssetType(persona=persona, type_id=type_id)
                for persona in PERSONAS
                for type_id in sorted(seen_types[persona])
                if type_id.startswith(half)
            ],
        }
    parts = {}
    for name, rows in records.items():
        content = "".join(row.model_dump_json() + "\n" for row in rows).encode()
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        (out / name).write_bytes(content)
        parts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "records": len(rows)}
    manifest = {
        "snapshot": SNAPSHOT,
        "venue": "routes",
        "parts": parts,
        "default_origin": {"door": "node/1", "edge": ENTRANCE},
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


# --- plans and a composer -------------------------------------------------------------------


def _plan(shape: str, mentions: list[tuple[str, str]], steps: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "shape": shape,
            "mentions": [
                {"id": f"m{n}", "text": text, "kind": kind}
                for n, (text, kind) in enumerate(mentions, 1)
            ],
            "steps": [{"id": f"s{n}", **step} for n, step in enumerate(steps, 1)],
        }
    )


def route_to(
    place: str, *, origin: str | None = None, described: bool = False, **route: Any
) -> str:
    """A route plan to a named place, or to whatever fits a description, from the default origin."""
    tool = "places_matching" if described else "resolve_place"
    mentions = [(place, "place")]
    steps: list[dict[str, Any]] = [{"tool": tool, "mention": "m1"}]
    last: dict[str, Any] = {"tool": "route", "destination": {"step": "s1"}, **route}
    if origin is not None:
        mentions.append((origin, "place"))
        steps.append({"tool": "resolve_place", "mention": "m2"})
        last["origin"] = {"step": "s2"}
    return _plan("route", mentions, [*steps, last])


def route_to_asset(kind: str) -> str:
    return _plan(
        "route",
        [(kind, "asset-type")],
        [
            {"tool": "resolve_asset_type", "mention": "m1"},
            {"tool": "assets_in_scope", "types": {"step": "s1"}},
            {"tool": "route", "destination": {"step": "s2"}},
        ],
    )


def nearest(kind: str, to: str | None = None) -> str:
    mentions = [(kind, "asset-type")]
    steps: list[dict[str, Any]] = [
        {"tool": "resolve_asset_type", "mention": "m1"},
        {"tool": "assets_in_scope", "types": {"step": "s1"}},
    ]
    last: dict[str, Any] = {"tool": "nearest", "assets": {"step": "s2"}}
    if to is not None:
        mentions.append((to, "place"))
        steps.append({"tool": "resolve_place", "mention": "m2"})
        last["to"] = {"step": "s3"}
    return _plan("proximity", mentions, [*steps, last])


def _ref(ref: str | None, otherwise: str) -> str:
    return f"{{{ref}}}" if ref else otherwise


def walked(payload: dict[str, Any]) -> str:
    """A composer for routes: one segment per route, by placeholder only."""
    segments = []
    for route in payload["routes"]:
        text = (
            f"From {_ref(route['origin'], 'here')} to {_ref(route['destination'], 'there')}: "
            f"{{{route['duration']}}}, {{{route['distance']}}}."
        )
        cites = [route["duration"], route["distance"]]
        for other in route["alternatives"]:
            text += f" Or {_ref(other['destination'], 'there')}: {{{other['duration']}}}."
            cites.append(other["duration"])
        segments.append({"text": text, "cites": cites})
    return json.dumps({"segments": segments})


def nearby(payload: dict[str, Any]) -> str:
    """A composer for proximity: what, where, and how far or what they share."""
    row = payload["nearest"][0]
    kinds = " and ".join(f"{{{ref}}}" for ref in payload["types"])
    counts = " and ".join(f"{{{ref}}}" for ref in row["counts"])
    text = f"Nearest {kinds}: {counts} in {_ref(row['in'], 'a room with no label')}"
    if payload["basis"] == "modelled-duration":
        text += f", {{{payload['duration']}}}, {{{payload['distance']}}}."
    else:
        text += f", in the same {_ref(payload.get('shared'), 'place')}."
    return json.dumps({"segments": [{"text": text, "cites": row["counts"]}]})
