"""Routes and proximity, end to end over HTTP, against a building with a circulation graph.

Every expected second and metre is worked out by hand in route_fixture's docstring
or beside the test, from the drawing, never by running the router.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg import sql
from psycopg.conninfo import make_conninfo
from pydantic import BaseModel

import fixture
import route_fixture as rf
from route_fixture import nearby, nearest, route_to, route_to_asset, walked
from spatial_core.records import PERSONAS, VisibleAsset, VisibleAssetType
from spatial_core.stores import READER_ROLE
from spatial_ingest.load import load
from spatial_ingest.promote import promote
from spatial_service.app import create_app
from spatial_service.store import PostgresStore, Store

CONSTANTS = {
    "walking_m_per_s": 1.34,
    "stairs_s_per_level": 12.0,
    "lift_s": 40.0,
    "lift_s_per_level": 4.0,
    "storey_m": 4.0,
}
NO_ROUTE = "There is no route on the mapped graph for this persona."


def ask(
    persona: str,
    question: str,
    plan: Any,
    composer: Any = walked,
    *,
    store: Store | None = None,
    inputs: list[str] | None = None,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    llm = fixture.llm(
        plan if callable(plan) else (lambda _p: plan),
        composer,
        inputs if inputs is not None else [],
    )
    app = create_app(store or rf.store(), llm)
    body = {"persona": persona, "question": question, "state": state}
    response = TestClient(app).post("/turn", json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


def passed(route: dict[str, Any]) -> list[str]:
    return [passage["space_id"] for passage in route["passages"]]


# --- a route ----------------------------------------------------------------------------------


def test_a_route_starts_at_the_default_origin_and_returns_seconds_metres_and_constants() -> None:
    turn = ask("visitor", "How do I get to the Seminarraum?", route_to("Seminarraum"))

    assert turn["kind"] == "answer"
    assert texts(turn) == ["From Foyer to Seminarraum: 23 seconds (modelled), 31 metres."]
    assert turn["trace"]["shape"] == "route"
    [route] = turn["routes"]
    # 10 m across the Foyer, 19.03 m along the corridor, 2 m into the room, at 1.34 m/s.
    assert (route["seconds"], route["metres"]) == (23, 31)
    assert route["modelled"] is True
    assert route["constants"] == CONSTANTS
    # Nobody said where to start, so it is the entrance every persona shares.
    assert route["origin"] == {
        "space_id": rf.FOYER,
        "label": "Foyer",
        "default": True,
        "door": "node/1",
        "asset_ids": [],
        "snapped_m": None,
    }
    assert route["destination"]["space_id"] == rf.SEMINAR
    assert passed(route) == [rf.FOYER, rf.CORRIDOR_0, rf.SEMINAR]
    assert turn["state"]["place"] == rf.SEMINAR


def test_the_duration_and_distance_are_cited_as_numbers_the_host_filled_in() -> None:
    turn = ask("visitor", "How do I get to the Seminarraum?", route_to("Seminarraum"))

    [segment] = turn["segments"]
    cited = {c["kind"]: c for c in segment["citations"]}
    assert cited["modelled-duration"]["seconds"] == 23
    assert cited["distance"]["metres"] == 31


def test_a_route_starts_where_the_question_says() -> None:
    turn = ask("visitor", "From the Seminarraum to the WC?", route_to("WC", origin="Seminarraum"))

    [route] = turn["routes"]
    # 2 m out of the room, 13.15 m along the corridor, 2 m into the WC.
    assert (route["seconds"], route["metres"]) == (13, 17)
    assert route["origin"]["space_id"] == rf.SEMINAR
    assert route["origin"]["default"] is False
    assert texts(turn) == ["From Seminarraum to WC: 13 seconds (modelled), 17 metres."]


def test_a_route_changes_level_by_the_stairs_when_they_are_quicker() -> None:
    turn = ask("visitor", "How do I get to room 202?", route_to("20.20 room 202"))

    [route] = turn["routes"]
    # 38.67 m at 1.34 m/s is 28.9 s, and one level by stairs is 12 s.
    assert (route["seconds"], route["metres"]) == (41, 39)
    assert passed(route) == [
        rf.FOYER,
        rf.CORRIDOR_0,
        rf.STAIRS,
        rf.CORRIDOR_1,
        rf.ROOM_201,
        rf.ROOM_202,
    ]
    stairs = route["passages"][2]
    assert (stairs["mode"], stairs["levels"]) == ("stairs", 1.0)
    # Metres are walked on the flat. The level adds time and no distance.
    assert (stairs["metres"], stairs["seconds"]) == (4.0, 15.0)
    assert route["step_free"] is False


# --- step-free --------------------------------------------------------------------------------


def test_a_step_free_route_takes_the_lift_and_no_stairs() -> None:
    turn = ask(
        "visitor",
        "How do I get to room 202 without stairs?",
        route_to("20.20 room 202", step_free=True),
    )

    [route] = turn["routes"]
    # 56.48 m is 42.2 s, and the lift is 40 s plus 4 s for the one level.
    assert (route["seconds"], route["metres"]) == (86, 56)
    assert route["step_free"] is True
    assert rf.STAIRS not in passed(route)
    lift = next(p for p in route["passages"] if p["space_id"] == rf.LIFT)
    assert (lift["mode"], lift["levels"]) == ("lift", 1.0)
    assert turn["trace"]["route"]["steps"][0]["step_free"] is True


@pytest.mark.parametrize("persona", ["visitor", "staff", "maintenance"])
def test_no_persona_implies_step_free(persona: str) -> None:
    # The lift is in the graph for everyone, and loses to the stairs for everyone.
    turn = ask(persona, "How do I get to room 202?", route_to("20.20 room 202"))

    [route] = turn["routes"]
    assert route["step_free"] is False
    assert (route["seconds"], rf.STAIRS in passed(route)) == (41, True)


def test_step_free_with_no_lift_is_the_same_abstention_and_the_trace_says_why() -> None:
    # Take the lift out of the snapshot: the only way up is the stairs.
    store = rf.store()
    store._edges = tuple(e for e in store._edges if rf.LIFT not in (e.from_id, e.to_id))

    turn = ask(
        "visitor",
        "Step-free to room 202?",
        route_to("20.20 room 202", step_free=True),
        store=store,
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-route")
    assert texts(turn) == [NO_ROUTE]
    # Not clearance: the same graph has a route with stairs. So the trace may say.
    [step] = turn["trace"]["route"]["steps"]
    assert step["reason"] == "there is a route, but not a step-free one"


# --- what a route may cross -------------------------------------------------------------------


def test_circulation_is_crossed_by_its_tags_and_an_untagged_room_by_a_derived_rule() -> None:
    turn = ask("visitor", "How do I get to room 202?", route_to("20.20 room 202"))

    [route] = turn["routes"]
    crossed = {p["space_id"]: (p["crossable"], p["rule"]) for p in route["passages"]}
    assert crossed[rf.CORRIDOR_0] == ("sourced", "crossable-by-tags@1")
    assert crossed[rf.STAIRS] == ("sourced", "crossable-by-tags@1")
    # Room 201 has no function tag and three doors. Flagged where the route uses it.
    assert crossed[rf.ROOM_201] == ("derived", "crossable-by-door-count@1")
    # The ends are entered, not crossed.
    assert crossed[rf.FOYER] == (None, None)
    assert crossed[rf.ROOM_202] == (None, None)
    [derived] = route["derived"]
    assert (derived["what"], derived["space_id"]) == ("crossable-space", rf.ROOM_201)
    assert derived["rule"] == "crossable-by-door-count@1"


def test_a_doorless_room_is_reached_by_its_snap_and_flagged_derived() -> None:
    turn = ask("visitor", "How do I get to the Teeküche?", route_to("Teeküche"))

    [route] = turn["routes"]
    # 10 + 12.04 + 4 + 15.63 along corridor 1 to the snap + 1.75 in: 43.42 m, 32.4 + 12 s.
    assert (route["seconds"], route["metres"]) == (44, 43)
    corridor = route["passages"][-2]
    assert corridor["space_id"] == rf.CORRIDOR_1
    assert (corridor["leaves_by"], corridor["leaves_by_rule"]) == (None, "snap-doorless-room@1")
    assert [(d["what"], d["rule"]) for d in route["derived"]] == [
        ("doorless-room-snap", "snap-doorless-room@1")
    ]


# --- destinations -----------------------------------------------------------------------------


def test_a_description_becomes_a_set_and_the_route_reaches_the_nearest() -> None:
    turn = ask(
        "visitor", "Where is the nearest toilet?", route_to("the nearest toilet", described=True)
    )

    [version] = turn["trace"]["plans"]
    matching = version["steps"][0]
    assert (matching["tool"], matching["how"]) == ("places_matching", "description")
    assert matching["ids"] == [rf.WC_0, rf.WC_1]
    [route] = turn["routes"]
    # The one on level 0: 10 + 6.08 + 2 m.
    assert route["destination"]["space_id"] == rf.WC_0
    assert (route["seconds"], route["metres"]) == (13, 18)
    assert route["alternatives"] == []


def test_a_route_step_returns_the_alternatives_it_is_asked_for() -> None:
    turn = ask(
        "visitor",
        "Which toilets are near?",
        route_to("the nearest toilet", described=True, alternatives=3),
    )

    [route] = turn["routes"]
    [other] = route["alternatives"]
    # Up the stairs: 36.29 m, 27.1 + 12 s. There are only two, so only one alternative.
    assert other["destination"]["space_id"] == rf.WC_1
    assert (other["seconds"], other["metres"]) == (39, 36)
    assert texts(turn) == [
        "From Foyer to WC: 13 seconds (modelled), 18 metres. "
        "Or 20.20 WC level 1: 39 seconds (modelled)."
    ]


def test_a_route_to_an_asset_goes_to_its_space() -> None:
    turn = ask("visitor", "Take me to a fire extinguisher", route_to_asset("fire extinguisher"))

    [route] = turn["routes"]
    assert route["destination"]["space_id"] == rf.CORRIDOR_0
    assert route["destination"]["asset_ids"] == [rf.EXTINGUISHER_0]
    # 10 m across the Foyer and 15 m to the middle of the corridor.
    assert (route["seconds"], route["metres"]) == (19, 25)


def test_an_asset_at_a_point_in_a_mapped_space_is_routed_to_that_space() -> None:
    # The defibrillator hangs on level 0 with no room. Its point lies in the Seminarraum.
    turn = ask("visitor", "Take me to the defibrillator", route_to_asset("defibrillator"))

    [route] = turn["routes"]
    assert route["destination"]["space_id"] == rf.SEMINAR
    assert route["destination"]["snapped_m"] is None
    assert route["derived"] == []
    assert (route["seconds"], route["metres"]) == (23, 31)


def test_an_asset_in_open_space_snaps_to_the_nearest_place_within_five_metres() -> None:
    # The hydrant hangs on the site, three metres outside the Foyer wall.
    turn = ask("visitor", "Take me to the hydrant", route_to_asset("hydrant"))

    [route] = turn["routes"]
    assert route["destination"]["space_id"] == rf.FOYER
    assert route["destination"]["snapped_m"] == 3.0
    assert [(d["what"], d["rule"]) for d in route["derived"]] == [
        ("open-space-snap", "snap-open-space-point@1")
    ]
    # From the door to the middle of the Foyer: 5 m.
    assert (route["seconds"], route["metres"]) == (4, 5)


def test_an_asset_too_far_out_in_open_space_has_no_route() -> None:
    # The standpipe is thirty metres from anything mapped.
    turn = ask("visitor", "Take me to the standpipe", route_to_asset("standpipe"))

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-route")
    assert texts(turn) == [NO_ROUTE]


# --- no route ---------------------------------------------------------------------------------


def test_a_destination_behind_a_restricted_corridor_has_no_route_for_a_visitor() -> None:
    inputs: list[str] = []
    turn = ask("visitor", "How do I get to room 210?", route_to("20.20 room 210"), inputs=inputs)

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-route")
    assert texts(turn) == [NO_ROUTE]
    assert turn["routes"] == []
    # Nothing of the staff corridor comes back or reaches a model, and nothing hints at it.
    assert rf.STAFF_CORRIDOR not in json.dumps(turn)
    assert rf.STAFF_CORRIDOR not in "".join(inputs)
    assert turn["trace"]["route"]["steps"][0]["reason"] is None


def test_staff_are_routed_through_the_corridor_a_visitor_cannot_take() -> None:
    turn = ask("staff", "How do I get to room 210?", route_to("20.20 room 210"))

    [route] = turn["routes"]
    # 10 + 12.04 + 4 + 18.03 + 6.08 + 2 = 52.15 m, 38.9 + 12 s.
    assert (route["seconds"], route["metres"]) == (51, 52)
    assert rf.STAFF_CORRIDOR in passed(route)
    # And the two personas' graphs are not the same size.
    visitor = ask("visitor", "How do I get to room 210?", route_to("20.20 room 210"))
    assert visitor["trace"]["route"]["subgraph_edges"] < turn["trace"]["route"]["subgraph_edges"]


def test_every_cause_of_no_route_is_the_same_abstention() -> None:
    def refusal(place: str, **more: Any) -> dict[str, Any]:
        turn = ask("visitor", "How do I get there?", route_to(place), **more)
        step = turn["trace"]["route"]["steps"][0]
        return {
            "kind": turn["kind"],
            "cause": turn["cause"],
            "segments": turn["segments"],
            "routes": turn["routes"],
            "candidates": turn["candidates"],
            "found": step["found"],
            "reason": step["reason"],
            "calls": [call["purpose"] for call in turn["llm_calls"]],
        }

    # Clearance: room 210 is off the staff corridor.
    restricted = refusal("20.20 room 210")
    # The crossing rule: room 205 is only reached through an office.
    behind_an_office = refusal("20.20 room 205")
    # A mapping gap: the Lager has no door and nothing to snap to.
    unmapped = refusal("Lager")

    assert restricted == behind_an_office == unmapped
    assert restricted["segments"] == [{"text": NO_ROUTE, "citations": []}]
    assert restricted["reason"] is None


def test_a_room_with_a_function_is_never_crossed_even_with_three_doors() -> None:
    # Room 203 is an office with three doors. Staff see everything and still cannot
    # get through it to room 205.
    turn = ask("staff", "How do I get to room 205?", route_to("20.20 room 205"))

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-route")


def test_a_place_the_persona_cannot_see_is_no_such_place_before_any_route() -> None:
    turn = ask("visitor", "How do I get to the Bibliothek?", route_to("Bibliothek"))

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-such-place")
    assert turn["trace"]["route"] is None


def test_with_no_default_origin_and_none_stated_there_is_no_route() -> None:
    turn = ask(
        "visitor",
        "How do I get to the Seminarraum?",
        route_to("Seminarraum"),
        store=rf.store(default_origin=None),
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-route")
    assert turn["trace"]["route"]["steps"][0]["reason"] == "the venue names no default origin"


# --- the plan ---------------------------------------------------------------------------------


def _two_routes(extra: list[dict[str, Any]] | None = None, **second: Any) -> str:
    steps = [
        {"id": "s1", "tool": "resolve_place", "mention": "m1"},
        {"id": "s2", "tool": "resolve_place", "mention": "m2"},
        {"id": "s3", "tool": "route", "destination": {"step": "s1"}},
        {"id": "s4", "tool": "route", "after": {"step": "s3"}, "destination": {"step": "s2"}}
        | second,
        *(extra or []),
    ]
    mentions = [
        {"id": "m1", "text": "Seminarraum", "kind": "place"},
        {"id": "m2", "text": "WC", "kind": "place"},
    ]
    return json.dumps({"shape": "route", "mentions": mentions, "steps": steps})


def test_a_second_route_starts_where_the_first_ended() -> None:
    turn = ask("visitor", "To the Seminarraum and then the WC?", _two_routes())

    first, second = turn["routes"]
    assert (first["step"], second["step"]) == ("s3", "s4")
    assert (first["seconds"], first["metres"]) == (23, 31)
    assert second["origin"] == {**second["origin"], "space_id": rf.SEMINAR, "default": False}
    assert (second["seconds"], second["metres"]) == (13, 17)
    assert texts(turn) == [
        "From Foyer to Seminarraum: 23 seconds (modelled), 31 metres.",
        "From Seminarraum to WC: 13 seconds (modelled), 17 metres.",
    ]


def test_a_plan_with_three_route_steps_is_rejected() -> None:
    third = {"id": "s5", "tool": "route", "after": {"step": "s4"}, "destination": {"step": "s1"}}
    turn = ask("visitor", "There and back and there again?", _two_routes([third]))

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    rejection = turn["trace"]["plans"][0]["rejection"]
    assert "the plan has 3 route steps, over the cap of 2" in rejection
    assert turn["trace"]["plans"][0]["steps"] == []


def test_a_route_step_asking_for_four_alternatives_is_rejected() -> None:
    turn = ask("visitor", "Show me every toilet", _two_routes(alternatives=4))

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    [reason] = turn["trace"]["plans"][0]["rejection"]
    assert reason.startswith("steps.3.alternatives: Input should be less than or equal to 3")


# --- proximity --------------------------------------------------------------------------------


def test_the_nearest_asset_of_a_type_is_by_modelled_duration_and_differs_by_persona() -> None:
    question = "Where is the nearest fire extinguisher to room 202?"
    plan = nearest("fire extinguisher", to="20.20 room 202")
    visitor = ask("visitor", question, plan, nearby)
    staff = ask("staff", question, plan, nearby)

    # A visitor knows of one, down the stairs: 19.79 m, 14.8 + 12 s.
    assert texts(visitor) == [
        "Nearest Fire extinguisher: 1 in a room with no label, 27 seconds (modelled), 20 metres."
    ]
    assert visitor["trace"]["shape"] == "proximity"
    assert visitor["routes"][0]["destination"]["space_id"] == rf.CORRIDOR_0
    assert visitor["trace"]["route"]["steps"][0]["basis"] == "modelled-duration"
    # Staff have one on the same level, in the staff corridor: 25.28 m and no stairs.
    assert texts(staff) == [
        "Nearest Fire extinguisher: 1 in a room with no label, 19 seconds (modelled), 25 metres."
    ]
    assert staff["routes"][0]["destination"]["space_id"] == rf.STAFF_CORRIDOR
    # The visitor's answer carries nothing of the one they cannot see.
    told = json.dumps(visitor)
    assert rf.EXTINGUISHER_STAFF not in told
    assert rf.STAFF_CORRIDOR not in told


def test_with_no_graph_nearest_is_by_shared_containment() -> None:
    # The Lager has no door, so nothing routes from it.
    question = "Where is the nearest fire extinguisher to the Lager?"
    plan = nearest("fire extinguisher", to="Lager")
    visitor = ask("visitor", question, plan, nearby)
    staff = ask("staff", question, plan, nearby)

    # A visitor's only one is on another level: they share the building.
    assert texts(visitor) == [
        "Nearest Fire extinguisher: 1 in a room with no label, in the same Physik."
    ]
    assert visitor["trace"]["route"]["steps"][0]["basis"] == "shared-containment"
    assert visitor["routes"] == []
    # Staff have one on the Lager's own level.
    assert texts(staff) == [
        "Nearest Fire extinguisher: 1 in a room with no label, in the same 20.20 level 1."
    ]


def test_the_nearest_with_no_place_named_is_nearest_the_default_origin() -> None:
    plan = nearest("fire extinguisher")
    turn = ask("visitor", "Where is the nearest fire extinguisher?", plan, nearby)

    # From the entrance: 10 m across the Foyer, 15 m to the middle of the corridor.
    assert turn["routes"][0]["origin"]["default"] is True
    assert (turn["routes"][0]["seconds"], turn["routes"][0]["metres"]) == (19, 25)


def test_a_type_the_persona_cannot_see_is_not_near_anything() -> None:
    plan = nearest("panelboard", to="20.20 room 202")
    visitor = ask("visitor", "Where is the nearest panelboard to room 202?", plan, nearby)
    technician = ask("maintenance", "Where is the nearest panelboard to room 202?", plan, nearby)

    assert (visitor["kind"], visitor["cause"]) == ("abstention", "no-such-asset-type")
    # It is in the room itself.
    assert texts(technician) == [
        "Nearest Panelboard P1: 1 generated in 20.20 room 202, 0 seconds (modelled), 0 metres."
    ]


# --- follow-ups -------------------------------------------------------------------------------


def _from_there(place: str) -> str:
    """A step-free route to ``place`` from "there": the place an earlier turn resolved."""
    mentions = [
        {"id": "m1", "text": "there", "kind": "place", "refers": {"to": "earlier-turn"}},
        {"id": "m2", "text": place, "kind": "place"},
    ]
    steps = [
        {"id": "s1", "tool": "resolve_place", "mention": "m1"},
        {"id": "s2", "tool": "resolve_place", "mention": "m2"},
        {
            "id": "s3",
            "tool": "route",
            "origin": {"step": "s1"},
            "destination": {"step": "s2"},
            "step_free": True,
        },
    ]
    return json.dumps({"shape": "route", "mentions": mentions, "steps": steps})


def test_the_last_route_is_carried_and_the_next_route_can_start_where_it_ended() -> None:
    first = ask("visitor", "Step-free to room 202?", route_to("20.20 room 202", step_free=True))

    assert first["state"]["route"] == {
        "kind": "route",
        "turn": 1,
        "space_ids": [rf.ROOM_202],
        "asset_ids": [],
        "asset_type_ids": [],
        "args": {"step_free": True, "default_origin": True},
    }

    heard: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        heard.append(payload)
        return _from_there("Teeküche")

    second = ask("visitor", "And from there to the Teeküche?", planner, state=first["state"])

    # The planner hears the last route by name, and that it was step-free.
    assert heard[0]["conversation"]["last_route"] == {"step_free": True, "to": "20.20 room 202"}
    [route] = second["routes"]
    assert route["origin"] == {**route["origin"], "space_id": rf.ROOM_202, "default": False}
    # All on level 1: 3 + 4.24 + 10.5 along the corridor to the snap + 1.75 = 19.49 m.
    assert (route["seconds"], route["metres"], route["step_free"]) == (15, 19, True)
    assert second["state"]["route"]["space_ids"] == [rf.ROOM_202, rf.KITCHEN]
    assert second["state"]["route"]["turn"] == 2


def test_a_last_route_naming_a_place_the_persona_cannot_see_is_dropped() -> None:
    # A state a visitor could not have been handed: a route through the staff corridor.
    state = {
        "snapshot": rf.SNAPSHOT,
        "persona": "visitor",
        "turns": 1,
        "route": {
            "kind": "route",
            "turn": 1,
            "space_ids": [rf.STAFF_CORRIDOR, rf.ROOM_210],
            "args": {"step_free": False, "default_origin": False},
        },
    }
    inputs: list[str] = []

    turn = ask(
        "visitor", "And to the Bibliothek?", route_to("Bibliothek"), state=state, inputs=inputs
    )

    assert turn["state"]["route"] is None
    # The instructions mention the key. The payload must not carry one.
    assert '"last_route": {' not in "".join(inputs)
    assert rf.STAFF_CORRIDOR not in json.dumps(turn)


def test_the_nearest_to_an_asset_from_an_earlier_turn_is_nearest_to_its_space() -> None:
    # The panelboard an earlier turn was about sits in room 202.
    state = {
        "snapshot": rf.SNAPSHOT,
        "persona": "maintenance",
        "turns": 1,
        "assets": ["seed:asset:P1"],
        "since": {"assets": 1},
    }
    mentions = [
        {"id": "m1", "text": "fire extinguisher", "kind": "asset-type"},
        {"id": "m2", "text": "that one", "kind": "asset", "refers": {"to": "earlier-turn"}},
    ]
    steps = [
        {"id": "s1", "tool": "resolve_asset_type", "mention": "m1"},
        {"id": "s2", "tool": "assets_in_scope", "types": {"step": "s1"}},
        {"id": "s3", "tool": "resolve_asset", "mention": "m2"},
        {"id": "s4", "tool": "nearest", "assets": {"step": "s2"}, "to": {"step": "s3"}},
    ]
    plan = json.dumps({"shape": "proximity", "mentions": mentions, "steps": steps})

    turn = ask(
        "maintenance", "Where is the nearest extinguisher to that one?", plan, nearby, state=state
    )

    [route] = turn["routes"]
    assert route["origin"]["space_id"] == rf.ROOM_202
    assert route["destination"]["space_id"] == rf.STAFF_CORRIDOR
    assert (route["seconds"], route["metres"]) == (19, 25)


# --- over Postgres ----------------------------------------------------------------------------

ADMIN_URL_ENV = "SPATIAL_TEST_DATABASE_URL"


def _bundle(root: Path) -> Path:
    bundle = root / rf.SNAPSHOT
    seen_assets, seen_types = rf.visible_assets()
    records: dict[str, list[BaseModel]] = {
        "osm/spaces.jsonl": list(rf.spaces()),
        "osm/edges.jsonl": list(rf.edges()),
        "osm/visible_spaces.jsonl": list(rf.visible()),
    }
    for half in ("osm", "seed"):
        records |= {
            f"{half}/asset_types.jsonl": [t for t in rf.asset_types() if t.id.startswith(half)],
            f"{half}/assets.jsonl": [a for a in rf.assets() if a.id.startswith(half)],
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
        (bundle / name).parent.mkdir(parents=True, exist_ok=True)
        (bundle / name).write_bytes(content)
        parts[name] = {"sha256": hashlib.sha256(content).hexdigest(), "records": len(rows)}
    manifest = {
        "snapshot": rf.SNAPSHOT,
        "venue": "routes",
        "parts": parts,
        "default_origin": {"door": "node/1", "edge": rf.ENTRANCE},
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    return bundle


@pytest.fixture(scope="module")
def reader_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A database holding the routes snapshot, current, and a reader's URL to it."""
    admin_url = os.environ.get(ADMIN_URL_ENV)
    if not admin_url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the store tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    name = f"spatial_test_{uuid4().hex[:12]}"
    reader = f"spatial_test_reader_{uuid4().hex[:8]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    owner_url = make_conninfo(admin_url, dbname=name)
    result = load(bundle=_bundle(tmp_path_factory.mktemp("bundles")), database_url=owner_url)
    assert result.ok, result.failures
    promote(snapshot=rf.SNAPSHOT, database_url=owner_url)
    with psycopg.connect(owner_url, autocommit=True) as conn:
        conn.execute(
            sql.SQL("CREATE ROLE {} LOGIN PASSWORD 'reader' IN ROLE {}").format(
                sql.Identifier(reader), sql.Identifier(READER_ROLE)
            )
        )
    yield make_conninfo(owner_url, user=reader, password="reader")
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
        conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(reader)))


def test_a_route_read_from_postgres_runs_over_the_personas_subgraph(reader_url: str) -> None:
    store = PostgresStore(reader_url)
    plan = route_to("20.20 room 210")

    visitor = ask("visitor", "How do I get to room 210?", plan, store=store)
    staff = ask("staff", "How do I get to room 210?", plan, store=store)

    assert (visitor["kind"], visitor["cause"]) == ("abstention", "no-route")
    assert (staff["routes"][0]["seconds"], staff["routes"][0]["metres"]) == (51, 52)
    # The default origin came from the manifest, the snap and the doors from the views.
    assert staff["routes"][0]["origin"]["door"] == "node/1"
    kitchen = ask("visitor", "How do I get to the Teeküche?", route_to("Teeküche"), store=store)
    assert (kitchen["routes"][0]["seconds"], kitchen["routes"][0]["metres"]) == (44, 43)


def test_nearest_of_a_type_read_from_postgres_differs_by_persona(reader_url: str) -> None:
    store = PostgresStore(reader_url)
    plan = nearest("fire extinguisher", to="20.20 room 202")
    question = "Where is the nearest fire extinguisher to room 202?"

    visitor = ask("visitor", question, plan, nearby, store=store)
    staff = ask("staff", question, plan, nearby, store=store)

    assert visitor["routes"][0]["destination"]["space_id"] == rf.CORRIDOR_0
    assert staff["routes"][0]["destination"]["space_id"] == rf.STAFF_CORRIDOR
    assert (visitor["routes"][0]["seconds"], staff["routes"][0]["seconds"]) == (27, 19)
