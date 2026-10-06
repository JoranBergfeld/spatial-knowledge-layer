"""The turn operation, end to end over HTTP, against the fixture snapshot and a scripted model."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fixture
from fixture import COMPOSER, PLANNER, located, where
from spatial_core.records import PERSONAS, Persona
from spatial_service.app import create_app
from spatial_service.llm import Llm
from spatial_service.plan import STEP_CAP

ROOM_107_A = "osm:way/1003"
ROOM_107_B = "osm:way/2001"
ROOM_107_C = "osm:way/3001"
SERVERRAUM = "osm:way/1009"
AUDIMAX = "osm:way/1002"


def client(llm: Llm) -> TestClient:
    return TestClient(create_app(fixture.store(), llm))


def ask(
    persona: str,
    question: str,
    planner: Any,
    composer: Any = located,
    *,
    state: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    snapshot: str | None = None,
    status: int = 200,
) -> dict[str, Any]:
    llm = fixture.llm(
        planner if callable(planner) or planner is None else (lambda _p: planner),
        composer,
        inputs if inputs is not None else [],
    )
    body: dict[str, Any] = {"persona": persona, "question": question, "state": state}
    if snapshot:
        body["snapshot"] = snapshot
    response = client(llm).post("/turn", json=body)
    assert response.status_code == status, response.text
    result: dict[str, Any] = response.json()
    return result


def visible_to(persona: Persona) -> set[str]:
    return {row.space_id for row in fixture.visible() if row.persona is persona}


# --- answers ------------------------------------------------------------------------------


def test_an_answer_names_the_place_and_what_holds_it() -> None:
    turn = ask("visitor", "Where is 50.34 room 107?", where("50.34 room 107"))

    assert turn["kind"] == "answer"
    assert [s["text"] for s in turn["segments"]] == [
        "50.34 room 107 is in 50.34 level 1, Informatik, Mini Campus."
    ]
    cited = [c["space_id"] for c in turn["segments"][0]["citations"]]
    assert cited == [ROOM_107_A, "floor:way/100:1", "osm:way/100", "site:mini"]
    assert turn["state"] == {
        "snapshot": fixture.SNAPSHOT,
        "persona": "visitor",
        "turns": 1,
        "place": ROOM_107_A,
        "pending": None,
    }


def test_the_trace_shows_shape_plan_steps_resolution_provenance_and_defaults() -> None:
    trace = ask("visitor", "Where is 50.34 room 107?", where("50.34 room 107"))["trace"]

    assert trace["shape"] == "place-location"
    assert trace["plan"] == json.loads(where("50.34 room 107"))
    assert trace["step_cap"] == STEP_CAP
    assert [(s["id"], s["outcome"], s["ids"]) for s in trace["steps"]] == [
        ("s1", "resolved", [ROOM_107_A])
    ]
    assert trace["resolved"] == [
        {
            "mention": "m1",
            "text": "50.34 room 107",
            "step": "s1",
            "space_id": ROOM_107_A,
            "label": "50.34 room 107",
            "how": "label",
        }
    ]
    room = trace["provenance"][0]
    assert room["label_provenance"] == "derived"
    assert room["label_rule"] == "room-label@1"
    assert room["source"] == "way/1003"
    # The corridor's clearance is an authored default, and it is what lets a visitor see it.
    assert trace["authored_defaults"] == [
        {"space_id": ROOM_107_A, "persona": "visitor", "rule": "clearance-authored-default@1"}
    ]


def test_each_llm_call_reports_its_tokens_and_deployment() -> None:
    turn = ask("visitor", "Where is the Audimax?", where("Audimax"))

    assert turn["llm_calls"] == [
        {
            "purpose": "planning",
            "deployment": PLANNER,
            "input_tokens": 120,
            "output_tokens": 40,
            "total_tokens": 160,
        },
        {
            "purpose": "composition",
            "deployment": COMPOSER,
            "input_tokens": 80,
            "output_tokens": 20,
            "total_tokens": 100,
        },
    ]


@pytest.mark.parametrize(
    "text",
    ["50.34/107", "room 107 in 50.34", "Raum 107 im Gebäude 50.34", "Informatik room 107"],
)
def test_a_room_label_parses_from_its_building_and_number(text: str) -> None:
    turn = ask("visitor", f"Where is {text}?", where(text))

    assert turn["kind"] == "answer"
    assert turn["trace"]["resolved"][0]["space_id"] == ROOM_107_A
    assert turn["trace"]["resolved"][0]["how"] == "building-and-room"


def test_a_later_step_works_within_what_an_earlier_one_resolved() -> None:
    turn = ask(
        "visitor", "Where is room 107 in the Informatik building?", where("Informatik", "107")
    )

    assert turn["kind"] == "answer"
    assert [(r["step"], r["space_id"]) for r in turn["trace"]["resolved"]] == [
        ("s1", "osm:way/100"),
        ("s2", ROOM_107_A),
    ]


# --- clarifications -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("persona", "expected"),
    [
        ("visitor", [ROOM_107_B, ROOM_107_A]),
        ("staff", [ROOM_107_C, ROOM_107_B, ROOM_107_A]),
    ],
)
def test_a_bare_room_number_asks_which_among_what_the_persona_sees(
    persona: str, expected: list[str]
) -> None:
    turn = ask(persona, "Where is 107?", where("107"))

    assert turn["kind"] == "clarification"
    assert [c["space_id"] for c in turn["candidates"]] == expected
    assert set(expected) <= visible_to(Persona(persona))
    assert turn["state"]["pending"] == {"mention": "107", "candidates": expected}
    assert [s["text"] for s in turn["segments"]] == [
        "More than one place matches. Which one do you mean?"
    ]


def test_a_bare_room_number_asks_even_when_one_room_carries_it() -> None:
    turn = ask("staff", "Where is 108?", where("108"))

    assert turn["kind"] == "clarification"
    assert [c["space_id"] for c in turn["candidates"]] == ["osm:way/1004"]


def test_several_close_names_ask_which() -> None:
    turn = ask("visitor", "Where is the Seminarraum?", where("Seminarraum"))

    assert turn["kind"] == "clarification"
    assert [c["label"] for c in turn["candidates"]] == ["Seminarraum Nord", "Seminarraum Süd"]
    assert [c["building"] for c in turn["candidates"]] == ["30.10", "30.10"]


def test_the_next_turn_answers_the_clarification() -> None:
    first = ask("visitor", "Where is 107?", where("107"))
    seen: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return where(payload["pending"]["candidates"][0])

    second = ask("visitor", "the one in 30.10", planner, state=first["state"])

    assert seen[0]["pending"] == {
        "mention": "107",
        "candidates": ["30.10 room 107", "50.34 room 107"],
    }
    assert second["kind"] == "answer"
    assert second["state"]["place"] == ROOM_107_B
    assert second["state"]["pending"] is None
    assert second["state"]["turns"] == 2


# --- abstentions --------------------------------------------------------------------------


def test_beyond_clearance_and_nonexistent_abstain_alike() -> None:
    hidden = ask("visitor", "Where is it?", where("Serverraum"))
    missing = ask("visitor", "Where is it?", where("Bibliothek"))

    for turn in (hidden, missing):
        assert turn["kind"] == "abstention"
        assert turn["cause"] == "no-such-place"
        assert turn["candidates"] == []
    assert hidden["segments"] == missing["segments"]
    assert hidden["state"] == missing["state"]
    assert hidden["llm_calls"] == missing["llm_calls"]

    def without_text(trace: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in trace.items() if k not in ("plan", "steps")} | {
            "steps": [{k: v for k, v in s.items() if k != "text"} for s in trace["steps"]]
        }

    assert without_text(hidden["trace"]) == without_text(missing["trace"])


def test_maintenance_does_see_the_serverraum() -> None:
    turn = ask("maintenance", "Where is the Serverraum?", where("Serverraum"))

    assert turn["kind"] == "answer"
    assert turn["state"]["place"] == SERVERRAUM


def test_a_question_of_another_shape_abstains() -> None:
    plan = json.dumps({"shape": "unsupported", "mentions": [], "steps": []})
    turn = ask("visitor", "How many rooms are there?", plan)

    assert (turn["kind"], turn["cause"]) == ("abstention", "unsupported-question")
    assert [call["purpose"] for call in turn["llm_calls"]] == ["planning"]


# --- rejected plans -----------------------------------------------------------------------


def _plan(**changes: Any) -> str:
    plan = json.loads(where("50.34 room 107"))
    plan.update(changes)
    return json.dumps(plan)


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            _plan(
                steps=[{"id": "s1", "tool": "resolve_place", "mention": "m1", "persona": "staff"}]
            ),
            "Extra inputs are not permitted",
        ),
        (_plan(snapshot="2026-01-01"), "Extra inputs are not permitted"),
        (where(*[f"place {n}" for n in range(STEP_CAP + 1)]), "over the cap"),
        (
            _plan(
                steps=[
                    {
                        "id": "s1",
                        "tool": "resolve_place",
                        "mention": "m1",
                        "within": {"step": "s2"},
                    }
                ]
            ),
            "not earlier",
        ),
        (_plan(steps=[{"id": "s1", "tool": "find_asset", "mention": "m1"}]), "steps.0.tool"),
        (_plan(shape="place-count"), "shape"),
        ("this is not a plan", "Invalid JSON"),
    ],
)
def test_a_plan_that_fails_its_checks_runs_nothing(plan: str, reason: str) -> None:
    turn = ask("visitor", "Where is 50.34 room 107?", plan)

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    assert any(reason in r for r in turn["trace"]["plan_rejection"]), turn["trace"]
    assert turn["trace"]["steps"] == []
    assert [call["purpose"] for call in turn["llm_calls"]] == ["planning"]


# --- composition --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "segment",
    [
        {"text": "{f1} is on floor 1 of {f3}.", "cites": ["f1", "f3"]},
        {"text": "{f1} is in building ５０.", "cites": ["f1"]},
    ],
)
def test_composed_output_with_a_digit_is_rejected_whole(segment: dict[str, Any]) -> None:
    def composer(_payload: dict[str, Any]) -> str:
        return json.dumps({"segments": [{"text": "{f1} is here.", "cites": ["f1"]}, segment]})

    turn = ask("visitor", "Where is 50.34 room 107?", where("50.34 room 107"), composer)

    assert (turn["kind"], turn["cause"]) == ("abstention", "composition-rejected")
    assert turn["trace"]["composition_rejection"] == ["segment 2 contains a digit"]
    # No partial answer: nothing of the first, clean segment survives.
    assert [s["text"] for s in turn["segments"]] == [
        "I found the place but couldn't word the answer."
    ]
    assert turn["state"]["place"] is None


def test_composition_naming_an_unknown_fact_is_rejected() -> None:
    def composer(_payload: dict[str, Any]) -> str:
        return json.dumps({"segments": [{"text": "{f1} is in {f9}.", "cites": ["f1"]}]})

    turn = ask("visitor", "Where is the Audimax?", where("Audimax"), composer)

    assert turn["cause"] == "composition-rejected"
    assert turn["trace"]["composition_rejection"] == ["segment 1 names f9, which is not a fact"]


# --- the model unreachable ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("planner", "composer", "calls"),
    [(None, located, []), (lambda _p: where("Audimax"), None, ["planning"])],
)
def test_an_unreachable_model_answers_nothing(
    planner: Any, composer: Any, calls: list[str]
) -> None:
    state = {"snapshot": fixture.SNAPSHOT, "persona": "visitor", "turns": 3, "place": AUDIMAX}

    turn = ask("visitor", "Where is the Audimax?", planner, composer, state=state, status=503)

    assert turn["kind"] == "unavailable"
    assert [s["text"] for s in turn["segments"]] == [
        "The language model could not be reached, so there is no answer."
    ]
    assert all(s["citations"] == [] for s in turn["segments"])
    assert turn["candidates"] == []
    assert turn["state"] == {**state, "pending": None}
    assert [call["purpose"] for call in turn["llm_calls"]] == calls


# --- snapshots and state ------------------------------------------------------------------


def test_a_conversation_may_stay_on_the_previous_snapshot() -> None:
    turn = ask("visitor", "Where is the Audimax?", where("Audimax"), snapshot=fixture.PREVIOUS)

    assert turn["state"]["snapshot"] == fixture.PREVIOUS


@pytest.mark.parametrize(
    ("body", "status"),
    [
        ({"snapshot": "2026-09-01"}, 409),
        ({"state": {"snapshot": fixture.SNAPSHOT, "persona": "staff"}}, 409),
        (
            {
                "snapshot": fixture.PREVIOUS,
                "state": {"snapshot": fixture.SNAPSHOT, "persona": "visitor"},
            },
            409,
        ),
        ({"persona": "janitor"}, 422),
        ({"question": ""}, 422),
    ],
)
def test_a_request_the_service_cannot_serve_is_refused(body: dict[str, Any], status: int) -> None:
    llm = fixture.llm(lambda _p: where("Audimax"), located, [])
    request = {"persona": "visitor", "question": "Where is the Audimax?", **body}

    assert client(llm).post("/turn", json=request).status_code == status


# --- what the model is shown --------------------------------------------------------------


@pytest.mark.parametrize("persona", [p for p in PERSONAS if p is not Persona.MAINTENANCE])
def test_no_id_the_persona_cannot_see_reaches_a_model(persona: Persona) -> None:
    hidden = [s for s in fixture.spaces() if s.id not in visible_to(persona)]
    assert hidden, "the fixture hides something from every persona but maintenance"
    inputs: list[str] = []
    every_space = [s.id for s in fixture.spaces()]
    # A client may send back state naming spaces the persona cannot see.
    forged = {
        "snapshot": fixture.SNAPSHOT,
        "persona": persona.value,
        "place": SERVERRAUM,
        "pending": {"mention": "107", "candidates": every_space},
    }

    def pick_first(payload: dict[str, Any]) -> str:
        return where(payload["pending"]["candidates"][0])

    for text in ("107", "Seminarraum", "Serverraum", "50.34 room 107", "Audimax"):
        ask(persona.value, "Where is it?", where(text), inputs=inputs)
    ask(persona.value, "the first one", pick_first, state=forged, inputs=inputs)

    assert len(inputs) > 10
    for text in inputs:
        for space in hidden:
            assert space.id not in text
            assert space.label is None or space.label.value not in text
