"""Follow-ups, end to end over HTTP: a later turn leaning on what an earlier one resolved."""

from __future__ import annotations

import json
from typing import Any

import pytest

import fixture
from fixture import counted, installed, located, most, where
from spatial_core.records import PERSONAS, Persona
from test_turn import ask

AUDIMAX = "osm:way/1002"
SERVERRAUM = "osm:way/1009"
INFORMATIK = "osm:way/100"
ROOM_107_A = "osm:way/1003"
ROOM_107_B = "osm:way/2001"
SEMINARRAUM_NORD = "osm:way/2002"
EXTINGUISHER = "osm:asset:node/9001"
# The one light in Seminarraum Nord, and the two in 50.34 room 107, the second unplaced.
LIGHT_NORD = "seed:asset:L9"
LIGHT_107, LIGHT_107_UNPLACED = "seed:asset:L6", "seed:asset:L7"
# In 50.34 room 108, which a visitor cannot see.
LIGHT_108 = "seed:asset:L8"

TOOLS = {"place": "resolve_place", "asset": "resolve_asset", "asset-type": "resolve_asset_type"}


def pointing(kind: str, text: str, ordinal: int | None = None) -> dict[str, Any]:
    """A mention that is a follow-up reference: at a candidate if ``ordinal``, else at a turn."""
    refers: dict[str, Any] = {"to": "earlier-turn" if ordinal is None else "candidate"}
    if ordinal is not None:
        refers["ordinal"] = ordinal
    return {"text": text, "kind": kind, "refers": refers}


def where_is(mention: dict[str, Any]) -> str:
    """A plan that locates the one thing a mention names: a place or an asset."""
    shape = "asset-location" if mention["kind"] == "asset" else "place-location"
    step = {"id": "s1", "tool": TOOLS[mention["kind"]], "mention": "m1"}
    return json.dumps({"shape": shape, "mentions": [{"id": "m1", **mention}], "steps": [step]})


def installed_in(place: dict[str, Any], kind: dict[str, Any] | None = None) -> str:
    """An asset-containment plan over mentions as given, so either can be a reference."""
    mentions = [{"id": "m1", **place}]
    steps: list[dict[str, Any]] = [{"id": "s1", "tool": "resolve_place", "mention": "m1"}]
    scope: dict[str, Any] = {"tool": "assets_in_scope", "scope": {"step": "s1"}}
    if kind is not None:
        mentions.append({"id": "m2", **kind})
        steps.append({"id": "s2", "tool": "resolve_asset_type", "mention": "m2"})
        scope["types"] = {"step": "s2"}
    steps.append({"id": f"s{len(steps) + 1}", **scope})
    return json.dumps({"shape": "asset-containment", "mentions": mentions, "steps": steps})


def named(text: str, kind: str = "place") -> dict[str, Any]:
    return {"text": text, "kind": kind}


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


def state_for(persona: str, **held: Any) -> dict[str, Any]:
    return {"snapshot": fixture.SNAPSHOT, "persona": persona, "turns": 1, **held}


def worded(payload: dict[str, Any]) -> str:
    """A composer for whichever shape the plan turned out to be."""
    return counted(payload) if "total" in payload else located(payload)


THAT_ONE = where_is(pointing("asset", "that one"))


# --- the candidates a clarification offered -----------------------------------------------


def test_the_second_one_is_the_second_candidate_the_clarification_offered() -> None:
    first = ask("visitor", "Where is 107?", where("107"))
    assert [c["space_id"] for c in first["candidates"]] == [ROOM_107_B, ROOM_107_A]

    plan = where_is(pointing("place", "the second one", 2))
    second = ask("visitor", "the second one", plan, state=first["state"])

    assert second["kind"] == "answer"
    assert texts(second) == ["50.34 room 107 is in 50.34 level 1, Informatik, Mini Campus."]
    assert second["trace"]["references"] == [
        {
            "mention": "m1",
            "text": "the second one",
            "step": "s1",
            "kind": "place",
            "to": "candidate",
            "ordinal": 2,
            "outcome": "resolved",
            "ids": [ROOM_107_A],
            "labels": ["50.34 room 107"],
            "turn": 1,
            "through": [],
        }
    ]
    assert second["trace"]["resolved"][0]["how"] == "follow-up"
    assert second["state"]["place"] == ROOM_107_A
    assert second["state"]["pending"] is None


def test_an_ordinal_past_the_last_candidate_offers_them_again() -> None:
    first = ask("visitor", "Where is 107?", where("107"))

    plan = where_is(pointing("place", "the fifth one", 5))
    second = ask("visitor", "the fifth one", plan, state=first["state"])

    assert second["kind"] == "clarification"
    assert second["candidates"] == first["candidates"]
    assert second["state"]["pending"] == first["state"]["pending"]
    assert [c["purpose"] for c in second["llm_calls"]] == ["planning"]
    [reference] = second["trace"]["references"]
    assert (reference["outcome"], reference["ids"], reference["ordinal"]) == ("none", [], 5)


def test_a_pending_mention_longer_than_a_mention_may_be_is_refused() -> None:
    # The state comes back from a client, and the pending mention goes on to the planner.
    # A plan's own mention is capped at 200 characters, so this one is too.
    def pending(mention: str) -> dict[str, Any]:
        held = {"mention": mention, "candidates": [ROOM_107_B, ROOM_107_A], "kind": "place"}
        return state_for("visitor", pending=held)

    plan = where_is(pointing("place", "the second one", 2))
    inputs: list[str] = []

    kept = ask("visitor", "the second one", plan, state=pending("x" * 200))
    refused = ask(
        "visitor", "the second one", plan, state=pending("x" * 201), status=422, inputs=inputs
    )

    assert kept["kind"] == "answer"
    assert refused["detail"][0]["loc"] == ["body", "state", "pending", "mention"]
    assert inputs == []


# --- an asset from an earlier turn --------------------------------------------------------


def test_that_one_is_the_asset_an_earlier_turn_found() -> None:
    first = ask(
        "visitor",
        "Where are the fire extinguishers in the Audimax?",
        installed("Audimax", "fire extinguishers"),
        counted,
    )
    assert first["state"]["assets"] == [EXTINGUISHER]
    assert first["state"]["since"] == {"place": 1, "asset_types": 1, "assets": 1}

    second = ask("visitor", "Where is that one?", THAT_ONE, state=first["state"])

    assert second["kind"] == "answer"
    assert second["trace"]["shape"] == "asset-location"
    assert texts(second) == [
        "Fire extinguisher is in Audimax, 50.34 level 0, Informatik, Mini Campus."
    ]
    assert second["trace"]["references"] == [
        {
            "mention": "m1",
            "text": "that one",
            "step": "s1",
            "kind": "asset",
            "to": "earlier-turn",
            "ordinal": None,
            "outcome": "resolved",
            "ids": [EXTINGUISHER],
            "labels": ["Fire extinguisher"],
            "turn": 1,
            "through": [],
        }
    ]
    subject = second["segments"][0]["citations"][0]
    assert (subject["kind"], subject["asset_id"]) == ("asset", EXTINGUISHER)
    assert subject["record_provenance"] == "sourced"
    [listed] = second["trace"]["assets"]["listed"]
    assert (listed["asset_id"], listed["position"]) == (EXTINGUISHER, "recorded")


def test_a_generated_asset_says_so_when_it_is_the_answer() -> None:
    first = ask("visitor", "What is in Seminarraum Nord?", installed("Seminarraum Nord"), counted)

    second = ask("visitor", "Where is that one?", THAT_ONE, state=first["state"])

    assert texts(second) == [
        "Light Fixture B1 13 (generated) is in Seminarraum Nord, 30.10 level 0, 30.10, Mini Campus."
    ]
    assert second["segments"][0]["citations"][0]["record_provenance"] == "seed"


def test_like_it_is_the_type_of_that_asset() -> None:
    first = ask("visitor", "What is in Seminarraum Nord?", installed("Seminarraum Nord"), counted)

    plan = installed_in(named("Informatik"), pointing("asset-type", "it"))
    second = ask(
        "visitor", "Are there more like it in Informatik?", plan, counted, state=first["state"]
    )

    # Its type, Light Fixture B1, and not every kind of light: the Q4 in the stairs is left out.
    assert texts(second) == [
        "Informatik holds 5 generated.",
        "Audimax: 3 generated.",
        "50.34 room 107: 2 generated.",
    ]
    assert second["trace"]["references"] == [
        {
            "mention": "m2",
            "text": "it",
            "step": "s2",
            "kind": "asset-type",
            "to": "earlier-turn",
            "ordinal": None,
            "outcome": "resolved",
            "ids": [fixture.LIGHT_B1],
            "labels": ["Light Fixture B1"],
            "turn": 1,
            "through": [LIGHT_NORD],
        }
    ]
    assert second["state"]["asset_types"] == [fixture.LIGHT_B1]


def test_a_third_turn_reaches_back_past_the_second() -> None:
    first = ask("visitor", "What is in Seminarraum Nord?", installed("Seminarraum Nord"), counted)
    second = ask("visitor", "Where is the Audimax?", where("Audimax"), state=first["state"])
    assert second["state"]["since"] == {"place": 2, "assets": 1}

    plan = installed_in(pointing("place", "there"), pointing("asset-type", "it"))
    third = ask("visitor", "Any like it in there?", plan, counted, state=second["state"])

    assert texts(third)[0] == "Audimax holds 3 generated."
    assert [(r["text"], r["ids"], r["turn"]) for r in third["trace"]["references"]] == [
        ("there", [AUDIMAX], 2),
        ("it", [fixture.LIGHT_B1], 1),
    ]
    assert third["state"]["questions"] == [
        "What is in Seminarraum Nord?",
        "Where is the Audimax?",
        "Any like it in there?",
    ]


def test_there_after_a_count_is_the_space_that_came_out_on_top() -> None:
    first = ask(
        "visitor", "Which building has the most lights?", most("building", "lights"), counted
    )
    assert first["state"]["place"] == INFORMATIK

    plan = installed_in(pointing("place", "there"))
    second = ask("visitor", "What is installed there?", plan, counted, state=first["state"])

    assert second["trace"]["references"][0]["ids"] == [INFORMATIK]
    assert second["trace"]["assets"]["scope"] == INFORMATIK


def test_an_answer_over_many_assets_leaves_none_to_point_at() -> None:
    first = ask("visitor", "Where are the lights?", installed(None, "lights"), counted)
    assert first["trace"]["assets"]["touched"] == 7
    assert first["state"]["assets"] == []

    second = ask("visitor", "Where is that one?", THAT_ONE, state=first["state"])

    assert (second["kind"], second["cause"]) == ("abstention", "nothing-referred-to")


def test_a_reference_with_nothing_before_it_abstains_and_buys_no_recovery() -> None:
    turn = ask("visitor", "Where is that one?", THAT_ONE)

    assert (turn["kind"], turn["cause"]) == ("abstention", "nothing-referred-to")
    assert texts(turn) == ["I don't know what that refers to."]
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    assert turn["trace"]["references"][0]["outcome"] == "none"


# --- an ambiguous reference ---------------------------------------------------------------


def test_that_one_among_several_asks_which_and_the_next_turn_picks() -> None:
    first = ask("visitor", "What is in 50.34 room 107?", installed("50.34 room 107"), counted)
    assert first["state"]["assets"] == [LIGHT_107, LIGHT_107_UNPLACED]

    second = ask("visitor", "Where is that one?", THAT_ONE, state=first["state"])

    assert second["kind"] == "clarification"
    assert texts(second) == ["That could be more than one asset. Which one do you mean?"]
    # Two of a type in one room: told apart by id, and by whether a position is recorded.
    assert second["candidates"] == [
        {
            "space_id": ROOM_107_A,
            "label": "Light Fixture B1 10",
            "building": "Informatik",
            "asset_id": LIGHT_107,
            "place": "50.34 room 107",
            "position": "recorded",
        },
        {
            "space_id": ROOM_107_A,
            "label": "Light Fixture B1 11",
            "building": "Informatik",
            "asset_id": LIGHT_107_UNPLACED,
            "place": "50.34 room 107",
            "position": "position not recorded",
        },
    ]
    assert second["state"]["pending"] == {
        "mention": "that one",
        "candidates": [LIGHT_107, LIGHT_107_UNPLACED],
        "kind": "asset",
        "turn": 2,
    }
    [reference] = second["trace"]["references"]
    assert (reference["outcome"], reference["turn"]) == ("ambiguous", 1)
    assert reference["ids"] == [LIGHT_107, LIGHT_107_UNPLACED]

    seen: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return where_is(pointing("asset", "the second one", 2))

    third = ask("visitor", "the second one", planner, state=second["state"])

    # The planner is given the question the clarification was about, and the candidates.
    assert seen[0]["conversation"]["earlier_questions"] == [
        "What is in 50.34 room 107?",
        "Where is that one?",
    ]
    assert seen[0]["pending"] == {
        "mention": "that one",
        "kind": "asset",
        "candidates": [
            "Light Fixture B1 10, in 50.34 room 107",
            "Light Fixture B1 11, in 50.34 room 107",
        ],
    }
    assert third["kind"] == "answer"
    assert texts(third)[0].startswith("Light Fixture B1 11 (generated) is in 50.34 room 107")
    [reference] = third["trace"]["references"]
    assert (reference["to"], reference["ids"], reference["turn"]) == (
        "candidate",
        [LIGHT_107_UNPLACED],
        2,
    )
    assert third["trace"]["assets"]["listed"][0]["position"] == "position not recorded"
    assert third["state"]["assets"] == [LIGHT_107_UNPLACED]


def test_like_it_among_assets_of_several_types_asks_which_asset() -> None:
    first = ask("visitor", "What is in the Audimax?", installed("Audimax"), counted)

    plan = installed_in(named("Informatik"), pointing("asset-type", "it"))
    second = ask("visitor", "More like it in Informatik?", plan, counted, state=first["state"])

    assert second["kind"] == "clarification"
    assert [c["asset_id"] for c in second["candidates"]] == first["state"]["assets"]
    assert second["state"]["pending"]["kind"] == "asset"

    third = ask(
        "visitor",
        "the first one",
        installed_in(named("Informatik"), pointing("asset-type", "the first one", 1)),
        counted,
        state=second["state"],
    )

    # The first candidate is the extinguisher, so "like it" is its type.
    [reference] = third["trace"]["references"]
    assert reference["ids"] == [fixture.EXTINGUISHER]
    assert reference["through"] == [EXTINGUISHER]
    assert texts(third)[0] == "Informatik holds 1."


def test_the_candidates_of_an_ambiguous_reference_are_only_what_the_persona_sees() -> None:
    # A staff client sends back the Audimax's assets with the panelboard slipped in.
    handed = ["seed:asset:P1", "seed:asset:L1", "seed:asset:L2"]
    state = state_for("staff", assets=handed, since={"assets": 1})

    staff = ask("staff", "Where is that one?", THAT_ONE, state=state)
    technician = ask(
        "maintenance", "Where is that one?", THAT_ONE, state={**state, "persona": "maintenance"}
    )

    assert staff["kind"] == "clarification"
    assert [c["asset_id"] for c in staff["candidates"]] == ["seed:asset:L1", "seed:asset:L2"]
    assert staff["state"]["pending"]["candidates"] == ["seed:asset:L1", "seed:asset:L2"]
    assert "seed:asset:P1" not in json.dumps(staff)
    assert "Panelboard" not in json.dumps(staff)
    # The same state under a persona cleared for it keeps all three.
    assert [c["asset_id"] for c in technician["candidates"]] == handed


# --- state a client sends back is never trusted -------------------------------------------


def test_an_id_the_persona_cannot_see_and_one_that_does_not_exist_are_dropped_alike() -> None:
    def handed(asset: str | None) -> dict[str, Any]:
        held = {"assets": [asset], "since": {"assets": 1}} if asset else {}
        return ask("visitor", "Where is that one?", THAT_ONE, state=state_for("visitor", **held))

    hidden, forged, nothing = handed(LIGHT_108), handed("seed:asset:nope"), handed(None)

    for turn in (hidden, forged, nothing):
        assert (turn["kind"], turn["cause"]) == ("abstention", "nothing-referred-to")
        assert turn["state"]["assets"] == []
        assert turn["state"]["since"] == {}
    assert hidden == forged == nothing
    # Staff, who can see room 108, do get it.
    seen = ask(
        "staff",
        "Where is that one?",
        THAT_ONE,
        state=state_for("staff", assets=[LIGHT_108], since={"assets": 1}),
    )
    assert seen["kind"] == "answer"
    assert seen["trace"]["references"][0]["ids"] == [LIGHT_108]


def test_a_tampered_place_is_dropped() -> None:
    state = state_for("visitor", place=SERVERRAUM, since={"place": 1})

    turn = ask(
        "visitor", "What is in there?", installed_in(pointing("place", "there")), state=state
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "nothing-referred-to")
    assert turn["state"]["place"] is None
    assert SERVERRAUM not in json.dumps(turn)


def test_a_tampered_candidate_is_dropped_before_the_ordinal_counts() -> None:
    pending = {"mention": "107", "candidates": [SERVERRAUM, "osm:way/nope", ROOM_107_A]}
    state = state_for("visitor", pending=pending)

    turn = ask(
        "visitor", "the first one", where_is(pointing("place", "the first one", 1)), state=state
    )

    assert turn["kind"] == "answer"
    assert turn["state"]["place"] == ROOM_107_A
    assert SERVERRAUM not in json.dumps(turn)


def test_a_constraint_or_route_naming_something_unseen_is_dropped_whole() -> None:
    def carried(kind: str, **ids: list[str]) -> dict[str, Any]:
        return {"kind": kind, "turn": 1, "args": {"before": "2027-01-01"}, **ids}

    kept = carried("service-due", space_ids=[AUDIMAX], asset_type_ids=[fixture.LIGHT_B1])
    state = state_for(
        "visitor",
        constraints=[
            kept,
            carried("service-due", space_ids=[AUDIMAX, SERVERRAUM]),
            carried("service-due", asset_ids=["seed:asset:P1"]),
            carried("service-due", asset_type_ids=[fixture.PANEL]),
        ],
        route=carried("route", space_ids=[AUDIMAX, SERVERRAUM]),
    )

    turn = ask("visitor", "Where is the Audimax?", where("Audimax"), state=state)

    assert turn["state"]["constraints"] == [{**kept, "asset_ids": []}]
    assert turn["state"]["route"] is None
    # A route the persona can see all of comes back as it went in.
    whole = carried("route", space_ids=[AUDIMAX, ROOM_107_A])
    turn = ask(
        "visitor", "Where is the Audimax?", where("Audimax"), state={**state, "route": whole}
    )
    assert turn["state"]["route"] == {**whole, "asset_ids": [], "asset_type_ids": []}


@pytest.mark.parametrize("persona", [p for p in PERSONAS if p is not Persona.MAINTENANCE])
def test_nothing_from_tampered_state_reaches_a_model(persona: Persona) -> None:
    seen_assets, seen_types = fixture.visible_assets()
    spaces = {row.space_id for row in fixture.visible() if row.persona is persona}
    hidden = [
        *(a.id for a in fixture.assets() if a.id not in seen_assets[persona]),
        *(a.name.value for a in fixture.assets() if a.id not in seen_assets[persona]),
        *(t.id for t in fixture.asset_types() if t.id not in seen_types[persona]),
        *(t.name.value for t in fixture.asset_types() if t.id not in seen_types[persona]),
        *(s.id for s in fixture.spaces() if s.id not in spaces),
        *(s.label.value for s in fixture.spaces() if s.id not in spaces and s.label),
    ]
    every_asset = [a.id for a in fixture.assets()]
    every_space = [s.id for s in fixture.spaces()]
    forged = state_for(
        persona.value,
        asset_types=[t.id for t in fixture.asset_types()],
        assets=["seed:asset:P1", "seed:asset:P2", LIGHT_108, "seed:asset:L4", EXTINGUISHER],
        since={"place": 1, "asset_types": 1, "assets": 1},
    )
    inputs: list[str] = []

    for state in (
        {**forged, "place": SERVERRAUM},
        {**forged, "place": "osm:way/3001"},
        {**forged, "pending": {"mention": "it", "kind": "asset", "candidates": every_asset}},
        {**forged, "pending": {"mention": "it", "kind": "place", "candidates": every_space}},
    ):
        for plan in (THAT_ONE, installed_in(named("Informatik"), pointing("asset-type", "it"))):
            turn = ask(persona.value, "And that one?", plan, worded, state=state, inputs=inputs)
            told = json.dumps([turn["state"], turn["candidates"], turn["segments"]])
            for secret in hidden:
                assert secret not in told

    assert len(inputs) > 15
    # What is left of the state does reach the planner, so the check above has teeth.
    assert any('"conversation"' in text and '"Fire extinguisher"' in text for text in inputs)
    for text in inputs:
        for secret in hidden:
            assert secret not in text


# --- plans --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("refers", "reason"),
    [
        ({"to": "candidate"}, "needs an ordinal"),
        ({"to": "candidate", "ordinal": 0}, "needs an ordinal"),
        ({"to": "earlier-turn", "ordinal": 2}, "only a candidate reference"),
        ({"to": "the-answer"}, "refers.to"),
    ],
)
def test_a_reference_that_does_not_say_where_it_points_is_a_rejected_plan(
    refers: dict[str, Any], reason: str
) -> None:
    plan = where_is({"text": "that one", "kind": "place", "refers": refers})

    turn = ask("visitor", "Where is that one?", plan)

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    assert turn["trace"]["plans"][0]["trigger"]["kind"] == "plan-invalid"
    assert any(reason in r for r in turn["trace"]["plans"][0]["rejection"])


@pytest.mark.parametrize("persona", ["visitor", "maintenance"])
def test_an_asset_named_in_words_is_a_malformed_argument_for_everyone(persona: str) -> None:
    turn = ask(persona, "Where is Panelboard P1 5?", where_is(named("Panelboard P1 5", "asset")))

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    assert turn["trace"]["plans"][0]["trigger"]["kind"] == "malformed-arguments"
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "recovery"]
