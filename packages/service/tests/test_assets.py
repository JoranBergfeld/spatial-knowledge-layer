"""Asset containment and aggregation, end to end over HTTP, against the fixture snapshot."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fixture
from fixture import counted, installed, most
from spatial_core.records import PERSONAS, Persona
from spatial_service.app import create_app
from spatial_service.plan import Plan
from test_turn import ask, repaired_by

AUDIMAX = "osm:way/1002"


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


# --- what is installed in a place ---------------------------------------------------------


def test_what_is_installed_in_a_room_is_listed_by_type() -> None:
    turn = ask("visitor", "What is installed in the Audimax?", installed("Audimax"), counted)

    assert turn["kind"] == "answer"
    assert texts(turn) == [
        "Audimax holds 3 generated and 1 sourced.",
        "Light Fixture B1: 3 generated.",
        "Fire extinguisher: 1.",
    ]
    assert turn["trace"]["shape"] == "asset-containment"
    assert turn["state"]["place"] == AUDIMAX


def test_the_same_question_under_two_personas_counts_differently() -> None:
    question = "What is installed in the Audimax?"
    visitor = ask("visitor", question, installed("Audimax"), counted)
    technician = ask("maintenance", question, installed("Audimax"), counted)

    assert texts(visitor)[0] == "Audimax holds 3 generated and 1 sourced."
    assert texts(technician) == [
        "Audimax holds 4 generated and 1 sourced.",
        "Light Fixture B1: 3 generated.",
        "Fire extinguisher: 1.",
        "Panelboard P1: 1 generated.",
    ]
    assert visitor["trace"]["assets"]["touched"] == 4
    assert technician["trace"]["assets"]["touched"] == 5
    # What a visitor is told carries nothing of the panelboard, and no sign one was left out.
    told = json.dumps(visitor)
    assert "Panelboard" not in told
    assert fixture.PANEL not in told
    assert "seed:asset:P1" not in told
    for word in ("hidden", "restricted", "clearance", "not shown", "omitted", "left out"):
        assert word not in " ".join(texts(visitor)).lower()


def test_where_assets_of_a_type_are_is_listed_by_space() -> None:
    turn = ask(
        "visitor",
        "Where are the lights on level 0 of 50.34?",
        installed("50.34 level 0", "lights"),
        counted,
    )

    assert texts(turn) == [
        "50.34 level 0 holds 4 generated.",
        "Audimax: 3 generated.",
        "Treppenhaus: 1 generated.",
    ]
    # "lights" names every type of light the persona can see, and nobody is asked which.
    assert turn["trace"]["resolved_types"] == [
        {
            "mention": "m2",
            "text": "lights",
            "step": "s2",
            "type_ids": [fixture.LIGHT_B1, fixture.LIGHT_Q4],
            "names": ["Light Fixture B1", "Light Fixture Q4"],
        }
    ]
    assert turn["state"]["asset_types"] == [fixture.LIGHT_B1, fixture.LIGHT_Q4]
    # One step covers every space under the scope: no call, and no step, per room.
    [version] = turn["trace"]["plans"]
    assert [s["tool"] for s in version["steps"]] == [
        "resolve_place",
        "resolve_asset_type",
        "assets_in_scope",
    ]
    assert version["steps"][-1]["ids"] == [
        "seed:asset:L1",
        "seed:asset:L2",
        "seed:asset:L3",
        "seed:asset:L5",
    ]
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "composition"]


def test_none_of_a_type_where_other_things_are_installed_is_a_count_of_zero() -> None:
    turn = ask(
        "visitor",
        "How many fire extinguishers are in Seminarraum Nord?",
        installed("Seminarraum Nord", "fire extinguishers"),
        counted,
    )

    assert turn["kind"] == "answer"
    assert texts(turn) == ["Seminarraum Nord holds 0."]


# --- no recorded installations ------------------------------------------------------------


@pytest.mark.parametrize("kind", [None, "lights"])
def test_a_scope_with_nothing_installed_is_a_coverage_gap_not_a_zero(kind: str | None) -> None:
    turn = ask("visitor", "What is in 30.10 room 107?", installed("30.10 room 107", kind), None)

    assert (turn["kind"], turn["cause"]) == ("coverage-gap", "no-recorded-installations")
    assert texts(turn) == ["No installations are recorded for 30.10 room 107."]
    assert [c["space_id"] for c in turn["segments"][0]["citations"]] == ["osm:way/2001"]
    # The host says it. No model is asked to word a gap, so none can dress it up as a count.
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    assert turn["trace"]["assets"]["installed_in_scope"] == 0
    assert turn["state"]["turns"] == 1


# --- counts across a scale level ----------------------------------------------------------


def test_which_building_has_the_most_differs_by_persona() -> None:
    question = "Which building has the most lights?"
    visitor = ask("visitor", question, most("building", "lights"), counted)
    technician = ask("maintenance", question, most("building", "lights"), counted)

    assert visitor["kind"] == "answer"
    assert texts(visitor) == [
        "Mini Campus holds 7 generated.",
        "Informatik: 6 generated.",
        "30.10: 1 generated.",
    ]
    assert texts(technician)[:3] == [
        "Mini Campus holds 9 generated.",
        "Informatik: 8 generated.",
        "30.10: 1 generated.",
    ]
    assert visitor["trace"]["assets"]["grouped_by"] == "building"
    assert [(r["label"], r["rank"]) for r in visitor["trace"]["assets"]["rows"]] == [
        ("Informatik", 1),
        ("30.10", 2),
    ]


def test_a_scale_level_in_the_question_sets_the_scope_and_the_grouping() -> None:
    seen: list[dict[str, Any]] = []

    def composer(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return counted(payload)

    turn = ask(
        "visitor",
        "Which floor of the Informatik building has the most installed?",
        most("floor", None, "Informatik"),
        composer,
    )

    # Mixed provenance: every count that mixes is two numbers, and there is none for the sum.
    assert texts(turn) == [
        "Informatik holds 6 generated and 1 sourced.",
        "50.34 level 0: 4 generated and 1 sourced.",
        "50.34 level 1: 2 generated.",
    ]
    assert turn["trace"]["assets"]["scope"] == "osm:way/100"
    assert turn["trace"]["assets"]["grouped_by"] == "floor"
    [leader] = seen[0]["leaders"]
    assert leader == seen[0]["rows"][0]["of"]
    # The composer is told references and kinds. No label, and no number to copy.
    told = json.dumps({k: v for k, v in seen[0].items() if k != "question"})
    assert "50.34" not in told
    assert "level" not in told
    assert not any(character.isdigit() for character in re.sub(r'"f[0-9]+"', "", told))
    # Scope is the building, so 30.10's floor is no part of it.
    assert "30.10" not in json.dumps(turn["segments"])


def test_a_space_spanning_floors_is_counted_once_on_the_floor_it_hangs_from() -> None:
    turn = ask("visitor", "Which floor has the most lights?", most("floor", "lights"), counted)

    rows = {r["label"]: r["generated"] for r in turn["trace"]["assets"]["rows"]}
    # The Treppenhaus covers levels 0 and 1 of 50.34 and hangs from level 0. Its one
    # light is in the 4 there and not in the 2 on level 1.
    assert rows == {"50.34 level 0": 4, "50.34 level 1": 2, "30.10 level 0": 1}
    assert sum(rows.values()) == turn["trace"]["assets"]["touched"] == 7
    assert turn["trace"]["assets"]["ungrouped"] == 0


def test_a_space_with_no_recorded_installations_is_said_so_and_never_ranked() -> None:
    question = "Which building has the most fire extinguishers?"
    plan = most("building", "fire extinguishers")
    staff = ask("staff", question, plan, counted)
    visitor = ask("visitor", question, plan, counted)

    # 30.10 has things installed and no extinguisher: a zero, ranked. The Mensa has
    # nothing recorded at all: a gap, said apart.
    assert texts(staff) == [
        "Mini Campus holds 1.",
        "Informatik: 1.",
        "30.10: 0.",
        "No recorded installations: Mensa.",
    ]
    rows = {r["label"]: (r["recorded"], r["rank"]) for r in staff["trace"]["assets"]["rows"]}
    assert rows == {"Informatik": (True, 1), "30.10": (True, 2), "Mensa": (False, None)}
    # A visitor does not see the Mensa at all, so for them it is not even a gap.
    assert "Mensa" not in json.dumps(visitor)
    assert texts(visitor) == ["Mini Campus holds 1.", "Informatik: 1.", "30.10: 0."]


def test_an_answer_that_leaves_out_a_gap_or_half_a_split_count_is_rejected() -> None:
    def no_gaps(payload: dict[str, Any]) -> str:
        return counted({**payload, "no_recorded_installations": []})

    turn = ask(
        "staff", "Which building has the most?", most("building", "fire extinguishers"), no_gaps
    )
    assert (turn["kind"], turn["cause"]) == ("abstention", "composition-rejected")
    assert "which the answer has to" in turn["trace"]["composition_rejection"][0]

    def half(payload: dict[str, Any]) -> str:
        return counted({**payload, "total": payload["total"][:1]})

    turn = ask("visitor", "What is in the Audimax?", installed("Audimax"), half)
    assert (turn["kind"], turn["cause"]) == ("abstention", "composition-rejected")
    assert any("one count with" in r for r in turn["trace"]["composition_rejection"])


# --- abstentions --------------------------------------------------------------------------


def test_a_type_beyond_clearance_and_one_that_does_not_exist_abstain_alike() -> None:
    hidden = ask("visitor", "Where are they?", installed("Audimax", "panelboards"), counted)
    missing = ask("visitor", "Where are they?", installed("Audimax", "flux capacitors"), counted)

    for turn in (hidden, missing):
        assert (turn["kind"], turn["cause"]) == ("abstention", "no-such-asset-type")
        assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    assert hidden["segments"] == missing["segments"]
    assert hidden["state"] == missing["state"]
    assert [s["outcome"] for s in hidden["trace"]["plans"][0]["steps"]] == [
        s["outcome"] for s in missing["trace"]["plans"][0]["steps"]
    ]
    # Maintenance does see them.
    seen = ask("maintenance", "Where are they?", installed("Audimax", "panelboards"), counted)
    assert texts(seen)[0] == "Audimax holds 1 generated."


def test_a_question_that_needs_one_result_to_choose_the_next_step_abstains() -> None:
    plan = json.dumps({"shape": "step-dependent", "mentions": [], "steps": []})

    turn = ask("visitor", "What is installed in the room with the most lights?", plan)

    assert (turn["kind"], turn["cause"]) == ("abstention", "step-dependent-question")
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    assert turn["trace"]["plans"][0]["steps"] == []


def test_a_plan_cannot_feed_a_count_back_into_a_lookup() -> None:
    # "Find the floor with the most lights, then look inside it", written as steps.
    plan = json.loads(most("floor", "lights"))
    plan["shape"] = "asset-containment"
    plan["steps"].append({"id": "s4", "tool": "assets_in_scope", "scope": {"step": "s3"}})

    turn = ask("visitor", "What is on the floor with the most lights?", json.dumps(plan))

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    first = turn["trace"]["plans"][0]
    assert first["trigger"]["kind"] == "unbound-reference"
    assert first["rejection"] == ["step s4 reads s3, which produces groups, not places"]
    assert first["steps"] == []


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"shape": "place-location"}, "a place-location plan ends in resolve_place"),
        (
            {"mentions": [{"id": "m1", "text": "lights", "kind": "place"}]},
            "step s1 is resolve_asset_type, but m1 is a mention of kind place",
        ),
    ],
)
def test_an_asset_plan_that_breaks_a_rule_is_rejected(change: dict[str, Any], reason: str) -> None:
    plan = {**json.loads(most("floor", "lights")), **change}

    turn = ask("visitor", "Which floor has the most lights?", json.dumps(plan))

    assert turn["cause"] == "plan-rejected"
    assert any(reason in r for r in turn["trace"]["plans"][0]["rejection"])


@pytest.mark.parametrize("persona", [p.value for p in PERSONAS])
def test_a_scale_level_the_venue_lacks_is_a_malformed_argument_for_everyone(persona: str) -> None:
    turn = ask(
        persona, "Which wing has the most?", repaired_by(most("wing"), most("floor")), counted
    )

    assert (turn["kind"], turn["recovery"]) == ("answer", "repaired")
    first = turn["trace"]["plans"][0]
    assert first["trigger"] == {
        "kind": "malformed-arguments",
        "step": "s2",
        "reasons": ["'wing' is not a scale level here; the venue has site, building, floor, room"],
    }


def test_the_plan_schema_has_no_one_of() -> None:
    # Strict structured output takes anyOf and refuses oneOf.
    assert "oneOf" not in json.dumps(Plan.model_json_schema())


# --- the collapsed tree -------------------------------------------------------------------


def planned_with(node_cap: int | None, persona: str = "visitor") -> tuple[Any, dict[str, Any]]:
    """What the planner was shown, and the turn's trace, at a node cap."""
    seen: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return installed("Audimax")

    llm = fixture.llm(planner, counted, [])
    if node_cap is None:
        app = create_app(fixture.store(), llm)
    else:
        app = create_app(fixture.store(), llm, tree_node_cap=node_cap)
    body = {"persona": persona, "question": "What is installed in the Audimax?"}
    turn = TestClient(app).post("/turn", json=body).json()
    return seen[0]["venue"], turn["trace"]


def test_the_planner_sees_the_whole_tree_when_it_fits() -> None:
    venue, trace = planned_with(None)

    assert trace["tree"] == {
        "depth": 3,
        "full_depth": 3,
        "nodes": 12,
        "full_nodes": 12,
        "node_cap": 200,
    }
    assert venue["tiers"] == ["site", "building", "floor", "room"]
    [site] = venue["tree"]
    assert (site["label"], site["tier"]) == ("Mini Campus", "site")
    assert [b["label"] for b in site["children"]] == ["30.10", "Informatik"]
    ground = site["children"][1]["children"][0]
    assert ground == {
        "label": "50.34 level 0",
        "tier": "floor",
        "children": [
            {"label": "Audimax", "tier": "room"},
            {"label": "Treppenhaus", "tier": "room"},
        ],
    }


def test_the_tree_is_cut_where_it_stops_fitting_and_the_depth_is_recorded() -> None:
    venue, trace = planned_with(8)

    # Site, two buildings and three floors fit under eight. Six rooms more do not.
    assert trace["tree"] == {
        "depth": 2,
        "full_depth": 3,
        "nodes": 6,
        "full_nodes": 12,
        "node_cap": 8,
    }
    [site] = venue["tree"]
    assert site["children"][1]["children"] == [
        {"label": "50.34 level 0", "tier": "floor", "below": {"room": 2}},
        {"label": "50.34 level 1", "tier": "floor", "below": {"room": 1}},
    ]
    # Code cuts it, so the same venue and cap give the same tree.
    assert planned_with(8)[0] == venue


def test_the_tree_holds_only_what_the_persona_sees() -> None:
    visitor, _ = planned_with(None, "visitor")
    technician, trace = planned_with(None, "maintenance")

    assert "Serverraum" not in json.dumps(visitor)
    assert "Mensa" not in json.dumps(visitor)
    assert "Serverraum" in json.dumps(technician)
    assert trace["tree"]["full_nodes"] == 16


# --- provenance in the trace --------------------------------------------------------------


def test_the_trace_shows_both_provenance_fields_for_every_asset_it_touches() -> None:
    turn = ask("maintenance", "What is installed in the Audimax?", installed("Audimax"), counted)
    assets = turn["trace"]["assets"]

    touched = turn["trace"]["plans"][0]["steps"][-1]["ids"]
    assert [a["asset_id"] for a in assets["listed"]] == touched
    by_id = {a["asset_id"]: a for a in assets["listed"]}
    assert by_id["osm:asset:node/9001"] == {
        "asset_id": "osm:asset:node/9001",
        "type_id": fixture.EXTINGUISHER,
        "space_id": AUDIMAX,
        "record_provenance": "sourced",
        "source": "node/9001",
        "name": "Fire extinguisher",
        "name_provenance": "derived",
        "name_rule": fixture.FIRE_RULE,
        "name_basis": None,
        "position": "recorded",
        "position_provenance": "sourced",
        "position_rule": None,
        "position_basis": None,
    }
    light = by_id["seed:asset:L1"]
    assert (light["record_provenance"], light["name_provenance"]) == ("seed", "inferred")
    assert (light["name_rule"], light["name_basis"]) == (None, fixture.BASIS)
    assert (light["position_provenance"], light["position_rule"]) == (
        "derived",
        "layout-in-outline@1",
    )
    for asset in assets["listed"]:
        assert asset["name_provenance"] == "sourced" or asset["name_rule"] or asset["name_basis"]
    # Every asset is accounted for by class too, which is what is left past the listing cap.
    assert sorted((c["record_provenance"], c["assets"]) for c in assets["classes"]) == [
        ("seed", 4),
        ("sourced", 1),
    ]
    assert sum(c["assets"] for c in assets["classes"]) == assets["touched"] == 5


def test_the_trace_shows_the_types_asked_for_and_the_rule_that_cleared_them() -> None:
    turn = ask(
        "maintenance", "Where are the panelboards?", installed("Informatik", "panelboards"), counted
    )

    assert turn["trace"]["asset_types"] == [
        {
            "type_id": fixture.PANEL,
            "record_provenance": "seed",
            "name": "Panelboard P1",
            "name_provenance": "inferred",
            "name_rule": None,
            "name_basis": fixture.BASIS,
            "omniclass": "23-80 30 20 11",
            "clearance": "maintenance",
            "clearance_rule": "clearance-from-omniclass@1",
            "authored_default": True,
            "unrecognised_category": False,
        }
    ]


def test_generated_assets_are_flagged_in_the_answer_and_its_citations() -> None:
    turn = ask("visitor", "What is installed in the Audimax?", installed("Audimax"), counted)

    total = turn["segments"][0]
    assert total["text"] == "Audimax holds 3 generated and 1 sourced."
    counts = [c for c in total["citations"] if c["kind"] == "count"]
    assert [(c["count"], c["record_provenance"], c["label"]) for c in counts] == [
        (3, "seed", "3 generated"),
        (1, "sourced", "1 sourced"),
    ]
    light = turn["segments"][1]["citations"][0]
    assert (light["kind"], light["type_id"], light["record_provenance"]) == (
        "asset-type",
        fixture.LIGHT_B1,
        "seed",
    )
    assert (light["provenance"], light["basis"]) == ("inferred", fixture.BASIS)


# --- placement ----------------------------------------------------------------------------


def test_an_asset_without_a_placement_reads_as_position_not_recorded() -> None:
    def composer(payload: dict[str, Any]) -> str:
        draft = json.loads(counted(payload))
        ref = payload["position_not_recorded"]
        draft["segments"].append({"text": f"Position not recorded: {{{ref}}}.", "cites": [ref]})
        return json.dumps(draft)

    turn = ask("visitor", "What is in 50.34 room 107?", installed("50.34 room 107"), composer)

    assert texts(turn)[-1] == "Position not recorded: 1."
    by_id = {a["asset_id"]: a for a in turn["trace"]["assets"]["listed"]}
    assert by_id["seed:asset:L7"]["position"] == "position not recorded"
    assert by_id["seed:asset:L7"]["position_provenance"] is None
    assert by_id["seed:asset:L6"]["position"] == "recorded"
    # A placement is never printed as coordinates, in the answer or the trace.
    assert "8.41" not in json.dumps(turn)
    assert "lon" not in turn["trace"]["assets"]["listed"][0]


# --- what the model is shown, and what comes back in state --------------------------------


@pytest.mark.parametrize("persona", [p for p in PERSONAS if p is not Persona.MAINTENANCE])
def test_nothing_the_persona_cannot_see_reaches_a_model(persona: Persona) -> None:
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
    assert hidden, "the fixture hides something from every persona but maintenance"
    inputs: list[str] = []
    forged = {
        "snapshot": fixture.SNAPSHOT,
        "persona": persona.value,
        "asset_types": [t.id for t in fixture.asset_types()],
    }

    for plan in (
        installed("Audimax"),
        installed("Informatik", "lights"),
        installed(None, None),
        most("building"),
        most("floor", "lights", "Informatik"),
        most("room", "fire extinguishers"),
    ):
        turn = ask(persona.value, "What is there?", plan, counted, inputs=inputs, state=forged)
        assert turn["kind"] == "answer"
        # State a client sent back is cut to what the persona can see.
        assert set(turn["state"]["asset_types"]) <= seen_types[persona]

    assert len(inputs) > 20
    for text in inputs:
        for secret in hidden:
            assert secret not in text
