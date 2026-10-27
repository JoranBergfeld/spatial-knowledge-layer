"""Planned maintenance, end to end over HTTP, against the fixture snapshot."""

from __future__ import annotations

import json
from typing import Any

import pytest

import fixture
from fixture import counted, installed
from spatial_service import maintenance
from spatial_service.plan_check import PlanRejected, parse_plan
from spatial_service.recovery import TriggerKind
from test_follow_ups import pointing
from test_turn import ask

LIGHT_JOB = f"seed:task:{fixture.LIGHT_B1}:1"
PANEL_JOB = f"seed:task:{fixture.PANEL}:1"
EXTINGUISHER_JOB = f"seed:task:{fixture.EXTINGUISHER}:1"
LEVEL_0 = "floor:way/100:0"
LEVEL_1 = "floor:way/100:1"


def jobs_of(place: str | None = None, kind: str | None = None) -> str:
    """An asset-maintenance plan: the jobs on what is in ``place``, of type ``kind`` if given."""
    plan = json.loads(installed(place, kind))
    plan["shape"] = "asset-maintenance"
    steps = plan["steps"]
    steps.append(
        {
            "id": f"s{len(steps) + 1}",
            "tool": "maintenance_tasks",
            "assets": {"step": steps[-1]["id"]},
        }
    )
    return json.dumps(plan)


THAT_ONES_JOBS = json.dumps(
    {
        "shape": "asset-maintenance",
        "mentions": [{"id": "m1", **pointing("asset", "that one")}],
        "steps": [
            {"id": "s1", "tool": "resolve_asset", "mention": "m1"},
            {"id": "s2", "tool": "maintenance_tasks", "assets": {"step": "s1"}},
        ],
    }
)


def _named(refs: list[str]) -> str:
    return " and ".join(f"{{{ref}}}" for ref in refs)


def lead_in(payload: dict[str, Any]) -> str:
    """A composer for maintenance: the total, the jobs per type, the gaps, then nearby."""
    scope = [payload["scope"]] if payload.get("scope") else []
    opening = f"{{{scope[0]}}} holds" if scope else "The venue holds"
    segments = [
        {"text": f"{opening} {_named(payload['total'])}.", "cites": [*scope, *payload["total"]]}
    ]
    for row in payload["jobs"]:
        segments.append(
            {
                "text": f"{{{row['type']}}} has {{{row['jobs']}}}.",
                "cites": [row["type"], row["jobs"]],
            }
        )
    for kind in payload["no_jobs"]:
        segments.append(
            {"text": f"No planned maintenance is recorded for {{{kind}}}.", "cites": [kind]}
        )
    if "more_no_jobs" in payload:
        more = payload["more_no_jobs"]
        segments.append({"text": f"{{{more}}} more types have none.", "cites": [more]})
    nearby = payload.get("nearby")
    if nearby:
        for row in nearby["rows"]:
            segments.append(
                {
                    "text": f"Also in {{{nearby['building']}}}: {_named(row['counts'])}.",
                    "cites": [nearby["building"], *row["counts"]],
                }
            )
    return json.dumps({"segments": segments})


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


def task_ids(turn: dict[str, Any]) -> list[str]:
    return [row["task_id"] for row in turn["maintenance"]["tasks"]]


# --- the jobs, and the same kit elsewhere in the building ---------------------------------


def test_the_jobs_on_the_types_asked_about_go_out_as_rows_with_a_lead_in() -> None:
    turn = ask(
        "staff",
        "What maintenance do the lights in 50.34 room 107 need?",
        jobs_of("50.34 room 107", "lights"),
        lead_in,
    )

    assert turn["kind"] == "answer"
    assert turn["trace"]["shape"] == "asset-maintenance"
    assert texts(turn)[:3] == [
        "50.34 room 107 holds 2 generated assets.",
        "Light Fixture B1 has 1 planned job.",
        "No planned maintenance is recorded for Light Fixture Q4.",
    ]
    [row] = turn["maintenance"]["tasks"]
    assert row == {
        "task_id": LIGHT_JOB,
        "type_id": fixture.LIGHT_B1,
        "type_name": "Light Fixture B1",
        "name": "Check light fixture",
        "category": "PM",
        "frequency": "12",
        "frequency_unit": "month",
        "duration": "0.1",
        "duration_unit": "hour",
        "description": "Check light fixture, and record it.",
        "resource_names": "Facility services",
        "task_number": "1",
        "record_provenance": "seed",
        "provenance": "inferred",
        "rule": None,
        "basis": fixture.BASIS,
    }
    assert turn["maintenance"]["tasks_left_out"] == 0
    traced = turn["trace"]["maintenance"]
    assert traced["types"] == [fixture.LIGHT_B1, fixture.LIGHT_Q4]
    assert traced["without_jobs"] == [fixture.LIGHT_Q4]
    assert traced["tasks"] == [LIGHT_JOB]


def test_the_same_kit_elsewhere_in_the_building_is_counted_per_floor() -> None:
    turn = ask(
        "staff",
        "What maintenance do the lights in 50.34 room 107 need?",
        jobs_of("50.34 room 107", "lights"),
        lead_in,
    )

    # Only the type with a job, and only what staff can see: the Audimax's three and
    # room 108's one, not the Serverraum's. Light Fixture Q4 has no job, so the
    # Treppenhaus light isn't counted.
    assert texts(turn)[3:] == ["Also in Informatik: 4 generated Light Fixture B1."]
    nearby = turn["maintenance"]["nearby"]
    assert (nearby["building"], nearby["label"], nearby["tier"]) == (
        "osm:way/100",
        "Informatik",
        "floor",
    )
    assert [(r["space_id"], r["label"], r["type_id"], r["generated"]) for r in nearby["rows"]] == [
        (LEVEL_0, "50.34 level 0", fixture.LIGHT_B1, 3),
        (LEVEL_1, "50.34 level 1", fixture.LIGHT_B1, 1),
    ]
    assert turn["trace"]["maintenance"]["nearby_types"] == [fixture.LIGHT_B1]


def test_a_scope_that_is_a_building_has_nothing_nearby_to_add() -> None:
    turn = ask(
        "staff",
        "What maintenance do the lights in the Informatik building need?",
        jobs_of("Informatik", "lights"),
        lead_in,
    )

    assert turn["kind"] == "answer"
    assert turn["maintenance"]["nearby"] is None
    assert turn["trace"]["maintenance"]["nearby_building"] is None
    assert not any(text.startswith("Also in") for text in texts(turn))


def test_a_place_with_no_types_named_gets_the_jobs_on_whatever_it_holds() -> None:
    turn = ask(
        "maintenance",
        "What planned maintenance is there in the Audimax?",
        jobs_of("Audimax"),
        lead_in,
    )

    assert task_ids(turn) == [EXTINGUISHER_JOB, LIGHT_JOB, PANEL_JOB]
    assert texts(turn)[1:4] == [
        "Fire extinguisher has 1 planned job.",
        "Light Fixture B1 has 1 planned job.",
        "Panelboard P1 has 1 planned job.",
    ]
    # The rest of 50.34's lights and panelboards. Its one extinguisher is the one asked about.
    rows = turn["maintenance"]["nearby"]["rows"]
    assert [(r["space_id"], r["type_id"], r["generated"]) for r in rows] == [
        (LEVEL_0, fixture.LIGHT_B1, 1),
        (LEVEL_0, fixture.PANEL, 2),
        (LEVEL_1, fixture.LIGHT_B1, 3),
    ]


def test_types_without_a_job_are_counted_not_named_when_the_question_named_none() -> None:
    turn = ask(
        "staff",
        "What planned maintenance is there on 50.34 level 0?",
        jobs_of("50.34 level 0"),
        lead_in,
    )

    assert turn["kind"] == "answer"
    assert texts(turn) == [
        "50.34 level 0 holds 4 generated assets and 1 sourced asset.",
        "Fire extinguisher has 1 planned job.",
        "Light Fixture B1 has 1 planned job.",
        "1 more types have none.",
        "Also in Informatik: 3 generated Light Fixture B1.",
    ]
    assert turn["trace"]["maintenance"]["without_jobs"] == [fixture.LIGHT_Q4]


def test_a_job_on_a_type_the_persona_cannot_see_is_not_there_for_them() -> None:
    turn = ask("staff", "What planned maintenance is there in the Audimax?", jobs_of("Audimax"))

    assert task_ids(turn) == [EXTINGUISHER_JOB, LIGHT_JOB]
    assert {r["type_id"] for r in turn["maintenance"]["nearby"]["rows"]} == {fixture.LIGHT_B1}


def test_more_job_rows_than_the_cap_are_cut_and_the_trace_keeps_every_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(maintenance, "TASK_CAP", 1)
    turn = ask(
        "maintenance",
        "What planned maintenance is there in the Audimax?",
        jobs_of("Audimax"),
        lead_in,
    )

    assert task_ids(turn) == [EXTINGUISHER_JOB]
    assert turn["maintenance"]["tasks_left_out"] == 2
    assert turn["trace"]["maintenance"]["tasks"] == [EXTINGUISHER_JOB, LIGHT_JOB, PANEL_JOB]


def test_that_ones_jobs_are_the_jobs_on_its_type() -> None:
    first = ask(
        "staff",
        "Where are the fire extinguishers in the Audimax?",
        installed("Audimax", "fire extinguishers"),
        counted,
    )

    second = ask(
        "staff",
        "What maintenance does that one need?",
        THAT_ONES_JOBS,
        lead_in,
        state=first["state"],
    )

    assert second["kind"] == "answer"
    assert texts(second) == [
        "Audimax holds 1 Fire extinguisher.",
        "Fire extinguisher has 1 planned job.",
    ]
    assert task_ids(second) == [EXTINGUISHER_JOB]
    assert second["maintenance"]["nearby"] is None


# --- who gets them, and when there are none -----------------------------------------------


def test_a_visitor_is_told_maintenance_plans_are_for_staff_and_nothing_runs() -> None:
    turn = ask(
        "visitor",
        "What maintenance do the lights in 50.34 room 107 need?",
        jobs_of("50.34 room 107", "lights"),
        lead_in,
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "staff-only")
    assert texts(turn) == ["Maintenance plans are only shown to staff."]
    assert turn["segments"][0]["citations"] == []
    assert turn["maintenance"] is None
    assert [call["purpose"] for call in turn["llm_calls"]] == ["planning"]
    assert turn["trace"]["plans"][0]["steps"] == []
    assert turn["trace"]["shape"] == "asset-maintenance"


def test_a_visitor_is_told_the_same_whether_or_not_the_type_exists() -> None:
    real = ask("visitor", "Maintenance on the lights?", jobs_of(None, "lights"))
    made_up = ask(
        "visitor", "Maintenance on the flux capacitors?", jobs_of(None, "flux capacitors")
    )

    assert (real["cause"], texts(real)) == (made_up["cause"], texts(made_up))


def test_types_with_no_job_on_record_are_a_coverage_gap_naming_them() -> None:
    turn = ask("staff", "What maintenance do the Q4 lights need?", jobs_of(None, "Q4"), None)

    assert (turn["kind"], turn["cause"], turn["gap"]) == (
        "abstention",
        "coverage-gap",
        "no-planned-maintenance",
    )
    assert texts(turn) == ["No planned maintenance is recorded for Light Fixture Q4."]
    [cited] = turn["segments"][0]["citations"]
    assert (cited["kind"], cited["type_id"]) == ("asset-type", fixture.LIGHT_Q4)
    # The host says it, with no composition call.
    assert [call["purpose"] for call in turn["llm_calls"]] == ["planning"]


# --- the plan -----------------------------------------------------------------------------


def test_an_asset_maintenance_plan_with_a_filter_is_rejected() -> None:
    plan = json.loads(jobs_of("Audimax", "lights"))
    scope = next(step for step in plan["steps"] if step["tool"] == "assets_in_scope")
    scope["filters"] = [{"attribute": "serial_number", "test": "recorded", "date": None}]

    with pytest.raises(PlanRejected) as rejected:
        parse_plan(json.dumps(plan))

    assert rejected.value.trigger is TriggerKind.PLAN_INVALID
    assert "an asset-maintenance plan has no filters: jobs are planned per type" in (
        rejected.value.reasons
    )


def test_maintenance_tasks_reads_assets_and_nothing_else() -> None:
    plan = {
        "shape": "asset-maintenance",
        "mentions": [{"id": "m1", "text": "Audimax", "kind": "place"}],
        "steps": [
            {"id": "s1", "tool": "resolve_place", "mention": "m1"},
            {"id": "s2", "tool": "maintenance_tasks", "assets": {"step": "s1"}},
        ],
    }

    with pytest.raises(PlanRejected) as rejected:
        parse_plan(json.dumps(plan))

    assert rejected.value.trigger is TriggerKind.UNBOUND_REFERENCE
    assert rejected.value.reasons == [
        "step s2 reads s1, which produces places, not assets or asset"
    ]
