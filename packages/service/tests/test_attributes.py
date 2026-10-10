"""Attribute filters and service due, end to end over HTTP, against the fixture snapshot."""

from __future__ import annotations

import json
from typing import Any

import pytest

import fixture
from fixture import counted, filtered, having, installed, where
from spatial_core.records import PERSONAS, Persona
from spatial_service.plan import Plan
from test_turn import ask, repaired_by


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


def filter_trace(turn: dict[str, Any]) -> dict[str, Any]:
    [only] = turn["trace"]["assets"]["filters"]
    result: dict[str, Any] = only
    return result


# --- filtering by an attribute, and its fill rate -----------------------------------------


def test_assets_are_filtered_by_an_attribute_and_the_trace_carries_its_fill_rate() -> None:
    turn = ask(
        "visitor",
        "Which lights in the Audimax have a serial number?",
        having("serial_number", "recorded", place="Audimax", kind="lights"),
        filtered,
    )

    assert turn["kind"] == "answer"
    assert turn["trace"]["shape"] == "attribute-filter"
    assert texts(turn) == [
        "Audimax holds 1 generated.",
        "Audimax: 1 generated.",
        "Counted: serial number recorded.",
        "2 generated with no serial number recorded.",
    ]
    assert turn["trace"]["plans"][0]["steps"][-1]["ids"] == ["seed:asset:L1"]
    # One of the three lights a visitor sees there carries one.
    fill = filter_trace(turn)
    assert (fill["attribute"], fill["test"]) == ("serial_number", "recorded")
    assert (fill["carried"], fill["of"], fill["matched"], fill["not_recorded"]) == (1, 3, 1, 2)
    assert fill["fill_rate"] == 1 / 3


def test_a_fill_rate_covers_only_what_the_persona_can_see() -> None:
    plan = having("serial_number", "recorded", place="Audimax")
    visitor = filter_trace(ask("visitor", "Which have a serial number?", plan, filtered))
    technician = filter_trace(ask("maintenance", "Which have a serial number?", plan, filtered))

    # The panelboard carries one too. A visitor's rate is over four assets, not five.
    assert (visitor["carried"], visitor["of"], visitor["fill_rate"]) == (1, 4, 0.25)
    assert (technician["carried"], technician["of"], technician["fill_rate"]) == (2, 5, 0.4)


def test_assets_that_lack_an_attribute_are_counted_as_not_recorded() -> None:
    turn = ask(
        "visitor",
        "Which lights in the Audimax have no serial number?",
        having("serial_number", "not-recorded", place="Audimax", kind="lights"),
        filtered,
    )

    assert texts(turn) == [
        "Audimax holds 2 generated.",
        "Audimax: 2 generated.",
        "Counted: no serial number recorded.",
    ]
    assert turn["trace"]["plans"][0]["steps"][-1]["ids"] == ["seed:asset:L2", "seed:asset:L3"]
    condition = turn["segments"][2]["citations"][0]
    assert (condition["kind"], condition["attribute"]) == ("condition", "serial_number")


def test_the_host_words_a_missing_attribute_and_the_answer_has_to_carry_it() -> None:
    seen: list[dict[str, Any]] = []

    def silent(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return counted(payload)

    plan = having("serial_number", "recorded", place="Audimax", kind="lights")
    turn = ask("visitor", "Which lights have a serial number?", plan, silent)

    # An answer that drops the condition, or the assets with nothing recorded, is rejected.
    assert (turn["kind"], turn["cause"]) == ("abstention", "composition-rejected")
    [told] = seen[0]["filters"]
    for ref in (told["condition"], told["not_recorded"]["as"], *told["not_recorded"]["counts"]):
        assert (
            f"no segment names {ref}, which the answer has to"
            in (turn["trace"]["composition_rejection"])
        )
    # The composer hears the test and not the words. Those the host fills in.
    assert "serial" not in json.dumps({k: v for k, v in seen[0].items() if k != "question"})


def test_an_answer_that_turns_a_gap_into_a_fact_about_the_world_is_rejected() -> None:
    def overreaching(payload: dict[str, Any]) -> str:
        draft = json.loads(filtered(payload))
        draft["segments"][-1]["text"] += " They were never given one."
        return json.dumps(draft)

    plan = having("serial_number", "recorded", place="Audimax", kind="lights")
    turn = ask("visitor", "Which lights have a serial number?", plan, overreaching)

    assert (turn["kind"], turn["cause"]) == ("abstention", "composition-rejected")
    assert turn["trace"]["composition_rejection"] == ["segment 4 says 'never'"]


def test_none_of_a_type_in_scope_is_still_a_count_of_zero_with_no_rate_to_give() -> None:
    turn = ask(
        "visitor",
        "Which fire extinguishers in Seminarraum Nord have a serial number?",
        having("serial_number", "recorded", place="Seminarraum Nord", kind="fire extinguishers"),
        filtered,
    )

    assert turn["kind"] == "answer"
    assert texts(turn) == ["Seminarraum Nord holds 0.", "Counted: serial number recorded."]
    fill = filter_trace(turn)
    assert (fill["carried"], fill["of"], fill["fill_rate"]) == (0, 0, None)


# --- a fill rate of zero ------------------------------------------------------------------


@pytest.mark.parametrize("test", ["recorded", "not-recorded"])
def test_an_attribute_nothing_in_scope_carries_is_a_coverage_gap_naming_it(test: str) -> None:
    turn = ask(
        "visitor",
        "Which lights in 50.34 room 107 have a serial number?",
        having("serial_number", test, place="50.34 room 107", kind="lights"),
        None,
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "coverage-gap")
    assert turn["gap"] == "attribute-not-recorded"
    assert texts(turn) == [
        "No serial number is recorded for the assets asked about in 50.34 room 107."
    ]
    # The host says it, at any tier: this gap is about an attribute, not a place.
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    fill = filter_trace(turn)
    assert (fill["carried"], fill["of"], fill["fill_rate"]) == (0, 2, 0.0)
    assert turn["trace"]["shape"] == "attribute-filter"


def test_service_due_that_cannot_be_worked_out_anywhere_in_scope_is_a_coverage_gap() -> None:
    turn = ask(
        "maintenance",
        "Which fire extinguishers fall due before 2027?",
        having("service_due", "before", "2027-01-01", kind="fire extinguishers"),
        None,
    )

    assert (turn["cause"], turn["gap"]) == ("coverage-gap", "attribute-not-recorded")
    assert texts(turn) == [
        "No service event or installation date is recorded for the assets asked about. "
        "Service due is counted from one of those, so it cannot be worked out."
    ]


# --- service due --------------------------------------------------------------------------


def test_service_due_is_a_year_on_from_the_latest_service_event_or_the_install_date() -> None:
    turn = ask(
        "visitor",
        "Which lights in the Audimax fall due for service before 2026?",
        having("service_due", "before", "2026-01-01", place="Audimax", kind="lights"),
        filtered,
    )

    # L1 was serviced in May 2025, so it is not due until May 2026. L2 has no event
    # on record and counts from its installation in March 2024. L3 has neither.
    assert texts(turn) == [
        "Audimax holds 1 generated.",
        "Audimax: 1 generated.",
        "Counted: service due before 2026-01-01.",
        "1 generated with no service event or installation date recorded.",
    ]
    assert turn["trace"]["plans"][0]["steps"][-1]["ids"] == ["seed:asset:L2"]
    fill = filter_trace(turn)
    assert (fill["carried"], fill["of"], fill["matched"], fill["not_recorded"]) == (2, 3, 1, 1)
    assert fill["due"] == [
        {
            "asset_id": "seed:asset:L1",
            "due": "2026-05-20",
            "counted_from": "service-event",
            "counted_on": "2025-05-20",
            "provenance": "inferred",
            "rule": None,
            "basis": fixture.BASIS,
        },
        {
            "asset_id": "seed:asset:L2",
            "due": "2025-03-01",
            "counted_from": "installation-date",
            "counted_on": "2024-03-01",
            "provenance": "inferred",
            "rule": None,
            "basis": fixture.BASIS,
        },
    ]


def test_service_due_counts_from_the_latest_event_and_needs_no_install_date() -> None:
    turn = ask(
        "visitor",
        "Which lights fall due after June 2026?",
        having("service_due", "after", "2026-06-30", kind="lights"),
        filtered,
    )

    due = {d["asset_id"]: (d["due"], d["counted_from"]) for d in filter_trace(turn)["due"]}
    # L9 has two events and the later one counts. L7 has an event and no install date.
    assert due["seed:asset:L9"] == ("2026-10-10", "service-event")
    assert due["seed:asset:L7"] == ("2026-11-11", "service-event")
    assert turn["trace"]["plans"][0]["steps"][-1]["ids"] == [
        "seed:asset:L5",
        "seed:asset:L7",
        "seed:asset:L9",
    ]


def test_service_due_is_flagged_as_resting_on_a_default_interval_and_is_stored_nowhere() -> None:
    turn = ask(
        "visitor",
        "Which lights in the Audimax fall due for service before 2026?",
        having("service_due", "before", "2026-01-01", place="Audimax", kind="lights"),
        filtered,
    )

    fill = filter_trace(turn)
    assert (fill["derived"], fill["default_interval"], fill["interval_years"]) == (True, True, 1)
    assert fill["rule"] == "service-due@1"
    condition = turn["segments"][2]["citations"][0]
    assert (condition["provenance"], condition["rule"]) == ("derived", "service-due@1")
    assert "default interval of 1 year" in condition["basis"]
    assert "standard" not in json.dumps(turn)
    # No record carries it. It exists only in the turn that worked it out.
    assert not any("service_due" in asset.attributes for asset in fixture.assets())
    # A stored attribute is flagged as neither.
    stored = ask(
        "visitor",
        "Which have a serial number?",
        having("serial_number", "recorded", place="Audimax"),
        filtered,
    )
    assert filter_trace(stored)["default_interval"] is False
    assert filter_trace(stored)["due"] == []


def test_a_count_across_a_scale_level_can_be_filtered() -> None:
    turn = ask(
        "visitor",
        "Which building has the most lights with no service event?",
        having("service_event", "not-recorded", kind="lights", by="building"),
        filtered,
    )

    assert turn["trace"]["shape"] == "asset-aggregation"
    assert texts(turn) == [
        "Mini Campus holds 3 generated.",
        "Informatik: 3 generated.",
        "30.10: 0.",
        "Counted: no service event recorded.",
    ]
    fill = filter_trace(turn)
    assert (fill["carried"], fill["of"]) == (4, 7)


# --- the filter vocabulary ----------------------------------------------------------------


def planned(persona: str) -> dict[str, Any]:
    """What the planner is shown of the venue."""
    seen: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return having("serial_number", "recorded", place="Audimax")

    ask(persona, "Which have a serial number?", planner, filtered)
    venue: dict[str, Any] = seen[0]["venue"]
    return venue


def test_the_planner_is_shown_the_attributes_the_persona_can_filter_on() -> None:
    everyday = ["installation_date", "serial_number", "service_due", "service_event"]

    assert planned("visitor")["attributes"] == everyday
    assert planned("staff")["attributes"] == everyday
    # Only a panelboard in the Serverraum carries a breaker rating.
    assert planned("maintenance")["attributes"] == ["breaker_rating", *everyday]


def test_a_constraint_outside_the_catalogue_is_a_typed_rejection() -> None:
    good = having("serial_number", "recorded", place="Audimax")
    turn = ask(
        "visitor",
        "Which assets in the Audimax are red?",
        repaired_by(having("colour", "recorded", place="Audimax"), good),
        filtered,
    )

    assert (turn["kind"], turn["recovery"]) == ("answer", "repaired")
    first = turn["trace"]["plans"][0]
    assert first["trigger"] == {
        "kind": "malformed-arguments",
        "step": "s2",
        "reasons": [
            "'colour' is not an attribute to filter on here; the venue has "
            "installation_date, serial_number, service_due, service_event"
        ],
    }
    # Rejected before any step ran, so not even the place was looked up.
    assert [s["outcome"] for s in first["steps"]] == ["rejected"]


def test_an_attribute_beyond_clearance_is_rejected_like_one_that_does_not_exist() -> None:
    hidden = ask("visitor", "?", having("breaker_rating", "recorded", place="Audimax"), filtered)
    missing = ask("visitor", "?", having("flux_level", "recorded", place="Audimax"), filtered)

    for turn in (hidden, missing):
        assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
        assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "recovery"]
    assert hidden["segments"] == missing["segments"]
    assert hidden["state"] == missing["state"]
    reasons = [t["trace"]["plans"][0]["trigger"]["reasons"][0] for t in (hidden, missing)]
    assert reasons[0].replace("breaker_rating", "flux_level") == reasons[1]
    # Maintenance does see the panelboard that carries one.
    seen = ask("maintenance", "?", having("breaker_rating", "recorded"), filtered)
    assert seen["kind"] == "answer"
    assert filter_trace(seen)["carried"] == 1


def test_before_and_after_apply_only_to_dates() -> None:
    turn = ask("visitor", "?", having("serial_number", "before", "2026-01-01"), filtered)

    assert turn["cause"] == "plan-rejected"
    assert turn["trace"]["plans"][0]["trigger"]["reasons"] == [
        "'serial_number' is not a date, so before does not apply; "
        "the dates are installation_date, service_due, service_event"
    ]


def with_filter(**fields: Any) -> str:
    plan = json.loads(having("installation_date", "recorded", place="Audimax"))
    plan["steps"][-1]["filters"] = [fields]
    return json.dumps(plan)


def with_step_field(**fields: Any) -> str:
    plan = json.loads(having("installation_date", "recorded", place="Audimax"))
    plan["steps"][-1] |= fields
    return json.dumps(plan)


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        # A predicate where the attribute's name goes.
        (
            with_filter(attribute="installation_date < '2025-01-01'", test="recorded", date=None),
            "filters.0.attribute: String should match pattern",
        ),
        # A test that is not one of the four.
        (
            with_filter(attribute="installation_date", test="matches '2024%'", date=None),
            "filters.0.test: Input should be 'recorded', 'not-recorded', 'before' or 'after'",
        ),
        # A field for one, on the filter or on the step.
        (
            with_filter(attribute="installation_date", test="recorded", date=None, sql="1=1"),
            "filters.0.sql: Extra inputs are not permitted",
        ),
        (
            with_step_field(where="installation_date < '2025-01-01'"),
            "steps.1.where: Extra inputs are not permitted",
        ),
    ],
)
def test_a_free_predicate_is_rejected(plan: str, reason: str) -> None:
    turn = ask("visitor", "Which were installed before 2025?", plan, filtered)

    assert (turn["kind"], turn["cause"]) == ("abstention", "plan-rejected")
    first = turn["trace"]["plans"][0]
    assert (first["verdict"], first["trigger"]["kind"]) == ("rejected", "plan-invalid")
    assert any(reason in r for r in first["rejection"]), first["rejection"]
    assert first["steps"] == []


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            with_filter(attribute="installation_date", test="before", date=None),
            "step s2 filter 1: before needs a date",
        ),
        (
            with_filter(attribute="installation_date", test="recorded", date="2026-01-01"),
            "step s2 filter 1: recorded takes no date",
        ),
        (
            json.dumps({**json.loads(installed("Audimax")), "shape": "attribute-filter"}),
            "an attribute-filter plan filters by at least one attribute",
        ),
        (
            json.dumps(
                {
                    **json.loads(having("serial_number", "recorded", place="Audimax")),
                    "shape": "asset-containment",
                }
            ),
            "an asset-containment plan has no filters; with one it is attribute-filter",
        ),
    ],
)
def test_a_filter_plan_that_breaks_a_rule_is_rejected(plan: str, reason: str) -> None:
    turn = ask("visitor", "Which were installed before 2025?", plan, filtered)

    assert turn["cause"] == "plan-rejected"
    assert reason in turn["trace"]["plans"][0]["rejection"]


def test_the_plan_schema_still_has_no_one_of() -> None:
    assert "oneOf" not in json.dumps(Plan.model_json_schema())


# --- what the model is shown --------------------------------------------------------------


@pytest.mark.parametrize("persona", [p for p in PERSONAS if p is not Persona.MAINTENANCE])
def test_a_filtered_turn_shows_a_model_nothing_the_persona_cannot_see(persona: Persona) -> None:
    seen_assets, seen_types = fixture.visible_assets()
    hidden_assets = [a for a in fixture.assets() if a.id not in seen_assets[persona]]
    hidden = [
        *(a.id for a in hidden_assets),
        *(a.name.value for a in hidden_assets),
        *(t.id for t in fixture.asset_types() if t.id not in seen_types[persona]),
        *(e.id for e in fixture.service_events() if e.asset_id not in seen_assets[persona]),
        "63 A",
        "P1-0002",
    ]
    inputs: list[str] = []

    for plan in (
        having("serial_number", "recorded", place="Audimax"),
        having("service_due", "before", "2027-01-01"),
        having("service_event", "not-recorded", kind="lights", by="floor"),
        having("installation_date", "after", "2024-01-01", place="Informatik"),
    ):
        turn = ask(persona.value, "What is there?", plan, filtered, inputs=inputs)
        assert turn["kind"] == "answer"

    assert len(inputs) > 15
    for text in inputs:
        assert fixture.BREAKER not in text
        for secret in hidden:
            assert secret not in text


# --- filters in the conversation state ----------------------------------------------------

DUE = {"attribute": "service_due", "test": "before", "date": "2026-01-01"}


def heard(persona: str, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """What the planner hears of the conversation on the next turn, and that turn."""
    seen: list[dict[str, Any]] = []

    def planner(payload: dict[str, Any]) -> str:
        seen.append(payload)
        return where("Audimax")

    turn = ask(persona, "And where is that?", planner, state=state)
    conversation: dict[str, Any] = seen[0].get("conversation", {})
    return conversation, turn


def test_a_follow_up_turn_sees_the_filter_in_force() -> None:
    first = ask(
        "visitor",
        "Which lights in the Audimax fall due for service before 2026?",
        having("service_due", "before", "2026-01-01", place="Audimax", kind="lights"),
        filtered,
    )

    # Ids in the id lists, and only there. The rest is words and a date.
    assert first["state"]["constraints"] == [
        {
            "kind": "attribute-filter",
            "turn": 1,
            "space_ids": ["osm:way/1002"],
            "asset_ids": [],
            "asset_type_ids": [fixture.LIGHT_B1, fixture.LIGHT_Q4],
            "args": DUE,
        }
    ]
    conversation, second = heard("visitor", first["state"])
    assert conversation["filters"] == [DUE]
    # A turn that is not about assets leaves it in force.
    assert second["kind"] == "answer"
    assert second["state"]["constraints"] == first["state"]["constraints"]

    # The next asset question sets its own, and one with no filter has none.
    third = ask(
        "visitor", "What is in the Audimax?", installed("Audimax"), counted, state=second["state"]
    )
    assert third["state"]["constraints"] == []
    assert "filters" not in heard("visitor", third["state"])[0]


def test_a_coverage_gap_on_an_attribute_still_leaves_the_filter_in_force() -> None:
    turn = ask(
        "visitor",
        "Which lights in 50.34 room 107 have a serial number?",
        having("serial_number", "recorded", place="50.34 room 107", kind="lights"),
        None,
    )

    assert turn["cause"] == "coverage-gap"
    [carried] = turn["state"]["constraints"]
    assert carried["args"] == {"attribute": "serial_number", "test": "recorded", "date": None}


@pytest.mark.parametrize(
    "tampered",
    [
        # A type a visitor cannot see, in the id list.
        {"asset_type_ids": [fixture.PANEL], "args": DUE},
        # An attribute only something hidden carries, and one nothing does.
        {"args": {"attribute": fixture.BREAKER, "test": "recorded", "date": None}},
        {"args": {"attribute": "flux_level", "test": "recorded", "date": None}},
        # Not a filter at all.
        {"args": {"attribute": "serial_number", "test": "matches", "date": None}},
    ],
)
def test_a_tampered_filter_in_state_never_reaches_the_planner(tampered: dict[str, Any]) -> None:
    state = {
        "snapshot": fixture.SNAPSHOT,
        "persona": "visitor",
        "turns": 1,
        "constraints": [{"kind": "attribute-filter", "turn": 1, **tampered}],
    }

    conversation, turn = heard("visitor", state)

    assert "filters" not in conversation
    assert turn["kind"] == "answer"
