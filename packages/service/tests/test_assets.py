"""Asset containment and aggregation, end to end over HTTP, against the fixture snapshot."""

from __future__ import annotations

import json
from typing import Any

import fixture
from fixture import counted, installed
from test_turn import ask

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
