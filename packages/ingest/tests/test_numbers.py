"""The numbers-in-words rule: small counts, clock times and floors in words, measures as digits."""

from __future__ import annotations

import pytest

from spatial_ingest.rules.numbers import in_words


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("There are 2 steps at the entrance.", "There are two steps at the entrance."),
        ("3 accessible toilets, 0 lifts.", "three accessible toilets, zero lifts."),
        ("It seats 20.", "It seats twenty."),
        ("Use one of the 12 doors", "Use one of the twelve doors"),
    ],
)
def test_a_small_standalone_count_is_written_in_words(text: str, expected: str) -> None:
    assert in_words(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "It seats 21.",
        "The hall has 350 seats.",
        "Built in 1962.",
    ],
)
def test_a_count_that_is_not_small_stays_as_digits(text: str) -> None:
    assert in_words(text) == text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Open from 8:00 to 16:30.", "Open from eight o'clock to sixteen thirty."),
        ("The lift is locked after 18:05.", "The lift is locked after eighteen oh five."),
        ("Doors open at 07:45", "Doors open at seven forty-five"),
        ("Closed from 0:15", "Closed from zero fifteen"),
        # A range of two times, however the dash is typed. The dash stays as written.
        ("Staffed 8:00-16:00.", "Staffed eight o'clock-sixteen o'clock."),
        ("Staffed 8:00–16:30 daily", "Staffed eight o'clock–sixteen thirty daily"),
        ("Staffed 7:30 - 9:05", "Staffed seven thirty - nine oh five"),
    ],
)
def test_a_clock_time_is_written_in_words(text: str, expected: str) -> None:
    assert in_words(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The hall is on the 2nd floor.", "The hall is on the second floor."),
        ("On the 1st upper floor", "On the first upper floor"),
        ("in the 3rd basement level", "in the third basement level"),
        ("Take the lift to floor 2.", "Take the lift to floor two."),
        ("Entrance on level 0", "Entrance on level zero"),
        ("the 12th storey", "the twelfth storey"),
    ],
)
def test_a_floor_ordinal_is_written_in_words(text: str, expected: str) -> None:
    assert in_words(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Frequency 37.1 MHz",
        "The ramp is 90 cm wide and 6 m long.",
        "The ramp rises 6 %.",
        "The ramp rises 6%.",
        "A door 2 m wide",
        "It weighs 12 kg",
        "Wait 5 minutes",
        "Check it every 12 months",
        "Runs at 5 °C",
        "Draws 16 A at 230 V",
        "Tighten to 8 Nm",
        # A handover manual measures in pounds and feet.
        "Charge to 5 psi",
        "Keep 2 ft clear in front of the panel",
        "Keep 3 ft. clear",
        "It weighs 12 lb",
        "Holds 10 gal",
        "Moves 15 cfm",
    ],
)
def test_a_measured_value_with_a_unit_stays_as_digits(text: str) -> None:
    assert in_words(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Building 10.11",
        "Room 7 is next door",
        "Channel 2",
        "Lecture hall 9",
        "Kaiserstr. 12, 76131 Karlsruhe",
        "Engesserstraße 5",
        "Tel. 608-42812",
        "Last changed 31.05.2012",
        "Steps 1-3 are loose",
        "A ratio of 1/2",
        "Model B1 and type 4",
        "See page 3, figure 2 and section 4",
        "Tram line 2",
        "Version 1.2",
        "The 3rd edition",
        "A gradient of 1:12",
        "A gradient of 1:12-1:20",
        "Tel. 608-42812, 8:00-9",
    ],
)
def test_a_number_that_names_something_is_left_alone(text: str) -> None:
    assert in_words(text) == text


def test_every_kind_in_one_passage() -> None:
    text = (
        "Hall 9 is on the 2nd floor of building 30.22. There are 2 steps, each 15 cm high. "
        "The side door is open from 7:30. Channel 2, 37.1 MHz."
    )

    assert in_words(text) == (
        "Hall 9 is on the second floor of building 30.22. There are two steps, each 15 cm high. "
        "The side door is open from seven thirty. Channel 2, 37.1 MHz."
    )


def test_the_rule_is_stable_on_its_own_output() -> None:
    once = in_words("2 steps on the 2nd floor, open from 8:00, 90 cm wide")

    assert in_words(once) == once
