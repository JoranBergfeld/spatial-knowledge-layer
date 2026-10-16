"""The readable-passage rule: OCR junk out, prose and tables in."""

from __future__ import annotations

import pytest

from spatial_ingest.rules.readable import BAD_SHARE, bad_share, readable

# The two passages #31 was opened over, as tesseract read two drawings of the test bench.
MECHANICAL_ROOM = (
    "YN HWP—2\n| 5-2\nOme J\nA ra\nC) )\nET-2/ . 80% | BOILER\nHWS _ Iw 100¢ B-1\n"
    "1508 WTR HT'R STACK\n5007 1009 208 CM\n- TUBE\n; CHEMICAL FEEDER x RevOVAL"
)
FLOOR_PLAN = (
    "l' ~-~~~~~~~I\n6,051mm\n56,950mm\nr-----'\nI I\nI I\nI I\nI I\n52,710mm\n40,859mm\n"
    "10,040mm\nFIRST FLOOR PLAN\nPART 1- Page 4-3"
)
# An upside-down page: tesseract reads it as letters, in no language.
UPSIDE_DOWN = (
    "Ayenuuy\n\nAjjenuury-ILWwasg VS\nApayend re) IUBYDSP\\ 48]Se)\\ W\n"
    "AIUJUO| NW olueyoey] UBWUABUINO¢ cr\nAIHOSNA M }WUN SUO SdIAIES olueyoey soluaiddy V\n"
    "Ayieq fal 0} posinba. au} JO} Jesoge’] pay|iysun O"
)

PROSE = (
    "The WELL-X-TROL WX Series vessels incorporate seamless bladder construction and "
    "contoured bladder design that ensures repeatable and predictable long-life expectancy. "
    "Note: Allow 18” (460mm) minimum clearance. All dimensions are approximate."
)
SCHEDULE = (
    "Location Room Name Fixture Description Quantity\n"
    "1AC1 Central Waiting R-2201 Dispenser Drinking Fountain 1\n"
    "1A11 Janitor Closet P-4700 Basin, Mop, Service 1\n"
    "1A09 Women's Toilet P-9050 Water Closet 1\n"
    "1B11 Staff Toilet P-3100 Lavatory, Clinic 1"
)
SPEC_SHEET = (
    "pH 700 Benchtop Meter. Measures pH, mV, and temperature (°C or °F).\n"
    "pHRange 0.00 to 14.00, TemperatureRange 0 to 100.0°C, ±5°C/±9°F.\n"
    "Dimensions 33-1/4 x 11-5/8 x 49-9/16 in., weight 207 lbs, $490.00.\n"
    "● Phone: 305-688-3501 ● E-mail: sales@panelfold.com ● Web Site: www.panelfold.com\n"
    "P.O. Box 680130, Miami. GFCIs and TradeMaster® receptacles, 4 ½” frame."
)
GERMAN = (
    "Der Hörsaal ist über den Aufzug im Gebäude 50.34 erreichbar. Zugang über 2 Stufen, "
    "Rampe 90 cm breit, Mikroport-Anlage auf Kanal 2."
)


@pytest.mark.parametrize("text", [MECHANICAL_ROOM, FLOOR_PLAN, UPSIDE_DOWN])
def test_ocr_junk_is_not_readable(text: str) -> None:
    assert bad_share(text) >= BAD_SHARE
    assert not readable(text)


@pytest.mark.parametrize("text", [PROSE, SCHEDULE, SPEC_SHEET])
def test_prose_and_tables_are_readable(text: str) -> None:
    assert bad_share(text) < 0.1
    assert readable(text)


def test_letter_pairs_are_only_held_against_english() -> None:
    # "zw", "tz" and "gk" are all over German.
    text = "Zwei Plätze zwischen den Sitzreihen, jetzt mit Möglichkeit zur Anmeldung."

    assert bad_share(text, english=False) == 0
    assert bad_share(text, english=True) > 0


def test_german_prose_is_readable() -> None:
    assert readable(GERMAN, english=False)


def test_a_passage_too_short_to_judge_is_kept() -> None:
    assert readable("| _ ~~ ¢ |")


def test_punctuation_and_bullets_count_neither_way() -> None:
    assert bad_share("• Switch off the breaker — then open the housing &") == 0
    assert bad_share("| Switch off the breaker _ then open the housing") > 0
