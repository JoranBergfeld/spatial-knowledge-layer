"""Counts in words: what a count placeholder fills in as, with the test bench's names.

The first real run on the test bench came back with "Medical-Dental Clinic has 1 1."
and "Medical-Dental Clinic has 2 Medical-Dental Clinic.". Its type names are long and
odd, so these use them, straight into ground() with no venue behind it.
"""

from __future__ import annotations

import json

import pytest

from spatial_service.compose import (
    ANY_ASSETS,
    Citation,
    CompositionRejected,
    Fact,
    FactKind,
    ground,
)

CLINIC = Fact(
    "f1", FactKind.PLACE, "Medical-Dental Clinic", Citation(label="Medical-Dental Clinic")
)
PANELBOARD = Fact(
    "f2",
    FactKind.ASSET_TYPE,
    "Lighting and Appliance Panelboard",
    Citation(
        kind=FactKind.ASSET_TYPE,
        type_id="type:panelboard",
        label="Lighting and Appliance Panelboard",
    ),
)
PANELBOARDS = Fact(
    "f3", FactKind.COUNT, "28", Citation(kind=FactKind.COUNT, label="28", count=28), noun="f2"
)
EQUIP = Fact(
    "f4",
    FactKind.ASSET_TYPE,
    "Equip-A1155",
    Citation(kind=FactKind.ASSET_TYPE, type_id="type:equip-a1155", label="Equip-A1155"),
)
EQUIPS = Fact(
    "f5",
    FactKind.COUNT,
    "3 generated",
    Citation(kind=FactKind.COUNT, label="3 generated", count=3),
    noun="f4",
)
ONE = Fact(
    "f6", FactKind.COUNT, "1", Citation(kind=FactKind.COUNT, label="1", count=1), noun=ANY_ASSETS
)
MANY = Fact(
    "f7", FactKind.COUNT, "22", Citation(kind=FactKind.COUNT, label="22", count=22), noun=ANY_ASSETS
)
FACTS = [CLINIC, PANELBOARD, PANELBOARDS, EQUIP, EQUIPS, ONE, MANY]


def said(text: str, *cites: str) -> str:
    return json.dumps({"segments": [{"text": text, "cites": list(cites)}]})


@pytest.mark.parametrize(
    ("text", "filled"),
    [
        ("{f1} has {f3}.", "Medical-Dental Clinic has 28 Lighting and Appliance Panelboard."),
        ("{f1} has {f5}.", "Medical-Dental Clinic has 3 generated Equip-A1155."),
        # The type named already: the count stays a number.
        ("{f2}: {f3}.", "Lighting and Appliance Panelboard: 28."),
        ("{f1} has {f3} {f2}.", "Medical-Dental Clinic has 28 Lighting and Appliance Panelboard."),
        ("{f1} has {f6}.", "Medical-Dental Clinic has 1 asset."),
        ("{f1} has {f7}.", "Medical-Dental Clinic has 22 assets."),
        # The model writing the noun anyway is not said twice.
        ("{f1} has {f7} assets.", "Medical-Dental Clinic has 22 assets."),
    ],
)
def test_a_count_reads_as_a_number_and_what_it_counts(text: str, filled: str) -> None:
    [segment], _ = ground(said(text, "f1"), FACTS)

    assert segment.text == filled


def test_the_type_a_count_put_in_for_itself_is_cited() -> None:
    [segment], _ = ground(said("{f1} has {f5}.", "f1", "f5"), FACTS)

    assert [c.label for c in segment.citations] == [
        "Medical-Dental Clinic",
        "3 generated",
        "Equip-A1155",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "{f1} has {f6} {f7}.",  # "has 1 1"
        "{f1} has {f3} {f1}.",  # "has 2 Medical-Dental Clinic"
        "{f1} has {f3} {f4}.",  # a type, but not the one counted
    ],
)
def test_a_count_run_into_a_placeholder_that_is_not_its_type_rejects_the_answer(text: str) -> None:
    with pytest.raises(CompositionRejected) as rejected:
        ground(said(text, "f1"), FACTS)

    [reason] = rejected.value.reasons
    assert reason.startswith("segment 1 runs {f")
    assert reason.endswith("a count is followed by words")


def test_counts_with_words_between_them_are_fine() -> None:
    [segment], _ = ground(said("{f1} has {f5} and {f3}.", "f1"), FACTS)

    assert segment.text == (
        "Medical-Dental Clinic has 3 generated Equip-A1155"
        " and 28 Lighting and Appliance Panelboard."
    )
