import pytest

from spatial_ingest.rules.levels import parse_levels


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        ("0", ("0",)),
        ("-1", ("-1",)),
        ("0.5", ("0.5",)),
        ("1;2", ("1", "2")),
        ("2;-1;0", ("-1", "0", "2")),
        ("0-2", ("0", "1", "2")),
        ("-1-1", ("-1", "0", "1")),
        ("-2--1", ("-2", "-1")),
        ("0-1.5", ("0", "1", "1.5")),
        (" 1 ; 1 ", ("1",)),
        ("01", ("1",)),
    ],
)
def test_level_tags_become_levels_lowest_first(level: str, expected: tuple[str, ...]) -> None:
    assert parse_levels(level) == expected


@pytest.mark.parametrize("level", [None, "", "EG", "roof"])
def test_a_missing_or_unreadable_level_gives_no_levels(level: str | None) -> None:
    assert parse_levels(level) == ()


def test_unreadable_pieces_are_skipped_and_the_rest_kept() -> None:
    assert parse_levels("1;EG;2") == ("1", "2")
