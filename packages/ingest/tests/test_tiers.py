import pytest

from spatial_ingest.rules.tiers import tier_for


@pytest.mark.parametrize(
    ("tags", "tier"),
    [
        ({"building": "university"}, "building"),
        ({"building": "yes", "ref": "50.34"}, "building"),
        ({"indoor": "room", "ref": "107"}, "room"),
        ({"indoor": "corridor"}, "room"),
        ({"indoor": "area"}, "room"),
    ],
)
def test_tags_give_a_space_its_tier(tags: dict[str, str], tier: str) -> None:
    assert tier_for(tags) == tier


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"building": "no"},
        {"indoor": "wall"},
        {"indoor": "level", "level": "1"},
        {"indoor": "door"},
        {"indoor": "yes"},
        {"highway": "footway", "level": "0"},
        {"building:part": "yes"},
    ],
)
def test_anything_else_is_not_a_space(tags: dict[str, str]) -> None:
    assert tier_for(tags) is None
