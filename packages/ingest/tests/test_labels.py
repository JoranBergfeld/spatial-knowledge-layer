from spatial_ingest.rules.labels import building_label, floor_label, room_label


def test_a_sourced_name_is_the_label() -> None:
    label = room_label(name="Gaede-Hörsaal", ref="001", building_number="30.22", building_name=None)

    assert label is not None
    assert label.text == "Gaede-Hörsaal"
    assert label.from_name is True


def test_an_unnamed_room_is_its_building_number_plus_its_room_number() -> None:
    label = room_label(name=None, ref="107", building_number="50.34", building_name="Informatik")

    assert label is not None
    assert label.text == "50.34 room 107"
    assert label.from_name is False


def test_an_unnumbered_building_lends_its_name() -> None:
    label = room_label(
        name=None, ref="107", building_number=None, building_name="Kollegiengebäude am Zirkel"
    )

    assert label is not None
    assert label.text == "Kollegiengebäude am Zirkel room 107"


def test_the_ref_prints_verbatim_and_only_the_first_of_several() -> None:
    several = room_label(name=None, ref="102;202;302", building_number="50.34", building_name=None)
    odd = room_label(name=None, ref="12$", building_number="50.34", building_name=None)

    assert several is not None and several.text == "50.34 room 102"
    assert odd is not None and odd.text == "50.34 room 12$"


def test_a_bare_room_number_is_never_a_label() -> None:
    assert room_label(name=None, ref="107", building_number=None, building_name=None) is None


def test_a_room_with_neither_name_nor_number_cannot_be_addressed() -> None:
    assert room_label(name=None, ref=None, building_number="50.34", building_name=None) is None
    assert room_label(name=None, ref=" ", building_number="50.34", building_name=None) is None


def test_a_building_is_its_name_or_else_its_number() -> None:
    named = building_label(name="Audimax", number="30.95")
    numbered = building_label(name=None, number="30.95")

    assert named is not None and named.text == "Audimax" and named.from_name
    assert numbered is not None and numbered.text == "30.95" and not numbered.from_name
    assert building_label(name=None, number=None) is None


def test_a_floor_is_its_building_plus_its_level() -> None:
    label = floor_label(building_number="50.34", building_name="Informatik", level="-1")

    assert label is not None
    assert label.text == "50.34 level -1"
    assert floor_label(building_number=None, building_name=None, level="1") is None
