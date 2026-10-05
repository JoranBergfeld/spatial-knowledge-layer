import pytest

from spatial_ingest.rules.names import strip_person_names


@pytest.mark.parametrize(
    "name",
    [
        "Gaede-Hörsaal",
        "Tulla-Hörsaal",
        "Thomas-Jordan-Berner-Hörsaal",
        "Fritz-Haber-Weg",
        "Projektionsraum",
        "Seminarraum 2",
        "Labor",
        "AG",
        "Büro",
        "Labor I",
        "Labor II",
        "Diplomanden Labor",
        "Sekretariat des IAI",
        "Sekretariat ITM Bereich Kontinuumsmechanik",
        "Service-Büro des Studierendenwerks",
        "Rudolf Criegee Hörsaal",
        "Raum (alt)",
    ],
)
def test_rooms_named_after_someone_and_plain_names_are_untouched(name: str) -> None:
    result = strip_person_names(name)

    assert result.text == name
    assert result.changed is False


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Büro Prof. Dr. Erika Mustermann", "Büro"),
        ("Sekretariat Mustermann", "Sekretariat"),
        ("Labor Dipl.-Ing. Max Mustermann", "Labor"),
        ("AG Mustermann", "AG"),
        ("Sekretariat IPEK Prof. Mustermann", "Sekretariat IPEK"),
        ("Besprechungsraum PD. Dr. Mustermann", "Besprechungsraum"),
        ("Dr. Mustermann, Werkstatt", "Werkstatt"),
        ("Büro (Prof. Mustermann)", "Büro"),
        ("Prof. Dr. José García", None),
        ("Dr. Łukasz Kowalski", None),
        ("Dr. McDonald", None),
        ("Dr. MUSTERMANN", None),
        ("Büro Mustermann und Schmidt", "Büro"),
        ("Büro: Mustermann", "Büro"),
        ("Prof. Mustermann & Dr. Schmidt", None),
    ],
)
def test_a_person_in_a_room_name_is_removed_and_the_rest_kept(
    name: str, expected: str | None
) -> None:
    result = strip_person_names(name)

    assert result.text == expected
    assert result.changed is True


def test_a_string_of_titles_goes_with_the_name() -> None:
    result = strip_person_names("Prof. Dr.-Ing. habil. Georg Bretthauer")

    assert result.text is None
    assert result.changed is True
