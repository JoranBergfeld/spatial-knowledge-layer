"""The places listing over HTTP: what a persona can see, to have something to ask about."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fixture
from spatial_service import listing
from spatial_service.app import create_app
from spatial_service.listing import BuildingPlaces, Places
from spatial_service.store import PostgresStore, Store

INFORMATIK = "osm:way/100"
MENSA = "osm:way/300"


def served(store: Store | None = None) -> TestClient:
    # The model is scripted with nothing: a listing that called it would fail.
    return TestClient(create_app(store or fixture.store(), fixture.llm(None, None, [])))


def get(path: str, persona: str, snapshot: str = fixture.SNAPSHOT, status: int = 200) -> Any:
    response = served().get(path, params={"persona": persona, "snapshot": snapshot})
    assert response.status_code == status, response.text
    return response.json()


def contract(name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((listing.CONTRACT_DIR / name).read_text())
    return found


def test_a_visitor_sees_the_buildings_they_can_reach_and_what_is_installed_there() -> None:
    body = get("/places", "visitor")

    assert body["snapshot"] == fixture.SNAPSHOT
    assert body["persona"] == "visitor"
    # The Mensa holds only a staff room, so for a visitor it isn't there.
    assert [b["id"] for b in body["buildings"]] == ["osm:way/200", INFORMATIK]
    informatik = body["buildings"][1]
    assert (informatik["name"], informatik["label"]) == ("Informatik", "Informatik")
    assert informatik["equipment"] == [
        {
            "type_id": fixture.EXTINGUISHER,
            "name": "Fire extinguisher",
            "count": 1,
            "provenance": "sourced",
        },
        # 3 in the Audimax and 2 in room 107. The Serverraum's and room 108's are not a visitor's.
        {"type_id": fixture.LIGHT_B1, "name": "Light Fixture B1", "count": 5, "provenance": "seed"},
        {"type_id": fixture.LIGHT_Q4, "name": "Light Fixture Q4", "count": 1, "provenance": "seed"},
    ]


def test_maintenance_sees_the_panelboards_nobody_else_does() -> None:
    def types(persona: str) -> set[str]:
        buildings = get("/places", persona)["buildings"]
        return {kind["type_id"] for b in buildings for kind in b["equipment"]}

    assert fixture.PANEL not in types("visitor")
    assert fixture.PANEL not in types("staff")
    assert fixture.PANEL in types("maintenance")


def test_a_building_lists_its_floors_rooms_and_which_rooms_hold_what() -> None:
    body = get(f"/places/{INFORMATIK}", "visitor")

    assert (body["id"], body["label"]) == (INFORMATIK, "Informatik")
    assert body["floors"] == [
        {
            "id": "floor:way/100:0",
            "label": "50.34 level 0",
            "rooms": [
                {"id": "osm:way/1002", "label": "Audimax", "ref": None},
                {"id": "osm:way/1010", "label": "Treppenhaus", "ref": None},
            ],
        },
        {
            "id": "floor:way/100:1",
            "label": "50.34 level 1",
            "rooms": [{"id": "osm:way/1003", "label": "50.34 room 107", "ref": "107"}],
        },
    ]
    assert body["floorless_rooms"] == []
    holding = {kind["type_id"]: kind["rooms"] for kind in body["equipment"]}
    assert holding == {
        fixture.EXTINGUISHER: ["osm:way/1002"],
        fixture.LIGHT_B1: ["osm:way/1002", "osm:way/1003"],
        fixture.LIGHT_Q4: ["osm:way/1010"],
    }


def test_staff_see_more_rooms_in_the_same_building() -> None:
    rooms = {r["id"] for f in get(f"/places/{INFORMATIK}", "staff")["floors"] for r in f["rooms"]}

    assert "osm:way/1004" in rooms
    assert "osm:way/1009" not in rooms


def test_rooms_on_no_floor_are_listed_apart() -> None:
    body = get(f"/places/{MENSA}", "staff")

    assert body["floors"] == []
    assert body["floorless_rooms"] == [
        {"id": "osm:way/3001", "label": "10.50 room 107", "ref": "107"}
    ]
    assert body["equipment"] == []


@pytest.mark.parametrize(
    "building",
    [
        MENSA,  # beyond a visitor's clearance
        "osm:way/1002",  # a room, not a building
        "osm:way/999",  # not there at all
    ],
)
def test_a_building_a_persona_cannot_see_does_not_exist(building: str) -> None:
    body = get(f"/places/{building}", "visitor", status=404)

    assert body == {"detail": f"no building {building}", "code": "no-such-building"}


@pytest.mark.parametrize("path", ["/places", f"/places/{INFORMATIK}"])
def test_a_snapshot_not_served_answers_as_a_turn_would(path: str) -> None:
    body = get(path, "visitor", snapshot="2020-01-01", status=409)

    assert body["code"] == "snapshot-dropped"


@pytest.mark.parametrize("path", ["/places", f"/places/{INFORMATIK}"])
def test_the_previous_snapshot_is_still_served(path: str) -> None:
    body = get(path, "visitor", snapshot=fixture.PREVIOUS)

    assert body["snapshot"] == fixture.PREVIOUS


@pytest.mark.parametrize("params", [{"persona": "visitor"}, {"persona": "visitor", "snapshot": ""}])
def test_the_snapshot_is_always_named(params: dict[str, str]) -> None:
    assert served().get("/places", params=params).status_code == 422


def test_an_unknown_persona_is_rejected() -> None:
    response = served().get("/places", params={"persona": "admin", "snapshot": fixture.SNAPSHOT})

    assert response.status_code == 422


def test_with_the_database_out_of_reach_the_listing_is_unavailable() -> None:
    # Nothing listens on port 1, so the connection is refused at once.
    store = PostgresStore("postgresql://nobody@127.0.0.1:1/nothing?connect_timeout=2")

    response = served(store).get("/places", params={"persona": "visitor", "snapshot": "x"})

    assert response.status_code == 503
    assert response.json()["code"] == "store-unreachable"
    assert "nobody" not in response.json()["detail"]


def test_nothing_says_where_anything_is_or_names_an_asset() -> None:
    bodies = [get("/places", p) for p in ("visitor", "staff", "maintenance")]
    bodies += [get(f"/places/{INFORMATIK}", p) for p in ("visitor", "staff", "maintenance")]
    text = json.dumps(bodies)

    for word in ("outline", "position", "lon", "lat", "coordinates"):
        assert f'"{word}"' not in text
    for asset in fixture.assets():
        assert asset.id not in text


@pytest.mark.parametrize(
    ("name", "model", "path", "persona"),
    [
        ("places.json", Places, "/places", "visitor"),
        ("building.json", BuildingPlaces, f"/places/{INFORMATIK}", "visitor"),
        ("building-floorless.json", BuildingPlaces, f"/places/{MENSA}", "staff"),
    ],
)
def test_the_contract_fixtures_are_what_the_service_answers(
    name: str, model: type[Places | BuildingPlaces], path: str, persona: str
) -> None:
    body = contract(name)

    model.model_validate(body)
    assert get(path, persona) == body


def test_the_models_refuse_a_field_they_do_not_know() -> None:
    body = contract("places.json")
    body["buildings"][0]["outline"] = None

    with pytest.raises(ValueError):
        Places.model_validate(body)
