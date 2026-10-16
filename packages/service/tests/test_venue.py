"""The venue's facts over HTTP: which snapshot is served, and what an answer is modelled on."""

from __future__ import annotations

from fastapi.testclient import TestClient

import fixture
from spatial_service.app import create_app
from spatial_service.store import PostgresStore, Store


def served(store: Store, inputs: list[str] | None = None) -> TestClient:
    return TestClient(create_app(store, fixture.llm(None, None, [] if inputs is None else inputs)))


def test_the_venue_names_the_snapshot_served_and_what_routes_are_modelled_on() -> None:
    store = fixture.store()
    store.sourced_share = 0.12

    response = served(store).get("/venue")

    assert response.status_code == 200
    assert response.json() == {
        "snapshot": "2026-09-15",
        "previous_snapshot": "2026-09-08",
        "sourced_clearance_share": 0.12,
        "personas": ["visitor", "staff", "maintenance"],
        "route_constants": {
            "walking_speed_m_per_s": 1.34,
            "stairs_s_per_level": 12,
            "lift_wait_s": 40,
            "lift_s_per_level": 4,
            "storey_height_m": 4.0,
        },
    }


def test_with_no_snapshot_current_the_venue_is_unavailable() -> None:
    store = fixture.store()
    store.snapshot, store.previous = None, None  # type: ignore[assignment]

    response = served(store).get("/venue")

    assert response.status_code == 503
    assert response.json()["code"] == "no-current-snapshot"
    assert response.json()["detail"]


def test_with_the_database_out_of_reach_the_venue_is_unavailable() -> None:
    # Nothing listens on port 1, so the connection is refused at once.
    store = PostgresStore("postgresql://nobody@127.0.0.1:1/nothing?connect_timeout=2")

    response = served(store).get("/venue")

    assert response.status_code == 503
    assert response.json()["code"] == "store-unreachable"
    assert "nobody" not in response.json()["detail"]
