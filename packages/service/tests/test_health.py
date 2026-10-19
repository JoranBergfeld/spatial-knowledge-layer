"""The probes: is the process up, and could it serve a turn right now."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import fixture
from spatial_core.records import Persona
from spatial_service.app import create_app
from spatial_service.metrics import Metrics
from spatial_service.store import PostgresStore, Store


class Untouchable:
    """A store that fails the test if anything reads it."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the store was read: {name}")


class Promotable:
    """A store whose current snapshot can be changed between requests, as a promote does."""

    def __init__(self) -> None:
        self.current: str | None = None

    def served(self) -> tuple[str | None, str | None]:
        return self.current, None

    def venue(self, snapshot: str, persona: Persona) -> Any:
        raise AssertionError("readiness does not read a venue")

    def sourced_clearance_share(self, snapshot: str) -> float | None:
        raise AssertionError("readiness does not read a manifest")


def probed(
    store: Store, inputs: list[str] | None = None, recorder: Metrics | None = None
) -> TestClient:
    # No reply scripted: a probe that called a model would fail, and leave an input behind.
    llm = fixture.llm(None, None, [] if inputs is None else inputs)
    return TestClient(create_app(store, llm, recorder))


def test_live_is_up_and_reads_nothing() -> None:
    inputs: list[str] = []

    response = probed(Untouchable(), inputs).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
    assert response.headers["content-type"] == "application/json"
    assert inputs == []


def test_live_is_up_with_the_database_out_of_reach() -> None:
    store = PostgresStore("postgresql://nobody@127.0.0.1:1/nothing?connect_timeout=2")

    response = probed(store).get("/health/live")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}


def test_ready_with_a_current_snapshot_is_up() -> None:
    inputs: list[str] = []

    response = probed(fixture.store(), inputs).get("/health/ready")

    assert response.status_code == 200
    assert response.json() == {"status": "UP"}
    assert response.headers["content-type"] == "application/json"
    assert inputs == []


def test_ready_with_no_snapshot_current_is_out_of_service() -> None:
    store = fixture.store()
    store.snapshot, store.previous = None, None  # type: ignore[assignment]

    response = probed(store).get("/health/ready")

    assert response.status_code == 503
    assert response.headers["content-type"] == "application/json"
    body = response.json()
    assert body["status"] == "OUT_OF_SERVICE"
    assert body["code"] == "no-current-snapshot"
    assert body["detail"]


def test_ready_with_the_database_out_of_reach_is_down() -> None:
    # Nothing listens on port 1, so the connection is refused at once.
    store = PostgresStore("postgresql://nobody@127.0.0.1:1/nothing?connect_timeout=2")

    response = probed(store).get("/health/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "DOWN"
    assert body["code"] == "store-unreachable"
    assert "nobody" not in body["detail"]


def test_a_promote_makes_ready_up_without_a_restart() -> None:
    store = Promotable()
    client = probed(store)

    assert client.get("/health/ready").status_code == 503
    store.current = fixture.SNAPSHOT
    assert client.get("/health/ready").json() == {"status": "UP"}


def test_no_probe_moves_a_metric() -> None:
    reader = InMemoryMetricReader()
    client = probed(fixture.store(), recorder=Metrics(MeterProvider(metric_readers=[reader])))

    client.get("/health/live")
    client.get("/health/ready")

    data = reader.get_metrics_data()
    points = [
        point
        for resource in (data.resource_metrics if data else [])
        for scope in resource.scope_metrics
        for metric in scope.metrics
        for point in metric.data.data_points
    ]
    assert points == []
