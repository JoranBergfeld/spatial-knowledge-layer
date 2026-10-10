"""What the service says about its own use: turns over HTTP, read back from an OTel reader."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from fastapi.testclient import TestClient
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

import fixture
from fixture import located, where
from spatial_service import metrics
from spatial_service.app import create_app
from spatial_service.metrics import Metrics
from spatial_service.turn import Turn

Labels = dict[str, str]


class Service:
    """The app with a meter that can be read back."""

    def __init__(self) -> None:
        self.reader = InMemoryMetricReader()
        self.metrics = Metrics(MeterProvider(metric_readers=[self.reader]))

    def ask(
        self, persona: str, question: str, planner: Any, composer: Any = located
    ) -> dict[str, Any]:
        llm = fixture.llm(
            planner if callable(planner) or planner is None else (lambda _p: planner),
            composer,
            [],
        )
        app = create_app(fixture.store(), llm, self.metrics)
        response = TestClient(app).post("/turn", json={"persona": persona, "question": question})
        turn: dict[str, Any] = response.json()
        return turn

    def series(self) -> dict[str, list[tuple[Labels, int]]]:
        """Every series emitted so far: its points as labels and a count."""
        data = self.reader.get_metrics_data()
        found: dict[str, list[tuple[Labels, int]]] = {}
        for resource in data.resource_metrics if data else []:
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    found[metric.name] = [
                        ({k: str(v) for k, v in (point.attributes or {}).items()}, point.value)
                        for point in metric.data.data_points
                    ]
        return found


def labels(persona: str, shape: str, kind: str) -> Labels:
    return {"persona": persona, "shape": shape, "turn_kind": kind, "snapshot": fixture.SNAPSHOT}


def broken(payload: dict[str, Any]) -> str:
    """A planner whose first plan does not parse and whose second finds the Audimax."""
    return where("Audimax") if "previous" in payload else "not a plan"


def test_a_turn_is_counted_by_kind_shape_and_persona() -> None:
    service = Service()

    service.ask("visitor", "Where is the Audimax?", where("Audimax"))
    service.ask("visitor", "Where is the Audimax?", where("Audimax"))
    service.ask("staff", "Where is 107?", where("107"))
    service.ask("visitor", "Where is the Bibliothek?", where("Bibliothek"))

    assert sorted(service.series()["spatial.turns"], key=str) == sorted(
        [
            (labels("visitor", "place-location", "answer"), 2),
            (labels("staff", "place-location", "clarification"), 1),
            (labels("visitor", "place-location", "abstention"), 1),
        ],
        key=str,
    )


def test_a_turn_no_plan_ran_for_has_an_unknown_shape() -> None:
    service = Service()

    service.ask("visitor", "Where is the Audimax?", None)
    service.ask("visitor", "Where is the Audimax?", "not a plan")

    assert sorted(service.series()["spatial.turns"], key=str) == sorted(
        [
            (labels("visitor", "unknown", "unavailable"), 1),
            (labels("visitor", "unknown", "abstention"), 1),
        ],
        key=str,
    )


def test_a_refused_request_is_not_a_turn() -> None:
    service = Service()
    app = create_app(fixture.store(), fixture.llm(None, None, []), service.metrics)

    response = TestClient(app).post(
        "/turn", json={"persona": "visitor", "question": "Where?", "snapshot": "1999-01-01"}
    )

    assert response.status_code == 409
    assert service.series() == {}


def test_a_turn_that_made_a_repair_call_is_counted_repaired_or_not() -> None:
    service = Service()

    service.ask("visitor", "Where is the Audimax?", where("Audimax"))
    assert "spatial.repair_calls" not in service.series()

    service.ask("visitor", "Where is the Audimax?", broken)
    service.ask("visitor", "Where is the Audimax?", "not a plan")

    assert sorted(service.series()["spatial.repair_calls"], key=str) == sorted(
        [
            (labels("visitor", "place-location", "answer"), 1),
            (labels("visitor", "unknown", "abstention"), 1),
        ],
        key=str,
    )
    # The rate is this over the turns, which count every turn, repaired or not.
    assert sum(count for _, count in service.series()["spatial.turns"]) == 3


def test_a_coverage_gap_is_a_turn_kind_of_its_own_and_is_counted() -> None:
    # Nothing ends a turn in a coverage gap yet, so this one is an abstention relabelled.
    service = Service()
    abstention = service.ask("visitor", "Where is the Bibliothek?", where("Bibliothek"))
    gap = Turn.model_validate({**abstention, "cause": "coverage-gap"})

    service.metrics.record(gap)

    series = service.series()
    assert series["spatial.coverage_gaps"] == [
        (labels("visitor", "place-location", "coverage-gap"), 1)
    ]
    assert (labels("visitor", "place-location", "coverage-gap"), 1) in series["spatial.turns"]
    assert (labels("visitor", "place-location", "abstention"), 1) in series["spatial.turns"]


def test_the_labels_are_shape_persona_turn_kind_and_snapshot_and_nothing_else() -> None:
    service = Service()
    service.ask("visitor", "Where is the Audimax?", where("Audimax"))
    service.ask("staff", "Where is 107?", where("107"))
    service.ask("visitor", "Where is the Audimax?", broken)
    service.ask("maintenance", "Where is the Audimax?", None)
    abstention = service.ask("visitor", "Where is the Bibliothek?", where("Bibliothek"))
    service.metrics.record(Turn.model_validate({**abstention, "cause": "coverage-gap"}))

    series = service.series()

    assert set(series) == {"spatial.turns", "spatial.repair_calls", "spatial.coverage_gaps"}
    for name, points in series.items():
        for point, _ in points:
            assert set(point) == {"shape", "persona", "turn_kind", "snapshot"}, name
    # No id a visitor, conversation, space or asset goes by is a label's value either.
    values = {
        value for points in series.values() for point, _ in points for value in point.values()
    }
    assert values == {
        "visitor",
        "staff",
        "maintenance",
        "place-location",
        "unknown",
        "answer",
        "clarification",
        "abstention",
        "coverage-gap",
        "unavailable",
        fixture.SNAPSHOT,
    }


# --- where they go ------------------------------------------------------------------------


def test_without_an_endpoint_nothing_is_set_up_and_a_turn_still_answers() -> None:
    quiet = metrics.from_env({})

    assert not quiet.exporting
    app = create_app(fixture.store(), fixture.llm(lambda _p: where("Audimax"), located, []), quiet)
    response = TestClient(app).post(
        "/turn", json={"persona": "visitor", "question": "Where is the Audimax?"}
    )
    assert response.json()["kind"] == "answer"
    quiet.close()


def test_with_an_endpoint_the_series_are_pushed_to_it() -> None:
    posts: list[tuple[str, bytes]] = []

    class Collector(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            posts.append((self.path, self.rfile.read(length)))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_: Any) -> None:
            pass

    server = HTTPServer(("127.0.0.1", 0), Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}"
        pushing = metrics.from_env({"OTEL_EXPORTER_OTLP_ENDPOINT": endpoint})
        assert pushing.exporting
        app = create_app(
            fixture.store(), fixture.llm(lambda _p: where("Audimax"), located, []), pushing
        )
        TestClient(app).post(
            "/turn", json={"persona": "visitor", "question": "Where is the Audimax?"}
        )
        # Closing flushes, which is what a stopping service does.
        pushing.close()
    finally:
        server.shutdown()

    [(path, body)] = posts
    assert path == "/v1/metrics"
    assert b"spatial.turns" in body
    assert b"spatial-service" in body
