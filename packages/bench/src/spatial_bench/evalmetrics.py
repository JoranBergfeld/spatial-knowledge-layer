"""An evaluation run's confusion matrix, pushed as metrics.

One counter, `spatial.eval.turns`: a count of questions per cell of the matrix.
Every point is labelled with the run it came from:

    run_id                   the run
    snapshot                 the snapshot it measured
    planning_deployment      the deployments it measured
    composition_deployment
    question_set             template or hand-written
    expected_turn_kind       what gold expects
    turn_kind                what came back

Empty cells are not pushed. Same endpoint variable as the service's live metrics
(OTEL_EXPORTER_OTLP_ENDPOINT), and nothing happens when it is unset. These are a
run's results, not live use, so they are a series of their own and never add to
spatial.turns.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence

from opentelemetry.metrics import MeterProvider, NoOpMeterProvider

from spatial_bench.score import Scored
from spatial_service.evaluation import Run

SERVICE_NAME = "spatial-bench"


class EvalMetrics:
    def __init__(self, provider: MeterProvider | None = None) -> None:
        self.provider = provider or NoOpMeterProvider()
        self._turns = self.provider.get_meter("spatial_bench").create_counter(
            "spatial.eval.turns",
            unit="{question}",
            description="Questions of an evaluation run, by expected and produced turn kind.",
        )

    def push(self, run: Run, scored: Sequence[Scored]) -> None:
        cells = Counter((one.set, one.expected, one.produced) for one in scored)
        for (question_set, expected, produced), count in sorted(cells.items()):
            self._turns.add(
                count,
                {
                    "run_id": run.id,
                    "snapshot": run.snapshot,
                    "planning_deployment": run.deployments.planning,
                    "composition_deployment": run.deployments.composition,
                    "question_set": question_set,
                    "expected_turn_kind": expected,
                    "turn_kind": produced,
                },
            )

    def close(self) -> None:
        """Push what is left and stop. Safe when nothing was set up."""
        shutdown = getattr(self.provider, "shutdown", None)
        if shutdown is not None:
            shutdown()


def from_env(env: Mapping[str, str] = os.environ) -> EvalMetrics:
    endpoint = env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return EvalMetrics()
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    exporter = OTLPMetricExporter(endpoint=endpoint.rstrip("/") + "/v1/metrics")
    resource = Resource.create({"service.name": env.get("OTEL_SERVICE_NAME", SERVICE_NAME)})
    return EvalMetrics(
        SdkMeterProvider(
            resource=resource, metric_readers=[PeriodicExportingMetricReader(exporter)]
        )
    )
