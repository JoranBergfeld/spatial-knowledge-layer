"""Live metrics: how the service is being used, and nothing about who is using it.

Three counters, each a count of turns:

    spatial.turns            every turn
    spatial.repair_calls     turns that spent a recovery call, repaired or not.
                             Over spatial.turns it is the repair-call rate.
    spatial.coverage_gaps    turns that ended in a coverage-gap abstention

Every point carries the same four labels and no others:

    shape       the question shape of the plan that ran, "unknown" when none did
    persona     visitor, staff or maintenance
    turn_kind   answer, clarification, abstention, coverage-gap or unavailable
    snapshot    the snapshot the turn was answered from

A coverage gap is an abstention on the wire. Here it is a kind of its own, so it
does not hide among the ordinary abstentions.

There is no visitor, conversation, space or asset id in a label, and there must not
be: labels() is the one place a label is made, and it only reads those four things
off the turn. A request the service refuses (a snapshot no longer served) is not a
turn and is not counted.

They are pushed over OTLP/HTTP to OTEL_EXPORTER_OTLP_ENDPOINT. With that unset
nothing is set up and recording does nothing.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from opentelemetry.metrics import MeterProvider, NoOpMeterProvider

from spatial_service.recovery import RecoveryOutcome
from spatial_service.turn import AbstentionCause, Turn, TurnKind

LABELS = ("shape", "persona", "turn_kind", "snapshot")
UNKNOWN_SHAPE = "unknown"
COVERAGE_GAP = "coverage-gap"
SERVICE_NAME = "spatial-service"


def labels(turn: Turn) -> dict[str, str]:
    gap = turn.kind is TurnKind.ABSTENTION and turn.cause is AbstentionCause.COVERAGE_GAP
    shape = turn.trace.shape
    return {
        "shape": shape.value if shape is not None else UNKNOWN_SHAPE,
        "persona": turn.trace.persona.value,
        "turn_kind": COVERAGE_GAP if gap else turn.kind.value,
        "snapshot": turn.trace.snapshot,
    }


class Metrics:
    def __init__(self, provider: MeterProvider | None = None) -> None:
        self.provider = provider or NoOpMeterProvider()
        self.exporting = provider is not None
        meter = self.provider.get_meter("spatial_service")
        self._turns = meter.create_counter("spatial.turns", unit="{turn}", description="Turns.")
        self._repair_calls = meter.create_counter(
            "spatial.repair_calls",
            unit="{turn}",
            description="Turns that spent a recovery call.",
        )
        self._coverage_gaps = meter.create_counter(
            "spatial.coverage_gaps",
            unit="{turn}",
            description="Turns that ended in a coverage-gap abstention.",
        )

    def record(self, turn: Turn) -> None:
        attributes = labels(turn)
        self._turns.add(1, attributes)
        if turn.recovery is not RecoveryOutcome.NOT_NEEDED:
            self._repair_calls.add(1, attributes)
        if attributes["turn_kind"] == COVERAGE_GAP:
            self._coverage_gaps.add(1, attributes)

    def close(self) -> None:
        """Push what is left and stop. Safe when nothing was set up."""
        shutdown = getattr(self.provider, "shutdown", None)
        if shutdown is not None:
            shutdown()


def from_env(env: Mapping[str, str] = os.environ) -> Metrics:
    endpoint = env.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return Metrics()
    # Imported here so a service with no endpoint never loads the SDK or the exporter.
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.metrics import MeterProvider as SdkMeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource

    # The exporter reads its headers, timeout and the export interval from the
    # usual OTEL_ variables itself. The provider is ours and not the global one, so
    # nothing else in the process gets to emit through it with labels of its own.
    exporter = OTLPMetricExporter(endpoint=endpoint.rstrip("/") + "/v1/metrics")
    resource = Resource.create({"service.name": env.get("OTEL_SERVICE_NAME", SERVICE_NAME)})
    return Metrics(
        SdkMeterProvider(
            resource=resource, metric_readers=[PeriodicExportingMetricReader(exporter)]
        )
    )
