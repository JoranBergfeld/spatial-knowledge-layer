"""The probes: GET /health/live and GET /health/ready, split the way Spring Boot's are.

Live says the process answers HTTP and touches nothing to say it. Ready says whether a
turn could be served right now: the store answers and a snapshot is current. Its "not
ready" codes are /venue's, so there is one vocabulary for it. Neither calls a model or
moves a metric. A probe that spends is a bug.
"""

from __future__ import annotations

from typing import Any

from spatial_service.store import Store, StoreUnreachable

UP: dict[str, Any] = {"status": "UP"}


def ready(store: Store) -> tuple[int, dict[str, Any]]:
    """The status code and body. Read per request, so a promote flips it without a restart."""
    try:
        current, _ = store.served()
    except StoreUnreachable:
        return 503, {
            "status": "DOWN",
            "code": "store-unreachable",
            "detail": "the store cannot be reached",
        }
    if current is None:
        # Nothing is broken, there is just nothing to serve: Spring's REFUSING_TRAFFIC.
        return 503, {
            "status": "OUT_OF_SERVICE",
            "code": "no-current-snapshot",
            "detail": "no snapshot is current",
        }
    return 200, UP
