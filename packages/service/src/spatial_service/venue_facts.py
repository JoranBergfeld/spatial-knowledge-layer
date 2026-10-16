"""What a caller needs to know about the venue before it asks anything: GET /venue.

Which snapshot is served now and which before it, how much of the clearance was
sourced rather than defaulted, who a conversation can be held as, and what a
modelled route duration is made of. All of it is read from the store and from the
constants routes.py computes with, never copied, and none of it calls a model.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from spatial_core.records import PERSONAS, Persona
from spatial_service.routes import CONSTANTS
from spatial_service.store import Store, StoreUnreachable


class VenueUnavailable(Exception):
    """No facts to give. The code is what the caller acts on; the message is for reading."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code


class _Facts(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RouteConstants(_Facts):
    """routes.Constants under the names the venue contract gives them."""

    walking_speed_m_per_s: float
    stairs_s_per_level: float
    lift_wait_s: float
    lift_s_per_level: float
    storey_height_m: float


class VenueFacts(_Facts):
    snapshot: str
    previous_snapshot: str | None
    sourced_clearance_share: float | None
    personas: tuple[Persona, ...]
    route_constants: RouteConstants


ROUTE_CONSTANTS = RouteConstants(
    walking_speed_m_per_s=CONSTANTS.walking_m_per_s,
    stairs_s_per_level=CONSTANTS.stairs_s_per_level,
    lift_wait_s=CONSTANTS.lift_s,
    lift_s_per_level=CONSTANTS.lift_s_per_level,
    storey_height_m=CONSTANTS.storey_m,
)


def read(store: Store) -> VenueFacts:
    try:
        current, previous = store.served()
        share = None if current is None else store.sourced_clearance_share(current)
    except StoreUnreachable as error:
        raise VenueUnavailable("store-unreachable", "the store cannot be reached") from error
    if current is None:
        raise VenueUnavailable("no-current-snapshot", "no snapshot is current")
    return VenueFacts(
        snapshot=current,
        previous_snapshot=previous,
        sourced_clearance_share=share,
        personas=PERSONAS,
        route_constants=ROUTE_CONSTANTS,
    )
