"""A route as the turn hands it back, and what the trace says about routing.

The route in a turn stands on its own: where it starts and ends, whole seconds and
metres, the constants they were modelled with, every space passed through, and
everything derived that it leaned on. Rounding further and wording are the page's.

The trace says how big the persona's graph was and what each route step did. When
there is no route it says so and gives no reason, whatever the cause: a restricted
corridor, a missing door, no lift and no default origin all look the same from
here, and are meant to.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from spatial_core import circulation
from spatial_core.records import Space
from spatial_service.executor import Execution
from spatial_service.places import Venue
from spatial_service.routes import (
    CONSTANTS,
    OPEN_SPACE_RULE,
    Constants,
    Found,
    Graph,
    Routed,
    Target,
)

SOURCED = "sourced"
DERIVED = "derived"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Endpoint(_Model):
    space_id: str
    label: str | None
    # True for an origin the question did not name: the persona's default origin.
    default: bool = False
    # The door a route from outside comes in by.
    door: str | None = None
    # The assets a destination was asked for by. The route goes to their space.
    asset_ids: list[str] = []
    # How far a point in open space was moved to reach this place.
    snapped_m: float | None = None


class Passage(_Model):
    """One space a route goes through, and what it does there."""

    space_id: str
    label: str | None
    # "walk", "stairs" or "lift".
    mode: str
    levels: float
    seconds: float
    metres: float
    # Why the route may pass through: "sourced" tags or a "derived" rule. None for
    # the spaces it starts and ends in.
    crossable: str | None
    rule: str | None
    # The door it leaves by, and the rule behind that way out. No door and a rule
    # is a doorless room snapped to circulation.
    leaves_by: str | None
    leaves_by_rule: str | None


class Derived(_Model):
    """Something a route used that no source says outright."""

    what: str
    space_id: str
    rule: str
    note: str


class RouteOut(_Model):
    # The plan step that found it.
    step: str
    origin: Endpoint
    destination: Endpoint
    # Modelled, never measured. Whole seconds and whole metres.
    seconds: int
    metres: int
    modelled: bool = True
    step_free: bool
    constants: Constants = CONSTANTS
    passages: list[Passage]
    derived: list[Derived]
    # Routes to the next nearest places of a set, when the step asked for some.
    alternatives: list[RouteOut] = []


class RouteStepTrace(_Model):
    step: str
    tool: str
    step_free: bool = False
    default_origin: bool
    # The places it could have ended at.
    targets: list[str]
    found: bool
    # What a nearest step measured by.
    basis: str | None = None


class RouteTrace(_Model):
    # The persona's graph: spaces with a way in or out, and the edges between them.
    subgraph_spaces: int
    subgraph_edges: int
    constants: Constants = CONSTANTS
    steps: list[RouteStepTrace]


def _label(space: Space) -> str | None:
    return space.label.value if space.label else None


def _one(venue: Venue, step: str, found: Found, target: Target | None, run: Routed) -> RouteOut:
    passages = []
    derived = []
    for leg in found.legs:
        space = venue.spaces[leg.space_id]
        crossable = None
        if leg.crossable is not None:
            crossable = SOURCED if leg.crossable == circulation.TAGS_RULE else DERIVED
        if crossable == DERIVED:
            derived.append(
                Derived(
                    what="crossable-space",
                    space_id=space.id,
                    rule=leg.crossable or "",
                    note="an untagged room with three or more doors, so it may still be an office",
                )
            )
        if leg.leaves_by is not None and leg.leaves_by.door is None:
            derived.append(
                Derived(
                    what="doorless-room-snap",
                    space_id=space.id,
                    rule=leg.leaves_by.rule,
                    note="no door is mapped here; a doorless room was joined to circulation",
                )
            )
        passages.append(
            Passage(
                space_id=space.id,
                label=_label(space),
                mode=leg.mode,
                levels=leg.levels,
                seconds=round(leg.seconds, 1),
                metres=round(leg.metres, 1),
                crossable=crossable,
                rule=leg.crossable,
                leaves_by=leg.leaves_by.door if leg.leaves_by else None,
                leaves_by_rule=leg.leaves_by.rule if leg.leaves_by else None,
            )
        )
    end = venue.spaces[found.destination]
    if target is not None and target.snapped_m is not None:
        derived.append(
            Derived(
                what="open-space-snap",
                space_id=end.id,
                rule=OPEN_SPACE_RULE,
                note="the destination is a point in open space, taken to mean the nearest place",
            )
        )
    start = venue.spaces[found.origin]
    return RouteOut(
        step=step,
        origin=Endpoint(
            space_id=start.id,
            label=_label(start),
            default=run.default_origin,
            door=found.entered_by.door if found.entered_by else None,
        ),
        destination=Endpoint(
            space_id=end.id,
            label=_label(end),
            asset_ids=list(target.assets) if target else [],
            snapped_m=round(target.snapped_m, 1)
            if target and target.snapped_m is not None
            else None,
        ),
        seconds=round(found.seconds),
        metres=round(found.metres),
        step_free=run.step_free,
        passages=passages,
        derived=derived,
    )


def describe(venue: Venue, step: str, run: Routed) -> RouteOut:
    """A route step's routes: the nearest, with the runners-up as its alternatives."""
    best, *rest = (
        _one(venue, step, found, run.target(found.destination), run) for found in run.found
    )
    return best.model_copy(update={"alternatives": rest})


def routes_of(venue: Venue, execution: Execution) -> list[RouteOut]:
    """Every route an execution found, in plan order."""
    found = []
    for result in execution.results:
        if result.routed is not None and result.routed.found:
            found.append(describe(venue, result.step, result.routed))
        elif result.near is not None and result.near.route is not None:
            run = Routed(result.near.origin, result.near.default_origin, False, ())
            found.append(_one(venue, result.step, result.near.route, None, run))
    return found


def trace(venue: Venue, execution: Execution) -> RouteTrace | None:
    steps = []
    for result in execution.results:
        if result.routed is not None:
            run = result.routed
            steps.append(
                RouteStepTrace(
                    step=result.step,
                    tool=result.tool,
                    step_free=run.step_free,
                    default_origin=run.default_origin,
                    targets=[target.space_id for target in run.targets],
                    found=bool(run.found),
                )
            )
        elif result.tool == "nearest":
            near = result.near
            steps.append(
                RouteStepTrace(
                    step=result.step,
                    tool=result.tool,
                    default_origin=near.default_origin if near else "to" not in result.args,
                    targets=[near.route.destination] if near and near.route else [],
                    found=near is not None,
                    basis=near.basis if near else None,
                )
            )
    if not steps:
        return None
    measure = Graph.of(venue).measure()
    return RouteTrace(
        subgraph_spaces=measure["spaces"], subgraph_edges=measure["edges"], steps=steps
    )
