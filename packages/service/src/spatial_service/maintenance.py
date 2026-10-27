"""Planned maintenance: the jobs on some assets' types, and the rest of them in the building.

A job (COBie's Job, a MaintenanceTask here) hangs off a type, not an asset. So the
jobs a question gets are the jobs on the types it is about: the types it asked for,
or, when it asked for none, the types of whatever the scope holds. A type with no job
on record is named as one, never quietly left out.

Only staff and maintenance get them (STAFF). A visitor asking this shape is declined
by the host before any step runs, whatever the question names, so the refusal says
nothing about which types exist.

Nearby is the contractor's view. Whoever services the lights in one room does the
rest of the building's while they're there. So when the scope sits below building
tier, the other visible assets in the same building, of the types asked about that
have a job and are in scope, are counted per space of the tier under the building
(floors, usually). A scope at building tier or above already holds all of them, so
there is nothing nearby to add. Plain counts. Due dates don't come into it yet.

The jobs go back to the page as rows of a table, as recorded: no grouping, and the
Clinic's "n/a" stays "n/a". At most TASK_CAP rows. The trace says how many were cut.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from spatial_core.records import MaintenanceTask, Persona, Value
from spatial_service.assets import ScopeResult, Tally
from spatial_service.places import BUILDING, Venue, building_or_above, label

# Who gets an answer of this shape.
STAFF = (Persona.STAFF, Persona.MAINTENANCE)
# How many job rows go back with an answer. The AHU on the Clinic has nineteen.
TASK_CAP = 30


@dataclass(frozen=True)
class NearbyGroup:
    # The space of the tier under the building, or None for one hung on the building itself.
    space_id: str | None
    type_id: str
    tally: Tally


@dataclass(frozen=True)
class Nearby:
    building: str
    # The tier counted by, under the building. None when the venue has none.
    tier: str | None
    groups: tuple[NearbyGroup, ...]

    def tally(self, type_id: str) -> Tally:
        """Every nearby asset of one type, across the building."""
        found = [g.tally for g in self.groups if g.type_id == type_id]
        return Tally(sum(t.sourced for t in found), sum(t.generated for t in found))

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(g.type_id for g in self.groups))


@dataclass(frozen=True)
class Jobs:
    # The types asked about, by name.
    types: tuple[str, ...]
    # Every job on them, in table order.
    tasks: tuple[MaintenanceTask, ...]
    # The types asked about with no job on record.
    without: tuple[str, ...]
    nearby: Nearby | None


def maintenance_tasks(venue: Venue, found: ScopeResult) -> Jobs:
    """The jobs on the types ``found`` is about, and the same kit elsewhere in its building."""
    if found.types is not None:
        types = set(found.types)
    else:
        types = {venue.assets[i].type_id for i in found.ids}
    ordered = sorted(types, key=lambda i: (venue.asset_types[i].name.value, i))
    tasks = [task for i in ordered for task in sorted(venue.maintenance_tasks.get(i, ()), key=_row)]
    without = tuple(i for i in ordered if not venue.maintenance_tasks.get(i))
    return Jobs(
        types=tuple(ordered),
        tasks=tuple(tasks),
        without=without,
        nearby=_nearby(venue, found, {t.type_id for t in tasks}),
    )


def _row(task: MaintenanceTask) -> tuple[str, int, str]:
    # The Clinic numbers a job's step rows 0, 1, 2, ... The seed numbers its jobs.
    number = task.task_number.value
    return task.name.value, int(number) if number.isdigit() else 1_000_000, task.id


def _nearby(venue: Venue, found: ScopeResult, with_jobs: set[str]) -> Nearby | None:
    if found.scope is None or BUILDING not in venue.tiers:
        return None
    scope = venue.spaces[found.scope]
    building = venue.building_of(scope)
    if building is None or building_or_above(venue, scope):
        return None
    asked = set(found.ids)
    present = {venue.assets[i].type_id for i in found.ids} & with_jobs
    others = [
        asset
        for asset in venue.assets_under.get(building.id, [])
        if asset.type_id in present and asset.id not in asked
    ]
    if not others:
        return None
    below = venue.tiers.index(BUILDING) + 1
    tier = venue.tiers[below] if below < len(venue.tiers) else None
    grouped: dict[tuple[str | None, str], list[str]] = {}
    for asset in others:
        home = next((a.id for a in asset.ancestor_path if a.tier == tier), None)
        grouped.setdefault((home, asset.type_id), []).append(asset.id)

    def order(key: tuple[str | None, str]) -> tuple[bool, str, str, str]:
        # By the floor's label, the unlabelled after, then by type.
        home, type_id = key
        place = _label_of(venue, home)
        return place is None, place or "", venue.asset_types[type_id].name.value, home or ""

    groups = tuple(
        NearbyGroup(home, type_id, Tally.of(venue.assets[i] for i in ids))
        for (home, type_id), ids in sorted(grouped.items(), key=lambda item: order(item[0]))
    )
    return Nearby(building.id, tier, groups)


# --- what goes back with the turn ----------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TaskRow(_Model):
    """One job record as a row of the table. Values as recorded, provenance alongside."""

    task_id: str
    type_id: str
    type_name: str
    name: str
    category: str
    frequency: str
    frequency_unit: str
    duration: str
    duration_unit: str
    description: str
    resource_names: str | None
    task_number: str
    # "sourced" or "seed": a seed row is generated, and the page says so.
    record_provenance: str
    # How the job's values were reached, and the rule or basis behind them.
    provenance: str
    rule: str | None
    basis: str | None


class NearbyRow(_Model):
    # The space of the tier under the building, or None for one on the building itself.
    space_id: str | None
    label: str | None
    type_id: str
    type_name: str
    sourced: int
    generated: int


class NearbyOut(_Model):
    building: str
    label: str | None
    tier: str | None
    rows: list[NearbyRow]


class MaintenanceOut(_Model):
    # The plan step that found them.
    step: str
    tasks: list[TaskRow]
    # Job rows past TASK_CAP, not sent.
    tasks_left_out: int
    nearby: NearbyOut | None


class MaintenanceTrace(_Model):
    step: str
    types: list[str]
    without_jobs: list[str]
    # Every job on the types asked about, whether or not it went out.
    tasks: list[str]
    task_cap: int = TASK_CAP
    nearby_building: str | None
    nearby_tier: str | None
    # The types counted nearby: those asked about, with a job, and in scope.
    nearby_types: list[str]


def _label_of(venue: Venue, space_id: str | None) -> str | None:
    space = venue.spaces.get(space_id) if space_id is not None else None
    return label(space) if space is not None else None


def _value(found: Value | None) -> str | None:
    return found.value if found is not None else None


def _task_row(venue: Venue, task: MaintenanceTask) -> TaskRow:
    return TaskRow(
        task_id=task.id,
        type_id=task.type_id,
        type_name=venue.asset_types[task.type_id].name.value,
        name=task.name.value,
        category=task.category.value,
        frequency=task.frequency.value,
        frequency_unit=task.frequency_unit.value,
        duration=task.duration.value,
        duration_unit=task.duration_unit.value,
        description=task.description.value,
        resource_names=_value(task.resource_names),
        task_number=task.task_number.value,
        record_provenance=task.record_provenance.value,
        provenance=task.name.provenance.value,
        rule=task.name.rule,
        basis=task.name.basis,
    )


def out(venue: Venue, step: str, jobs: Jobs) -> MaintenanceOut:
    nearby = None
    if jobs.nearby is not None:
        building = venue.spaces[jobs.nearby.building]
        nearby = NearbyOut(
            building=building.id,
            label=label(building),
            tier=jobs.nearby.tier,
            rows=[
                NearbyRow(
                    space_id=group.space_id,
                    label=_label_of(venue, group.space_id),
                    type_id=group.type_id,
                    type_name=venue.asset_types[group.type_id].name.value,
                    sourced=group.tally.sourced,
                    generated=group.tally.generated,
                )
                for group in jobs.nearby.groups
            ],
        )
    return MaintenanceOut(
        step=step,
        tasks=[_task_row(venue, task) for task in jobs.tasks[:TASK_CAP]],
        tasks_left_out=max(len(jobs.tasks) - TASK_CAP, 0),
        nearby=nearby,
    )


def trace(step: str, jobs: Jobs) -> MaintenanceTrace:
    return MaintenanceTrace(
        step=step,
        types=list(jobs.types),
        without_jobs=list(jobs.without),
        tasks=[task.id for task in jobs.tasks],
        nearby_building=jobs.nearby.building if jobs.nearby else None,
        nearby_tier=jobs.nearby.tier if jobs.nearby else None,
        nearby_types=list(jobs.nearby.types) if jobs.nearby else [],
    )


def type_names(venue: Venue, type_ids: Iterable[str]) -> list[str]:
    return [venue.asset_types[i].name.value for i in type_ids]
