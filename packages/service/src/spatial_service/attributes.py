"""Attributes: which ones a question may filter on, who carries them, and service due.

The attribute catalogue is, per asset type, the attributes at least one asset of
the type carries. It is read off the assets, never declared, and here it is the
persona's cut of the snapshot's: only assets they can see count. So an attribute
that only hidden assets carry is not in it, and filtering on it is rejected in the
same words as filtering on one nothing carries. The planner's filter vocabulary is
the catalogue's attribute names and nothing more.

Two attributes are not stored on an asset and are in the catalogue anyway, once
anything visible can carry them:

    service_event   the start of the asset's latest service event
    service_due     that, or the installation date where there is no event, plus
                    one year. Worked out here, every time it is asked for, and
                    stored nowhere. The year is a default interval, the same for
                    every type. Nobody's standard, and the trace says so.

An asset that does not carry an attribute is not recorded for it. That is all it
is: a gap in the data. It never passes or fails a test on the value, it is counted
apart, and nothing here turns it into a statement about the asset.

The fill rate of an attribute is how many of the assets asked about carry it: the
visible ones under the scope, of the types asked for. It is worked out per
question and is never part of the catalogue.
"""

from __future__ import annotations

import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from spatial_core.records import Asset, Value
from spatial_service.places import Venue
from spatial_service.plan import DATED, Filter, FilterTest

INSTALLED = "installation_date"
SERVICE_EVENT = "service_event"
SERVICE_DUE = "service_due"
SERVICE_DUE_RULE = "service-due@1"
SERVICE_INTERVAL_YEARS = 1
# How many assets the trace lists a service due date for, one by one.
DUE_TRACE_CAP = 200

_MEMO = "attribute-catalogue"


class CountedFrom(StrEnum):
    SERVICE_EVENT = "service-event"
    INSTALLATION_DATE = "installation-date"


@dataclass(frozen=True)
class Reading:
    """One asset's value for one attribute."""

    value: str
    # The value as a date, when it is one.
    on: datetime.date | None
    # For service due: what it was counted from, and that date with its provenance.
    counted_from: CountedFrom | None = None
    source: Value | None = None


def _day(text: str) -> datetime.date | None:
    try:
        return datetime.date.fromisoformat(text[:10])
    except ValueError:
        return None


def _a_year_on(day: datetime.date) -> datetime.date:
    try:
        return day.replace(year=day.year + SERVICE_INTERVAL_YEARS)
    except ValueError:
        # 29 February.
        return day.replace(year=day.year + SERVICE_INTERVAL_YEARS, day=28)


def _latest_event(venue: Venue, asset: Asset) -> Value | None:
    """The start of the asset's latest service event. An undated start sorts first."""
    starts = [event.start for event in venue.service_events.get(asset.id, ())]
    if not starts:
        return None
    return max(starts, key=lambda start: (_day(start.value) or datetime.date.min, start.value))


def read(venue: Venue, asset: Asset, attribute: str) -> Reading | None:
    """What ``asset`` carries for ``attribute``. None is not recorded."""
    if attribute == SERVICE_EVENT:
        start = _latest_event(venue, asset)
        return Reading(start.value, _day(start.value)) if start is not None else None
    if attribute == SERVICE_DUE:
        return service_due(venue, asset)
    carried = asset.attributes.get(attribute)
    return Reading(carried.value, _day(carried.value)) if carried is not None else None


def service_due(venue: Venue, asset: Asset) -> Reading | None:
    """The latest service event, or the installation date where there is none, plus a year."""
    for counted_from, source in (
        (CountedFrom.SERVICE_EVENT, _latest_event(venue, asset)),
        (CountedFrom.INSTALLATION_DATE, asset.attributes.get(INSTALLED)),
    ):
        day = _day(source.value) if source is not None else None
        if day is not None:
            due = _a_year_on(day)
            return Reading(due.isoformat(), due, counted_from, source)
    return None


@dataclass(frozen=True)
class Catalogue:
    # Per visible asset type, the attributes at least one visible asset of it carries.
    by_type: Mapping[str, frozenset[str]]
    # Those whose every value is a date, so before and after mean something.
    dated: frozenset[str]

    @property
    def attributes(self) -> tuple[str, ...]:
        return tuple(sorted({name for carried in self.by_type.values() for name in carried}))


def catalogue(venue: Venue) -> Catalogue:
    """The attribute catalogue as this persona sees it. Worked out once per venue."""
    if _MEMO not in venue.memo:
        by_type: dict[str, set[str]] = {}
        undated: set[str] = set()
        for asset in venue.assets.values():
            for name in (*asset.attributes, SERVICE_EVENT, SERVICE_DUE):
                reading = read(venue, asset, name)
                if reading is None:
                    continue
                by_type.setdefault(asset.type_id, set()).add(name)
                if reading.on is None:
                    undated.add(name)
        names = {name for carried in by_type.values() for name in carried}
        venue.memo[_MEMO] = Catalogue(
            {type_id: frozenset(carried) for type_id, carried in by_type.items()},
            frozenset(names - undated),
        )
    found = venue.memo[_MEMO]
    assert isinstance(found, Catalogue)
    return found


def check_filters(venue: Venue, filters: Sequence[Filter]) -> list[str]:
    """Why ``filters`` are no arguments for assets_in_scope: an attribute outside the catalogue.

    Reads the persona's catalogue and looks nothing up, so the answer is the same
    for an attribute beyond clearance and one that does not exist.
    """
    known = catalogue(venue)
    reasons = []
    for one in filters:
        if one.attribute not in known.attributes:
            listed = ", ".join(known.attributes) or "none"
            reasons.append(
                f"{one.attribute!r} is not an attribute to filter on here; the venue has {listed}"
            )
        elif one.test in DATED and one.attribute not in known.dated:
            listed = ", ".join(sorted(known.dated)) or "none"
            reasons.append(
                f"{one.attribute!r} is not a date, so {one.test} does not apply; "
                f"the dates are {listed}"
            )
    return reasons


@dataclass(frozen=True)
class FillRate:
    """One filter over the assets asked about: who carries its attribute, and who passes."""

    filter: Filter
    # Of the assets asked about, how many carry the attribute, with what.
    of: int
    readings: tuple[tuple[str, Reading], ...]
    # The assets that do not carry it. Not recorded, and nothing more is known.
    lacking: tuple[str, ...]
    # How many pass this filter alone.
    matched: int

    @property
    def carried(self) -> int:
        return len(self.readings)

    @property
    def gap(self) -> bool:
        """Assets were asked about and not one carries the attribute: a coverage gap."""
        return self.of > 0 and self.carried == 0


def _passes(one: Filter, reading: Reading | None) -> bool:
    if one.test is FilterTest.NOT_RECORDED:
        return reading is None
    if reading is None:
        return False
    if one.test is FilterTest.RECORDED:
        return True
    if reading.on is None or one.date is None:
        return False
    return reading.on < one.date if one.test is FilterTest.BEFORE else reading.on > one.date


def apply(
    venue: Venue, assets: Sequence[Asset], filters: Sequence[Filter]
) -> tuple[list[Asset], tuple[FillRate, ...]]:
    """The assets passing every filter, and each filter's fill rate over all of ``assets``."""
    kept = list(assets)
    rates = []
    for one in filters:
        read_out = {asset.id: read(venue, asset, one.attribute) for asset in assets}
        passing = {asset_id for asset_id, reading in read_out.items() if _passes(one, reading)}
        kept = [asset for asset in kept if asset.id in passing]
        rates.append(
            FillRate(
                filter=one,
                of=len(assets),
                readings=tuple((i, r) for i, r in read_out.items() if r is not None),
                lacking=tuple(i for i, r in read_out.items() if r is None),
                matched=len(passing),
            )
        )
    return kept, tuple(rates)


# --- in words -----------------------------------------------------------------------------

# What an asset lacks when a derived attribute cannot be worked out for it.
_LACKS = {SERVICE_DUE: "service event or installation date"}


def name(attribute: str) -> str:
    return attribute.replace("_", " ")


def not_recorded(attribute: str) -> str:
    """How a missing attribute is put: a claim about the data, never about the asset."""
    return f"no {_LACKS.get(attribute, name(attribute))} recorded"


def condition(one: Filter) -> str:
    """What a filter tested, as the answer prints it."""
    if one.test is FilterTest.NOT_RECORDED:
        return not_recorded(one.attribute)
    if one.test is FilterTest.RECORDED:
        return f"{name(one.attribute)} recorded"
    assert one.date is not None
    return f"{name(one.attribute)} {one.test.value} {one.date.isoformat()}"


# --- in the trace -------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DueTrace(_Model):
    """One asset's service due, and the date it was counted from."""

    asset_id: str
    due: str
    counted_from: CountedFrom
    counted_on: str
    # Value provenance of the date counted from, with its rule or basis.
    provenance: str
    rule: str | None
    basis: str | None


class FilterTrace(_Model):
    attribute: str
    test: FilterTest
    date: str | None
    # The fill rate: how many of the visible assets asked about carry the attribute.
    carried: int
    of: int
    # None when no assets were asked about, so there is no rate to give.
    fill_rate: float | None
    # How many pass this filter, and how many are not recorded for it.
    matched: int
    not_recorded: int
    # True for an attribute worked out at query time and stored nowhere.
    derived: bool
    rule: str | None = None
    # Service due rests on one default interval for every type. Not an industry standard.
    default_interval: bool = False
    interval_years: int | None = None
    # Each asset's service due, up to DUE_TRACE_CAP of them.
    due: list[DueTrace] = []


def trace(rates: Sequence[FillRate]) -> list[FilterTrace]:
    traced = []
    for rate in rates:
        one = rate.filter
        due = one.attribute == SERVICE_DUE
        traced.append(
            FilterTrace(
                attribute=one.attribute,
                test=one.test,
                date=one.date.isoformat() if one.date else None,
                carried=rate.carried,
                of=rate.of,
                fill_rate=rate.carried / rate.of if rate.of else None,
                matched=rate.matched,
                not_recorded=len(rate.lacking),
                derived=due,
                rule=SERVICE_DUE_RULE if due else None,
                default_interval=due,
                interval_years=SERVICE_INTERVAL_YEARS if due else None,
                due=[
                    DueTrace(
                        asset_id=asset_id,
                        due=reading.value,
                        counted_from=reading.counted_from,
                        counted_on=reading.source.value[:10],
                        provenance=reading.source.provenance.value,
                        rule=reading.source.rule,
                        basis=reading.source.basis,
                    )
                    for asset_id, reading in rate.readings[:DUE_TRACE_CAP]
                    if reading.counted_from is not None and reading.source is not None
                ],
            )
        )
    return traced
