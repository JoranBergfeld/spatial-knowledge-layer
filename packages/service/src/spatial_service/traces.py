"""The trace's parts, worked out from what a turn ran and what it answered.

Ids, provenance and the rule or basis behind each label and name. Never a
position: an asset's is said to be recorded or not, and that is all.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from spatial_core.records import Asset, AssetType, Space
from spatial_service import attributes
from spatial_service.conversation import name
from spatial_service.executor import Execution, StepResult
from spatial_service.facts import POSITION_NOT_RECORDED, POSITION_RECORDED, Answer
from spatial_service.places import Outcome, Venue, labelled
from spatial_service.plan import resolves
from spatial_service.turn_models import (
    TRACE_ASSET_CAP,
    AssetsTrace,
    AssetTrace,
    AuthoredDefault,
    CountRow,
    Provenance,
    ProvenanceClass,
    Reference,
    Resolved,
    ResolvedTypes,
    TraceStep,
    TypeTrace,
)


def steps(execution: Execution) -> list[TraceStep]:
    return [
        TraceStep(
            id=r.step,
            tool=r.tool,
            args=r.args,
            text=r.text,
            outcome=r.outcome,
            ids=list(r.ids),
            how=r.how,
            rejection=list(r.rejection),
        )
        for r in execution.results
    ]


def resolved(execution: Execution, venue: Venue) -> list[Resolved]:
    return [
        Resolved(
            mention=r.mention or "",
            text=r.text or "",
            step=r.step,
            space_id=r.ids[0],
            label=labelled(venue.spaces[r.ids[0]]),
            how=r.how,
        )
        for r in execution.by_tool("resolve_place")
        if r.outcome is Outcome.RESOLVED
    ]


def resolved_types(execution: Execution, venue: Venue) -> list[ResolvedTypes]:
    return [
        ResolvedTypes(
            mention=r.mention or "",
            text=r.text or "",
            step=r.step,
            type_ids=list(r.ids),
            names=[venue.asset_types[i].name.value for i in r.ids],
        )
        for r in execution.by_tool("resolve_asset_type")
        if r.outcome is Outcome.RESOLVED
    ]


def references(execution: Execution, venue: Venue) -> list[Reference]:
    found = []
    for r in execution.results:
        referred = r.referred
        if referred is None:
            continue
        found.append(
            Reference(
                mention=r.mention or "",
                text=r.text or "",
                step=r.step,
                kind=resolves(r.tool),
                to=referred.to,
                ordinal=referred.ordinal,
                outcome=r.outcome,
                ids=list(r.ids),
                labels=[name(venue, referred.kind, i) for i in r.ids],
                turn=referred.turn,
                through=list(referred.through),
            )
        )
    return found


def provenance(space: Space) -> Provenance:
    assert space.label is not None
    return Provenance(
        space_id=space.id,
        record_provenance=space.record_provenance.value,
        source=space.source,
        label=space.label.value,
        label_provenance=space.label.provenance.value,
        label_rule=space.label.rule,
    )


def defaults(spaces: list[Space]) -> list[AuthoredDefault]:
    return [
        AuthoredDefault(space_id=s.id, persona=s.clearance.persona, rule=s.clearance.rule)
        for s in spaces
        if s.clearance is not None and s.clearance.authored_default
    ]


def type_trace(kind: AssetType) -> TypeTrace:
    return TypeTrace(
        type_id=kind.id,
        record_provenance=kind.record_provenance.value,
        name=kind.name.value,
        name_provenance=kind.name.provenance.value,
        name_rule=kind.name.rule,
        name_basis=kind.name.basis,
        omniclass=kind.omniclass.value,
        clearance=kind.clearance.persona,
        clearance_rule=kind.clearance.rule,
        authored_default=kind.clearance.authored_default,
        unrecognised_category=kind.clearance.unrecognised_category,
    )


def _asset_provenance(asset: Asset) -> dict[str, Any]:
    """Both provenance fields of an asset, with the rule or basis. Never where it is."""
    position = asset.position
    return {
        "record_provenance": asset.record_provenance.value,
        "name_provenance": asset.name.provenance.value,
        "name_rule": asset.name.rule,
        "name_basis": asset.name.basis,
        "position": POSITION_RECORDED if position else POSITION_NOT_RECORDED,
        "position_provenance": position.provenance.value if position else None,
        "position_rule": position.rule if position else None,
        "position_basis": position.basis if position else None,
    }


def _asset_trace(asset: Asset, fields: dict[str, Any]) -> AssetTrace:
    """One asset as the trace lists it, with the provenance fields worked out for it."""
    return AssetTrace(
        asset_id=asset.id,
        type_id=asset.type_id,
        space_id=asset.space_id,
        source=asset.source,
        name=asset.name.value,
        **fields,
    )


def one_asset_trace(venue: Venue, asset: Asset) -> AssetsTrace:
    """The trace of an answer about one asset: itself, and how much its space holds."""
    fields = _asset_provenance(asset)
    return AssetsTrace(
        scope=asset.space_id,
        types=[asset.type_id],
        grouped_by=None,
        touched=1,
        installed_in_scope=len(venue.assets_under.get(asset.space_id, [])),
        listed=[_asset_trace(asset, fields)],
        classes=[ProvenanceClass(**fields, assets=1)],
        rows=[],
    )


def assets_trace(venue: Venue, last: StepResult, answer: Answer) -> AssetsTrace:
    found = last.found
    assert found is not None
    assets = [venue.assets[i] for i in found.ids]
    provenance = [_asset_provenance(asset) for asset in assets]
    classes = Counter(tuple(fields.items()) for fields in provenance)
    return AssetsTrace(
        scope=found.scope,
        types=list(found.types) if found.types is not None else None,
        grouped_by=last.aggregation.by if last.aggregation else None,
        touched=len(assets),
        installed_in_scope=found.installed,
        listed=[
            _asset_trace(asset, fields)
            for asset, fields in list(zip(assets, provenance, strict=True))[:TRACE_ASSET_CAP]
        ],
        classes=[ProvenanceClass(**dict(key), assets=count) for key, count in classes.items()],
        rows=[
            CountRow(
                space_id=row.space_id,
                type_id=row.type_id,
                label=row.label,
                sourced=row.tally.sourced,
                generated=row.tally.generated,
                recorded=row.recorded,
                rank=row.rank,
            )
            for row in answer.rows
        ],
        ungrouped=last.aggregation.ungrouped if last.aggregation else 0,
        filters=attributes.trace(found.filters),
    )
