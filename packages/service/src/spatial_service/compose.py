"""Composition: the answer in words, with every number and room label filled in from facts.

The composing model never sees a label, a name or a number. It is given facts by
reference ({f1}, {f2}, ...) with what kind each is, and writes segments that name
things only through those placeholders, each citing the facts it rests on. The host
checks what comes back and fills the placeholders from the facts. Output carrying a
digit is rejected, whatever it says: a number the model typed is a number nobody
checked.

The facts and their placeholders are in facts.py. Checking a draft and filling
it in is grounding.py. This is the answers: one builder per question shape.

Each answer builder also says which facts the answer has to name. An answer that
leaves one out is rejected too.

To add an answer: build its facts with Facts, return an Answer with the payload
the composer reads, and give the turn an instruction for the shape (instructions.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from spatial_core.records import Asset, Space
from spatial_service.assets import Aggregation, ScopeResult, Tally
from spatial_service.facts import (
    ANY_ASSETS,
    Answer,
    Citation,
    Fact,
    FactKind,
    Facts,
    Row,
)
from spatial_service.grounding import CompositionRejected, ground
from spatial_service.nearest import Nearest
from spatial_service.places import Venue, label
from spatial_service.retrieval import Retrieval
from spatial_service.routes import Found, Routed

# The builders, and what the tests have always taken from here.
__all__ = [
    "ANY_ASSETS",
    "ROW_CAP",
    "Citation",
    "CompositionRejected",
    "Fact",
    "FactKind",
    "aggregation",
    "asset_documents",
    "asset_location",
    "containment",
    "ground",
    "location",
    "place_documents",
    "proximity",
    "route",
]

# How many rows an answer spells out. The rest are counted, not named.
ROW_CAP = 12


def location(venue: Venue, question: str, subject: Space) -> Answer:
    """The place asked about, then each labelled place enclosing it, innermost first."""
    facts = Facts()
    for space in [subject, *venue.ancestors(subject)]:
        if space.label:
            facts.place(space)
    payload = {
        "question": question,
        "subject": facts.items[0].ref,
        "facts": [{"ref": f.ref, "tier": f.told["tier"]} for f in facts.items],
    }
    return Answer(facts.items, payload, required=[facts.items[0].ref])


def asset_location(venue: Venue, question: str, asset: Asset) -> Answer:
    """One asset, then the space it is in and each labelled place enclosing that."""
    facts = Facts()
    subject = facts.asset(asset)
    home = venue.spaces[asset.space_id]
    for space in [home, *venue.ancestors(home)]:
        if space.label:
            facts.place(space)
    payload = {
        "question": question,
        "shape": "asset-location",
        "subject": subject,
        "required": [subject],
        "facts": facts.told(),
    }
    return Answer(facts.items, payload, required=[subject])


def containment(venue: Venue, question: str, found: ScopeResult) -> Answer:
    """What is installed under a scope: a total, then rows by type, or by space for given types."""
    facts = Facts()
    assets = [venue.assets[i] for i in found.ids]
    scope = _scope(venue, facts, found)
    noun = facts.noun(venue, found.types)
    total = facts.counts(Tally.of(assets), space_id=found.scope, noun=noun)

    by_space = found.types is not None
    grouped: dict[str, list[Asset]] = {}
    for asset in assets:
        grouped.setdefault(asset.space_id if by_space else asset.type_id, []).append(asset)

    rows: list[Row] = []
    for key, members in grouped.items():
        tally = Tally.of(members)
        if by_space:
            label = venue.spaces[key].label
            rows.append(Row(tally, space_id=key, label=label.value if label else None))
        else:
            rows.append(Row(tally, type_id=key, label=venue.asset_types[key].name.value))
    rows.sort(key=lambda r: (-r.tally.total, r.label or "", r.space_id or r.type_id or ""))

    payload: dict[str, Any] = {
        "question": question,
        "shape": "asset-containment",
        "scope": scope,
        "total": total,
        "rows_by": "space" if by_space else "asset-type",
    }
    together = [total]
    told_rows = []
    # A space no rule can address still holds its assets. It has no label to print,
    # so those go in one row of their own.
    nameable = [r for r in rows if r.label is not None]
    for row in nameable[:ROW_CAP]:
        if row.space_id is not None:
            of = facts.place(venue.spaces[row.space_id])
        else:
            assert row.type_id is not None
            of = facts.asset_type(venue.asset_types[row.type_id])
        counts = facts.counts(
            row.tally,
            space_id=row.space_id or found.scope,
            type_id=row.type_id,
            noun=noun if row.space_id is not None else of,
        )
        told_rows.append({"of": of, "counts": counts})
        together.append(counts)
    payload["rows"] = told_rows
    if len(nameable) > ROW_CAP:
        payload["more_rows"] = facts.number(len(nameable) - ROW_CAP, about="rows left out")
    unlabelled = Tally.of(a for r in rows if r.label is None for a in grouped[r.space_id or ""])
    if unlabelled.total:
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled, noun=noun)
        together.append(payload["in_spaces_with_no_label"])
    unplaced = sum(1 for asset in assets if asset.position is None)
    if unplaced:
        payload["position_not_recorded"] = facts.number(
            unplaced, about="assets whose position is not recorded", space_id=found.scope
        )

    required = [*([scope] if scope else []), *total]
    payload["required"] = required
    payload["facts"] = facts.told()
    return Answer(facts.items, payload, required, together, rows)


def aggregation(venue: Venue, question: str, found: ScopeResult, counted: Aggregation) -> Answer:
    """Counts per space of a scale level: ranked, leaders named, gaps apart from zeros."""
    facts = Facts()
    scope = _scope(venue, facts, found)
    noun = facts.noun(venue, found.types)
    total = facts.counts(
        Tally.of(venue.assets[i] for i in found.ids), space_id=found.scope, noun=noun
    )

    rows = []
    for group in counted.ranked:
        label = venue.spaces[group.space_id].label
        rows.append(
            Row(
                group.tally,
                space_id=group.space_id,
                label=label.value if label else None,
                rank=group.rank,
            )
        )
    for space_id in counted.unrecorded:
        label = venue.spaces[space_id].label
        rows.append(
            Row(Tally(), space_id=space_id, label=label.value if label else None, recorded=False)
        )

    together = [total]
    told_rows = []
    by_space: dict[str, dict[str, Any]] = {}
    nameable = [r for r in rows if r.recorded and r.label is not None]
    for row in nameable[:ROW_CAP]:
        assert row.space_id is not None
        of = facts.place(venue.spaces[row.space_id])
        counts = facts.counts(row.tally, space_id=row.space_id, noun=noun)
        by_space[row.space_id] = {"of": of, "counts": counts}
        told_rows.append(by_space[row.space_id])
        together.append(counts)

    # The most, among the spaces that can be named. Ties share it.
    top = nameable[0].tally.total if nameable else 0
    leaders = [
        by_space[r.space_id or ""] for r in nameable[:ROW_CAP] if top and r.tally.total == top
    ]
    gaps = [r for r in rows if not r.recorded and r.label is not None]
    gap_refs = [facts.place(venue.spaces[r.space_id or ""]) for r in gaps[:ROW_CAP]]

    payload: dict[str, Any] = {
        "question": question,
        "shape": "asset-aggregation",
        "scope": scope,
        "grouped_by": counted.by,
        "total": total,
        "rows": told_rows,
        "leaders": [leader["of"] for leader in leaders],
        "no_recorded_installations": gap_refs,
    }
    required = [ref for leader in leaders for ref in (leader["of"], *leader["counts"])]
    if not leaders:
        required = list(total)
    required += gap_refs
    if len(nameable) > ROW_CAP:
        payload["more_rows"] = facts.number(len(nameable) - ROW_CAP, about="rows left out")
    if len(gaps) > ROW_CAP:
        payload["more_no_recorded_installations"] = facts.number(
            len(gaps) - ROW_CAP, about="more spaces with no recorded installations"
        )
        required.append(payload["more_no_recorded_installations"])
    unlabelled = Tally(
        sum(r.tally.sourced for r in rows if r.label is None),
        sum(r.tally.generated for r in rows if r.label is None),
    )
    if unlabelled.total:
        payload["in_spaces_with_no_label"] = facts.counts(unlabelled, noun=noun)
        together.append(payload["in_spaces_with_no_label"])

    payload["required"] = required
    payload["facts"] = facts.told()
    return Answer(facts.items, payload, required, together, rows)


def asset_documents(
    venue: Venue, question: str, found: ScopeResult, retrieved: Retrieval
) -> Answer:
    """The assets a question is about, as containment gives them, and what was retrieved."""
    base = containment(venue, question, found)
    payload = {
        **base.payload,
        "shape": "asset-document",
        "passages": _told_passages(retrieved),
    }
    return replace(base, payload=payload, passages=retrieved.passages)


def place_documents(venue: Venue, question: str, subject: Space, retrieved: Retrieval) -> Answer:
    """The place a question is about, and what was retrieved for it."""
    base = location(venue, question, subject)
    payload = {
        **base.payload,
        "shape": "place-document",
        "required": base.required,
        "passages": _told_passages(retrieved),
    }
    return replace(base, payload=payload, passages=retrieved.passages)


def _told_passages(retrieved: Retrieval) -> list[dict[str, str]]:
    """What the composer reads of each passage: its words. No id, no link, no date."""
    return [
        {"ref": p.ref, "kind": p.chunk.kind.value, "title": p.chunk.title, "text": p.chunk.text}
        for p in retrieved.passages
    ]


def _place(venue: Venue, facts: Facts, space_id: str) -> str | None:
    space = venue.spaces[space_id]
    return facts.place(space) if space.label else None


def _told_route(venue: Venue, facts: Facts, found: Found) -> dict[str, Any]:
    """One route by reference: its ends, what it passes through, how long and how far."""
    via: list[str] = []
    for leg in found.legs[1:-1]:
        ref = _place(venue, facts, leg.space_id)
        if ref is not None and ref not in via:
            via.append(ref)
    return {
        "destination": _place(venue, facts, found.destination),
        "duration": facts.duration(round(found.seconds), space_id=found.destination),
        "distance": facts.distance(round(found.metres), space_id=found.destination),
        "via": via[:ROW_CAP],
        "changes_level_by": sorted({leg.mode for leg in found.legs if leg.levels}),
    }


def route(venue: Venue, question: str, runs: Sequence[Routed]) -> Answer:
    """Each route found, in order: where from, where to, how long, how far."""
    facts = Facts()
    told = []
    required: list[str] = []
    for run in runs:
        best = run.found[0]
        origin = _place(venue, facts, best.origin)
        one = {
            "origin": origin,
            "origin_is_default": run.default_origin,
            "step_free": run.step_free,
            **_told_route(venue, facts, best),
            "alternatives": [_told_route(venue, facts, other) for other in run.found[1:]],
        }
        told.append(one)
        required += [ref for ref in (one["destination"], one["duration"], one["distance"]) if ref]
    payload = {
        "question": question,
        "shape": "route",
        "routes": told,
        "required": required,
        "facts": facts.told(),
    }
    return Answer(facts.items, payload, required)


def proximity(venue: Venue, question: str, near: Nearest) -> Answer:
    """The nearest assets: what they are, where, and what nearest was measured by."""
    facts = Facts()
    assets = [venue.assets[i] for i in near.asset_ids]
    by_space: dict[str, list[Asset]] = {}
    for asset in assets:
        by_space.setdefault(asset.space_id, []).append(asset)
    rows = [
        Row(Tally.of(members), space_id=space_id, label=label(venue.spaces[space_id]))
        for space_id, members in sorted(by_space.items())
    ]
    payload: dict[str, Any] = {
        "question": question,
        "shape": "proximity",
        "basis": near.basis,
        "anchor": _place(venue, facts, near.anchor),
        "anchor_is_default": near.default_origin,
        "types": [
            facts.asset_type(venue.asset_types[type_id])
            for type_id in sorted({asset.type_id for asset in assets})
        ],
    }
    noun = payload["types"][0] if len(payload["types"]) == 1 else ANY_ASSETS
    together = []
    told_rows: list[dict[str, Any]] = []
    for row in rows[:ROW_CAP]:
        assert row.space_id is not None
        counts = facts.counts(row.tally, space_id=row.space_id, noun=noun)
        told_rows.append({"in": _place(venue, facts, row.space_id), "counts": counts})
        together.append(counts)
    payload["nearest"] = told_rows
    required = [ref for row in told_rows[:1] for ref in (row["in"], *row["counts"]) if ref]
    required += payload["types"]
    if near.route is not None:
        payload["duration"] = facts.duration(
            round(near.route.seconds), space_id=near.route.destination
        )
        payload["distance"] = facts.distance(
            round(near.route.metres), space_id=near.route.destination
        )
        required += [payload["duration"], payload["distance"]]
    elif near.shared is not None:
        payload["shared"] = _place(venue, facts, near.shared)
        if payload["shared"]:
            required.append(payload["shared"])
    payload["required"] = required
    payload["facts"] = facts.told()
    return Answer(facts.items, payload, required, together, rows)


def _scope(venue: Venue, facts: Facts, found: ScopeResult) -> str | None:
    """The scope as a fact, where it has a label to print. The whole venue is its root."""
    space = venue.spaces[found.scope] if found.scope is not None else venue.root
    return facts.place(space) if space is not None and space.label else None
