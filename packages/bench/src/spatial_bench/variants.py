"""Copies of a bundle with the restricted things taken out, and with more of them put in.

"Restricted" is whatever the lowest persona cannot see: assets hidden by their
type or by their room, and the documents that go with them. The counterfactual
leak check loads each copy into stores of its own and asks the visitor's
questions again. Nothing a visitor can see is touched in either copy, so nothing
a visitor is told may change.

    removed   every restricted asset, its service events and every restricted
              document are gone
    added     every restricted asset and document is there twice. Every restricted
              type also gets an asset in rooms a visitor walks through, a type
              nobody had before shows up there too, all with an attribute only
              they carry. And every document a visitor can read gets a twin with
              the same words that only a restricted type is anchored to. If the
              filter let it through, it would rank right beside the real one.

Both copies still pass every load invariant. That is checked by loading them.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel

from spatial_bench.bundle import Bundle, replace_part
from spatial_core import clearance
from spatial_core.records import (
    PERSONAS,
    Ancestor,
    Anchor,
    AnchorKind,
    Asset,
    AssetType,
    Chunk,
    Document,
    DocumentKind,
    Persona,
    ServiceEvent,
    Space,
    Value,
    ValueProvenance,
    VisibleAsset,
    VisibleAssetType,
)
from spatial_ingest import summaries
from spatial_ingest.documents import (
    CHUNKS_PART,
    DOCUMENTS_PART,
    TRANSLATIONS_PART,
    VECTORS_PART,
    Translation,
    Vector,
)
from spatial_ingest.load import HALVES
from spatial_ingest.summaries import Summary

MARK = "~cf"
TWIN = "~twin"
MARKER_ATTRIBUTE = "counterfactual_marker"
HIDDEN_TYPE_NAME = "Counterfactual Hidden Unit"
# How many rooms a restricted type is dropped into, and how many documents get a twin.
ROOM_CAP = 25
TWIN_CAP = 25


def _spread[T](items: Sequence[T], count: int) -> list[T]:
    if len(items) <= count:
        return list(items)
    return [items[(index * len(items)) // count] for index in range(count)]


def _path(space: Space) -> tuple[Ancestor, ...]:
    return (*space.ancestor_path, Ancestor(tier=space.tier.value, id=space.id))


def _set_visible(bundle: Bundle) -> None:
    """Work the visible assets and types out again, by the shared rule, into both halves."""
    spaces = {persona: bundle.visible(persona).spaces for persona in PERSONAS}
    asset_sets, type_sets = clearance.visible_assets(bundle.assets, bundle.types, spaces)
    for half in HALVES:
        assets = {a.id for a in bundle.parts.get(f"{half}/assets.jsonl", [])}  # type: ignore[attr-defined]
        types = {t.id for t in bundle.parts.get(f"{half}/asset_types.jsonl", [])}  # type: ignore[attr-defined]
        replace_part(
            bundle,
            f"{half}/visible_assets.jsonl",
            [
                VisibleAsset(persona=persona, asset_id=asset_id)
                for persona in PERSONAS
                for asset_id in sorted(asset_sets[persona] & assets)
            ],
        )
        replace_part(
            bundle,
            f"{half}/visible_asset_types.jsonl",
            [
                VisibleAssetType(persona=persona, type_id=type_id)
                for persona in PERSONAS
                for type_id in sorted(type_sets[persona] & types)
            ],
        )


def _ancestors(bundle: Bundle, chunk: Chunk) -> tuple[str, ...]:
    spaces = {space.id: space for space in bundle.spaces}
    assets = {asset.id: asset for asset in bundle.assets}
    found: set[str] = set()
    for space_id in chunk.space_ids:
        found.add(space_id)
        found.update(step.id for step in spaces[space_id].ancestor_path)
    for asset_id in chunk.asset_ids:
        found.update(step.id for step in assets[asset_id].ancestor_path)
    return tuple(sorted(found))


def _set_documents(
    bundle: Bundle,
    documents: Sequence[Document],
    chunks: Sequence[Chunk],
    made: Sequence[Summary],
    vectors: Sequence[Vector],
) -> None:
    kept = {chunk.id for chunk in chunks} | {s.id for s in made}
    replace_part(bundle, DOCUMENTS_PART, documents)
    replace_part(bundle, CHUNKS_PART, chunks)
    replace_part(bundle, summaries.PART, made)
    replace_part(bundle, VECTORS_PART, [v for v in vectors if v.chunk_id in kept])
    translations = bundle.parts.get(TRANSLATIONS_PART, [])
    replace_part(
        bundle,
        TRANSLATIONS_PART,
        [t for t in translations if isinstance(t, Translation) and t.chunk_id in kept],
    )


def without_restricted(bundle: Bundle, below: Persona = Persona.VISITOR) -> Bundle:
    """The bundle with everything ``below`` cannot see taken out: assets, events, documents."""
    seen = bundle.visible(below)
    read = {document.id for document in bundle.documents if seen.reads(document)}
    changed = bundle.copy()
    for half in HALVES:
        assets = [a for a in bundle.parts.get(f"{half}/assets.jsonl", []) if a.id in seen.assets]  # type: ignore[attr-defined]
        events = [
            e
            for e in bundle.parts.get(f"{half}/service_events.jsonl", [])
            if e.asset_id in seen.assets  # type: ignore[attr-defined]
        ]
        replace_part(changed, f"{half}/assets.jsonl", assets)
        replace_part(changed, f"{half}/service_events.jsonl", events)
    _set_visible(changed)

    # A document a visitor reads may also be tied to an asset that is now gone.
    documents = [
        document.model_copy(
            update={
                "anchors": tuple(
                    anchor
                    for anchor in document.anchors
                    if anchor.kind is not AnchorKind.ASSET or anchor.id in seen.assets
                )
            }
        )
        for document in bundle.documents
        if document.id in read
    ]
    chunks = []
    for chunk in bundle.chunks:
        if chunk.document_id not in read:
            continue
        kept = chunk.model_copy(
            update={"asset_ids": tuple(i for i in chunk.asset_ids if i in seen.assets)}
        )
        chunks.append(kept.model_copy(update={"ancestor_ids": _ancestors(changed, kept)}))
    made = [s for s in bundle.summaries if s.document_id in read]
    vectors = [v for v in bundle.parts.get(VECTORS_PART, []) if isinstance(v, Vector)]
    _set_documents(changed, documents, chunks, made, vectors)
    return changed


def _marked(asset: Asset, asset_id: str, space: Space | None = None) -> Asset:
    marker = Value(
        value="set", provenance=ValueProvenance.INFERRED, basis="counterfactual leak check"
    )
    update: dict[str, object] = {
        "id": asset_id,
        "attributes": {**asset.attributes, MARKER_ATTRIBUTE: marker},
    }
    if space is not None:
        update |= {"space_id": space.id, "ancestor_path": _path(space), "position": None}
        update |= {"door": None, "source": None}
    return asset.model_copy(update=update)


def with_restricted_added(bundle: Bundle, below: Persona = Persona.VISITOR) -> Bundle:
    """The bundle with more of what ``below`` cannot see. See the module's docstring."""
    seen = bundle.visible(below)
    read = {document.id for document in bundle.documents if seen.reads(document)}
    changed = bundle.copy()
    parents = {space.parent_id for space in bundle.spaces}
    rooms = _spread(
        sorted(
            (s for s in bundle.spaces if s.id in seen.spaces and s.id not in parents),
            key=lambda s: s.id,
        ),
        ROOM_CAP,
    )
    restricted_types = sorted(
        (t for t in bundle.types if not below.clears(t.clearance)), key=lambda t: t.id
    )

    added: dict[str, list[Asset]] = {half: [] for half in HALVES}
    added_types: dict[str, list[AssetType]] = {half: [] for half in HALVES}
    added_events: dict[str, list[ServiceEvent]] = {half: [] for half in HALVES}
    by_type: dict[str, Asset] = {}
    for asset in bundle.assets:
        by_type.setdefault(asset.type_id, asset)
        if asset.id not in seen.assets:
            added[bundle.half_of("assets", asset.id)].append(_marked(asset, asset.id + MARK))
    for event in bundle.events:
        if event.asset_id not in seen.assets:
            copy = event.model_copy(
                update={"id": event.id + MARK, "asset_id": event.asset_id + MARK}
            )
            added_events[bundle.half_of("service_events", event.id)].append(copy)
    for kind in restricted_types:
        template = by_type.get(kind.id)
        if template is None:
            continue
        half = bundle.half_of("assets", template.id)
        for number, room in enumerate(rooms):
            added[half].append(_marked(template, f"{template.id}{MARK}-in-{number}", room))
    if restricted_types and restricted_types[0].id in by_type:
        kind, template = restricted_types[0], by_type[restricted_types[0].id]
        half = bundle.half_of("asset_types", kind.id)
        renamed = kind.name.model_copy(update={"value": HIDDEN_TYPE_NAME})
        new = kind.model_copy(update={"id": kind.id + MARK, "name": renamed})
        added_types[half].append(new)
        asset_half = bundle.half_of("assets", template.id)
        if asset_half == half:
            for number, room in enumerate(rooms[:3]):
                moved = _marked(template, f"{template.id}{MARK}-new-{number}", room)
                added[half].append(moved.model_copy(update={"type_id": new.id}))
    for half in HALVES:
        extras: tuple[tuple[str, Sequence[BaseModel]], ...] = (
            ("assets", added[half]),
            ("asset_types", added_types[half]),
            ("service_events", added_events[half]),
        )
        for table, extra in extras:
            part = f"{half}/{table}.jsonl"
            if extra or part in bundle.parts:
                replace_part(changed, part, [*bundle.parts.get(part, []), *extra])
    _set_visible(changed)

    documents = list(bundle.documents)
    chunks = list(bundle.chunks)
    made = list(bundle.summaries)
    vectors = [v for v in bundle.parts.get(VECTORS_PART, []) if isinstance(v, Vector)]
    by_chunk = {vector.chunk_id: vector for vector in vectors}

    def copy_of(document: Document, suffix: str, **update: object) -> None:
        """A second document with the first one's words, under another id."""
        new_id = document.id + suffix

        def renamed(old: str) -> str:
            return new_id + old.removeprefix(document.id)

        documents.append(document.model_copy(update={"id": new_id, **update}))
        on_chunk = {key: value for key, value in update.items() if key in Chunk.model_fields}
        if "anchors" in update:
            anchors: tuple[Anchor, ...] = update["anchors"]  # type: ignore[assignment]
            on_chunk |= {
                "space_ids": tuple(a.id for a in anchors if a.kind is AnchorKind.SPACE),
                "asset_type_ids": tuple(a.id for a in anchors if a.kind is AnchorKind.ASSET_TYPE),
                "asset_ids": tuple(a.id for a in anchors if a.kind is AnchorKind.ASSET),
            }
        for chunk in bundle.chunks:
            if chunk.document_id != document.id:
                continue
            twin = chunk.model_copy(
                update={"id": renamed(chunk.id), "document_id": new_id, **on_chunk}
            )
            chunks.append(twin.model_copy(update={"ancestor_ids": _ancestors(changed, twin)}))
            vectors.append(by_chunk[chunk.id].model_copy(update={"chunk_id": twin.id}))
        for summary in bundle.summaries:
            if summary.document_id != document.id:
                continue
            classed = {"clearance_class": update["clearance_class"]} if on_chunk else {}
            made.append(
                summary.model_copy(
                    update={
                        "id": renamed(summary.id),
                        "document_id": new_id,
                        "source_chunk_ids": tuple(renamed(i) for i in summary.source_chunk_ids),
                        **classed,
                    }
                )
            )
            if summary.id in by_chunk:
                vectors.append(
                    by_chunk[summary.id].model_copy(update={"chunk_id": renamed(summary.id)})
                )

    for document in bundle.documents:
        if document.id not in read:
            copy_of(document, MARK)
    placed = [kind for kind in restricted_types if kind.id in by_type]
    if placed:
        kind = placed[0]
        anchors = (
            Anchor(kind=AnchorKind.ASSET_TYPE, id=kind.id, provenance=ValueProvenance.SOURCED),
        )
        for document in _spread([d for d in bundle.documents if d.id in read], TWIN_CAP):
            copy_of(
                document,
                TWIN,
                kind=DocumentKind.MANUAL,
                anchors=anchors,
                clearance_class=kind.clearance.persona,
            )
    _set_documents(changed, documents, chunks, made, vectors)
    return changed
