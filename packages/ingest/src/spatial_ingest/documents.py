"""The snapshot's prose: a corpus of sources in, anchored English chunks out.

For each source of the corpus, build

1. resolves its anchors against the snapshot, and drops the source when none is
   left. A manual is only ingested for a type that is actually placed.
2. fetches the original, which is also the link check, and keeps it under
   documents/originals/. Nothing ever serves that file again.
3. reads its text, cuts it into passages, translates each passage that is not
   English, and writes small numbers out in words.
4. reads a document date where the page states one, works out the clearance
   class from the anchors, and embeds each chunk.

Fetching, translating and embedding are the three things here that are not
deterministic, so each keeps what it got in the bundle: documents/fetched.json,
translations.jsonl and vectors.jsonl. A second build reads those instead of
asking again, which is why it writes the same bytes and costs nothing.

    anchor-building-by-ref@1   a building number in the corpus is the building
                               space whose sourced ref is that number
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict

from spatial_core import clearance
from spatial_core.records import (
    PERSONAS,
    Anchor,
    AnchorKind,
    Asset,
    AssetType,
    Chunk,
    Document,
    DocumentKind,
    LinkCheck,
    LinkStatus,
    Persona,
    RecordProvenance,
    Space,
    TranslationSource,
    Value,
    ValueProvenance,
)
from spatial_ingest.corpus import AnchorSpec, Corpus, Source
from spatial_ingest.models import PROMPT_VERSION, Models
from spatial_ingest.rules import numbers, passages, tiers

ANCHOR_RULE = "anchor-building-by-ref@1"
RULES = (
    ANCHOR_RULE,
    passages.TEXT_RULE,
    passages.CHUNK_RULE,
    passages.DATE_RULE,
    numbers.RULE,
    clearance.DOCUMENT_RULE,
)

DIR = "documents"
ORIGINALS_DIR = "originals"
FETCHED_FILE = "fetched.json"
DOCUMENTS_PART = f"{DIR}/documents.jsonl"
CHUNKS_PART = f"{DIR}/chunks.jsonl"
TRANSLATIONS_PART = f"{DIR}/translations.jsonl"
VECTORS_PART = f"{DIR}/vectors.jsonl"

_USER_AGENT = "spatial-knowledge-layer-ingest (github.com/JoranBergfeld/spatial-knowledge-layer)"
_EXTENSIONS = {"html": ".html", "pdf": ".pdf", "text": ".txt", "other": ".bin"}


class DocumentsError(Exception):
    pass


class _Stored(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Translation(_Stored):
    """One passage's original and its English, kept so it is never translated twice.

    The original text lives here and in originals/, and nowhere outside the bundle.
    """

    chunk_id: str
    language: str
    original_text_sha256: str
    original: str
    # As the model wrote it, before numbers-in-words.
    english: str
    deployment: str
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int


class Vector(_Stored):
    """The embedding of one chunk's title and text."""

    chunk_id: str
    text_sha256: str
    deployment: str
    model: str
    input_tokens: int
    vector: tuple[float, ...]


@dataclass(frozen=True)
class Ingested:
    documents: list[Document]
    chunks: list[Chunk]
    translations: list[Translation]
    vectors: list[Vector]
    # What the manifest says about the corpus: counts, links, translations, spend.
    summary: dict[str, Any]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def embedded_text(chunk: Chunk) -> str:
    """What a chunk's vector is the embedding of: its document's title, then its text."""
    return f"{chunk.title}\n\n{chunk.text}"


def _is_english(language: str) -> bool:
    return language.split("-")[0].lower() == "en"


def fetch(url: str) -> tuple[bytes, str] | None:
    """The original and its content type, or None when the source does not answer."""
    try:
        response = httpx.get(
            url, headers={"User-Agent": _USER_AGENT}, follow_redirects=True, timeout=60
        )
    except httpx.HTTPError:
        return None
    if not response.is_success:
        return None
    return response.content, response.headers.get("content-type", "")


def _read_jsonl[R: BaseModel](path: Path, model: type[R]) -> list[R]:
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return [model.model_validate_json(line) for line in lines]
    except ValueError as error:
        raise DocumentsError(f"{path} is not what build wrote: {error}") from error


def _rates() -> dict[str, dict[str, Any]]:
    table = json.loads((resources.files("spatial_ingest") / "rates.json").read_text("utf-8"))
    return {row["deployment"]: row for row in table["rates"]}


def _spend(translations: Sequence[Translation], vectors: Sequence[Vector]) -> dict[str, Any]:
    """Ingestion spend: tokens per purpose and deployment, times a dated rate where one is listed.

    It counts what the bundle's translations and vectors cost to make, whichever
    build made them.
    """
    used: dict[tuple[str, str], list[int]] = {}
    for made in translations:
        entry = used.setdefault(("translation", made.deployment), [0, 0, 0])
        entry[0] += 1
        entry[1] += made.input_tokens
        entry[2] += made.output_tokens
    for vector in vectors:
        entry = used.setdefault(("embedding", vector.deployment), [0, 0, 0])
        entry[0] += 1
        entry[1] += vector.input_tokens
    rates = _rates()
    million = Decimal(1_000_000)
    calls = []
    total = Decimal(0)
    for (purpose, deployment), (count, tokens_in, tokens_out) in sorted(used.items()):
        rate = rates.get(deployment)
        cost = None
        if rate is not None:
            priced = Decimal(str(rate.get("usd_per_million_input", 0))) * tokens_in / million
            priced += Decimal(str(rate.get("usd_per_million_output", 0))) * tokens_out / million
            total += priced
            cost = float(round(priced, 6))
        calls.append(
            {
                "purpose": purpose,
                "deployment": deployment,
                "calls": count,
                "input_tokens": tokens_in,
                "output_tokens": tokens_out,
                "rate": rate,
                "cost_usd": cost,
            }
        )
    return {
        "calls": calls,
        "total_usd": float(round(total, 6)),
        # Deployments with no row in rates.json. Their tokens are above, their cost is not.
        "unpriced": sorted({call["deployment"] for call in calls if call["rate"] is None}),
    }


class _Snapshot:
    """What anchors resolve against: the spaces, types and assets of this snapshot."""

    def __init__(
        self, spaces: Iterable[Space], types: Iterable[AssetType], assets: Iterable[Asset]
    ) -> None:
        self.spaces = {space.id: space for space in spaces}
        self.types = {kind.id: kind for kind in types}
        self.assets = {asset.id: asset for asset in assets}
        self.placed = {asset.type_id for asset in self.assets.values()}
        self.buildings: dict[str, list[str]] = {}
        for space in self.spaces.values():
            if space.tier.value == tiers.BUILDING and space.ref is not None:
                self.buildings.setdefault(space.ref.value, []).append(space.id)

    def resolve(self, spec: AnchorSpec) -> tuple[list[Anchor], str | None]:
        """The anchors a corpus entry stands for, or why it stands for none."""

        def sourced(kind: AnchorKind, anchor_id: str) -> list[Anchor]:
            return [Anchor(kind=kind, id=anchor_id, provenance=ValueProvenance.SOURCED)]

        if spec.building is not None:
            found = sorted(self.buildings.get(spec.building, []))
            if not found:
                return [], f"no building numbered {spec.building}"
            return [
                Anchor(
                    kind=AnchorKind.SPACE,
                    id=space_id,
                    provenance=ValueProvenance.DERIVED,
                    rule=ANCHOR_RULE,
                )
                for space_id in found
            ], None
        if spec.space is not None:
            if spec.space not in self.spaces:
                return [], f"no space {spec.space}"
            return sourced(AnchorKind.SPACE, spec.space), None
        if spec.asset_type is not None:
            if spec.asset_type not in self.types or spec.asset_type not in self.placed:
                return [], f"no asset of type {spec.asset_type} is placed"
            return sourced(AnchorKind.ASSET_TYPE, spec.asset_type), None
        assert spec.asset is not None
        if spec.asset not in self.assets:
            return [], f"no asset {spec.asset}"
        return sourced(AnchorKind.ASSET, spec.asset), None

    def clearance_of(self, anchor: Anchor) -> Persona | None:
        if anchor.kind is AnchorKind.SPACE:
            cleared = self.spaces[anchor.id].clearance
            return cleared.persona if cleared is not None else None
        type_id = (
            anchor.id if anchor.kind is AnchorKind.ASSET_TYPE else self.assets[anchor.id].type_id
        )
        return self.types[type_id].clearance.persona

    def ancestors(self, anchors: Sequence[Anchor]) -> tuple[str, ...]:
        found: set[str] = set()
        for anchor in anchors:
            if anchor.kind is AnchorKind.SPACE:
                found.add(anchor.id)
                found.update(step.id for step in self.spaces[anchor.id].ancestor_path)
            elif anchor.kind is AnchorKind.ASSET:
                found.update(step.id for step in self.assets[anchor.id].ancestor_path)
        return tuple(sorted(found))


def _original_name(source_id: str, kind: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", source_id).strip("_") or "source"
    return f"{safe}-{sha256(source_id)[:8]}{_EXTENSIONS[kind]}"


def ingest(
    *,
    corpus: Corpus | None,
    bundle: Path,
    spaces: Iterable[Space],
    types: Iterable[AssetType],
    assets: Iterable[Asset],
    models: Callable[[], Models],
    today: str,
    refetch: bool = False,
) -> Ingested:
    """The corpus as documents and chunks for this snapshot.

    ``models`` is only called once there is a passage to translate or a chunk to
    embed that the bundle does not already hold.
    """
    directory = bundle / DIR
    originals = directory / ORIGINALS_DIR
    snapshot = _Snapshot(spaces, types, assets)

    fetched_path = directory / FETCHED_FILE
    fetched: dict[str, dict[str, Any]] = (
        json.loads(fetched_path.read_text(encoding="utf-8")) if fetched_path.is_file() else {}
    )
    # Whichever deployment made them: the bundle is the record of what was said,
    # and a rebuild pointed at another deployment should not quietly redo it all.
    # A translation made under another prompt version is made again.
    known_translations = {
        (made.original_text_sha256, made.language): made
        for made in _read_jsonl(bundle / TRANSLATIONS_PART, Translation)
        if made.prompt_version == PROMPT_VERSION
    }
    known_vectors = {
        vector.text_sha256: vector for vector in _read_jsonl(bundle / VECTORS_PART, Vector)
    }
    client: list[Models] = []

    def model() -> Models:
        if not client:
            client.append(models())
        return client[0]

    documents: list[Document] = []
    chunks: list[Chunk] = []
    translations: list[Translation] = []
    vectors: list[Vector] = []
    skipped: list[dict[str, str]] = []
    dropped: list[dict[str, str]] = []
    links: list[dict[str, str]] = []
    translated: list[dict[str, Any]] = []
    kept: dict[str, dict[str, Any]] = {}

    for source in sorted(corpus.sources if corpus else (), key=lambda entry: entry.id):
        anchors: dict[tuple[AnchorKind, str], Anchor] = {}
        for spec in source.anchors:
            resolved, reason = snapshot.resolve(spec)
            if reason is not None:
                dropped.append({"document": source.id, "reason": reason})
            for anchor in resolved:
                anchors.setdefault((anchor.kind, anchor.id), anchor)
        if not anchors:
            skipped.append({"id": source.id, "reason": "none of its anchors is in the snapshot"})
            continue
        anchored = tuple(anchors[key] for key in sorted(anchors))

        entry = _original(source, fetched.get(source.id), originals, today, refetch)
        kept[source.id] = entry
        link = LinkCheck.model_validate(entry["link"])
        links.append({"document": source.id, "url": source.url, **link.model_dump(mode="json")})
        if entry["file"] is None:
            skipped.append(
                {"id": source.id, "reason": "its source is offline and the bundle has no original"}
            )
            continue
        content = (originals / entry["file"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"]:
            raise DocumentsError(
                f"{originals / entry['file']} no longer matches the hash it was stored with"
            )
        try:
            assert corpus is not None
            text = passages.text_of(content, entry["content_type"], corpus.html_content_between)
        except passages.UnreadableSource as error:
            skipped.append({"id": source.id, "reason": str(error)})
            continue
        cut = passages.chunk(text)
        if not cut:
            skipped.append({"id": source.id, "reason": "its original holds no text"})
            continue

        english = _is_english(source.language)
        cleared = clearance.document_clearance(
            source.kind, [(anchor.kind, snapshot.clearance_of(anchor)) for anchor in anchored]
        )
        dated = passages.last_change(content, entry["content_type"])
        served: TranslationSource | None = None
        for ordinal, original in enumerate(cut):
            chunk_id = f"{source.id}#{ordinal}"
            original_hash = sha256(original)
            body = original
            if not english:
                made = known_translations.get((original_hash, source.language))
                if made is None:
                    answer = model().translate(original, source.language)
                    made = Translation(
                        chunk_id=chunk_id,
                        language=source.language,
                        original_text_sha256=original_hash,
                        original=original,
                        english=answer.text,
                        deployment=answer.deployment,
                        model=answer.model,
                        prompt_version=PROMPT_VERSION,
                        input_tokens=answer.input_tokens,
                        output_tokens=answer.output_tokens,
                    )
                made = made.model_copy(update={"chunk_id": chunk_id})
                translations.append(made)
                served = TranslationSource(
                    deployment=made.deployment,
                    model=made.model,
                    prompt_version=made.prompt_version,
                )
                body = made.english
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=source.id,
                    ordinal=ordinal,
                    kind=source.kind,
                    title=source.title,
                    text=numbers.in_words(body),
                    language=source.language,
                    url=source.url,
                    original_text_sha256=original_hash,
                    translated=not english,
                    space_ids=_ids(anchored, AnchorKind.SPACE),
                    asset_type_ids=_ids(anchored, AnchorKind.ASSET_TYPE),
                    asset_ids=_ids(anchored, AnchorKind.ASSET),
                    ancestor_ids=snapshot.ancestors(anchored),
                    clearance_class=cleared,
                    document_date=dated,
                    link_status=link.status,
                    link_checked_on=link.checked_on,
                )
            )
        if served is not None:
            translated.append(
                {"document": source.id, "chunks": len(cut), **served.model_dump(mode="json")}
            )
        documents.append(
            Document(
                id=source.id,
                record_provenance=RecordProvenance.SOURCED,
                kind=source.kind,
                title=source.title,
                url=source.url,
                language=source.language,
                original_sha256=entry["sha256"],
                anchors=anchored,
                clearance_class=cleared,
                date=Value.derived(dated, passages.DATE_RULE) if dated else None,
                link=link,
                translation=served,
                chunks=len(cut),
            )
        )

    for chunk in chunks:
        text_hash = sha256(embedded_text(chunk))
        stored = known_vectors.get(text_hash)
        if stored is None:
            embedded = model().embed(embedded_text(chunk))
            stored = Vector(
                chunk_id=chunk.id,
                text_sha256=text_hash,
                deployment=embedded.deployment,
                model=embedded.model,
                input_tokens=embedded.input_tokens,
                vector=embedded.vector,
            )
        vectors.append(stored.model_copy(update={"chunk_id": chunk.id}))
    sizes = sorted({len(vector.vector) for vector in vectors})
    if len(sizes) > 1:
        raise DocumentsError(f"the chunk vectors come in more than one size: {sizes}")
    served_by = sorted({(vector.deployment, vector.model) for vector in vectors})
    if len(served_by) > 1:
        raise DocumentsError(
            f"the chunk vectors come from more than one embedding deployment: {served_by}. "
            f"Delete {bundle / VECTORS_PART} to embed them all again."
        )

    if corpus is not None:
        directory.mkdir(parents=True, exist_ok=True)
        fetched_path.write_text(
            json.dumps(kept, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    summary = {
        "documents": len(documents),
        "chunks": len(chunks),
        "by_kind": {
            kind.value: sum(1 for document in documents if document.kind is kind)
            for kind in DocumentKind
        },
        "chunks_by_clearance_class": {
            persona.value: sum(1 for chunk in chunks if chunk.clearance_class is persona)
            for persona in PERSONAS
        },
        "dated": sum(1 for document in documents if document.date is not None),
        "undated": sum(1 for document in documents if document.date is None),
        # Sources the corpus lists and this snapshot does not hold, each with why.
        "skipped": skipped,
        "dropped_anchors": dropped,
        # Every source asked for, with whether it answered and the day build asked.
        "links": links,
        "offline": sum(1 for link_ in links if link_["status"] == LinkStatus.OFFLINE.value),
        "translations": translated,
        "embedding": (
            {"deployment": served_by[0][0], "model": served_by[0][1], "dimensions": sizes[0]}
            if vectors
            else None
        ),
        "originals": {
            source_id: entry["sha256"] for source_id, entry in sorted(kept.items()) if entry["file"]
        },
        "spend": _spend(translations, vectors),
    }
    return Ingested(documents, chunks, translations, vectors, summary)


def _ids(anchors: Sequence[Anchor], kind: AnchorKind) -> tuple[str, ...]:
    return tuple(anchor.id for anchor in anchors if anchor.kind is kind)


def _original(
    source: Source,
    stored: dict[str, Any] | None,
    originals: Path,
    today: str,
    refetch: bool,
) -> dict[str, Any]:
    """Where a source's original is in the bundle, fetching it when it has to.

    One GET is both the fetch and the link check. A source that does not answer
    keeps the original the bundle already has, and is recorded as offline.
    """
    if stored is not None and stored.get("url") != source.url:
        stored = None
    if stored is not None and not refetch:
        return stored
    have = stored if stored is not None and stored.get("file") else None
    got = fetch(source.url)
    if got is None:
        link = {"status": LinkStatus.OFFLINE.value, "checked_on": today}
        if have is not None and (originals / have["file"]).is_file():
            return {**have, "link": link}
        return {"url": source.url, "file": None, "sha256": None, "content_type": None, "link": link}
    content, content_type = got
    try:
        kind = passages.kind_of(content, content_type)
    except passages.UnreadableSource:
        kind = "other"
    name = _original_name(source.id, kind)
    originals.mkdir(parents=True, exist_ok=True)
    (originals / name).write_bytes(content)
    return {
        "url": source.url,
        "file": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "content_type": content_type,
        "link": {"status": LinkStatus.ONLINE.value, "checked_on": today},
    }
