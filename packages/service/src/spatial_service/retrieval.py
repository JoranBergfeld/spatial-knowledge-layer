"""Document retrieval: the passages anchored to what an earlier step found.

The tool takes anchors from an earlier step and a query, and searches the
snapshot's index with hybrid ranking. Like every tool it is handed a venue cut
down to one persona, and the anchors are built from that venue alone:

    from a place      the place and every visible space under it, the visible
                      assets under it, the types of those assets, and the spaces
                      above it up to its building
    from assets       those assets and their types
    from one asset    the same, for the one a follow-up points at: it and its type

The filter the index gets is then: clearance class within the persona's, and at
least one of the chunk's anchors among those ids. That is
the spec's filter with its last two clauses folded into one. "Scope matches, or
an asset type is visible in scope" and "an anchor is in the visible set" are the
same test once the scope is spelled out as the visible ids in it. It is stricter
in one corner, on purpose: a document tied to the scope only through something
the persona cannot see does not come back, even if it has a visible anchor
elsewhere.

A type is only ever in the venue through a visible asset of it, so a manual
anchored to a type is retrievable through such an asset and no other way.

The whole filter runs in the index, before ranking and the cut to the top few.
Filtering afterwards would let a hidden document take a slot and shorten the
answer, which tells. The host still checks every hit against the same filter and
refuses what does not pass, in case an index ever gets it wrong.

Rough edge: for a place high in the tree the id lists get long (every visible
space and asset under it). Fine on the mini campus, not measured on the real one.

Summaries. The index also holds one summary per document (role "summary"), with
its document's anchors and clearance, so the same filter admits it. The search
covers both roles, and a summary that hits is a hit for its document: each of the
document's passages gets the summary's rank added to its own, RRF style. If none
of that document's passages made the candidates, one more search fetches them
(same filter, that document, passages only). Then every summary is dropped. Only
a chunk that is evidence leaves this module: the passages list, the ids a step
reports and the trace are built from that and nothing else. The trace says a
document was found through its summary and never shows the summary.

Passages and facts are separate lists from here to composition. Nothing here
scores a fact. The fusion above is over document search results only.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from spatial_core.records import PERSONAS, Chunk, ChunkRole, Persona, Space
from spatial_core.search import RRF_K, Filter, Hit, Query, SearchQueries
from spatial_service.assets import ScopeResult
from spatial_service.places import BUILDING, Venue, tier

# How many passages a retrieval returns.
TOP = 5
# How many chunks, of either role, the index is asked for per passage returned.
CANDIDATES_PER_PASSAGE = 4

ANCHORED_BY_PLACE = "places"
ANCHORED_BY_ASSETS = "assets"


def check_query(text: str) -> list[str]:
    """Why ``text`` is no query. Reads no venue, so no persona."""
    if not re.search(r"\w", text):
        return [f"{text!r} has nothing to search for"]
    return []


@dataclass(frozen=True)
class Anchors:
    """The ids a retrieval may match a chunk's anchors against. All visible to the persona."""

    # What the earlier step produced: "places" or "assets".
    by: str
    # The place asked about, or the scope the assets were found under.
    scope: str | None
    space_ids: frozenset[str] = frozenset()
    asset_ids: frozenset[str] = frozenset()
    asset_type_ids: frozenset[str] = frozenset()


def anchors_of_place(venue: Venue, place: str) -> Anchors:
    """A place as anchors: itself, what the persona can see under it, and up to its building."""
    under = venue.assets_under.get(place, [])
    return Anchors(
        by=ANCHORED_BY_PLACE,
        scope=place,
        space_ids=frozenset(
            s.id
            for s in venue.spaces.values()
            if s.id == place or any(a.id == place for a in s.ancestor_path)
        )
        | _up_to_building(venue, venue.spaces[place]),
        asset_ids=frozenset(asset.id for asset in under),
        asset_type_ids=frozenset(asset.type_id for asset in under) & venue.asset_types.keys(),
    )


def _up_to_building(venue: Venue, space: Space) -> frozenset[str]:
    """The spaces between a place and its building, the building included. None past it.

    Only the spaces themselves, not what is in them: a building's page, yes, but not
    the manual of every light on the other floors.
    """
    above = venue.ancestors(space)
    if not any(tier(a) == BUILDING for a in above):
        return frozenset()
    upto = next(i for i, a in enumerate(above) if tier(a) == BUILDING)
    return frozenset(a.id for a in above[: upto + 1])


def anchors_of_assets(venue: Venue, found: ScopeResult) -> Anchors:
    """The assets a step found as anchors: each of them, and their types."""
    return Anchors(
        by=ANCHORED_BY_ASSETS,
        scope=found.scope,
        asset_ids=frozenset(found.ids),
        asset_type_ids=frozenset(venue.assets[i].type_id for i in found.ids)
        & venue.asset_types.keys(),
    )


def document_filter(persona: Persona, anchors: Anchors) -> Filter:
    """The retrieval filter for one persona. Both roles: summaries rank, passages return."""
    return Filter(
        clearance_classes=tuple(p for p in PERSONAS if p.rank <= persona.rank),
        space_ids=anchors.space_ids,
        asset_type_ids=anchors.asset_type_ids,
        asset_ids=anchors.asset_ids,
    )


@dataclass(frozen=True)
class Embedded:
    vector: tuple[float, ...]
    deployment: str
    input_tokens: int | None


class Embedder(Protocol):
    def embed(self, deployment: str, text: str) -> Embedded:
        """The query's embedding, from the deployment that embedded the chunks."""


class RetrievalUnavailable(Exception):
    """The index or the embedding deployment could not be reached. The turn answers nothing."""

    purpose = "retrieval"


@dataclass(frozen=True)
class Passage:
    """One retrieved chunk, as composition refers to it."""

    ref: str
    rank: int
    chunk: Chunk
    # Whether its document's summary matched the query better than the passage did.
    via_summary: bool = False


@dataclass(frozen=True)
class Retrieval:
    query: str
    anchors: Anchors
    filter: Filter
    passages: tuple[Passage, ...] = ()
    index: str | None = None
    semantic_ranker: bool = False
    top: int = TOP
    # Hits the index returned that the filter does not pass. Should always be zero.
    refused: int = 0
    embedding: Embedded | None = None


class Documents(Protocol):
    def retrieve(self, venue: Venue, anchors: Anchors, query: str) -> Retrieval: ...


def _cut(embedded: Embedded, dimensions: int | None) -> Embedded:
    """The vector cut to the index's size and normalised again: Matryoshka's cut.

    A vector already that size is left alone, and so is one shorter, which the
    index refuses on its own.
    """
    if dimensions is None or len(embedded.vector) <= dimensions:
        return embedded
    head = embedded.vector[:dimensions]
    norm = math.sqrt(sum(value * value for value in head)) or 1.0
    return replace(embedded, vector=tuple(value / norm for value in head))


class Retriever:
    """The retrieval tool, bound by the host to an index and an embedding client."""

    def __init__(
        self,
        index: SearchQueries,
        embedder: Embedder | None = None,
        *,
        semantic_ranker: bool = False,
        top: int = TOP,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.semantic_ranker = semantic_ranker
        self.top = top

    def retrieve(self, venue: Venue, anchors: Anchors, query: str) -> Retrieval:
        chosen = document_filter(venue.persona, anchors)
        ref = venue.index
        result = Retrieval(
            query=query,
            anchors=anchors,
            filter=chosen,
            index=ref.name if ref else None,
            semantic_ranker=self.semantic_ranker,
            top=self.top,
        )
        # No index, or nothing to anchor to: nothing is searched, so nothing is embedded.
        if ref is None or not chosen.anchored:
            return result
        embedded = None
        if self.embedder is not None and ref.embedding_deployment is not None:
            embedded = self.embedder.embed(ref.embedding_deployment, ref.query_prompt + query)
            embedded = _cut(embedded, ref.dimensions)
        asked = Query(
            text=query,
            filter=chosen,
            vector=embedded.vector if embedded else None,
            top=self.top * CANDIDATES_PER_PASSAGE,
            semantic_ranker=self.semantic_ranker,
        )
        hits = self.index.search(ref.name, asked)
        passed = [hit for hit in hits if chosen.matches(hit.chunk)]
        refused = len(hits) - len(passed)

        # A summary's rank is its document's. The best one, if an index held two.
        by_summary: dict[str, int] = {}
        for hit in passed:
            if not hit.chunk.is_evidence:
                by_summary.setdefault(hit.chunk.document_id, hit.rank)
        own = {hit.chunk.id: hit for hit in passed if hit.chunk.is_evidence}
        missing = by_summary.keys() - {hit.chunk.document_id for hit in own.values()}
        fetched: dict[str, Hit] = {}
        if missing:
            only = replace(chosen, roles=(ChunkRole.PASSAGE,), document_ids=frozenset(missing))
            # A plain fetch, not a search: these passages did not match the query themselves.
            more = self.index.search(ref.name, Query(text=None, filter=only, top=asked.top))
            fetched = {hit.chunk.id: hit for hit in more if only.matches(hit.chunk)}
            refused += len(more) - len(fetched)

        def fused(chunk: Chunk) -> float:
            score = 0.0
            if chunk.id in own:
                score += 1 / (RRF_K + own[chunk.id].rank)
            if chunk.document_id in by_summary:
                score += 1 / (RRF_K + by_summary[chunk.document_id])
            return score

        def order(chunk: Chunk) -> tuple[float, int, str]:
            # Passages fetched for a summary have no rank of their own: after those that do.
            rank = own[chunk.id].rank if chunk.id in own else len(passed) + fetched[chunk.id].rank
            return (-fused(chunk), rank, chunk.id)

        def lifted(chunk: Chunk) -> bool:
            # Its summary matched better than it did itself, or it did not match at all.
            summary = by_summary.get(chunk.document_id)
            return summary is not None and (chunk.id not in own or summary < own[chunk.id].rank)

        # Evidence only, whatever came back. A summary never gets past this line.
        found = (*own.values(), *fetched.values())
        evidence = [hit.chunk for hit in found if hit.chunk.is_evidence]
        best = sorted(evidence, key=order)[: self.top]
        return replace(
            result,
            passages=tuple(
                Passage(
                    ref=f"p{number}",
                    rank=number,
                    chunk=chunk,
                    via_summary=lifted(chunk),
                )
                for number, chunk in enumerate(best, start=1)
            ),
            refused=refused,
            embedding=embedded,
        )


# --- what the trace says of it -------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RetrievedDocument(_Model):
    """One retrieved passage's document, for the trace. Its date is here and nowhere else."""

    ref: str
    rank: int
    chunk_id: str
    document_id: str
    kind: str
    title: str
    clearance_class: Persona
    # ISO, or None for undated.
    document_date: str | None
    translated: bool
    language: str
    # None when the source was offline at build. Then it is cited unlinked.
    url: str | None
    link_status: str
    link_checked_on: str
    # The anchors it was retrieved through. Its other anchors are not listed:
    # the persona may not be able to see them.
    space_ids: list[str]
    asset_type_ids: list[str]
    asset_ids: list[str]
    # Its document's summary ranked above the passage itself. The summary is not shown.
    via_summary: bool = False


class DocumentsTrace(_Model):
    step: str
    query: str
    index: str | None
    ranking: str = "hybrid"
    semantic_ranker: bool
    top: int
    # What the anchors came from, and the scope they were found under.
    anchored_by: str
    scope: str | None
    # How many ids the filter named, and the types by id.
    spaces: int
    assets: int
    asset_types: list[str]
    clearance_classes: list[Persona]
    retrieved: list[RetrievedDocument]
    refused: int = 0


class DroppedSegment(_Model):
    """A segment the provenance check dropped, so a shortened answer explains itself."""

    stage: str = "provenance"
    process: str = "spatial-service"
    # Its place in what the composer wrote, from 1.
    segment: int
    reasons: list[str]


class EmbeddingCall(_Model):
    """One query embedding's token usage, and the deployment that served it."""

    deployment: str
    input_tokens: int | None


def link(chunk: Chunk) -> str | None:
    """Where the original lives, unless build found it offline."""
    return chunk.url if chunk.link_status.value == "online" else None


def trace(step: str, found: Retrieval) -> DocumentsTrace:
    chosen = found.filter
    return DocumentsTrace(
        step=step,
        query=found.query,
        index=found.index,
        semantic_ranker=found.semantic_ranker,
        top=found.top,
        anchored_by=found.anchors.by,
        scope=found.anchors.scope,
        spaces=len(chosen.space_ids),
        assets=len(chosen.asset_ids),
        asset_types=sorted(chosen.asset_type_ids),
        clearance_classes=list(chosen.clearance_classes),
        retrieved=[
            RetrievedDocument(
                ref=passage.ref,
                rank=passage.rank,
                chunk_id=passage.chunk.id,
                document_id=passage.chunk.document_id,
                kind=passage.chunk.kind.value,
                title=passage.chunk.title,
                clearance_class=passage.chunk.clearance_class,
                document_date=passage.chunk.document_date,
                translated=passage.chunk.translated,
                language=passage.chunk.language,
                url=link(passage.chunk),
                link_status=passage.chunk.link_status.value,
                link_checked_on=passage.chunk.link_checked_on,
                space_ids=sorted(chosen.space_ids.intersection(passage.chunk.space_ids)),
                asset_type_ids=sorted(
                    chosen.asset_type_ids.intersection(passage.chunk.asset_type_ids)
                ),
                asset_ids=sorted(chosen.asset_ids.intersection(passage.chunk.asset_ids)),
                via_summary=passage.via_summary,
            )
            for passage in found.passages
        ],
        refused=found.refused,
    )
