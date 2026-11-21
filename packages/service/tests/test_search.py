"""The Azure AI Search and embedding clients, against a transport that answers from a script.

Nothing else exercises these: every other test searches the stand-in index. So this
only pins the request the service sends and how it reads the answer. Whether Azure
agrees is not something a test here can say.
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import httpx
import pytest

import fixture
import papers
from spatial_core.records import ChunkRole, Persona
from spatial_core.search import Filter, IndexRef, Query
from spatial_service.retrieval import Anchors, Embedded, RetrievalUnavailable, Retriever
from spatial_service.search import (
    AzureEmbedder,
    AzureSearch,
    OpenAIEmbedder,
    RoutedEmbedder,
    embedder_from_env,
    odata,
)

FILTER = Filter(
    clearance_classes=(Persona.VISITOR, Persona.STAFF),
    asset_type_ids=frozenset({"seed:type:b", "seed:type:a"}),
    asset_ids=frozenset({"osm:asset:node/1"}),
)


def serving(answer: dict[str, Any], seen: list[httpx.Request], status: int = 200) -> Any:
    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=answer)

    return httpx.MockTransport(handle)


def test_the_filter_is_rendered_clause_for_clause() -> None:
    assert odata(FILTER) == (
        "search.in(role, 'passage|summary', '|')"
        " and search.in(clearance_class, 'staff|visitor', '|')"
        " and (asset_type_ids/any(x: search.in(x, 'seed:type:a|seed:type:b', '|'))"
        " or asset_ids/any(x: search.in(x, 'osm:asset:node/1', '|')))"
    )


def test_a_fetch_of_one_documents_passages_narrows_role_and_document() -> None:
    only = Filter(
        clearance_classes=(Persona.VISITOR,),
        space_ids=frozenset({"osm:way/1"}),
        roles=(ChunkRole.PASSAGE,),
        document_ids=frozenset({"manual:q'4"}),
    )

    assert odata(only) == (
        "search.in(role, 'passage', '|')"
        " and search.in(clearance_class, 'visitor', '|')"
        " and (space_ids/any(x: search.in(x, 'osm:way/1', '|')))"
        " and search.in(document_id, 'manual:q''4', '|')"
    )


def test_an_id_with_the_delimiter_in_it_is_refused_not_split() -> None:
    with pytest.raises(ValueError, match="splits on"):
        odata(Filter(clearance_classes=(Persona.VISITOR,), space_ids=frozenset({"a|b"})))


@pytest.mark.parametrize("semantic", [False, True])
def test_a_search_sends_words_vector_and_filter_and_the_ranker_only_when_on(
    semantic: bool,
) -> None:
    [chunk] = [c for c in papers.passages() if c.id == "manual:b1#0"]
    row = {**chunk.model_dump(mode="json"), "key": "abc", "@search.score": 0.03}
    seen: list[httpx.Request] = []
    index = AzureSearch(
        endpoint="https://search.example/", api_key="k", transport=serving({"value": [row]}, seen)
    )

    query = Query("lamp", FILTER, vector=(0.1, 0.2), top=20, semantic_ranker=semantic)
    hits = index.search("spatial-mini-20260915", query)

    [request] = seen
    assert request.url.path == "/indexes/spatial-mini-20260915/docs/search"
    assert request.url.params["api-version"] == "2024-07-01"
    assert request.headers["api-key"] == "k"
    body = json.loads(request.content)
    assert body["search"] == "lamp"
    assert body["filter"] == odata(FILTER)
    assert body["top"] == 20
    assert body["vectorQueries"] == [
        {"kind": "vector", "vector": [0.1, 0.2], "fields": "vector", "k": 50}
    ]
    if semantic:
        assert (body["queryType"], body["semanticConfiguration"]) == ("semantic", "chunks")
    else:
        assert "queryType" not in body and "semanticConfiguration" not in body
    assert [(hit.chunk, hit.rank) for hit in hits] == [(chunk, 1)]


def test_a_search_the_service_cannot_make_is_retrieval_unavailable() -> None:
    index = AzureSearch(
        endpoint="https://search.example", api_key="k", transport=serving({}, [], status=503)
    )

    with pytest.raises(RetrievalUnavailable):
        index.search("spatial-mini-20260915", Query("lamp", FILTER))


def test_a_query_is_embedded_by_the_deployment_named() -> None:
    seen: list[httpx.Request] = []
    answer = {"data": [{"embedding": [0.5, 0.25]}], "usage": {"prompt_tokens": 3}}
    embedder = AzureEmbedder(
        endpoint="https://openai.example",
        api_key="k",
        api_version="2024-10-21",
        transport=serving(answer, seen),
    )

    embedded = embedder.embed("text-embedding-3-small", "lamp")

    [request] = seen
    assert request.url.path == "/openai/deployments/text-embedding-3-small/embeddings"
    assert json.loads(request.content) == {"input": ["lamp"]}
    assert (embedded.vector, embedded.deployment, embedded.input_tokens) == (
        (0.5, 0.25),
        "text-embedding-3-small",
        3,
    )


def test_a_query_is_embedded_on_a_v1_server_by_model_name() -> None:
    seen: list[httpx.Request] = []
    answer = {"data": [{"embedding": [0.6, 0.8]}], "usage": {"prompt_tokens": 4}}
    embedder = OpenAIEmbedder(
        base_url="http://spark.example:8103/v1/", api_key="k", transport=serving(answer, seen)
    )

    embedded = embedder.embed("google/embeddinggemma-2", "task: search result | query: lamp")

    [request] = seen
    assert request.url.path == "/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer k"
    assert json.loads(request.content) == {
        "model": "google/embeddinggemma-2",
        "input": ["task: search result | query: lamp"],
    }
    assert (embedded.vector, embedded.deployment) == ((0.6, 0.8), "google/embeddinggemma-2")


def test_only_the_v1_model_goes_to_the_v1_server_so_an_older_snapshot_still_embeds_on_azure() -> (
    None
):
    env = {
        "SPATIAL_EMBEDDING_BASE_URL": "http://spark.example:8103/v1",
        "SPATIAL_EMBEDDING_API_KEY": "k",
        "SPATIAL_EMBEDDING_MODEL": "google/embeddinggemma-2",
    }
    azure = papers.Embedder()
    v1 = papers.Embedder()
    routed = RoutedEmbedder({"google/embeddinggemma-2": v1}, azure)

    routed.embed("google/embeddinggemma-2", "lamp")
    routed.embed("text-embedding-3-small", "lamp")

    assert v1.asked == [("google/embeddinggemma-2", "lamp")]
    assert azure.asked == [("text-embedding-3-small", "lamp")]
    azure_only = embedder_from_env({}, endpoint="https://o.example", api_key="k", api_version="v")
    assert isinstance(azure_only, AzureEmbedder)
    both = embedder_from_env(env, endpoint="https://o.example", api_key="k", api_version="v")
    assert isinstance(both, RoutedEmbedder)
    with pytest.raises(ValueError, match="SPATIAL_EMBEDDING_MODEL"):
        embedder_from_env(
            {k: v for k, v in env.items() if k != "SPATIAL_EMBEDDING_MODEL"},
            endpoint="https://o.example",
            api_key="k",
            api_version="v",
        )


class Longer(papers.Embedder):
    """A model whose native vector is twice the index's: the index holds a Matryoshka cut."""

    def embed(self, deployment: str, text: str) -> Embedded:
        full = super().embed(deployment, text)
        return replace(full, vector=tuple(2 * x for x in full.vector) + (9.0,) * papers.DIMENSIONS)


def test_the_query_gets_the_snapshots_prompt_and_is_cut_to_the_index_size() -> None:
    store = papers.store()
    store.index = IndexRef(
        papers.INDEX, papers.EMBEDDING, query_prompt="task: q | ", dimensions=papers.DIMENSIONS
    )
    venue = store.venue(fixture.SNAPSHOT, Persona.VISITOR)
    embedder = Longer()
    index = papers.Recording(papers.index())
    anchors = Anchors(by="places", scope=None, space_ids=frozenset(venue.spaces))

    Retriever(index, embedder).retrieve(venue, anchors, "lamp")

    assert embedder.asked == [(papers.EMBEDDING, "task: q | lamp")]
    [query] = index.queries
    assert query.text == "lamp"
    assert query.vector is not None
    assert query.vector == pytest.approx(papers.vector("task: q | lamp"))
