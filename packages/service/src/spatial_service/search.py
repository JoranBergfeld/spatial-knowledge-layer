"""The service's side of the search index: querying it, and embedding the query.

Azure AI Search runs the keyword search and the vector search and fuses them with
RRF itself, when a request carries both. The filter goes in the same request, so
it cuts the chunks down before anything is ranked. The semantic ranker is one
more field on the request, and off unless the configuration turns it on.

Which index client the service gets comes from the environment, like ingestion's:
Azure AI Search when AZURE_SEARCH_ENDPOINT is set, else the stand-in directory
named by SPATIAL_SEARCH_DIR, else none, and then no document is ever retrieved.

Not run against a real Azure service yet. The request shape is from the REST
reference (api-version 2024-07-01) and the tests only check it is what I meant.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import httpx

from spatial_core.records import Chunk
from spatial_core.search import (
    SEARCH_DIR_ENV,
    SEMANTIC_CONFIGURATION,
    Filter,
    Hit,
    LocalIndex,
    Query,
    SearchQueries,
)
from spatial_service.retrieval import Embedded, RetrievalUnavailable

ENDPOINT_ENV = "AZURE_SEARCH_ENDPOINT"
API_KEY_ENV = "AZURE_SEARCH_API_KEY"
API_VERSION = "2024-07-01"
VECTOR_FIELD = "vector"
# How many neighbours the vector half hands to the fusion.
_NEIGHBOURS = 50
# Splits the values of a search.in list. No id has one in it.
_DELIMITER = "|"


def _in(field: str, values: Collection[str]) -> str:
    listed = sorted(values)
    if any(_DELIMITER in value for value in listed):
        raise ValueError(f"a {field} value has a {_DELIMITER!r} in it, which the filter splits on")
    joined = _DELIMITER.join(listed).replace("'", "''")
    return f"search.in({field}, '{joined}', '{_DELIMITER}')"


def odata(chosen: Filter) -> str:
    """A Filter as the OData the index takes. Says what Filter.matches says, no more."""
    anchors = [
        f"{field}/any(x: {_in('x', values)})"
        for field, values in (
            ("space_ids", chosen.space_ids),
            ("asset_type_ids", chosen.asset_type_ids),
            ("asset_ids", chosen.asset_ids),
        )
        if values
    ]
    return " and ".join(
        [
            _in("kind", [kind.value for kind in chosen.kinds]),
            _in("clearance_class", [persona.value for persona in chosen.clearance_classes]),
            f"({' or '.join(anchors)})",
        ]
    )


class AzureSearch:
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        timeout_seconds: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            params={"api-version": API_VERSION},
            headers={"api-key": api_key},
            timeout=timeout_seconds,
            transport=transport,
        )

    def search(self, name: str, query: Query) -> list[Hit]:
        if not query.filter.anchored:
            return []
        body: dict[str, Any] = {
            "search": query.text,
            "searchFields": "title,text",
            "filter": odata(query.filter),
            "select": ",".join(Chunk.model_fields),
            "top": query.top,
        }
        if query.vector is not None:
            body["vectorQueries"] = [
                {
                    "kind": "vector",
                    "vector": list(query.vector),
                    "fields": VECTOR_FIELD,
                    "k": _NEIGHBOURS,
                }
            ]
        if query.semantic_ranker:
            body["queryType"] = "semantic"
            body["semanticConfiguration"] = SEMANTIC_CONFIGURATION
        try:
            response = self._client.post(f"/indexes/{name}/docs/search", json=body)
            response.raise_for_status()
            rows = response.json()["value"]
            return [
                Hit(
                    chunk=Chunk.model_validate({f: row.get(f) for f in Chunk.model_fields}),
                    rank=rank,
                    score=float(row.get("@search.rerankerScore") or row.get("@search.score") or 0),
                )
                for rank, row in enumerate(rows, start=1)
            ]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as error:
            raise RetrievalUnavailable(f"search on {name} failed: {error!r}") from error


class AzureEmbedder:
    """Embeds a query with the deployment the snapshot's chunks were embedded with."""

    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        api_version: str,
        timeout_seconds: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=endpoint.rstrip("/"),
            params={"api-version": api_version},
            headers={"api-key": api_key},
            timeout=timeout_seconds,
            transport=transport,
        )

    def embed(self, deployment: str, text: str) -> Embedded:
        try:
            response = self._client.post(
                f"/openai/deployments/{deployment}/embeddings", json={"input": [text]}
            )
            response.raise_for_status()
            answer = response.json()
            return Embedded(
                vector=tuple(float(value) for value in answer["data"][0]["embedding"]),
                deployment=deployment,
                input_tokens=int(answer["usage"]["prompt_tokens"]),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise RetrievalUnavailable(f"embedding on {deployment} failed: {error!r}") from error


def from_env(env: Mapping[str, str]) -> SearchQueries | None:
    """The index the service searches, or None when the environment names none."""
    if env.get(ENDPOINT_ENV):
        if not env.get(API_KEY_ENV):
            raise ValueError(f"{ENDPOINT_ENV} is set, so set {API_KEY_ENV} too")
        return AzureSearch(endpoint=env[ENDPOINT_ENV], api_key=env[API_KEY_ENV])
    if env.get(SEARCH_DIR_ENV):
        return LocalIndex(Path(env[SEARCH_DIR_ENV]))
    return None
