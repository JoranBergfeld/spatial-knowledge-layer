"""The service's side of the search index: querying it, and embedding the query.

Postgres runs it as one statement: the filter first, then a keyword ranking over
the words column and an exact cosine ranking over the vectors, fused with RRF.
The keyword half is ts_rank over English stems, which is not BM25: it has no
idea how rare a word is. There is no semantic ranker, so that switch does nothing.

Azure AI Search runs the keyword search and the vector search and fuses them with
RRF itself, when a request carries both. The filter goes in the same request, so
it cuts the chunks down before anything is ranked. The semantic ranker is one
more field on the request, and off unless the configuration turns it on.

The query is embedded by whatever embedded the snapshot's chunks: the Azure
deployment the manifest names, or, when the environment sets
SPATIAL_EMBEDDING_BASE_URL and the manifest names SPATIAL_EMBEDDING_MODEL, that
OpenAI-compatible /v1 server. A snapshot from before the switch still embeds with
Azure, so promoting it back is still a rollback.

Which index client the service gets comes from the environment, like ingestion's:
Azure AI Search when AZURE_SEARCH_ENDPOINT is set, else the stand-in directory
named by SPATIAL_SEARCH_DIR, else the tables in the service's own database.

The Azure client was only ever run against the bench's service. The request shape
is from the REST reference (api-version 2024-07-01).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import httpx
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from spatial_core.records import Chunk
from spatial_core.search import (
    RRF_K,
    SEARCH_DIR_ENV,
    SEMANTIC_CONFIGURATION,
    Filter,
    Hit,
    LocalIndex,
    Query,
    SearchQueries,
)
from spatial_core.stores import SEARCH_SCHEMA
from spatial_service.retrieval import Embedded, Embedder, RetrievalUnavailable

ENDPOINT_ENV = "AZURE_SEARCH_ENDPOINT"
API_KEY_ENV = "AZURE_SEARCH_API_KEY"
API_VERSION = "2024-07-01"
# The same names build reads (spatial_ingest.models), for the same server.
EMBEDDING_BASE_URL_ENV = "SPATIAL_EMBEDDING_BASE_URL"
EMBEDDING_API_KEY_ENV = "SPATIAL_EMBEDDING_API_KEY"
EMBEDDING_MODEL_ENV = "SPATIAL_EMBEDDING_MODEL"
VECTOR_FIELD = "vector"
# How many neighbours the vector half hands to the fusion.
_NEIGHBOURS = 50
# How many keyword matches go to the fusion, at most. Azure's default for hybrid.
_KEYWORD_RECALL = 1000
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
    clauses = [
        _in("role", [role.value for role in chosen.roles]),
        _in("clearance_class", [persona.value for persona in chosen.clearance_classes]),
        f"({' or '.join(anchors)})",
    ]
    if chosen.document_ids:
        clauses.append(_in("document_id", chosen.document_ids))
    return " and ".join(clauses)


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
            # "*" matches everything: a fetch by filter, with nothing to rank by.
            "search": query.text if query.text is not None else "*",
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
        elif query.text is None:
            body["orderby"] = "id asc"
        if query.semantic_ranker and query.text is not None:
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


def where(chosen: Filter) -> tuple[sql.Composable, dict[str, Any]]:
    """A Filter as a WHERE clause over search.<index> and its parameters. As Filter.matches."""
    clauses: list[sql.Composable] = [
        sql.SQL("role = ANY(%(roles)s)"),
        sql.SQL("clearance_class = ANY(%(classes)s)"),
        # An empty array overlaps nothing, as an empty set intersects nothing.
        sql.SQL(
            "(space_ids && %(spaces)s::text[] OR asset_type_ids && %(types)s::text[]"
            " OR asset_ids && %(assets)s::text[])"
        ),
    ]
    if chosen.document_ids:
        clauses.append(sql.SQL("document_id = ANY(%(documents)s)"))
    return sql.SQL(" AND ").join(clauses), {
        "roles": [role.value for role in chosen.roles],
        "classes": [persona.value for persona in chosen.clearance_classes],
        "spaces": sorted(chosen.space_ids),
        "types": sorted(chosen.asset_type_ids),
        "assets": sorted(chosen.asset_ids),
        "documents": sorted(chosen.document_ids),
    }


def keywords(text: str) -> str:
    """The words of a question as a tsquery that any one of them matches, as Azure's "any" does."""
    # Letters and digits only, so nothing in a question can be tsquery syntax.
    words = dict.fromkeys(re.findall(r"[^\W_]+", text.casefold()))
    return " | ".join(words)


class PostgresSearch:
    """Searches the search.<name> tables load wrote, through the service's reader login."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url

    def search(self, name: str, query: Query) -> list[Hit]:
        if not query.filter.anchored:
            return []
        table = sql.Identifier(SEARCH_SCHEMA, name)
        passes, params = where(query.filter)
        params |= {"top": query.top, "rrf_k": RRF_K}
        columns = sql.SQL(", ").join(sql.Identifier("chunk", f) for f in Chunk.model_fields)
        if query.text is None and query.vector is None:
            statement = sql.SQL(
                "SELECT {columns}, 1.0 AS score FROM {table} chunk WHERE {passes}"
                " ORDER BY id LIMIT %(top)s"
            ).format(columns=columns, table=table, passes=passes)
            return self._hits(name, statement, params)
        rankings: list[sql.Composable] = []
        if query.text is not None and (words := keywords(query.text)):
            params |= {"words": words, "recall": _KEYWORD_RECALL}
            rankings.append(
                sql.SQL(
                    "(SELECT id, row_number() OVER"
                    " (ORDER BY ts_rank(words, q) DESC, id) AS position"
                    " FROM {table}, to_tsquery('english', %(words)s) q"
                    " WHERE words @@ q AND {passes} ORDER BY position LIMIT %(recall)s)"
                ).format(table=table, passes=passes)
            )
        if query.vector is not None:
            params |= {
                "vector": "[" + ",".join(repr(float(x)) for x in query.vector) + "]",
                "neighbours": _NEIGHBOURS,
            }
            rankings.append(
                sql.SQL(
                    "(SELECT id, row_number() OVER"
                    " (ORDER BY embedding <=> %(vector)s::vector, id) AS position"
                    " FROM {table} WHERE {passes} ORDER BY position LIMIT %(neighbours)s)"
                ).format(table=table, passes=passes)
            )
        if not rankings:
            return []
        statement = sql.SQL(
            "SELECT {columns}, fused.score FROM"
            " (SELECT id, sum(1.0 / (%(rrf_k)s + position)) AS score"
            " FROM ({rankings}) ranked GROUP BY id) fused"
            " JOIN {table} chunk USING (id)"
            " ORDER BY fused.score DESC, id LIMIT %(top)s"
        ).format(
            columns=columns,
            rankings=sql.SQL(" UNION ALL ").join(rankings),
            table=table,
        )
        return self._hits(name, statement, params)

    def _hits(self, name: str, statement: sql.Composed, params: dict[str, Any]) -> list[Hit]:
        try:
            with psycopg.connect(
                self._url,
                row_factory=dict_row,
                options="-c default_transaction_read_only=on",
            ) as conn:
                rows = conn.execute(statement, params).fetchall()
            return [
                Hit(
                    chunk=Chunk.model_validate({f: row[f] for f in Chunk.model_fields}),
                    rank=rank,
                    score=float(row["score"]),
                )
                for rank, row in enumerate(rows, start=1)
            ]
        except (psycopg.Error, KeyError, ValueError) as error:
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


class OpenAIEmbedder:
    """Embeds a query on an OpenAI-compatible /v1 server. The deployment is the model's name.

    It asks for the model's native size. The retriever cuts the vector to the
    index's, which for a Matryoshka model is the same vector the server would cut.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 30,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
            transport=transport,
        )

    def embed(self, deployment: str, text: str) -> Embedded:
        try:
            response = self._client.post("/embeddings", json={"model": deployment, "input": [text]})
            response.raise_for_status()
            answer = response.json()
            return Embedded(
                vector=tuple(float(value) for value in answer["data"][0]["embedding"]),
                deployment=deployment,
                input_tokens=int(answer["usage"]["prompt_tokens"]),
            )
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as error:
            raise RetrievalUnavailable(f"embedding on {deployment} failed: {error!r}") from error


class RoutedEmbedder:
    """One embedder for named deployments, another for the rest."""

    def __init__(self, routes: Mapping[str, Embedder], otherwise: Embedder) -> None:
        self._routes = dict(routes)
        self._otherwise = otherwise

    def embed(self, deployment: str, text: str) -> Embedded:
        return self._routes.get(deployment, self._otherwise).embed(deployment, text)


def embedder_from_env(
    env: Mapping[str, str],
    *,
    endpoint: str,
    api_key: str,
    api_version: str,
    timeout_seconds: float = 30,
) -> Embedder:
    """Azure for every deployment, unless the environment names a /v1 server for one model."""
    azure = AzureEmbedder(
        endpoint=endpoint, api_key=api_key, api_version=api_version, timeout_seconds=timeout_seconds
    )
    if not env.get(EMBEDDING_BASE_URL_ENV):
        return azure
    missing = [name for name in (EMBEDDING_API_KEY_ENV, EMBEDDING_MODEL_ENV) if not env.get(name)]
    if missing:
        raise ValueError(f"{EMBEDDING_BASE_URL_ENV} is set, so set {', '.join(missing)} too")
    v1 = OpenAIEmbedder(
        base_url=env[EMBEDDING_BASE_URL_ENV],
        api_key=env[EMBEDDING_API_KEY_ENV],
        timeout_seconds=timeout_seconds,
    )
    return RoutedEmbedder({env[EMBEDDING_MODEL_ENV]: v1}, azure)


def from_env(env: Mapping[str, str], database_url: str) -> SearchQueries:
    """The index the service searches. The service's own database unless told otherwise."""
    if env.get(ENDPOINT_ENV):
        if not env.get(API_KEY_ENV):
            raise ValueError(f"{ENDPOINT_ENV} is set, so set {API_KEY_ENV} too")
        return AzureSearch(endpoint=env[ENDPOINT_ENV], api_key=env[API_KEY_ENV])
    if env.get(SEARCH_DIR_ENV):
        return LocalIndex(Path(env[SEARCH_DIR_ENV]))
    return PostgresSearch(database_url)
