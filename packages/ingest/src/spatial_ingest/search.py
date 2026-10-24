"""Ingestion's side of the search index: writing a snapshot's chunks to Postgres or Azure.

One index per snapshot, hybrid: the text and title are searchable by keyword,
a vector column or field takes the embedding, and the service fuses the two with
RRF. The anchor ids, the ancestor ids and the clearance class are filterable,
since the retrieval filter is made of exactly those.

In Postgres an index is a table in the search schema of the load's own database,
next to the snapshot it belongs to (ddl/search.sql in spatial_core). That needs
pgvector, created once by the superuser. Load never creates it.

Which index load writes to comes from the environment: an Azure AI Search
service when AZURE_SEARCH_ENDPOINT is set, else the stand-in directory named by
SPATIAL_SEARCH_DIR, else Postgres.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx
import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from spatial_core.records import Chunk
from spatial_core.search import (
    SEARCH_DIR_ENV,
    SEMANTIC_CONFIGURATION,
    IndexEntry,
    LocalIndex,
    SearchIndex,
)
from spatial_core.stores import READER_ROLE, SEARCH_SCHEMA, ddl

ENDPOINT_ENV = "AZURE_SEARCH_ENDPOINT"
API_KEY_ENV = "AZURE_SEARCH_API_KEY"
API_VERSION = "2024-07-01"

VECTOR_FIELD = "vector"
_PROFILE = "chunks"
_BATCH = 500
_PAGE = 1000

_COLLECTIONS = ("space_ids", "asset_type_ids", "asset_ids", "ancestor_ids")
_FILTERABLE = (
    "id",
    "document_id",
    "role",
    "kind",
    "language",
    "clearance_class",
    "link_status",
)
_STORED = ("url", "original_text_sha256", "document_date", "link_checked_on")


class SearchError(Exception):
    pass


def definition(name: str, dimensions: int) -> dict[str, Any]:
    """The index: a field per Chunk field, under the same name, and the vector."""

    def text(field: str, **options: Any) -> dict[str, Any]:
        return {"name": field, "type": "Edm.String", "searchable": False, **options}

    fields: list[dict[str, Any]] = [
        # Azure keys take letters, digits, dashes and underscores. Chunk ids have more.
        text("key", key=True),
        *(text(field, filterable=True, sortable=field == "id") for field in _FILTERABLE),
        {"name": "ordinal", "type": "Edm.Int32", "filterable": True, "sortable": True},
        text("title", searchable=True, analyzer="en.microsoft"),
        text("text", searchable=True, analyzer="en.microsoft"),
        {"name": "translated", "type": "Edm.Boolean", "filterable": True},
        *(text(field) for field in _STORED),
        *(
            {
                "name": field,
                "type": "Collection(Edm.String)",
                "searchable": False,
                "filterable": True,
            }
            for field in _COLLECTIONS
        ),
        {
            "name": VECTOR_FIELD,
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": False,
            "dimensions": dimensions,
            "vectorSearchProfile": _PROFILE,
        },
    ]
    listed = {field["name"] for field in fields} - {"key", VECTOR_FIELD}
    assert listed == set(Chunk.model_fields), listed ^ set(Chunk.model_fields)
    return {
        "name": name,
        "fields": fields,
        "vectorSearch": {
            "algorithms": [{"name": "hnsw", "kind": "hnsw"}],
            "profiles": [{"name": _PROFILE, "algorithm": "hnsw"}],
        },
        # For the semantic ranker, which the service leaves off unless told otherwise.
        # Having the configuration costs nothing until a query asks for it.
        "semantic": {
            "configurations": [
                {
                    "name": SEMANTIC_CONFIGURATION,
                    "prioritizedFields": {
                        "titleField": {"fieldName": "title"},
                        "prioritizedContentFields": [{"fieldName": "text"}],
                    },
                }
            ]
        },
    }


def _key(chunk_id: str) -> str:
    return hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()


class AzureSearchIndex:
    def __init__(self, *, endpoint: str, api_key: str, settle_seconds: float = 60) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        # How long to wait for uploaded chunks to show up in the count.
        self._settle_seconds = settle_seconds
        self._uploaded: dict[str, set[str]] = {}

    def _send(
        self, method: str, path: str, *, body: Any = None, missing_is_fine: bool = False
    ) -> httpx.Response:
        try:
            response = httpx.request(
                method,
                f"{self._endpoint}{path}",
                params={"api-version": API_VERSION},
                headers={"api-key": self._api_key},
                json=body,
                timeout=120,
            )
        except httpx.HTTPError as error:
            raise SearchError(f"{method} {path} failed: {error}") from error
        if not response.is_success and not (missing_is_fine and response.status_code == 404):
            raise SearchError(
                f"{method} {path} answered {response.status_code}: {response.text[:500]}"
            )
        return response

    def create(self, name: str, dimensions: int) -> None:
        self.delete(name)
        self._send("PUT", f"/indexes/{name}", body=definition(name, dimensions))
        self._uploaded[name] = set()

    def upload(self, name: str, entries: Sequence[IndexEntry]) -> None:
        for start in range(0, len(entries), _BATCH):
            batch = [
                {
                    "@search.action": "mergeOrUpload",
                    "key": _key(entry.chunk.id),
                    **entry.chunk.model_dump(mode="json"),
                    VECTOR_FIELD: list(entry.vector),
                }
                for entry in entries[start : start + _BATCH]
            ]
            answer = self._send("POST", f"/indexes/{name}/docs/index", body={"value": batch})
            refused = [row for row in answer.json().get("value", []) if not row.get("status")]
            if refused:
                raise SearchError(f"{name} refused {len(refused)} chunks: {refused[:3]}")
        expected = self._uploaded.setdefault(name, set())
        expected.update(entry.chunk.id for entry in entries)
        self._settle(name, len(expected))

    def _count(self, name: str) -> int:
        answer = self._send("GET", f"/indexes/{name}/docs/$count")
        return int(answer.text.strip().lstrip("﻿"))

    def _settle(self, name: str, expected: int) -> None:
        """Wait until the index counts what was uploaded. It indexes a moment behind."""
        deadline = time.monotonic() + self._settle_seconds
        while (count := self._count(name)) < expected:
            if time.monotonic() >= deadline:
                raise SearchError(f"{name} holds {count} chunks after upload, expected {expected}")
            time.sleep(1)

    def chunks(self, name: str) -> list[Chunk]:
        found: list[Chunk] = []
        while True:
            answer = self._send(
                "POST",
                f"/indexes/{name}/docs/search",
                body={
                    "search": "*",
                    "select": ",".join(Chunk.model_fields),
                    "orderby": "id asc",
                    "top": _PAGE,
                    "skip": len(found),
                },
            )
            page = answer.json().get("value", [])
            found += [
                Chunk.model_validate({field: row.get(field) for field in Chunk.model_fields})
                for row in page
            ]
            if len(page) < _PAGE:
                return sorted(found, key=lambda chunk: chunk.id)

    def delete(self, name: str) -> None:
        self._send("DELETE", f"/indexes/{name}", missing_is_fine=True)
        self._uploaded.pop(name, None)

    def exists(self, name: str) -> bool:
        return self._send("GET", f"/indexes/{name}", missing_is_fine=True).is_success


class PostgresSearchIndex:
    """Each index a table, search.<name>, that the service's reader role can SELECT."""

    def __init__(self, database_url: str) -> None:
        self._url = database_url

    def _connect(self) -> psycopg.Connection[dict[str, Any]]:
        try:
            return psycopg.connect(self._url, autocommit=True, row_factory=dict_row)
        except psycopg.Error as error:
            raise SearchError(f"the search index's database cannot be reached: {error}") from error

    @staticmethod
    def _table(name: str) -> sql.Identifier:
        return sql.Identifier(SEARCH_SCHEMA, name)

    def create(self, name: str, dimensions: int) -> None:
        definition = sql.SQL(ddl("search")).format(
            reader=sql.Identifier(READER_ROLE),
            table=sql.Identifier(name),
            dimensions=sql.Literal(int(dimensions)),
        )
        with self._connect() as conn:
            if conn.execute("SELECT FROM pg_extension WHERE extname = 'vector'").fetchone() is None:
                raise SearchError(
                    "this database has no vector extension. Its superuser creates it once"
                    " (CREATE EXTENSION vector); load does not"
                )
            try:
                with conn.transaction():
                    conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table(name)))
                    conn.execute(definition)
            except psycopg.Error as error:
                raise SearchError(f"could not create the index {name}: {error}") from error

    def upload(self, name: str, entries: Sequence[IndexEntry]) -> None:
        columns = [*Chunk.model_fields, "embedding"]
        query = sql.SQL(
            "INSERT INTO {} ({}) VALUES ({}, %s::vector) ON CONFLICT (id) DO UPDATE SET {}"
        ).format(
            self._table(name),
            sql.SQL(", ").join(sql.Identifier(column) for column in columns),
            sql.SQL(", ").join(sql.Placeholder() for _ in Chunk.model_fields),
            sql.SQL(", ").join(
                sql.SQL("{0} = excluded.{0}").format(sql.Identifier(column))
                for column in columns[1:]
            ),
        )
        rows = [
            [
                *entry.chunk.model_dump(mode="json").values(),
                # pgvector's text form: [0.1,0.2,...]. Saves a dependency on its Python package.
                "[" + ",".join(repr(float(x)) for x in entry.vector) + "]",
            ]
            for entry in entries
        ]
        with self._connect() as conn:
            try:
                with conn.transaction(), conn.cursor() as cursor:
                    cursor.executemany(query, rows)
            except psycopg.Error as error:
                raise SearchError(f"{name} refused the chunks: {error}") from error

    def chunks(self, name: str) -> list[Chunk]:
        query = sql.SQL("SELECT {} FROM {} ORDER BY id").format(
            sql.SQL(", ").join(sql.Identifier(field) for field in Chunk.model_fields),
            self._table(name),
        )
        with self._connect() as conn:
            try:
                rows = conn.execute(query).fetchall()
            except psycopg.Error as error:
                raise SearchError(f"could not read the index {name}: {error}") from error
        return [Chunk.model_validate(row) for row in rows]

    def delete(self, name: str) -> None:
        with self._connect() as conn:
            conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(self._table(name)))

    def exists(self, name: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT to_regclass(%s) IS NOT NULL AS there",
                (self._table(name).as_string(conn),),
            ).fetchone()
        return bool(row and row["there"])


def from_env(database_url: str, env: Mapping[str, str] | None = None) -> SearchIndex:
    """The index store load writes to, from the environment. Postgres unless told otherwise."""
    env = os.environ if env is None else env
    if env.get(ENDPOINT_ENV):
        if not env.get(API_KEY_ENV):
            raise SearchError(f"{ENDPOINT_ENV} is set, so set {API_KEY_ENV} too")
        return AzureSearchIndex(endpoint=env[ENDPOINT_ENV], api_key=env[API_KEY_ENV])
    if env.get(SEARCH_DIR_ENV):
        return LocalIndex(Path(env[SEARCH_DIR_ENV]))
    return PostgresSearchIndex(database_url)
