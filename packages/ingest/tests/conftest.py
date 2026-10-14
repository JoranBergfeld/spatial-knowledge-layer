from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

import prose
from mini_campus import VENUE
from spatial_core.search import SEARCH_DIR_ENV
from spatial_ingest import models
from stubs import OpenAI, Overpass, Site, serve_openai, serve_overpass, serve_site


@pytest.fixture(autouse=True)
def no_model_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model call still retries in tests, it just does not wait in between."""
    monkeypatch.setattr(models, "MAX_WAIT_SECONDS", 0.0)


@pytest.fixture
def overpass() -> Iterator[Overpass]:
    with serve_overpass() as state:
        yield state


@pytest.fixture
def venue_file(tmp_path: Path, overpass: Overpass) -> Path:
    path = tmp_path / "mini-campus.json"
    path.write_text(json.dumps({**VENUE, "overpass_url": overpass.url}))
    return path


@pytest.fixture
def site() -> Iterator[Site]:
    with serve_site() as state:
        yield state


@pytest.fixture
def openai(monkeypatch: pytest.MonkeyPatch) -> Iterator[OpenAI]:
    """A stand-in Azure OpenAI resource, with the environment pointing build at it."""
    with serve_openai() as state:
        monkeypatch.setenv(models.ENDPOINT_ENV, state.url)
        monkeypatch.setenv(models.API_KEY_ENV, "not-a-key")
        monkeypatch.setenv(models.TRANSLATION_DEPLOYMENT_ENV, prose.TRANSLATION_DEPLOYMENT)
        monkeypatch.setenv(models.EMBEDDING_DEPLOYMENT_ENV, prose.EMBEDDING_DEPLOYMENT)
        yield state


@pytest.fixture
def corpus_file(tmp_path: Path, site: Site, openai: OpenAI) -> Path:
    """The mini campus's corpus, linking to the stand-in site."""
    path = tmp_path / "mini-corpus.json"
    path.write_text(json.dumps(prose.corpus(site.url), ensure_ascii=False))
    return path


@pytest.fixture(autouse=True)
def search_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in search index's directory, one to each test here."""
    directory = tmp_path / "search"
    monkeypatch.setenv(SEARCH_DIR_ENV, str(directory))
    return directory


ADMIN_URL_ENV = "SPATIAL_TEST_DATABASE_URL"


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the store tests"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return url


@pytest.fixture
def database_url(admin_url: str) -> Iterator[str]:
    name = f"spatial_test_{uuid4().hex[:12]}"
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    yield make_conninfo(admin_url, dbname=name)
    with psycopg.connect(admin_url, autocommit=True) as conn:
        conn.execute(sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(name)))
