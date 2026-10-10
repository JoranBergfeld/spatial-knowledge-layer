from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import prose
from mini_campus import VENUE
from spatial_core.search import SEARCH_DIR_ENV
from spatial_ingest import models
from stubs import OpenAI, Overpass, Site, serve_openai, serve_overpass, serve_site


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


@pytest.fixture
def search_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Where the stand-in search index keeps its files for this test."""
    directory = tmp_path / "search"
    monkeypatch.setenv(SEARCH_DIR_ENV, str(directory))
    return directory
