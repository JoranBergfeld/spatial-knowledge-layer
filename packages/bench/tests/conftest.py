"""The two venues the checks run on in CI, built once per test run.

The test bench here is the hand-made handover in packages/ingest/tests, never the
real one. The demo venue is the mini campus. Both are built by the real build,
with the model, the pages and Overpass all stand-ins on localhost, and loaded
into a real Postgres by the code under test.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

import mini_handover
import prose
from mini_campus import VENUE
from spatial_bench import questions
from spatial_bench.questions import QuestionSet
from spatial_bench.run import Loaded, loaded
from spatial_ingest import models
from spatial_ingest.cli import main
from stubs import serve_openai, serve_overpass, serve_site

ADMIN_URL_ENV = "SPATIAL_TEST_DATABASE_URL"
DATE = "2026-09-15"


@pytest.fixture(scope="session")
def admin_url() -> str:
    url = os.environ.get(ADMIN_URL_ENV)
    if not url:
        message = f"set {ADMIN_URL_ENV} to a Postgres superuser URL to run the bench checks"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return url


@pytest.fixture(scope="session")
def stand_in_model() -> Iterator[None]:
    """A stand-in Azure OpenAI resource for build's summaries and embeddings."""
    patch = pytest.MonkeyPatch()
    with serve_openai() as state:
        patch.setenv(models.ENDPOINT_ENV, state.url)
        patch.setenv(models.API_KEY_ENV, "not-a-key")
        patch.setenv(models.TRANSLATION_DEPLOYMENT_ENV, prose.TRANSLATION_DEPLOYMENT)
        patch.setenv(models.EMBEDDING_DEPLOYMENT_ENV, prose.EMBEDDING_DEPLOYMENT)
        yield
    patch.undo()


@pytest.fixture(scope="session")
def bench_bundle(tmp_path_factory: pytest.TempPathFactory, stand_in_model: None) -> Path:
    """The test bench's bundle, built from the hand-made handover."""
    out = tmp_path_factory.mktemp("bench-bundles")
    built = ["build", "--date", DATE, "--venue", str(mini_handover.DEFINITION), "--out", str(out)]
    assert main([*built, "--handover", str(mini_handover.DIR)]) == 0
    return out / DATE


@pytest.fixture(scope="session")
def campus_bundle(tmp_path_factory: pytest.TempPathFactory, stand_in_model: None) -> Path:
    """The demo venue's stand-in: the mini campus with its corpus."""
    root = tmp_path_factory.mktemp("campus")
    with serve_overpass() as overpass, serve_site() as site:
        venue = root / "mini-campus.json"
        venue.write_text(json.dumps({**VENUE, "overpass_url": overpass.url}))
        corpus = root / "mini-corpus.json"
        corpus.write_text(json.dumps(prose.corpus(site.url), ensure_ascii=False))
        built = ["build", "--date", DATE, "--venue", str(venue), "--out", str(root / "bundles")]
        assert main([*built, "--documents", str(corpus)]) == 0
    return root / "bundles" / DATE


@pytest.fixture(scope="session")
def bench_gold(tmp_path_factory: pytest.TempPathFactory) -> QuestionSet:
    """Gold for the test bench, from the workbook alone."""
    return questions.from_cobie(
        mini_handover.DIR / mini_handover.WORKBOOK,
        tmp_path_factory.mktemp("gold") / "bench.json",
        ocr=mini_handover.DIR / mini_handover.OCR_DIR,
    )


@pytest.fixture(scope="session")
def campus_gold(tmp_path_factory: pytest.TempPathFactory, campus_bundle: Path) -> QuestionSet:
    return questions.from_bundle(campus_bundle, tmp_path_factory.mktemp("gold") / "campus.json")


@pytest.fixture(scope="session")
def bench_loaded(
    bench_bundle: Path, admin_url: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Loaded]:
    with loaded(bench_bundle, admin_url, tmp_path_factory.mktemp("bench-index")) as stores:
        yield stores


@pytest.fixture(scope="session")
def campus_loaded(
    campus_bundle: Path, admin_url: str, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Loaded]:
    with loaded(campus_bundle, admin_url, tmp_path_factory.mktemp("campus-index")) as stores:
        yield stores
