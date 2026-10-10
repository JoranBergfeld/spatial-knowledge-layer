"""For every test in the workspace: the search index is the stand-in, never a real service."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from spatial_core.search import SEARCH_DIR_ENV


@pytest.fixture(scope="session", autouse=True)
def _stand_in_search_index(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Load writes its index to a directory of this test run.

    Session wide, so a module's fixtures can load a snapshot. A test that wants
    a directory to itself sets the variable again.
    """
    patch = pytest.MonkeyPatch()
    patch.delenv("AZURE_SEARCH_ENDPOINT", raising=False)
    patch.setenv(SEARCH_DIR_ENV, str(tmp_path_factory.mktemp("search")))
    yield
    patch.undo()
