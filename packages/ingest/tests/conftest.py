from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from mini_campus import VENUE
from stubs import Overpass, serve_overpass


@pytest.fixture
def overpass() -> Iterator[Overpass]:
    with serve_overpass() as state:
        yield state


@pytest.fixture
def venue_file(tmp_path: Path, overpass: Overpass) -> Path:
    path = tmp_path / "mini-campus.json"
    path.write_text(json.dumps({**VENUE, "overpass_url": overpass.url}))
    return path
