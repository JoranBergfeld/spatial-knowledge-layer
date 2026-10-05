"""Venue definitions: what the build needs to know about a place before it pulls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Venue:
    id: str
    # The one authored string in the structure. The part polygons carry no name in OSM.
    site_name: str
    # south, west, north, east
    bbox: tuple[float, float, float, float]
    # The closed ways that together are the site. Everything outside them is dropped.
    part_ways: tuple[int, ...]
    overpass_url: str


def load_venue(name_or_path: str) -> Venue:
    """A built-in venue by id, or a venue definition file by path."""
    path = Path(name_or_path)
    if path.suffix == ".json":
        raw = path.read_text(encoding="utf-8")
    else:
        builtin = resources.files("spatial_ingest.venues") / f"{name_or_path}.json"
        if not builtin.is_file():
            raise ValueError(f"unknown venue {name_or_path!r}")
        raw = builtin.read_text(encoding="utf-8")
    data = json.loads(raw)
    south, west, north, east = (float(value) for value in data["bbox"])
    return Venue(
        id=data["id"],
        site_name=data["site_name"],
        bbox=(south, west, north, east),
        part_ways=tuple(int(way) for way in data["part_ways"]),
        overpass_url=data["overpass_url"],
    )
