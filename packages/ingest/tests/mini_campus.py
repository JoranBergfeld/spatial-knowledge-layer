"""A campus small enough to read, with one of every case the build has to handle.

Coordinates are a grid: one unit is 0.0001 degrees. Three part polygons (9001-9003)
make up the site, as the three outer ways of the KIT relation do on the real one.

    Part 1 (0,0)-(40,40)
      A  way 100   50.34 "Informatik"            rooms 1001-1009
      B  way 200   "Kollegiengebäude am Zirkel"  rooms 2001-2002, no number
      C  rel 300   30.10, with a courtyard       room 3001, and 3002 out in the courtyard
      D  way 400   no number, no name            room 4001
    Part 2 (50,0)-(70,20)
      E  way 500   10.50                         room 5001
    Part 3 (50,30)-(70,50)
      F  way 600   20.30                         no rooms
    Between the parts, inside the bounding box
      G  way 700   99.99                         room 7001, not Campus Süd
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

BBOX = (49.0090, 8.4090, 49.0160, 8.4180)  # south, west, north, east
PART_WAYS = (9001, 9002, 9003)
PERSON = "Erika Mustermann"

VENUE = {
    "id": "mini-campus",
    "site_name": "Mini Campus",
    "bbox": list(BBOX),
    "part_ways": list(PART_WAYS),
    "overpass_url": "http://overpass.invalid/api/interpreter",
}


def _lon(x: float) -> float:
    return round(8.4100 + x * 0.0001, 7)


def _lat(y: float) -> float:
    return round(49.0100 + y * 0.0001, 7)


class _Osm:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self.ways: list[dict[str, Any]] = []
        self.relations: list[dict[str, Any]] = []
        self._next_node = 1

    def node(self, x: float, y: float, node_id: int | None = None, **tags: str) -> int:
        if node_id is None:
            node_id = self._next_node
            self._next_node += 1
        element: dict[str, Any] = {"type": "node", "id": node_id, "lat": _lat(y), "lon": _lon(x)}
        if tags:
            element["tags"] = dict(tags)
        self.nodes.append(element)
        return node_id

    def rect(self, way_id: int, x0: float, y0: float, x1: float, y1: float, **tags: str) -> None:
        corners = [self.node(x, y) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
        element: dict[str, Any] = {"type": "way", "id": way_id, "nodes": [*corners, corners[0]]}
        if tags:
            element["tags"] = dict(tags)
        self.ways.append(element)

    def multipolygon(self, rel_id: int, outer: int, inner: int, **tags: str) -> None:
        members = [
            {"type": "way", "ref": outer, "role": "outer"},
            {"type": "way", "ref": inner, "role": "inner"},
        ]
        tags = {"type": "multipolygon", **tags}
        self.relations.append(
            {"type": "relation", "id": rel_id, "members": members, "tags": dict(tags)}
        )


def mini_campus() -> _Osm:
    osm = _Osm()

    osm.rect(9001, 0, 0, 40, 40)
    osm.rect(9002, 50, 0, 70, 20)
    osm.rect(9003, 50, 30, 70, 50)

    osm.rect(
        100,
        2,
        2,
        18,
        18,
        building="university",
        ref="50.34",
        name="Informatik",
        description=f"Leitung Dr. {PERSON}",
    )
    osm.rect(1001, 3, 3, 6, 6, indoor="room", ref="107", level="1", access="private")
    osm.rect(1002, 7, 3, 10, 6, indoor="room", ref="001", level="0", name="Gaede-Hörsaal")
    osm.rect(
        1003, 3, 7, 6, 10, indoor="room", ref="108", level="1", name=f"Büro Prof. Dr. {PERSON}"
    )
    osm.rect(1004, 11, 3, 13, 6, indoor="room", ref="T1", level="0-2", room="stairs")
    osm.rect(1005, 7, 7, 10, 10, indoor="room", level="0")
    osm.rect(1006, 3, 11, 13, 12, indoor="corridor", level="0")
    osm.node(15, 15, node_id=1007, indoor="room", ref="109", level="1")
    osm.rect(1008, 14, 3, 16, 6, indoor="room", ref="110")
    osm.rect(
        1009, 3, 13, 6, 16, indoor="room", ref="111", level="2", description=f"Prof. Dr. {PERSON}"
    )
    osm.rect(800, 7, 13, 10, 13.2, indoor="wall", level="0")

    osm.rect(200, 22, 2, 38, 18, building="yes", name="Kollegiengebäude am Zirkel")
    osm.rect(2001, 23, 3, 26, 6, indoor="room", ref="107", level="1")
    osm.rect(2002, 27, 3, 30, 6, indoor="room", ref="102;202", level="1")

    osm.rect(301, 2, 22, 18, 38)
    osm.rect(302, 6, 26, 14, 34)
    osm.multipolygon(300, outer=301, inner=302, building="university", ref="30.10")
    osm.rect(3001, 3, 23, 5, 25, indoor="room", ref="001", level="0")
    osm.rect(3002, 8, 28, 10, 30, indoor="room", ref="002", level="0")

    osm.rect(400, 22, 22, 30, 30, building="yes")
    osm.rect(4001, 23, 23, 26, 26, indoor="room", ref="5", level="0")

    osm.rect(500, 52, 2, 60, 10, building="yes", ref="10.50")
    osm.rect(5001, 53, 3, 56, 6, indoor="room", ref="101", level="1")

    osm.rect(600, 52, 32, 60, 40, building="yes", ref="20.30")

    osm.rect(700, 42, 42, 48, 48, building="yes", ref="99.99")
    osm.rect(7001, 43, 43, 46, 46, indoor="room", ref="1", level="0")
    return osm


def overpass_response() -> bytes:
    osm = mini_campus()
    body = {
        "version": 0.6,
        "generator": "mini campus",
        "elements": [*osm.nodes, *osm.ways, *osm.relations],
    }
    return json.dumps(body, ensure_ascii=False).encode()


def write_pbf(path: Path) -> None:
    """The same campus as a regional extract, plus a building far outside the box."""
    import osmium
    from osmium.osm import mutable

    osm = mini_campus()
    osm.rect(880, 500, 500, 510, 510, building="yes", ref="77.77")
    with osmium.SimpleWriter(str(path), overwrite=True) as writer:
        for node in sorted(osm.nodes, key=lambda n: n["id"]):
            writer.add_node(
                mutable.Node(
                    id=node["id"], location=(node["lon"], node["lat"]), tags=node.get("tags", {})
                )
            )
        for way in sorted(osm.ways, key=lambda w: w["id"]):
            writer.add_way(mutable.Way(id=way["id"], nodes=way["nodes"], tags=way.get("tags", {})))
        for rel in osm.relations:
            members = [(m["type"][0], m["ref"], m["role"]) for m in rel["members"]]
            writer.add_relation(mutable.Relation(id=rel["id"], members=members, tags=rel["tags"]))
