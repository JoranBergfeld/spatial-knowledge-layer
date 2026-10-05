"""The pinned OSM pull: fetching it, cutting the fallback, and reading it back.

A stored pull is a directory holding the raw data and a source.json that says
where it came from. Its pin is the file plus its sha256, whichever way it got here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import osmium
from osmium.osm import Node, Relation, Way

from spatial_ingest.osm import OsmData, from_osm_file, from_overpass_json
from spatial_ingest.venue import Venue

SOURCE_FILE = "source.json"
QUERY_FILE = "query.overpassql"
OVERPASS_FILE = "overpass.json"
EXTRACT_FILE = "extract.osm.pbf"

_USER_AGENT = "spatial-knowledge-layer-ingest (github.com/JoranBergfeld/spatial-knowledge-layer)"


class PullError(Exception):
    pass


class RateLimited(PullError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def overpass_query(venue: Venue, date: str) -> str:
    """The dated query over the bounding box.

    Bounding box and not area: an area filter always applies today's boundaries,
    whatever the date says. The part ways come along so the cut to the site uses
    the boundaries of the same date. The selection is wider than the structure
    needs (doors, footways, emergency) so later build steps read the same pull.
    """
    south, west, north, east = venue.bbox
    parts = ",".join(str(way) for way in venue.part_ways)
    return f"""[out:json][timeout:300][date:"{date}T00:00:00Z"][bbox:{south},{west},{north},{east}];
(
  way(id:{parts});
  nwr["building"];
  nwr["indoor"];
  nwr["room"];
  nwr["level"];
  node["door"];
  node["entrance"];
  way["highway"~"^(footway|corridor|steps|elevator|pedestrian|path)$"];
  nwr["emergency"];
);
out body;
>;
out skel qt;
"""


def fetch_overpass(venue: Venue, date: str, pull_dir: Path) -> None:
    query = overpass_query(venue, date)
    try:
        response = httpx.post(
            venue.overpass_url,
            data={"data": query},
            headers={"User-Agent": _USER_AGENT},
            timeout=600,
        )
    except httpx.HTTPError as error:
        raise PullError(f"Overpass is unreachable at {venue.overpass_url}: {error}") from error
    if response.status_code == 429:
        raise RateLimited(
            "Overpass is rate limiting (HTTP 429). Wait, or build from a dated regional "
            "extract with --extract <file.osm.pbf>."
        )
    if response.status_code != 200:
        raise PullError(
            f"Overpass answered HTTP {response.status_code}. If it keeps doing that, build "
            "from a dated regional extract with --extract <file.osm.pbf>."
        )
    try:
        body = response.json()
    except ValueError as error:
        raise PullError("Overpass answered something that is not JSON") from error
    # Overpass reports a timeout or an out-of-memory as HTTP 200 with a remark.
    if "elements" not in body or "remark" in body:
        raise PullError(f"Overpass did not finish the query: {body.get('remark', 'no elements')}")

    pull_dir.mkdir(parents=True, exist_ok=True)
    (pull_dir / OVERPASS_FILE).write_bytes(response.content)
    (pull_dir / QUERY_FILE).write_text(query, encoding="utf-8")
    _write_source(
        pull_dir,
        {
            "sha256": hashlib.sha256(response.content).hexdigest(),
            "kind": "overpass",
            "venue": venue.id,
            "date": date,
            "endpoint": venue.overpass_url,
            "bbox": venue.bbox,
        },
    )


def cut_extract(venue: Venue, date: str, extract: Path, pull_dir: Path) -> None:
    """Cuts a regional extract down to the venue's box, keeping ways and relations whole."""
    if not extract.is_file():
        raise PullError(f"no extract at {extract}")
    south, west, north, east = venue.bbox
    source = str(extract)

    inside: set[int] = set()
    ways: set[int] = set()
    relations: set[int] = set()
    member_ways: set[int] = set()
    for obj in osmium.FileProcessor(source):
        if isinstance(obj, Node):
            location = obj.location
            if south <= location.lat <= north and west <= location.lon <= east:
                inside.add(obj.id)
        elif isinstance(obj, Way):
            if any(node.ref in inside for node in obj.nodes):
                ways.add(obj.id)
        elif isinstance(obj, Relation):
            members = [(member.type, member.ref) for member in obj.members]
            if any(
                (kind == "n" and ref in inside) or (kind == "w" and ref in ways)
                for kind, ref in members
            ):
                relations.add(obj.id)
                member_ways.update(ref for kind, ref in members if kind == "w")
    ways |= member_ways

    nodes = set(inside)
    for obj in osmium.FileProcessor(source, osmium.osm.WAY):
        if isinstance(obj, Way) and obj.id in ways:
            nodes.update(node.ref for node in obj.nodes)

    pull_dir.mkdir(parents=True, exist_ok=True)
    with osmium.SimpleWriter(str(pull_dir / EXTRACT_FILE), overwrite=True) as writer:
        for obj in osmium.FileProcessor(source):
            if isinstance(obj, Node) and obj.id in nodes:
                writer.add_node(obj)
            elif isinstance(obj, Way) and obj.id in ways:
                writer.add_way(obj)
            elif isinstance(obj, Relation) and obj.id in relations:
                writer.add_relation(obj)

    reader = osmium.io.Reader(source)
    replication_timestamp = reader.header().get("osmosis_replication_timestamp") or None
    reader.close()
    _write_source(
        pull_dir,
        {
            "kind": "extract",
            "sha256": sha256_file(pull_dir / EXTRACT_FILE),
            "venue": venue.id,
            "date": date,
            "bbox": venue.bbox,
            "cut_from": {
                "name": extract.name,
                "sha256": sha256_file(extract),
                "replication_timestamp": replication_timestamp,
            },
        },
    )


def _write_source(pull_dir: Path, source: dict[str, Any]) -> None:
    text = json.dumps(source, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (pull_dir / SOURCE_FILE).write_text(text, encoding="utf-8")


@dataclass(frozen=True)
class StoredPull:
    data: OsmData
    # What the manifest records under inputs.pull.
    pin: dict[str, Any]


def has_pull(pull_dir: Path) -> bool:
    return (pull_dir / SOURCE_FILE).is_file()


def load_pull(pull_dir: Path, venue: Venue, date: str) -> StoredPull:
    source = json.loads((pull_dir / SOURCE_FILE).read_text(encoding="utf-8"))
    if source["venue"] != venue.id:
        raise PullError(f"the stored pull is of {source['venue']}, not {venue.id}")
    if source["date"] != date:
        raise PullError(f"the stored pull is dated {source['date']}, not {date}")
    files = {"overpass": OVERPASS_FILE, "extract": EXTRACT_FILE}
    if source["kind"] not in files:
        raise PullError(f"unknown kind of pull {source['kind']!r}")
    path = pull_dir / files[source["kind"]]
    found = sha256_file(path)
    if found != source["sha256"]:
        raise PullError(
            f"{path} has sha256 {found} but was pinned as {source['sha256']}. "
            "The pull changed after it was stored."
        )
    pin: dict[str, Any] = {**source, "file": f"pull/{path.name}"}
    if source["kind"] == "extract":
        return StoredPull(from_osm_file(path), pin)
    pin |= {
        "query_file": f"pull/{QUERY_FILE}",
        "query": (pull_dir / QUERY_FILE).read_text(encoding="utf-8"),
    }
    return StoredPull(from_overpass_json(path.read_bytes()), pin)
