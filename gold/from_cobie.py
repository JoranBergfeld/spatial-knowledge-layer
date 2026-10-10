"""The test bench as a World, read from its COBie workbook and nothing else.

Every rule here is typed in again from the decisions it comes from, not imported:
the Table 13 rule for rooms (decision 69), the Table 23 rule for asset types
(decision 70), who sees what and how a document is cleared (decisions 35 and 44).
If the layer and this file disagree, one of them read the decision wrong, and
that is the point of having two.

Ids follow the convention the bench's ingestion publishes: the sheet's name,
lowercased, with every run of anything but a letter or digit turned into one
hyphen, behind a prefix for the sheet (building:, floor:, space:, type:, asset:,
doc:).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import xlsx
from world import PERSONAS, Asset, AssetType, Document, Space, World, clears

# Decision 69. Circulation, restrooms, waiting and reception are visitor. Plant and
# service rooms are maintenance. Anything else, a missing category too, is staff.
SPACE_VISITOR = (
    "13-11 11 31",
    "13-41 11 14 21",
    "13-51 31 11",
    "13-85 11 11",
    "13-85 21 11",
    "13-85 31",
)
SPACE_MAINTENANCE = (
    "13-75 41 24",
    "13-81 11 99",
    "13-81 21 11",
    "13-81 21 17",
    "13-81 21 21",
    "13-81 21 31",
    "13-81 31",
    "13-91 11 11 33",
)

# Decision 70. The longest matching prefix wins. No match is maintenance.
TYPE_TIERS = {
    "maintenance": (
        "23-15",
        "23-60",
        "23-65 35",
        "23-65 50",
        "23-65 55",
        "23-75 10",
        "23-75 35",
        "23-75 50",
        "23-75 70 21 24",
        "23-75 85",
        "23-80 30",
    ),
    "staff": ("23-20", "23-25 30", "23-35", "23-40 35", "23-40 40", "23-40 70", "23-85"),
    "visitor": (
        "23-25 40",
        "23-30",
        "23-40 10",
        "23-40 20",
        "23-40 90",
        "23-40 95",
        "23-45",
        "23-50",
        "23-55",
        "23-65 70",
        "23-75 70 21 27",
        "23-80 50",
        "23-80 70",
    ),
}

ATTRIBUTES = {
    "SerialNumber": "serial_number",
    "InstallationDate": "installation_date",
    "WarrantyStartDate": "warranty_start_date",
    "TagNumber": "tag_number",
    "BarCode": "bar_code",
    "AssetIdentifier": "asset_identifier",
}
MANUALS = {
    "product data",
    "operation and maintenance",
    "manufacturer's instructions",
    "manufacturer instructions",
}


@dataclass
class _Listed:
    """One file of the Document sheet, over every row that lists it."""

    title: str
    manual: bool = False
    anchors: list[tuple[str, str]] = field(default_factory=list)


def _code(category: str | None, table: str) -> list[str] | None:
    """The numeric code a category starts with, as its two-digit groups."""
    found = re.match(rf"\s*({table}-\d\d(?: \d\d)*)(?!\d)", category or "")
    return found.group(1).replace("-", " ").split() if found else None


def _starts(code: list[str], prefix: str) -> bool:
    groups = prefix.replace("-", " ").split()
    return code[: len(groups)] == groups


def space_persona(category: str | None) -> str:
    code = _code(category, "13")
    if code is None:
        return "staff"
    if any(_starts(code, prefix) for prefix in SPACE_VISITOR):
        return "visitor"
    if any(_starts(code, prefix) for prefix in SPACE_MAINTENANCE):
        return "maintenance"
    return "staff"


def type_persona(category: str | None) -> str:
    code = _code(category, "23")
    best: tuple[int, str] = (0, "maintenance")
    for persona, prefixes in TYPE_TIERS.items():
        for prefix in prefixes:
            if code is not None and _starts(code, prefix):
                best = max(best, (len(prefix.replace("-", " ").split()), persona))
    return best[1]


def slug(prefix: str, name: str) -> str:
    return f"{prefix}:" + re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def document_persona(kind: str, anchors: list[tuple[str, str | None]]) -> str:
    """The strictest anchor's. A manual's type anchors count once, as the least strict of them."""
    ranks = [
        PERSONAS.index(lowest)
        for anchor, lowest in anchors
        if lowest is not None and not (kind == "manual" and anchor == "asset_type")
    ]
    shared = [
        PERSONAS.index(lowest)
        for anchor, lowest in anchors
        if lowest is not None and kind == "manual" and anchor == "asset_type"
    ]
    if shared:
        ranks.append(min(shared))
    return PERSONAS[max(ranks, default=0)]


def read(workbook: Path, ocr: Path | None = None) -> World:
    """The handover as a World. With ``ocr``, only documents that have text there count."""
    book = xlsx.read(workbook)
    (facility,) = book["Facility"]
    number = facility["Name"] or ""
    title = facility.get("Description") or facility.get("ProjectName")
    building = slug("building", number)
    spaces = {building: Space(building, "building", None, title or number, number, title, None)}

    floors: dict[str, str] = {}
    for row in book["Floor"]:
        name = row["Name"] or ""
        floors[name] = slug("floor", name)
        spaces[floors[name]] = Space(floors[name], "floor", building, name, None, name, None)

    rooms: dict[str, str] = {}
    for row in book["Space"]:
        name = row["Name"] or ""
        rooms[name] = slug("space", name)
        described = row.get("Description")
        spaces[rooms[name]] = Space(
            id=rooms[name],
            tier="room",
            parent=floors[row["FloorName"] or ""],
            # The sourced name where there is one, else the building's number and the room's.
            label=described or f"{number} room {name}",
            ref=name,
            name=described,
            lowest=space_persona(row.get("Category")),
        )

    types: dict[str, AssetType] = {}
    type_ids: dict[str, str] = {}
    for row in book["Type"]:
        name = row["Name"] or ""
        type_ids[name] = slug("type", name)
        types[type_ids[name]] = AssetType(type_ids[name], name, type_persona(row.get("Category")))

    assets: dict[str, Asset] = {}
    asset_ids: dict[str, str] = {}
    for row in book["Component"]:
        name = row["Name"] or ""
        asset_ids[name] = slug("asset", name)
        # Listed in several spaces, it sits in the first.
        first = (row["Space"] or "").split(",")[0].strip()
        assets[asset_ids[name]] = Asset(
            id=asset_ids[name],
            type=type_ids[row["TypeName"] or ""],
            space=rooms[first],
            attributes={
                key: row[column] or ""
                for column, key in ATTRIBUTES.items()
                if row.get(column) is not None
            },
        )

    systems: dict[str, list[str]] = {}
    for row in book.get("System", []):
        members = systems.setdefault(row.get("Name") or "", [])
        for member in (row.get("ComponentNames") or "").split(","):
            if member.strip() in asset_ids:
                members.append(asset_ids[member.strip()])

    by_file: dict[str, _Listed] = {}
    ordered = sorted(
        (row for row in book.get("Document", []) if row.get("File") and row.get("SheetName")),
        key=lambda r: (r["File"] or "", r["Name"] or "", r["SheetName"] or "", r["RowName"] or ""),
    )
    for row in ordered:
        entry = by_file.setdefault(row["File"] or "", _Listed(row["Name"] or row["File"] or ""))
        entry.manual = entry.manual or (row.get("Category") or "").lower() in MANUALS
        sheet, named = row["SheetName"], row["RowName"] or ""
        found: list[tuple[str, str]] = []
        if sheet == "Facility":
            found = [("space", building)]
        elif sheet == "Floor" and named in floors:
            found = [("space", floors[named])]
        elif sheet == "Space" and named in rooms:
            found = [("space", rooms[named])]
        elif sheet == "Type" and named in type_ids:
            found = [("asset_type", type_ids[named])]
        elif sheet == "Component" and named in asset_ids:
            found = [("asset", asset_ids[named])]
        elif sheet == "System":
            found = [("asset", member) for member in systems.get(named, [])]
        entry.anchors.extend(found)

    documents: dict[str, Document] = {}
    for file, entry in by_file.items():
        anchors = tuple(sorted(set(entry.anchors)))
        if not anchors:
            continue
        if ocr is not None:
            text = ocr / f"{Path(file).stem}.txt"
            if not text.is_file() or not text.read_text(encoding="utf-8").strip():
                continue
        kind = "manual" if entry.manual else "handover-record"

        def lowest(anchor: tuple[str, str]) -> str | None:
            if anchor[0] == "space":
                return spaces[anchor[1]].lowest
            return types[anchor[1] if anchor[0] == "asset_type" else assets[anchor[1]].type].lowest

        documents[slug("doc", file)] = Document(
            id=slug("doc", file),
            title=entry.title,
            kind=kind,
            anchors=anchors,
            lowest=document_persona(kind, [(anchor[0], lowest(anchor)) for anchor in anchors]),
        )

    world = World(spaces, types, assets, documents)
    for persona in PERSONAS:
        # A room by its own clearance, and whatever encloses it with it.
        seen: set[str] = set()
        for space in spaces.values():
            if space.lowest is not None and clears(persona, space.lowest):
                seen.update([space.id, *world.path(space.id)])
        # An asset when its type is cleared and its space is visible. A type only
        # through an asset.
        reached = {
            asset.id
            for asset in assets.values()
            if clears(persona, types[asset.type].lowest) and asset.space in seen
        }
        world.visible[persona] = {
            "spaces": frozenset(seen),
            "assets": frozenset(reached),
            "types": frozenset(assets[asset_id].type for asset_id in reached),
        }
    return world
