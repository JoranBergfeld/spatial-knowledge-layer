"""A COBie handover as the layer's records: the tree, the types, the assets, the documents.

    tree-from-cobie-sheets@1         the Facility is the building and the root. Every
                                     Floor row is a floor under it, whatever its
                                     category. Every Space is a room on its FloorName.
    asset-in-first-space@1           a Component listed in several spaces (a door in
                                     both rooms it joins) sits in the first
    position-from-cobie-coordinate@1 a Component's point from the Coordinate sheet, or
                                     the middle of its box
    position-from-ifc-placement@1    else the placement of the IFC element whose
                                     GlobalId is the Component's ExtIdentifier
    document-per-file@1              one document per File, anchored to every row the
                                     Document sheet lists it for
    anchors-from-system@1            a document keyed to a System is anchored to each
                                     of that system's components
    document-kind-from-category@1    product data, O&M and maker's instructions are
                                     manuals. Anything else is a handover record.

Ids are the sheet's names, slugged: "1A01" is space:1a01, "AC Unit Type 1 AC-1" is
asset:ac-unit-type-1-ac-1. Two names that slug the same stop the build.

Clearance is the shared code's: clearance-from-space-category@1 for the rooms and
clearance-from-omniclass@1 for the types. A handover has no doors and no edges
here, so every room is visible by its own clearance.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spatial_core import clearance
from spatial_core.records import (
    Ancestor,
    Asset,
    AssetType,
    DocumentKind,
    MaintenanceTask,
    Position,
    RecordProvenance,
    Space,
    Value,
    ValueProvenance,
)
from spatial_ingest.corpus import AnchorSpec, Source
from spatial_ingest.handover import cobie, ifc
from spatial_ingest.handover.definition import Definition
from spatial_ingest.rules import labels, tiers
from spatial_ingest.seed.generate import MANUFACTURER_RULE, manufacturer_from_email

TREE_RULE = "tree-from-cobie-sheets@1"
FIRST_SPACE_RULE = "asset-in-first-space@1"
COORDINATE_RULE = "position-from-cobie-coordinate@1"
IFC_RULE = "position-from-ifc-placement@1"
DOCUMENT_RULE = "document-per-file@1"
SYSTEM_RULE = "anchors-from-system@1"
KIND_RULE = "document-kind-from-category@1"

STRUCTURE_RULES = (TREE_RULE, labels.RULE, clearance.SPACE_CATEGORY_RULE)
ASSET_RULES = (
    FIRST_SPACE_RULE,
    COORDINATE_RULE,
    IFC_RULE,
    MANUFACTURER_RULE,
    clearance.TYPE_RULE,
)
DOCUMENT_RULES = (DOCUMENT_RULE, SYSTEM_RULE, KIND_RULE)

# The Component columns kept as attributes, under the names the seed inventory uses.
ATTRIBUTES = {
    "SerialNumber": "serial_number",
    "InstallationDate": "installation_date",
    "WarrantyStartDate": "warranty_start_date",
    "TagNumber": "tag_number",
    "BarCode": "bar_code",
    "AssetIdentifier": "asset_identifier",
}
_MANUAL_CATEGORIES = {
    "product data",
    "operation and maintenance",
    "manufacturer's instructions",
    "manufacturer instructions",
}
_METRES_PER_DEGREE = 111_320.0


class HandoverError(Exception):
    pass


@dataclass(frozen=True)
class Handover:
    spaces: tuple[Space, ...]
    types: tuple[AssetType, ...]
    assets: tuple[Asset, ...]
    tasks: tuple[MaintenanceTask, ...]
    # The documents as corpus entries. Their text is the OCR's, read at build.
    sources: tuple[Source, ...]
    # Corpus URL to the OCR text file behind it. A document with no text has no entry.
    texts: Mapping[str, Path]
    counts: dict[str, Any]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class _Ids:
    """Ids from names, one prefix per sheet. A clash is an error, never a merge."""

    def __init__(self) -> None:
        self._taken: dict[str, str] = {}

    def make(self, prefix: str, name: str) -> str:
        made = f"{prefix}:{slug(name)}"
        if made == f"{prefix}:" or self._taken.setdefault(made, name) != name:
            raise HandoverError(
                f"{name!r} and {self._taken.get(made)!r} both come out as the id {made}"
            )
        return made


def _sourced(text: str | None) -> Value | None:
    return Value.sourced(text) if text is not None else None


def _values(row: cobie.Row, columns: Mapping[str, str]) -> dict[str, Value]:
    """The columns a row fills, as sourced values under the names given. A blank is left out."""
    found = {}
    for column, key in columns.items():
        value = row.get(column)
        if value is not None:
            found[key] = Value.sourced(value)
    return found


def _label(label: labels.Label | None) -> Value | None:
    if label is None:
        return None
    return Value.sourced(label.text) if label.from_name else Value.derived(label.text, labels.RULE)


def _named(rows: Iterable[cobie.Row], sheet: str) -> dict[str, cobie.Row]:
    found: dict[str, cobie.Row] = {}
    for row in rows:
        name = row.get("Name")
        if name is None:
            raise HandoverError(f"a {sheet} row has no Name")
        if name in found:
            raise HandoverError(f"the {sheet} sheet lists {name!r} twice")
        found[name] = row
    return found


def _point(definition: Definition, x: float, y: float, rule: str) -> Position:
    """A point of the model, in metres from its own origin, as the WGS84 point the schema takes."""
    latitude = definition.origin.lat + y / _METRES_PER_DEGREE
    across = _METRES_PER_DEGREE * math.cos(math.radians(definition.origin.lat))
    return Position(
        lon=round(definition.origin.lon + x / across, 9),
        lat=round(latitude, 9),
        provenance=ValueProvenance.DERIVED,
        rule=rule,
    )


def _coordinates(book: cobie.Workbook) -> dict[str, tuple[float, float]]:
    """Each component's point from the Coordinate sheet: its point, else the middle of its box."""
    points: dict[str, tuple[float, float]] = {}
    corners: dict[str, list[tuple[float, float]]] = {}
    for row in book.get("Coordinate", []):
        if (row.get("SheetName") or "").strip() != "Component" or row.get("RowName") is None:
            continue
        try:
            at = (float(row.get("CoordinateXAxis") or ""), float(row.get("CoordinateYAxis") or ""))
        except ValueError:
            continue
        name = row["RowName"] or ""
        if (row.get("Category") or "").lower() == "point":
            points[name] = at
        else:
            corners.setdefault(name, []).append(at)
    for name, box in corners.items():
        if name not in points and len(box) == 2:
            points[name] = ((box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2)
    return points


def read_handover(definition: Definition, directory: Path) -> Handover:
    """The handover in ``directory`` as records. The pins are checked before this."""
    book = cobie.read(directory / definition.cobie.file)
    ids = _Ids()

    facilities = cobie.sheet(book, "Facility", "Name")
    if len(facilities) != 1:
        raise HandoverError(f"the Facility sheet has {len(facilities)} rows, not one")
    facility = facilities[0]
    number, title = facility["Name"], facility.get("Description") or facility.get("ProjectName")
    building_label = labels.building_label(name=title, number=number)
    building = Space(
        id=ids.make("building", number or ""),
        record_provenance=RecordProvenance.SOURCED,
        source=f"cobie:Facility/{number}",
        tier=Value.derived(tiers.BUILDING, TREE_RULE),
        parent_id=None,
        ancestor_path=(),
        name=_sourced(title),
        ref=_sourced(number),
        label=_label(building_label),
        addressable=building_label is not None,
        attributes=_values(facility, {"Category": "category"}),
    )
    spaces: dict[str, Space] = {building.id: building}
    below_building = (Ancestor(tier=tiers.BUILDING, id=building.id),)

    floor_ids: dict[str, str] = {}
    for name, row in _named(cobie.sheet(book, "Floor", "Name"), "Floor").items():
        floor_id = floor_ids[name] = ids.make("floor", name)
        attributes = _values(row, {"Category": "category", "Elevation": "elevation"})
        spaces[floor_id] = Space(
            id=floor_id,
            record_provenance=RecordProvenance.SOURCED,
            source=f"cobie:Floor/{name}",
            tier=Value.derived(tiers.FLOOR, TREE_RULE),
            parent_id=building.id,
            ancestor_path=below_building,
            name=Value.sourced(name),
            label=Value.sourced(name),
            addressable=True,
            attributes=attributes,
        )

    space_ids: dict[str, str] = {}
    uncategorised = 0
    for name, row in _named(cobie.sheet(book, "Space", "Name", "FloorName"), "Space").items():
        floor = row["FloorName"]
        if floor not in floor_ids:
            raise HandoverError(f"space {name!r} is on {floor!r}, which is not a floor")
        space_id = space_ids[name] = ids.make("space", name)
        label = labels.room_label(
            name=row.get("Description"), ref=name, building_number=number, building_name=title
        )
        cleared = clearance.space_category_clearance(row.get("Category"))
        uncategorised += row.get("Category") is None
        attributes = _values(
            row, {"Category": "category", "NetArea": "net_area", "GrossArea": "gross_area"}
        )
        spaces[space_id] = Space(
            id=space_id,
            record_provenance=RecordProvenance.SOURCED,
            source=f"cobie:Space/{name}",
            tier=Value.derived(tiers.ROOM, TREE_RULE),
            parent_id=floor_ids[floor],
            ancestor_path=(*below_building, Ancestor(tier=tiers.FLOOR, id=floor_ids[floor])),
            name=_sourced(row.get("Description")),
            ref=Value.sourced(name),
            label=_label(label),
            addressable=label is not None,
            attributes=attributes,
            clearance=cleared,
        )

    type_ids: dict[str, str] = {}
    types: list[AssetType] = []
    for name, row in _named(cobie.sheet(book, "Type", "Name"), "Type").items():
        type_id = type_ids[name] = ids.make("type", name)
        maker = manufacturer_from_email(
            row.get("Manufacturer"), definition.placeholder_manufacturers
        )
        years = (row.get("WarrantyDurationUnit") or "year").lower().startswith("year")
        types.append(
            AssetType(
                id=type_id,
                record_provenance=RecordProvenance.SOURCED,
                name=Value.sourced(name),
                # A type with no category still needs a value. It fails closed below.
                omniclass=Value.sourced(row.get("Category") or cobie.BLANK),
                manufacturer=Value.derived(maker, MANUFACTURER_RULE) if maker else None,
                model=_sourced(row.get("ModelNumber")),
                warranty_parts_years=_sourced(row.get("WarrantyDurationParts")) if years else None,
                warranty_labor_years=_sourced(row.get("WarrantyDurationLabor")) if years else None,
                expected_life_years=_sourced(row.get("ExpectedLife"))
                if (row.get("DurationUnit") or "year").lower().startswith("year")
                else None,
                clearance=clearance.type_clearance(row.get("Category")),
            )
        )

    coordinates = _coordinates(book)
    modelled: dict[str, ifc.Point] = {}
    for pinned in definition.ifc:
        modelled.update(ifc.placements(directory / pinned.file))

    asset_ids: dict[str, str] = {}
    assets: list[Asset] = []
    in_several = by_coordinate = by_model = 0
    components = cobie.sheet(book, "Component", "Name", "TypeName", "Space")
    for name, row in _named(components, "Component").items():
        if row["TypeName"] not in type_ids:
            raise HandoverError(f"component {name!r} is a {row['TypeName']!r}, which is not a type")
        listed = cobie.names(row["Space"])
        unknown = [space for space in listed if space not in space_ids]
        if not listed or unknown:
            raise HandoverError(
                f"component {name!r} is in {row['Space']!r}, which is not a space of the handover"
            )
        in_several += len(listed) > 1
        space = spaces[space_ids[listed[0]]]
        position = None
        if name in coordinates:
            position = _point(definition, *coordinates[name], COORDINATE_RULE)
            by_coordinate += 1
        elif row.get("ExtIdentifier") in modelled:
            x, y, _ = modelled[row["ExtIdentifier"] or ""]
            position = _point(definition, x, y, IFC_RULE)
            by_model += 1
        asset_id = asset_ids[name] = ids.make("asset", name)
        assets.append(
            Asset(
                id=asset_id,
                record_provenance=RecordProvenance.SOURCED,
                source=f"cobie:Component/{name}",
                type_id=type_ids[row["TypeName"] or ""],
                space_id=space.id,
                ancestor_path=(*space.ancestor_path, Ancestor(tier=tiers.ROOM, id=space.id)),
                name=Value.sourced(name),
                position=position,
                attributes=_values(row, ATTRIBUTES),
            )
        )

    tasks = _tasks(book, type_ids, ids)

    systems: dict[str, list[str]] = {}
    for row in book.get("System", []):
        if row.get("Name") is None:
            continue
        members = systems.setdefault(row["Name"] or "", [])
        members.extend(m for m in cobie.names(row.get("ComponentNames")) if m in asset_ids)

    sources, texts, keyed, unkeyed = _documents(
        book, definition, directory, building.id, floor_ids, space_ids, type_ids, asset_ids, systems
    )
    counts = {
        "spaces": len(spaces),
        "by_tier": {
            tier: sum(1 for space in spaces.values() if space.tier.value == tier)
            for tier in (tiers.BUILDING, tiers.FLOOR, tiers.ROOM)
        },
        "unaddressable": sum(1 for space in spaces.values() if not space.addressable),
        # Rooms the handover gives no category. They are staff, like any unlisted one.
        "rooms_without_category": uncategorised,
        "asset_types": len(types),
        "assets": len(assets),
        "assets_listed_in_several_spaces": in_several,
        "placed": {
            "by_coordinate_sheet": by_coordinate,
            "by_ifc_placement": by_model,
            "share": round((by_coordinate + by_model) / len(assets), 4) if assets else 0.0,
        },
        "maintenance_tasks": len(tasks),
        "systems": len(systems),
        "documents_listed": len(sources),
        "documents_keyed_by": keyed,
        # Files whose every row names something the handover does not have.
        "documents_keyed_to_nothing": unkeyed,
        "documents_without_text": sorted(s.id for s in sources if s.url not in texts),
    }
    return Handover(
        tuple(spaces.values()), tuple(types), tuple(assets), tasks, sources, texts, counts
    )


def _tasks(
    book: cobie.Workbook, type_ids: Mapping[str, str], ids: _Ids
) -> tuple[MaintenanceTask, ...]:
    """The Job sheet, row for row. What a row leaves blank stays the sheet's own "n/a"."""

    def said(row: cobie.Row, column: str) -> Value:
        return Value.sourced(row.get(column) or cobie.BLANK)

    tasks = []
    for row in book.get("Job", []):
        name, kind = row.get("Name"), row.get("TypeName")
        if name is None or kind is None or kind not in type_ids:
            continue
        number = row.get("TaskNumber") or "0"
        tasks.append(
            MaintenanceTask(
                id=ids.make("task", f"{kind} {name} {number}"),
                record_provenance=RecordProvenance.SOURCED,
                type_id=type_ids[kind],
                name=Value.sourced(name),
                category=said(row, "Category"),
                status=said(row, "Status"),
                description=said(row, "Description"),
                duration=said(row, "Duration"),
                duration_unit=said(row, "DurationUnit"),
                start=said(row, "Start"),
                task_start_unit=said(row, "TaskStartUnit"),
                frequency=said(row, "Frequency"),
                frequency_unit=said(row, "FrequencyUnit"),
                task_number=Value.sourced(number),
                priors=_sourced(row.get("Priors")),
                resource_names=_sourced(row.get("ResourceNames")),
            )
        )
    return tuple(tasks)


def _documents(
    book: cobie.Workbook,
    definition: Definition,
    directory: Path,
    building_id: str,
    floor_ids: Mapping[str, str],
    space_ids: Mapping[str, str],
    type_ids: Mapping[str, str],
    asset_ids: Mapping[str, str],
    systems: Mapping[str, list[str]],
) -> tuple[tuple[Source, ...], dict[str, Path], dict[str, int], list[str]]:
    """The Document sheet as corpus sources, one per file, and where each one's text is."""
    base = definition.documents.url if definition.documents and definition.documents.url else None
    ocr = directory / definition.ocr_dir
    titles: dict[str, str] = {}
    manual: dict[str, bool] = {}
    anchors: dict[str, dict[tuple[str, str], AnchorSpec]] = {}
    keyed: dict[str, int] = {}
    for row in sorted(
        cobie.sheet(book, "Document", "Name", "SheetName", "RowName", "File"),
        key=lambda r: (r["File"] or "", r["Name"] or "", r["SheetName"] or "", r["RowName"] or ""),
    ):
        file, sheet, named = row["File"], row["SheetName"], row["RowName"] or ""
        if file is None or sheet is None:
            continue
        titles.setdefault(file, row["Name"] or file)
        category = (row.get("Category") or "").lower()
        manual[file] = manual.get(file, False) or category in _MANUAL_CATEGORIES
        found = anchors.setdefault(file, {})
        keyed[sheet] = keyed.get(sheet, 0) + 1
        specs: list[AnchorSpec] = []
        if sheet == "Facility":
            specs = [AnchorSpec(space=building_id)]
        elif sheet == "Floor" and named in floor_ids:
            specs = [AnchorSpec(space=floor_ids[named])]
        elif sheet == "Space" and named in space_ids:
            specs = [AnchorSpec(space=space_ids[named])]
        elif sheet == "Type" and named in type_ids:
            specs = [AnchorSpec(asset_type=type_ids[named])]
        elif sheet == "Component" and named in asset_ids:
            specs = [AnchorSpec(asset=asset_ids[named])]
        elif sheet == "System":
            specs = [
                AnchorSpec(asset=asset_ids[member], rule=SYSTEM_RULE)
                for member in systems.get(named, [])
            ]
        for spec in specs:
            key = next((k, v) for k, v in spec.model_dump().items() if v and k != "rule")
            # Named outright beats reached through a system.
            if key not in found or spec.rule is None:
                found[key] = spec

    sources = []
    texts: dict[str, Path] = {}
    unkeyed: list[str] = []
    ids = _Ids()
    for file, title in titles.items():
        if not anchors[file]:
            unkeyed.append(file)
            continue
        url = f"{base}#{file}" if base else f"handover:{definition.id}/{file}"
        text = ocr / f"{Path(file).stem}.txt"
        if text.is_file():
            texts[url] = text
        sources.append(
            Source(
                id=ids.make("doc", file),
                kind=DocumentKind.MANUAL if manual[file] else DocumentKind.HANDOVER_RECORD,
                title=title,
                url=url,
                language="en",
                anchors=tuple(anchors[file][key] for key in sorted(anchors[file])),
            )
        )
    return tuple(sources), texts, dict(sorted(keyed.items())), sorted(unkeyed)
