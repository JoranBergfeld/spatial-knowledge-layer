"""Just enough of the xlsx format to read a COBie workbook, with the standard library only.

A workbook is a zip of XML. This reads the sheet list, the shared strings and
each sheet's cells, and gives back rows of text keyed by the header row. Dates in
COBie are written as text, so number formats are not looked at.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_PACKAGE = "{http://schemas.openxmlformats.org/package/2006/relationships}"

Row = dict[str, str | None]


def _column(reference: str) -> int:
    """The column of a cell reference, from 0: "A1" is 0, "AB7" is 27."""
    number = 0
    for letter in re.match(r"[A-Z]+", reference).group():  # type: ignore[union-attr]
        number = number * 26 + ord(letter) - ord("A") + 1
    return number - 1


def _text(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text.lower() == "n/a":
        return None
    # 1.0 is how a spreadsheet may store a count somebody typed as 1.
    if re.fullmatch(r"-?\d+\.0+", text):
        return text.split(".")[0]
    return text


def _strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    return ["".join(part.text or "" for part in item.iter(f"{_MAIN}t")) for item in root]


def _cell(cell: ElementTree.Element, strings: list[str]) -> str | None:
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(part.text or "" for part in cell.iter(f"{_MAIN}t"))
    value = cell.find(f"{_MAIN}v")
    if value is None or value.text is None:
        return None
    if kind == "s":
        return strings[int(value.text)]
    if kind == "b":
        return "true" if value.text == "1" else "false"
    return value.text


def read(path: Path) -> dict[str, list[Row]]:
    """Every sheet as rows keyed by its header. "n/a" and empty cells are None."""
    with zipfile.ZipFile(path) as archive:
        strings = _strings(archive)
        targets = {
            rel.get("Id"): rel.get("Target", "")
            for rel in ElementTree.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            if rel.tag == f"{_PACKAGE}Relationship"
        }
        book = ElementTree.fromstring(archive.read("xl/workbook.xml"))
        sheets: dict[str, list[Row]] = {}
        for sheet in book.iter(f"{_MAIN}sheet"):
            target = targets[sheet.get(f"{_REL}id")].lstrip("/")
            member = target if target.startswith("xl/") else f"xl/{target}"
            root = ElementTree.fromstring(archive.read(member))
            header: dict[int, str] = {}
            rows: list[Row] = []
            for number, line in enumerate(root.iter(f"{_MAIN}row")):
                cells = {
                    _column(cell.get("r", "A")): _text(_cell(cell, strings))
                    for cell in line.iter(f"{_MAIN}c")
                }
                if number == 0:
                    header = {at: name for at, name in cells.items() if name}
                    continue
                row: Row = {name: cells.get(at) for at, name in header.items()}
                if any(value is not None for value in row.values()):
                    rows.append(row)
            sheets[(sheet.get("name") or "").strip()] = rows
        return sheets
