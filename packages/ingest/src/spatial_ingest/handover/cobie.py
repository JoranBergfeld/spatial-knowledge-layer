"""Reading a COBie 2.4 workbook: each sheet as rows of text.

COBie writes "n/a" where it has nothing, so that is a blank here. Names are
stripped, since a few of the real ones carry a trailing space and the sheets
refer to each other by name. A numeric cell comes back as the text a person
would type: 1, not 1.0.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import openpyxl

BLANK = "n/a"

Row = dict[str, str | None]
Workbook = dict[str, list[Row]]


class CobieError(Exception):
    pass


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, datetime.datetime | datetime.date):
        return value.isoformat()
    text = str(value).strip()
    return None if not text or text.lower() == BLANK else text


def read(path: Path) -> Workbook:
    """Every sheet with a header row, as rows keyed by column name."""
    try:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    except (OSError, ValueError, KeyError) as error:
        raise CobieError(f"cannot read the COBie workbook {path}: {error}") from error
    sheets: Workbook = {}
    try:
        for sheet in book.worksheets:
            rows = sheet.iter_rows(values_only=True)
            header = [_text(cell) for cell in next(rows, ())]
            found = []
            for cells in rows:
                row = {name: _text(cell) for name, cell in zip(header, cells, strict=False) if name}
                if any(value is not None for value in row.values()):
                    found.append(row)
            # A sheet name may be padded too.
            sheets[sheet.title.strip()] = found
    finally:
        book.close()
    return sheets


def sheet(book: Workbook, name: str, *columns: str) -> list[Row]:
    """A sheet's rows, each with at least ``columns``. A missing sheet is an error."""
    if name not in book:
        raise CobieError(f"the workbook has no {name} sheet")
    rows = book[name]
    for number, row in enumerate(rows, start=2):
        missing = [column for column in columns if column not in row]
        if missing:
            raise CobieError(f"{name} row {number} has no {', '.join(missing)} column")
    return rows


def names(text: str | None) -> list[str]:
    """The names in a cell that lists several, comma separated."""
    return [name.strip() for name in (text or "").split(",") if name.strip()]
