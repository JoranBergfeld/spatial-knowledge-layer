"""A tiny hand-made building handover: a COBie workbook, an IFC file and OCR'd documents.

It stands in for the real test bench in tests and CI, where the real handover is
never downloaded. Thirteen spaces on three floors, ten types, a few dozen
components, and one of every awkward case I found in the real file: "n/a" for a
blank, a padded name, a component listed in two spaces, a system sheet with one
row per member, a document keyed to a system, to several types at once, and a
sheet name with a trailing space.

The files under handover_fixture/ are checked in. Run this module to write them
again after changing the tables below:

    uv run python packages/ingest/tests/mini_handover.py

Who sees what, worked out by hand from the two published tables:

    space  category                   lowest persona
    1G01    Reception Space            visitor
    1G02    Corridor                   visitor
    1G03    Restroom                   visitor
    1G04    Office                     staff
    1G05    Electrical Room            maintenance
    1G06    Waiting Room               visitor
    2F01    Corridor                   visitor
    2F02    Physical Examination Room  staff
    2F03    Office                     staff
    2F04    Mechanical Room            maintenance
    2F05    Restroom                   visitor
    2F06    (none)                     staff
    3R01    Equipment Room             maintenance

    type             Table 23 code       lowest persona
    Light Fixture A  23-80 70 11 14 11   visitor
    Door Type A      23-30 10            visitor
    Lavatory L1      23-45 05 14 14      visitor
    Sprinkler Head   23-65 70 11 11      visitor
    Exam Table       23-40 70 11 14 11   staff
    Work Station     23-40 70 17 14 14   staff
    Panelboard P1    23-80 30 11 17      maintenance
    Ball Valve 50    23-65 55 14 17      maintenance
    AHU              23-75 35 14 17      maintenance
    Smoke Detector   23-99 99            maintenance (unlisted, fails closed)

A visitor sees 14 assets, staff 23, maintenance all 35.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

VENUE = "mini-bench"
DIR = Path(__file__).parent / "handover_fixture"
WORKBOOK = "handover.xlsx"
IFC = "model.ifc"
OCR_DIR = "ocr"
DEFINITION = DIR / "definition.json"

BY = "someone@example.org"
ON = "2019-05-06T10:00:00"
NA = "n/a"

FACILITY = [
    {
        "Name": "B 12",
        "Category": "11-13 24 14: Clinic",
        "ProjectName": "Harbour Clinic",
        "SiteName": "Default",
        "LinearUnits": "millimeters",
        "Description": "Harbour Clinic",
        "Phase": "Handover",
    }
]

FLOORS = [
    {"Name": "Ground Floor", "Category": "Floor", "Elevation": 0},
    {"Name": "First Floor", "Category": "Floor", "Elevation": 4.2},
    {"Name": "Roof", "Category": "Roof", "Elevation": 8.4},
]

# (name, category, floor, description)
_SPACES = [
    ("1G01", "13-11 11 31: Reception Space", "Ground Floor", "RECEPTION"),
    ("1G02", "13-85 11 11: Corridor", "Ground Floor", "CORRIDOR"),
    ("1G03", "13-41 11 14 21: Restroom", "Ground Floor", "RESTROOM"),
    ("1G04", "13-15 11 34 11: Office", "Ground Floor", "OFFICE"),
    ("1G05", "13-81 21 21: Electrical Room", "Ground Floor", "ELEC. ROOM"),
    ("1G06", "13-51 31 11: Waiting Room", "Ground Floor", "WAITING"),
    ("2F01", "13-85 11 11: Corridor", "First Floor", "CORRIDOR"),
    ("2F02", "13-41 41 14: Physical Examination Room", "First Floor", "EXAM"),
    ("2F03", "13-15 11 34 11: Office", "First Floor", "OFFICE"),
    ("2F04", "13-81 21 17: Mechanical Room", "First Floor", "MECH. ROOM"),
    ("2F05", "13-41 11 14 21: Restroom", "First Floor", NA),
    ("2F06", NA, "First Floor", "STORE"),
    ("3R01", "13-81 21 11: Equipment Room", "Roof", "ROOF PLANT"),
]
SPACES = [
    {"Name": name, "Category": category, "FloorName": floor, "Description": description}
    for name, category, floor, description in _SPACES
]

# (name, category, manufacturer, model, parts years, labour years, expected life)
_TYPES = [
    ("Light Fixture A", "23-80 70 11 14 11: Downlights", "sales@lithonia.com", "GT2 MV", 2, 2, NA),
    ("Door Type A", "23-30 10: Doors", "fibrdor@fibrdor.com", "Standard", 2, 0, 15),
    ("Lavatory L1", "23-45 05 14 14: Sinks/Lavatories", "sales@kohler.com", "K-2211", 1, 0, NA),
    ("Sprinkler Head", "23-65 70 11 11: Sprinklers", "supplycompany@usace.army.mil", NA, 1, 1, NA),
    ("Exam Table", "23-40 70 11 14 11: Medical Equipment", "info@midmark.com", "204", 1, 1, 10),
    # Padded, as three of the real ones are.
    ("Work Station ", "23-40 70 17 14 14: Work Stations ", NA, NA, NA, NA, NA),
    ("Panelboard P1", "23-80 30 11 17: Distribution Boards", "sales@eaton.com", "PRL4", 1, 1, 20),
    ("Ball Valve 50", "23-65 55 14 17: Adjusting Valves", "sales@apollovalves.com", "64", 2, 0, NA),
    ("AHU", "23-75 35 14 17: Modular Indoor Air Handling Units", "sales@York.com", "AP", 1, 1, NA),
    ("Smoke Detector", "23-99 99: Not In The Table", NA, "SD-1", NA, NA, NA),
]
TYPES = [
    {
        "Name": name,
        "Category": category,
        "Manufacturer": maker,
        "ModelNumber": model,
        "WarrantyDurationParts": parts,
        "WarrantyDurationLabor": labour,
        "WarrantyDurationUnit": "year",
        "ExpectedLife": life,
        "DurationUnit": "year",
    }
    for name, category, maker, model, parts, labour, life in _TYPES
]

# (name, type, space, serial, installed, IFC GlobalId)
_COMPONENTS = [
    (
        "Light A-01",
        "Light Fixture A",
        "1G01",
        "LA-0001",
        "2018-03-01T09:00:00",
        "0aLightA01GlobalIdxxx1",
    ),
    ("Light A-02", "Light Fixture A", "1G01", NA, "2018-03-01T09:00:00", NA),
    ("Light A-03", "Light Fixture A", "1G02", "LA-0003", "2018-03-02T09:00:00", NA),
    ("Light A-04", "Light Fixture A", "1G02", NA, "2018-03-02T09:00:00", NA),
    ("Light A-05", "Light Fixture A", "1G03", NA, "2018-03-05T09:00:00", NA),
    ("Light A-06", "Light Fixture A", "1G04", "LA-0006", "2018-03-05T09:00:00", NA),
    ("Light A-07", "Light Fixture A", "1G04", NA, "2018-03-05T09:00:00", NA),
    ("Light A-08", "Light Fixture A", "1G06", NA, "2018-04-10T09:00:00", NA),
    ("Light A-09", "Light Fixture A", "2F01", "LA-0009", "2018-04-10T09:00:00", NA),
    ("Light A-10", "Light Fixture A", "2F01", NA, "2018-04-11T09:00:00", NA),
    ("Light A-11", "Light Fixture A", "2F02", NA, "2018-04-11T09:00:00", NA),
    ("Light A-12", "Light Fixture A", "2F03", NA, "2018-04-11T09:00:00", NA),
    ("Light A-13", "Light Fixture A", "2F05", NA, NA, NA),
    ("Light A-14", "Light Fixture A", "1G05", NA, "2018-04-12T09:00:00", NA),
    ("Light A-15", "Light Fixture A", "2F04", NA, "2018-04-12T09:00:00", NA),
    # A door is listed in both rooms it joins. It sits in the first.
    ("Door D-G04", "Door Type A", "1G04, 1G02", NA, "2017-11-20T09:00:00", NA),
    ("Door D-G03", "Door Type A", "1G03, 1G02", NA, "2017-11-20T09:00:00", NA),
    ("Door D-F02", "Door Type A", "2F02, 2F01", NA, "2017-11-21T09:00:00", NA),
    ("Lavatory L1-01", "Lavatory L1", "1G03", NA, "2018-01-15T09:00:00", NA),
    ("Lavatory L1-02", "Lavatory L1", "2F05", NA, "2018-01-15T09:00:00", NA),
    ("Sprinkler S-01", "Sprinkler Head", "1G01", NA, "2017-12-01T09:00:00", NA),
    ("Sprinkler S-02", "Sprinkler Head", "2F01", NA, "2017-12-01T09:00:00", NA),
    ("Exam Table ET-01", "Exam Table", "2F02", "ET-77", "2018-05-02T09:00:00", NA),
    ("Work Station WS-01", "Work Station", "1G04", NA, "2018-05-03T09:00:00", NA),
    ("Work Station WS-02", "Work Station", "2F03", NA, "2018-05-03T09:00:00", NA),
    (
        "Panel P1-01",
        "Panelboard P1",
        "1G05",
        "P1-0001",
        "2017-10-02T09:00:00",
        "0aPanelP101GlobalIdxx2",
    ),
    ("Panel P1-02", "Panelboard P1", "1G05", "P1-0002", "2017-10-02T09:00:00", NA),
    # A maintenance type in a room anyone walks through.
    ("Panel P1-03", "Panelboard P1", "1G02", "P1-0003", "2017-10-03T09:00:00", NA),
    ("Ball Valve BV-01", "Ball Valve 50", "1G03", NA, "2017-09-12T09:00:00", NA),
    ("Ball Valve BV-02", "Ball Valve 50", "2F04", NA, "2017-09-12T09:00:00", NA),
    ("AHU-1", "AHU", "3R01", "AHU-9001", "2017-08-01T09:00:00", NA),
    ("AHU-2", "AHU", "2F04", "AHU-9002", "2017-08-01T09:00:00", NA),
    ("Smoke Detector SD-01", "Smoke Detector", "1G02", NA, "2018-02-01T09:00:00", NA),
    ("Smoke Detector SD-02", "Smoke Detector", "2F01", NA, "2018-02-01T09:00:00", NA),
    ("Smoke Detector SD-03", "Smoke Detector", "1G05", NA, "2018-02-01T09:00:00", NA),
]
COMPONENTS = [
    {
        "Name": name,
        "TypeName": kind,
        "Space": space,
        "ExtIdentifier": global_id,
        "SerialNumber": serial,
        "InstallationDate": installed,
        "WarrantyStartDate": installed,
        "TagNumber": NA,
        "BarCode": NA,
        "AssetIdentifier": NA,
    }
    for name, kind, space, serial, installed, global_id in _COMPONENTS
]

# One row per member, as the real sheet has it.
_SYSTEMS = {
    "Fire Alarm": ["Smoke Detector SD-01", "Smoke Detector SD-02", "Smoke Detector SD-03"],
    "HVAC-01": ["AHU-1", "AHU-2", "Ball Valve BV-02"],
    "Sprinklers": ["Sprinkler S-01", "Sprinkler S-02"],
}
SYSTEMS = [
    {"Name": system, "Category": "21-51 51 16: Air Distribution", "ComponentNames": member}
    for system, members in _SYSTEMS.items()
    for member in members
]

JOBS = [
    {
        "Name": "AHU Annual Maintenance",
        "Category": "PM",
        "Status": "Not Yet Started",
        "TypeName": "AHU",
        "Description": "AHU PM",
        "Duration": 180,
        "DurationUnit": "minute",
        "Start": ON,
        "TaskStartUnit": "year",
        "Frequency": 1,
        "FrequencyUnit": "year",
        "TaskNumber": 0,
        "Priors": 0,
        "ResourceNames": "Grease gun, Vacuum",
    },
    {
        "Name": "Air Handler Lockout",
        "Category": "Safety",
        "Status": "Not Yet Started",
        "TypeName": "AHU",
        "Description": "Turn off power at the control panel.",
        "Duration": 30,
        "DurationUnit": "minute",
        "Start": ON,
        "TaskStartUnit": "year",
        "Frequency": NA,
        "FrequencyUnit": NA,
        "TaskNumber": 1,
        "Priors": 0,
        "ResourceNames": NA,
    },
]

# (name, category, sheet, row, file)
_DOCUMENTS = [
    (
        "Light Fixture A Product Data",
        "Product Data",
        "Type",
        "Light Fixture A",
        "LightA-ProductData.pdf",
    ),
    (
        "Panelboard Manual",
        "Operation and Maintenance",
        "Type",
        "Panelboard P1",
        "Panelboard-OM.pdf",
    ),
    # One file for three types. A manual takes the least restrictive of them.
    ("General Warranty", "Warranty Certificate", "Type", "Door Type A", "Warranty-General.pdf"),
    ("General Warranty", "Warranty Certificate", "Type", "Lavatory L1", "Warranty-General.pdf"),
    ("General Warranty", "Product Data", "Type", "Panelboard P1", "Warranty-General.pdf"),
    ("Fire Alarm Test Report", "Test Reports", "System", "Fire Alarm", "FireAlarm-Test.pdf"),
    ("Sprinkler Test Report", "Test Reports", "System ", "Sprinklers", "Sprinkler-Test.pdf"),
    (
        "HVAC Operation and Maintenance",
        "Operation and Maintenance",
        "System",
        "HVAC-01",
        "HVAC-OM.pdf",
    ),
    ("Building Closeout", "Closeout Submittals", "Facility", "B 12", "Closeout.pdf"),
    ("First Floor Plan Notes", "Design Data", "Floor", "First Floor", "FirstFloor-Notes.pdf"),
    ("Electrical Room Schedule", "Design Data", "Space", "1G05", "ElecRoom-Schedule.pdf"),
    ("Reception Finish Schedule", "Design Data", "Space", "1G01", "Reception-Finishes.pdf"),
    ("AHU Startup Report", "Test Reports", "Component", "AHU-1", "AHU-Startup.pdf"),
    (
        "Exam Table Instructions",
        "Manufacturer's Instructions",
        "Type",
        "Exam Table",
        "ExamTable.pdf",
    ),
    # Listed, but the scan has no text behind it.
    ("Lost Scan", "Product Data", "Type", "Light Fixture A", "Lost.pdf"),
    # Keyed to a type the handover does not have.
    ("Ghost", "Product Data", "Type", "No Such Type", "Ghost.pdf"),
]
DOCUMENTS = [
    {
        "Name": name,
        "Category": category,
        "ApprovalBy": "Contractor Certified",
        "Stage": "As Built",
        "SheetName": sheet,
        "RowName": row,
        "Directory": "doc",
        "File": file,
    }
    for name, category, sheet, row, file in _DOCUMENTS
]

# A point for one, a box for another. Metres, whatever the facility sheet says.
# (category, sheet, row, x, y, z)
_COORDINATES = [
    ("point", "Component", "AHU-1", 12.5, 3.0, 8.4),
    ("box-lowerleft", "Component", "Door D-G03", 4.0, 6.0, 0),
    ("box-upperright", "Component", "Door D-G03", 5.0, 6.2, 2.1),
    ("box-lowerleft", "Space", "1G01", 0, 0, 0),
]
COORDINATES = [
    {
        "Name": f"{row}-{category}",
        "Category": category,
        "SheetName": sheet,
        "RowName": row,
        "CoordinateXAxis": x,
        "CoordinateYAxis": y,
        "CoordinateZAxis": z,
    }
    for category, sheet, row, x, y, z in _COORDINATES
]

SHEETS: dict[str, list[dict[str, Any]]] = {
    "Facility": FACILITY,
    "Floor": FLOORS,
    "Space": SPACES,
    "Type": TYPES,
    "Component": COMPONENTS,
    "System": SYSTEMS,
    "Job": JOBS,
    "Document": DOCUMENTS,
    "Coordinate": COORDINATES,
}

# Millimetres, like the real models. The light hangs off a storey placement that is
# itself turned a quarter, so its point only comes out right if the chain is followed.
IFC_TEXT = """ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');
FILE_NAME('model.ifc','2019-05-06T10:00:00',(''),(''),'hand','hand','');
FILE_SCHEMA(('IFC2X3'));
ENDSEC;
DATA;
#1=IFCSIUNIT(*,.LENGTHUNIT.,.MILLI.,.METRE.);
#2=IFCCARTESIANPOINT((0.,0.,0.));
#3=IFCDIRECTION((0.,0.,1.));
#4=IFCDIRECTION((0.,1.,0.));
#5=IFCAXIS2PLACEMENT3D(#2,$,$);
#6=IFCLOCALPLACEMENT($,#5);
#7=IFCCARTESIANPOINT((10000.,2000.,0.));
#8=IFCAXIS2PLACEMENT3D(#7,#3,#4);
#9=IFCLOCALPLACEMENT(#6,#8);
#10=IFCCARTESIANPOINT((1000.,500.,2700.));
#11=IFCAXIS2PLACEMENT3D(#10,$,$);
#12=IFCLOCALPLACEMENT(#9,#11);
#13=IFCFLOWTERMINAL('0aLightA01GlobalIdxxx1',$,'Light, with a comma (and a bracket',$,$,#12,$,$);
#14=IFCCARTESIANPOINT((7000.,
  1500.,0.));
#15=IFCAXIS2PLACEMENT3D(#14,$,$);
#16=IFCLOCALPLACEMENT(#6,#15);
#17=IFCFLOWCONTROLLER('0aPanelP101GlobalIdxx2',$,'Panel',$,$,#16,$,$);
#18=IFCFLOWTERMINAL('0aNobodyListsThisOnexx',$,'Unlisted',$,$,#16,$,$);
#19=IFCBUILDINGSTOREY('0aStoreyGlobalIdxxxxx4',$,'Ground Floor',$,$,#6,$,$,.ELEMENT.,0.);
#20=IFCRELCONTAINEDINSPATIALSTRUCTURE('0aRelationGlobalIdxxx3',$,$,$,(#13,#17),#19);
ENDSEC;
END-ISO-10303-21;
"""

# What the OCR made of each file. No text for Lost.pdf or Ghost.pdf.
OCR_TEXT = {
    "LightA-ProductData": (
        "Light Fixture A product data.\n\n"
        "The downlight takes a compact fluorescent lamp. Replace the lamp when it flickers "
        "or fails to strike. Wipe the reflector with a dry cloth."
    ),
    "Panelboard-OM": (
        "Panelboard operation and maintenance.\n\n"
        "Before servicing, isolate the supply and remove the cover. Tighten every lug to the "
        "torque on the label. Only a qualified electrician may open the board."
    ),
    "Warranty-General": (
        "General warranty certificate.\n\n"
        "The contractor warrants doors and lavatories against defects in material and "
        "workmanship. Report a defect to the facility office in writing."
    ),
    "FireAlarm-Test": (
        "Fire alarm test report.\n\n"
        "Every smoke detector was tested with canned smoke and reported to the panel. "
        "The sounders were heard in each corridor."
    ),
    "Sprinkler-Test": (
        "Sprinkler test report.\n\n"
        "The sprinkler heads were inspected for paint and damage. The flow switch reported "
        "to the fire alarm panel during the drain test."
    ),
    "HVAC-OM": (
        "HVAC operation and maintenance.\n\n"
        "Change the air handler filters when the gauge reads high. Grease the fan bearings "
        "and check the belt for wear. Close the isolation valve before opening the coil."
    ),
    "Closeout": (
        "Building closeout submittal.\n\n"
        "The clinic was handed over with keys, record drawings and this set of documents. "
        "Visitors enter through reception on the ground floor."
    ),
    "FirstFloor-Notes": (
        "First floor plan notes.\n\n"
        "The first floor holds the examination rooms and offices. The corridor runs the "
        "length of the building between the two stairs."
    ),
    "ElecRoom-Schedule": (
        "Electrical room schedule.\n\n"
        "The electrical room holds the panelboards that feed both floors. Keep the door "
        "locked and the floor in front of the boards clear."
    ),
    "Reception-Finishes": (
        "Reception finish schedule.\n\n"
        "Reception has a rubber floor and painted walls. The desk front is laminate with a "
        "lowered section for wheelchair users."
    ),
    "AHU-Startup": (
        "Air handling unit startup report.\n\n"
        "The roof air handler was started and balanced. Supply fan rotation was checked "
        "and the vibration isolators were released."
    ),
    "ExamTable": (
        "Examination table instructions.\n\n"
        "Lock the casters before a patient gets on. Wipe the upholstery with a mild "
        "disinfectant after each use."
    ),
}


def write(directory: Path = DIR) -> None:
    """Write the workbook, the IFC file and the OCR text."""
    import openpyxl

    book = openpyxl.Workbook()
    book.remove(book.active)
    for title, rows in SHEETS.items():
        sheet = book.create_sheet(title)
        columns = ["Name", "CreatedBy", "CreatedOn", *dict.fromkeys(k for r in rows for k in r)]
        columns = list(dict.fromkeys(columns))
        sheet.append(columns)
        for row in rows:
            filled = {"CreatedBy": BY, "CreatedOn": ON, **row}
            sheet.append([filled.get(column, NA) for column in columns])
    directory.mkdir(parents=True, exist_ok=True)
    book.save(directory / WORKBOOK)
    (directory / IFC).write_text(IFC_TEXT, encoding="utf-8")
    ocr = directory / OCR_DIR
    ocr.mkdir(exist_ok=True)
    for stem, text in OCR_TEXT.items():
        (ocr / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")

    def pinned(name: str) -> dict[str, str]:
        return {"file": name, "sha256": hashlib.sha256((directory / name).read_bytes()).hexdigest()}

    definition = {
        "id": VENUE,
        "source": "handover",
        "handover": {"name": "A hand-made handover for tests", "licence": "none, it is made up"},
        "cobie": pinned(WORKBOOK),
        "ifc": [pinned(IFC)],
        "ocr_dir": OCR_DIR,
        "origin": {"lon": 0.0, "lat": 0.0},
        "placeholder_manufacturers": ["supplycompany@usace.army.mil"],
    }
    (directory / DEFINITION.name).write_text(json.dumps(definition, indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    write()
