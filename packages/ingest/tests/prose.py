"""A corpus for the mini campus, with one of every case document ingestion has to handle.

hall:gaede     page, German, building 50.34, dated 31.05.2012, numbers of every kind
hall:bau       page, German, building 10.50, says nothing of when it changed
hall:nowhere   page for building 77.77, which the campus does not have
hall:gone      page for building 20.30, and the site answers 404
room:107       page, English, room 1001 (access=private) and its building
manual:b1      manual for the B1 light (visitor), long enough for two chunks
manual:air     one PDF for the VAV box (maintenance) and the diffuser (visitor)
manual:valve   manual for the ball valve (maintenance) and a chiller nobody placed
manual:chiller manual for that chiller alone
manual:ext     manual anchored to one extinguisher, the asset and not its type
"""

from __future__ import annotations

import hashlib
from typing import Any

CONTENT_BETWEEN = ["<!-- CONTENT-AREA START -->", "<!-- CONTENT-AREA END -->"]
TRANSLATION_DEPLOYMENT = "gpt-5.4-mini"
TRANSLATION_MODEL = "gpt-5.4-mini-2026-03-17"
EMBEDDING_DEPLOYMENT = "text-embedding-3-small"
EMBEDDING_MODEL = "text-embedding-3-small-1"
DIMENSIONS = 4


def pdf(*lines: str) -> bytes:
    """A one page PDF saying those lines, written by hand."""
    stream = "BT /F1 12 Tf 72 720 Td " + " ".join(f"({line}) Tj 0 -14 Td" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    body = "%PDF-1.4\n"
    offsets = []
    for number, content in enumerate(objects, start=1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{content}\nendobj\n"
    xref = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return body.encode("latin-1")


def _page(content: str, footer: str = "") -> bytes:
    return f"""<!doctype html>
<html lang="de-de"><head><title>KIT - Hörsäle</title></head>
<body>
<nav><a href="/index.php">Startseite</a></nav>
{CONTENT_BETWEEN[0]}
{content}
{CONTENT_BETWEEN[1]}
<footer>{footer}</footer>
</body></html>
""".encode()


GAEDE_GERMAN = (
    "50.34 Gaede Hs\n\n"
    "Stockwerk: 2. OG\n"
    "Zugang zum HS: über 2 Stufen, Rampe 90 cm breit\n"
    "Geöffnet ab 7:30\n"
    "Mikroport Anlage: Kanal 2, Frequenz 37.1 MHz"
)
# What the stand-in translator answers, digits and all.
GAEDE_TRANSLATED = (
    "50.34 Gaede Lecture Hall\n\n"
    "Floor: 2nd upper floor\n"
    "Access to the lecture hall: via 2 steps, ramp 90 cm wide\n"
    "Open from 7:30\n"
    "Microport system: Channel 2, frequency 37.1 MHz"
)
# And what the chunk says once numbers-in-words has run.
GAEDE_ENGLISH = (
    "50.34 Gaede Lecture Hall\n\n"
    "Floor: second upper floor\n"
    "Access to the lecture hall: via two steps, ramp 90 cm wide\n"
    "Open from seven thirty\n"
    "Microport system: Channel 2, frequency 37.1 MHz"
)
BAU_GERMAN = "10.50 Großer Hs\n\nZugang ins Gebäude: ja"
BAU_ENGLISH = "10.50 Large Lecture Hall\n\nAccess to the building: yes"
TRANSLATIONS = {GAEDE_GERMAN: GAEDE_TRANSLATED, BAU_GERMAN: BAU_ENGLISH}

ROOM_107 = "Room 107 is reached through the staff corridor. Ring the bell at the door."
B1_PARAGRAPHS = [
    "B1 light fixture. Maintenance instructions.",
    "Lamp replacement. " + "Switch off the breaker before opening the housing. " * 14,
    "Cleaning. " + "Wipe the diffuser with a dry cloth. " * 22,
]
AIR_LINES = ["Air terminal units.", "Check the damper every 12 months."]
VALVE = "Ball valve, 50 mm. Close the valve before servicing the line."
EXTINGUISHER = "Powder extinguisher. Check the pressure gauge is in the green."

PAGES: dict[str, tuple[bytes, str]] = {
    "/halls/gaede.html": (
        _page(
            "<h3>50.34 Gaede Hs</h3><ul>"
            '<li><span class="title">Stockwerk:</span><span>2. OG</span></li>'
            '<li><span class="title">Zugang zum HS:</span>'
            "<span>&uuml;ber 2 Stufen, Rampe 90 cm breit</span></li>"
            '<li><span class="title">Ge&ouml;ffnet ab</span><span>7:30</span></li>'
            '<li><span class="title">Mikroport Anlage:</span>'
            "<span>Kanal 2, Frequenz 37.1 MHz</span></li></ul>",
            footer='<span class="last_change">letzte Änderung:\n 31.05.2012</span>',
        ),
        "text/html; charset=utf-8",
    ),
    "/halls/bau.html": (
        _page(
            "<h3>10.50 Gro&szlig;er Hs</h3><ul>"
            '<li><span class="title">Zugang ins Geb&auml;ude:</span><span>ja</span></li></ul>'
        ),
        "text/html; charset=utf-8",
    ),
    "/halls/nowhere.html": (_page("<h3>77.77 Nirgendwo</h3>"), "text/html"),
    "/rooms/107.html": (f"<html><body><p>{ROOM_107}</p></body></html>".encode(), "text/html"),
    "/manuals/b1.txt": ("\n\n".join(B1_PARAGRAPHS).encode(), "text/plain; charset=utf-8"),
    "/manuals/air.pdf": (pdf(*AIR_LINES), "application/pdf"),
    "/manuals/valve.txt": (VALVE.encode(), "text/plain"),
    "/manuals/chiller.txt": (b"Chiller. Not placed anywhere.", "text/plain"),
    "/manuals/ext.txt": (EXTINGUISHER.encode(), "text/plain"),
}

B1, VAV, DIFFUSER, VALVE_TYPE, CHILLER = (
    "seed:type:light-fixture-b1",
    "seed:type:vav-200-mm",
    "seed:type:supply-diffuser-600-face",
    "seed:type:ball-valve-50-mm",
    "seed:type:chiller-x",
)
EXTINGUISHER_ASSET = "osm:asset:node/1201"


def corpus(site: str) -> dict[str, Any]:
    """The corpus file, with every source on the stand-in site at ``site``."""

    def page(source_id: str, path: str, title: str, *anchors: dict[str, str]) -> dict[str, Any]:
        return {
            "id": source_id,
            "kind": "accessibility-page",
            "language": "de",
            "title": title,
            "url": site + path,
            "anchors": list(anchors),
        }

    def manual(source_id: str, path: str, title: str, *anchors: dict[str, str]) -> dict[str, Any]:
        return {**page(source_id, path, title, *anchors), "kind": "manual", "language": "en"}

    return {
        "venue": "mini-campus",
        "html_content_between": CONTENT_BETWEEN,
        "documents": [
            page("hall:gaede", "/halls/gaede.html", "50.34 Gaede Hs", {"building": "50.34"}),
            page("hall:bau", "/halls/bau.html", "10.50 Großer Hs", {"building": "10.50"}),
            page("hall:nowhere", "/halls/nowhere.html", "77.77 Nirgendwo", {"building": "77.77"}),
            page("hall:gone", "/halls/gone.html", "20.30 Hs", {"building": "20.30"}),
            {
                **page(
                    "room:107",
                    "/rooms/107.html",
                    "50.34 room 107",
                    {"space": "osm:way/1001"},
                    {"building": "50.34"},
                ),
                "language": "en",
            },
            manual("manual:b1", "/manuals/b1.txt", "B1 light fixture manual", {"asset_type": B1}),
            manual(
                "manual:air",
                "/manuals/air.pdf",
                "Air terminal units manual",
                {"asset_type": VAV},
                {"asset_type": DIFFUSER},
            ),
            manual(
                "manual:valve",
                "/manuals/valve.txt",
                "Ball valve manual",
                {"asset_type": VALVE_TYPE},
                {"asset_type": CHILLER},
            ),
            manual("manual:chiller", "/manuals/chiller.txt", "Chiller", {"asset_type": CHILLER}),
            manual(
                "manual:ext",
                "/manuals/ext.txt",
                "Extinguisher manual",
                {"asset": EXTINGUISHER_ASSET},
            ),
        ],
    }


def translation(text: str) -> str:
    return TRANSLATIONS.get(text, "EN: " + text)


def asks_for_summary(system: str) -> bool:
    return system.startswith("Summarise")


SUMMARY_WORDS = 40


def summary(text: str) -> str:
    """What the stand-in summariser answers: the first words it was shown, and nothing else.

    So whatever a summary says, it says because it was in what the model read.
    """
    return "Covers: " + " ".join(text.split()[:SUMMARY_WORDS])


def embedding(text: str) -> list[float]:
    """A small vector that depends on nothing but the text."""
    digest = hashlib.sha256(text.encode()).digest()
    return [round(byte / 255, 4) for byte in digest[:DIMENSIONS]]
