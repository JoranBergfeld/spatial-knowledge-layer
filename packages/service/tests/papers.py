"""Documents for the fixture snapshot, in a stand-in index, and a stand-in embedder.

One passage per document, and one summary per document beside it, as load writes them:

    manual:b1     manual, type Light Fixture B1 (visitor)
    manual:q4     manual, type Light Fixture Q4 (visitor). Only the Treppenhaus has one.
    manual:p1     manual, type Panelboard P1: maintenance only. Dated.
    page:audimax  page of the Audimax, German and translated. Dated.
    page:sued     page of Seminarraum Süd. Its source was offline at build. Undated.
    page:server   page of the Serverraum: maintenance only
    note:l8       a note on one asset, the light in 50.34 room 108. Its type is a
                  visitor's, so the note is visitor class, but the room is staff's:
                  a visitor cannot see the asset it hangs on.

Every summary says "Precis:" and words its passage does not, so a test can tell a
summary's text from a passage's anywhere it turns up.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import fixture
from spatial_core import clearance
from spatial_core.records import (
    SUMMARY_ORDINAL,
    AnchorKind,
    Chunk,
    ChunkRole,
    DocumentKind,
    LinkStatus,
    Persona,
)
from spatial_core.search import Hit, IndexEntry, IndexRef, LocalIndex, Query
from spatial_service.app import create_app
from spatial_service.retrieval import Documents, Embedded, Retriever
from spatial_service.store import MemoryStore

INDEX = "spatial-mini-20260915"
EMBEDDING = "embed-deployment"
DIMENSIONS = 64
CHECKED_ON = "2026-09-14"

AUDIMAX = "osm:way/1002"
SERVERRAUM = "osm:way/1009"
SUED = "osm:way/2003"
LIGHT_IN_108 = "seed:asset:L8"

B1_TEXT = (
    "Lamp replacement. Switch off the breaker before servicing the fixture. "
    "Wipe the diffuser with a dry cloth."
)
Q4_TEXT = "Replace the starter together with the tube."
P1_TEXT = (
    "Before servicing, isolate the supply and remove the cover. "
    "Torque the busbar bolts once a year."
)
AUDIMAX_TEXT = (
    "Access to the lecture hall is via two steps.\nAn induction loop covers the front rows."
)
SUED_TEXT = "Step-free access is from the courtyard."
SERVER_TEXT = "Access is only with a second person present."
L8_TEXT = "Service note. The ballast of this fixture was replaced."


def _chunk(
    document_id: str,
    kind: DocumentKind,
    title: str,
    text: str,
    *,
    spaces: tuple[str, ...] = (),
    types: tuple[str, ...] = (),
    assets: tuple[str, ...] = (),
    date: str | None = None,
    language: str = "en",
    offline: bool = False,
) -> Chunk:
    by_space = {space.id: space for space in fixture.spaces()}
    by_type = {t.id: t for t in fixture.asset_types()}
    by_asset = {asset.id: asset for asset in fixture.assets()}
    above: set[str] = set()
    for space_id in spaces:
        above.update([space_id, *(a.id for a in by_space[space_id].ancestor_path)])
    for asset_id in assets:
        above.update(a.id for a in by_asset[asset_id].ancestor_path)
    cleared = [
        *((AnchorKind.SPACE, c.persona if (c := by_space[i].clearance) else None) for i in spaces),
        *((AnchorKind.ASSET_TYPE, by_type[i].clearance.persona) for i in types),
        *((AnchorKind.ASSET, by_type[by_asset[i].type_id].clearance.persona) for i in assets),
    ]
    return Chunk(
        id=f"{document_id}#0",
        document_id=document_id,
        ordinal=0,
        kind=kind,
        title=title,
        text=text,
        language=language,
        url=f"https://docs.example/{document_id.replace(':', '/')}",
        original_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        translated=language != "en",
        space_ids=spaces,
        asset_type_ids=types,
        asset_ids=assets,
        ancestor_ids=tuple(sorted(above)),
        clearance_class=clearance.document_clearance(kind, cleared),
        document_date=date,
        link_status=LinkStatus.OFFLINE if offline else LinkStatus.ONLINE,
        link_checked_on=CHECKED_ON,
    )


def passages() -> list[Chunk]:
    manual, page = DocumentKind.MANUAL, DocumentKind.ACCESSIBILITY_PAGE
    return [
        _chunk("manual:b1", manual, "Light Fixture B1 manual", B1_TEXT, types=(fixture.LIGHT_B1,)),
        _chunk("manual:q4", manual, "Light Fixture Q4 manual", Q4_TEXT, types=(fixture.LIGHT_Q4,)),
        _chunk(
            "manual:p1",
            manual,
            "Panelboard P1 manual",
            P1_TEXT,
            types=(fixture.PANEL,),
            date="2021-03-04",
        ),
        _chunk(
            "page:audimax",
            page,
            "Audimax",
            AUDIMAX_TEXT,
            spaces=(AUDIMAX,),
            date="2012-05-31",
            language="de",
        ),
        _chunk("page:sued", page, "Seminarraum Süd", SUED_TEXT, spaces=(SUED,), offline=True),
        _chunk("page:server", page, "Serverraum", SERVER_TEXT, spaces=(SERVERRAUM,)),
        _chunk("note:l8", manual, "Service note", L8_TEXT, assets=(LIGHT_IN_108,)),
    ]


# What each document's summary says. Words its passage does not have.
SUMMARIES = {
    "manual:b1": "Precis: relamping and cleaning of the luminaire.",
    "manual:q4": "Precis: fluorescent upkeep for the stairwell luminaire.",
    "manual:p1": "Precis: annual switchgear inspection routine.",
    "page:audimax": "Precis: wheelchair users need assistance at the entrance.",
    "page:sued": "Precis: barrier-free entry.",
    "page:server": "Precis: escorted entry rules.",
    "note:l8": "Precis: gear swapped in an office luminaire.",
}


def summaries() -> list[Chunk]:
    """One summary chunk per document, with its passage's anchors and clearance."""
    return [
        passage.model_copy(
            update={
                "id": f"{passage.document_id}#summary",
                "role": ChunkRole.SUMMARY,
                "ordinal": SUMMARY_ORDINAL,
                "text": SUMMARIES[passage.document_id],
                "language": "en",
                "translated": False,
            }
        )
        for passage in passages()
    ]


def vector(text: str) -> tuple[float, ...]:
    """A bag of words, hashed into a few buckets. Shared words pull two texts together."""
    buckets = [0.0] * DIMENSIONS
    for word in re.findall(r"\w+", text.casefold()):
        buckets[hashlib.sha256(word.encode()).digest()[0] % DIMENSIONS] += 1
    norm = math.sqrt(sum(x * x for x in buckets)) or 1.0
    return tuple(round(x / norm, 6) for x in buckets)


def entries() -> list[IndexEntry]:
    return [
        IndexEntry(chunk=chunk, vector=vector(f"{chunk.title}\n\n{chunk.text}"))
        for chunk in (*passages(), *summaries())
    ]


class Embedder:
    """Embeds like the index was, and keeps what it was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def embed(self, deployment: str, text: str) -> Embedded:
        self.asked.append((deployment, text))
        return Embedded(vector=vector(text), deployment=deployment, input_tokens=7)


class Recording:
    """An index that keeps every query it gets, and answers from the one behind it."""

    def __init__(self, behind: LocalIndex) -> None:
        self.behind = behind
        self.queries: list[Query] = []

    def search(self, name: str, query: Query) -> list[Hit]:
        self.queries.append(query)
        return self.behind.search(name, query)


@cache
def index() -> LocalIndex:
    """The stand-in index holding the fixture's documents. Written once per run."""
    local = LocalIndex(Path(tempfile.mkdtemp(prefix="spatial-prose-")))
    local.create(INDEX, DIMENSIONS)
    local.upload(INDEX, entries())
    return local


def documents() -> Retriever:
    return Retriever(index(), Embedder())


def store() -> MemoryStore:
    held = fixture.store()
    held.index = IndexRef(INDEX, EMBEDDING)
    return held


# --- plans and composers -----------------------------------------------------------------


def manual(query: str, place: str | None = None, kind: str | None = None) -> str:
    """An asset-document plan: the assets in ``place`` of type ``kind``, then their documents."""
    plan = json.loads(fixture.installed(place, kind))
    plan["shape"] = "asset-document"
    plan["steps"].append(
        {
            "id": f"s{len(plan['steps']) + 1}",
            "tool": "retrieve_documents",
            "anchors": {"step": plan["steps"][-1]["id"]},
            "query": query,
        }
    )
    return json.dumps(plan)


def page(query: str, place: str) -> str:
    """A place-document plan: the place, then the documents anchored there."""
    plan = json.loads(fixture.where(place))
    plan["shape"] = "place-document"
    plan["steps"].append(
        {"id": "s2", "tool": "retrieve_documents", "anchors": {"step": "s1"}, "query": query}
    )
    return json.dumps(plan)


def first_sentence(text: str) -> str:
    return text.split(". ")[0].split("\n")[0].rstrip(".") + "."


def _about(payload: dict[str, Any]) -> dict[str, Any]:
    """The segment of facts a document answer opens with."""
    if "subject" in payload:
        return {"text": f"About {{{payload['subject']}}}.", "cites": [payload["subject"]]}
    refs = " and ".join(f"{{{ref}}}" for ref in payload["total"])
    return {
        "text": f"{{{payload['scope']}}} holds {refs}.",
        "cites": [payload["scope"], *payload["total"]],
    }


def quoting(payload: dict[str, Any]) -> str:
    """A composer that says what was found, then quotes each passage's first sentence."""
    segments = [_about(payload)]
    for passage in payload["passages"]:
        segments.append(
            {
                "text": f'It says "{first_sentence(passage["text"])}"',
                "cites": [passage["ref"]],
            }
        )
    return json.dumps({"segments": segments})


def saying(*segments: dict[str, Any]) -> Any:
    """A composer that opens with the facts and then writes exactly these segments."""

    def composer(payload: dict[str, Any]) -> str:
        return json.dumps({"segments": [_about(payload), *segments]})

    return composer


def ask(
    persona: str,
    question: str,
    planner: Any,
    composer: Any = quoting,
    *,
    retriever: Documents | None = None,
    inputs: list[str] | None = None,
    status: int = 200,
) -> dict[str, Any]:
    llm = fixture.llm(
        planner if callable(planner) or planner is None else (lambda _p: planner),
        composer,
        inputs if inputs is not None else [],
    )
    app = create_app(store(), llm, documents=retriever or documents())
    response = TestClient(app).post("/turn", json={"persona": persona, "question": question})
    assert response.status_code == status, response.text
    result: dict[str, Any] = response.json()
    return result


def retrieved(turn: dict[str, Any]) -> list[str]:
    return [row["chunk_id"] for row in turn["trace"]["documents"]["retrieved"]]


def texts(turn: dict[str, Any]) -> list[str]:
    return [segment["text"] for segment in turn["segments"]]


def visible_to(persona: Persona) -> set[str]:
    return {row.space_id for row in fixture.visible() if row.persona is persona}
