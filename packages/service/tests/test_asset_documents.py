"""Asset plus document and place document answers, end to end over HTTP.

The fixture snapshot, with papers.py's documents in a stand-in index and a
scripted model. The search and the embedder are stand-ins too.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

import fixture
import papers
from papers import ask, manual, page, quoting, retrieved, saying, texts
from spatial_core.search import Hit, Query
from spatial_service.app import create_app
from spatial_service.retrieval import RetrievalUnavailable, Retriever

P1 = "manual:p1#0"
B1 = "manual:b1#0"
Q4 = "manual:q4#0"


# --- the manual for what is in a room -----------------------------------------------------


def test_the_asset_and_the_cited_passage_come_in_one_answer() -> None:
    turn = ask(
        "maintenance",
        "What does the manual say about servicing the panelboard in the Audimax?",
        manual("servicing", "Audimax", "panelboard"),
    )

    assert turn["kind"] == "answer"
    assert turn["trace"]["shape"] == "asset-document"
    assert texts(turn) == [
        "Audimax holds 1 generated Panelboard P1.",
        'It says "Before servicing, isolate the supply and remove the cover."',
    ]
    fact, said = turn["segments"]
    assert fact["passages"] == []
    assert said["citations"] == []
    # The segment carries the whole passage it cites, for the entailment check elsewhere.
    assert said["passages"] == [
        {
            "ref": "p1",
            "chunk_id": P1,
            "document_id": "manual:p1",
            "kind": "manual",
            "title": "Panelboard P1 manual",
            "text": papers.P1_TEXT,
            "quotes": ["Before servicing, isolate the supply and remove the cover."],
            "url": "https://docs.example/manual/p1",
            "link_status": "online",
            "link_checked_on": papers.CHECKED_ON,
            "translated": False,
            "language": "en",
        }
    ]
    [version] = turn["trace"]["plans"]
    assert [s["tool"] for s in version["steps"]] == [
        "resolve_place",
        "resolve_asset_type",
        "assets_in_scope",
        "retrieve_documents",
    ]
    assert version["steps"][-1]["ids"] == [P1]
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "composition"]
    assert turn["embedding_calls"] == [{"deployment": papers.EMBEDDING, "input_tokens": 7}]
    assert turn["state"]["place"] == papers.AUDIMAX


def test_the_trace_lists_what_was_retrieved_with_its_document_date() -> None:
    turn = ask(
        "maintenance", "Servicing the panelboard?", manual("servicing", "Audimax", "panelboard")
    )

    found = turn["trace"]["documents"]
    assert (found["step"], found["query"], found["index"]) == ("s4", "servicing", papers.INDEX)
    assert (found["ranking"], found["semantic_ranker"], found["top"]) == ("hybrid", False, 5)
    assert (found["anchored_by"], found["scope"]) == ("assets", papers.AUDIMAX)
    assert found["asset_types"] == [fixture.PANEL]
    assert found["clearance_classes"] == ["visitor", "staff", "maintenance"]
    [row] = found["retrieved"]
    assert row["chunk_id"] == P1
    assert row["document_date"] == "2021-03-04"
    assert row["clearance_class"] == "maintenance"
    assert row["asset_type_ids"] == [fixture.PANEL]


def test_a_maintenance_only_manual_is_there_for_the_technician_and_not_for_a_visitor() -> None:
    question = "What do the manuals say about servicing what is installed in the Audimax?"
    seen: list[str] = []
    plan = manual("isolate the supply before servicing", "Audimax")
    visitor = ask("visitor", question, plan, inputs=seen)
    technician = ask("maintenance", question, plan)

    assert retrieved(technician) == [P1, B1]
    assert retrieved(visitor) == [B1]
    assert texts(visitor) == [
        "Audimax holds 3 generated assets and 1 sourced asset.",
        'It says "Lamp replacement."',
    ]
    assert texts(technician)[1:] == [
        'It says "Before servicing, isolate the supply and remove the cover."',
        'It says "Lamp replacement."',
    ]
    # Nothing of the panelboard's manual is in what a visitor gets, or in what a model was shown.
    for told in (json.dumps(visitor), "\n".join(seen)):
        for word in ("Panelboard", "manual:p1", "busbar", "remove the cover", fixture.PANEL):
            assert word not in told
    assert visitor["trace"]["documents"]["clearance_classes"] == ["visitor"]
    for word in ("hidden", "restricted", "clearance", "not shown", "omitted", "left out"):
        assert word not in " ".join(texts(visitor)).lower()


def test_asking_for_a_manual_beyond_clearance_is_asking_for_one_that_does_not_exist() -> None:
    beyond = ask("visitor", "The panelboard manual?", manual("servicing", "Audimax", "panelboard"))
    nowhere = ask("visitor", "The flux manual?", manual("servicing", "Audimax", "flux capacitor"))

    assert (beyond["kind"], beyond["cause"]) == ("abstention", "no-such-asset-type")
    assert beyond["segments"] == nowhere["segments"]
    assert beyond["trace"]["documents"] is None


def test_a_type_anchored_manual_comes_only_through_a_visible_asset_of_that_type() -> None:
    # Seminarraum Nord has a B1 light and no Q4. Both manuals are a visitor's to read.
    nord = ask("visitor", "Manuals for Seminarraum Nord?", page("servicing", "Seminarraum Nord"))
    stairs = ask("visitor", "Manuals for the Treppenhaus?", page("servicing", "Treppenhaus"))

    assert retrieved(nord) == [B1]
    assert retrieved(stairs) == [Q4]


def test_a_document_tied_to_the_scope_only_by_something_unseen_is_not_retrieved() -> None:
    # The note hangs on a light in a staff room of 50.34. It is visitor class, and
    # 50.34 is on its ancestor path, but a visitor cannot see the light.
    visitor = ask("visitor", "Service notes for 50.34?", page("service note ballast", "50.34"))
    staff = ask("staff", "Service notes for 50.34?", page("service note ballast", "50.34"))

    assert "note:l8#0" not in retrieved(visitor)
    assert retrieved(staff)[0] == "note:l8#0"
    assert staff["trace"]["documents"]["retrieved"][0]["asset_ids"] == [papers.LIGHT_IN_108]
    assert papers.LIGHT_IN_108 not in json.dumps(visitor)


# --- facts and passages -------------------------------------------------------------------


def test_facts_and_passages_reach_composition_as_two_lists() -> None:
    shown: list[dict[str, Any]] = []
    seen: list[str] = []

    def composer(payload: dict[str, Any]) -> str:
        shown.append(payload)
        return quoting(payload)

    ask("maintenance", "?", manual("servicing", "Audimax", "panelboard"), composer, inputs=seen)

    [payload] = shown
    assert payload["passages"] == [
        {"ref": "p1", "kind": "manual", "title": "Panelboard P1 manual", "text": papers.P1_TEXT}
    ]
    # A fact is a reference and a kind. No value, and no score next to either list.
    assert all(set(fact) <= {"ref", "kind", "tier", "assets", "about"} for fact in payload["facts"])
    assert "score" not in json.dumps(payload)
    # A passage's id, link and date stay with the host.
    for told in seen:
        for word in (P1, "docs.example", "2021-03-04", papers.CHECKED_ON):
            assert word not in told


@pytest.mark.parametrize(
    "segment",
    [
        {"text": "Torque the bolts every 12 months.", "cites": ["p1"]},
        {"text": 'It says "torque them 1 time a year".', "cites": ["p1"]},
    ],
)
def test_a_number_written_into_a_document_segment_drops_that_segment(
    segment: dict[str, Any],
) -> None:
    plan = manual("servicing", "Audimax", "panelboard")
    turn = ask("maintenance", "?", plan, saying(segment))

    # The only passage segment is gone, so there is no document answer left.
    assert (turn["kind"], turn["cause"]) == ("abstention", "ungrounded")
    assert turn["trace"]["composition_rejection"] == []
    [note] = turn["trace"]["dropped_segments"]
    assert note["segment"] == 2
    assert "writes a digit" in note["reasons"]


# --- provenance ---------------------------------------------------------------------------


def test_a_quote_that_is_not_in_the_passage_drops_its_segment_and_the_trace_says_so() -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying(
            {"text": 'It says "isolate the supply and remove the cover".', "cites": ["p1"]},
            {"text": 'It also says "never open the cover alone".', "cites": ["p1"]},
        ),
    )

    assert turn["kind"] == "answer"
    assert texts(turn) == [
        "Audimax holds 1 generated Panelboard P1.",
        'It says "isolate the supply and remove the cover".',
    ]
    assert turn["trace"]["dropped_segments"] == [
        {
            "stage": "provenance",
            "process": "spatial-service",
            "segment": 3,
            "reasons": ["quotes text that is not in a passage it cites"],
        }
    ]


def test_a_segment_citing_a_passage_not_retrieved_this_turn_is_dropped() -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying(
            {"text": "Isolate the supply first.", "cites": ["p1"]},
            {"text": "Wear gloves.", "cites": ["p7"]},
        ),
    )

    assert texts(turn) == ["Audimax holds 1 generated Panelboard P1.", "Isolate the supply first."]
    [dropped] = turn["trace"]["dropped_segments"]
    assert dropped["segment"] == 3
    assert dropped["reasons"] == ["cites p7, which is not a passage retrieved this turn"]


def test_an_answer_with_no_document_segment_left_is_an_abstention() -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying({"text": 'It says "call the manufacturer".', "cites": ["p1"]}),
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "ungrounded")
    assert len(turn["trace"]["dropped_segments"]) == 1
    assert turn["state"]["turns"] == 1


@pytest.mark.parametrize(
    "segment",
    [
        # What the first real run did: a fact said, and a passage cited for it.
        {"text": "{f1} is shown in the manual.", "cites": ["f1", "p1"]},
        {"text": "It is installed there.", "cites": ["f1", "p1"]},
    ],
)
def test_a_segment_naming_a_fact_and_citing_a_passage_is_dropped(segment: dict[str, Any]) -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying(segment, {"text": "Isolate the supply first.", "cites": ["p1"]}),
    )

    assert texts(turn) == ["Audimax holds 1 generated Panelboard P1.", "Isolate the supply first."]
    [dropped] = turn["trace"]["dropped_segments"]
    assert dropped["segment"] == 2
    assert dropped["reasons"] == [
        "names a fact and cites a passage, and a segment rests on one or the other"
    ]


@pytest.mark.parametrize(
    "text",
    [
        "The documents do not say anything about this.",
        "The manual doesn't mention torque.",
        "It says nothing about the cover.",
        "There is no information on that.",
        "None of the documents cover it.",
    ],
)
def test_saying_the_documents_say_nothing_is_no_document_answer(text: str) -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying({"text": text, "cites": ["p1"]}),
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "ungrounded")
    [dropped] = turn["trace"]["dropped_segments"]
    assert dropped["reasons"] == ["says what the documents do not say"]


def test_a_passage_telling_you_not_to_do_something_is_not_saying_nothing() -> None:
    turn = ask(
        "maintenance",
        "?",
        manual("servicing", "Audimax", "panelboard"),
        saying({"text": "The manual says do not open the cover alone.", "cites": ["p1"]}),
    )

    assert turn["kind"] == "answer"
    assert turn["trace"]["dropped_segments"] == []


def test_a_quote_may_run_across_a_line_break_of_the_passage() -> None:
    turn = ask(
        "visitor",
        "?",
        page("access", "Audimax"),
        saying({"text": 'It says "via two steps. An induction loop".', "cites": ["p1"]}),
    )

    assert turn["kind"] == "answer"
    assert turn["segments"][1]["passages"][0]["quotes"] == ["via two steps. An induction loop"]


# --- place documents, translations and dead links -----------------------------------------


def test_a_place_page_is_quoted_in_translation_and_links_the_original() -> None:
    turn = ask(
        "visitor", "What does the Audimax's page say about access?", page("access", "Audimax")
    )

    assert turn["trace"]["shape"] == "place-document"
    assert texts(turn)[:2] == [
        "About Audimax.",
        'It says "Access to the lecture hall is via two steps."',
    ]
    cited = turn["segments"][1]["passages"][0]
    assert (cited["translated"], cited["language"]) == (True, "de")
    assert cited["url"] == "https://docs.example/page/audimax"
    assert cited["text"] == papers.AUDIMAX_TEXT
    row = turn["trace"]["documents"]["retrieved"][0]
    assert (row["chunk_id"], row["translated"], row["language"]) == ("page:audimax#0", True, "de")
    assert row["document_date"] == "2012-05-31"
    assert row["space_ids"] == [papers.AUDIMAX]
    assert turn["trace"]["documents"]["anchored_by"] == "places"


def test_a_source_that_was_offline_at_build_is_cited_unlinked_with_that_date() -> None:
    turn = ask("visitor", "Is Seminarraum Süd step-free?", page("step-free", "Seminarraum Süd"))

    cited = turn["segments"][1]["passages"][0]
    assert cited["url"] is None
    assert (cited["link_status"], cited["link_checked_on"]) == ("offline", "2026-09-14")
    row = turn["trace"]["documents"]["retrieved"][0]
    assert (row["url"], row["link_status"], row["link_checked_on"]) == (
        None,
        "offline",
        "2026-09-14",
    )
    # It says nothing of when it changed.
    assert row["document_date"] is None
    assert "docs.example" not in json.dumps(turn)


# --- retrieval returning nothing ----------------------------------------------------------


def test_retrieval_returning_nothing_buys_one_recovery_and_then_abstains() -> None:
    plan = page("access", "30.10 room 107")
    turn = ask("visitor", "What does the page of 30.10 room 107 say?", plan, None)

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-document")
    assert texts(turn) == ["I don't know of a document about that."]
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "recovery"]
    assert turn["recovery"] == "unrepaired"
    for version in turn["trace"]["plans"]:
        assert version["trigger"] == {
            "kind": "empty-retrieval",
            "step": "s2",
            "reasons": ["retrieval returned nothing"],
        }
        assert version["steps"][-1]["outcome"] == "none"
    assert turn["trace"]["documents"] is None


def test_a_second_plan_that_retrieves_something_is_a_repair() -> None:
    plans = iter([page("access", "30.10 room 107"), page("access", "Audimax")])
    turn = ask("visitor", "?", lambda _payload: next(plans))

    assert turn["kind"] == "answer"
    assert turn["recovery"] == "repaired"
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning", "recovery", "composition"]


def test_the_page_of_a_place_beyond_clearance_ends_like_the_page_of_no_place() -> None:
    # The Serverraum has a page. Both are a technician's alone.
    beyond = ask("staff", "?", page("access", "Serverraum"), None)
    nowhere = ask("staff", "?", page("access", "Bibliothek"), None)
    technician = ask("maintenance", "?", page("access", "Serverraum"))

    assert (beyond["kind"], beyond["cause"]) == ("abstention", "no-such-place")
    assert beyond["segments"] == nowhere["segments"]
    assert [c["purpose"] for c in beyond["llm_calls"]] == ["planning"]
    assert retrieved(technician)[0] == "page:server#0"


def test_a_query_with_nothing_to_search_for_is_a_malformed_argument() -> None:
    turn = ask("visitor", "?", page("?!", "Audimax"), None)

    [first, _] = turn["trace"]["plans"]
    assert first["trigger"]["kind"] == "malformed-arguments"
    assert first["steps"][0]["tool"] == "retrieve_documents"


@pytest.mark.parametrize(
    ("plan", "reason"),
    [
        (
            json.loads(page("access", "Audimax")) | {"shape": "asset-document"},
            "a asset-document plan retrieves by assets, but step s2 reads places",
        ),
        (
            json.loads(manual("servicing", "Audimax")) | {"shape": "place-document"},
            "a place-document plan retrieves by places, but step s3 reads assets",
        ),
    ],
)
def test_a_document_plan_anchored_by_the_wrong_thing_is_invalid(
    plan: dict[str, Any], reason: str
) -> None:
    turn = ask("visitor", "?", json.dumps(plan), None)

    assert turn["cause"] == "plan-rejected"
    assert turn["trace"]["plans"][0]["rejection"] == [reason]


# --- the manual of the one asset an earlier turn found --------------------------------------

ROOM_108 = "osm:way/1004"
LIGHT_NORD = "seed:asset:L9"
NOTE = "note:l8#0"


def held(persona: str, asset: str) -> dict[str, Any]:
    """The state an earlier turn about one asset leaves behind."""
    return {
        "snapshot": fixture.SNAPSHOT,
        "persona": persona,
        "turns": 1,
        "assets": [asset],
        "since": {"assets": 1},
    }


def test_its_manual_is_retrieved_by_the_one_asset_an_earlier_turn_found() -> None:
    # Staff asked about the light in room 108. A note hangs on that very asset.
    index = papers.Recording(papers.index())
    turn = ask(
        "staff",
        "What does its manual say?",
        papers.its_manual("ballast"),
        retriever=Retriever(index, papers.Embedder()),
        state=held("staff", papers.LIGHT_IN_108),
    )

    assert turn["kind"] == "answer"
    assert turn["trace"]["shape"] == "asset-document"
    # The asset itself, and its type. Nothing else in the room, nothing else of the type.
    [query] = index.queries
    assert query.filter.asset_ids == {papers.LIGHT_IN_108}
    assert query.filter.asset_type_ids == {fixture.LIGHT_B1}
    assert query.filter.space_ids == frozenset()
    assert sorted(retrieved(turn)) == [B1, NOTE]
    assert texts(turn)[0] == "50.34 room 108 holds 1 generated Light Fixture B1."
    assert 'It says "Service note."' in texts(turn)
    documents = turn["trace"]["documents"]
    assert (documents["anchored_by"], documents["scope"], documents["assets"]) == (
        "assets",
        ROOM_108,
        1,
    )
    [reference] = turn["trace"]["references"]
    assert (reference["kind"], reference["ids"]) == ("asset", [papers.LIGHT_IN_108])
    # It is still the asset the conversation is about.
    assert turn["state"]["assets"] == [papers.LIGHT_IN_108]
    assert turn["state"]["place"] == ROOM_108


def test_a_visitor_gets_the_manual_of_their_own_asset_and_never_the_note_on_another() -> None:
    # The light in Seminarraum Nord is of the same type as the one in room 108.
    inputs: list[str] = []
    turn = ask(
        "visitor",
        "What does its manual say?",
        papers.its_manual("ballast"),
        inputs=inputs,
        state=held("visitor", LIGHT_NORD),
    )

    assert turn["kind"] == "answer"
    assert retrieved(turn) == [B1]
    assert texts(turn)[0] == "Seminarraum Nord holds 1 generated Light Fixture B1."
    for told in (json.dumps(turn), "".join(inputs)):
        assert "Service note" not in told
        assert papers.LIGHT_IN_108 not in told


def test_its_manual_for_an_asset_the_persona_cannot_see_refers_to_nothing() -> None:
    # A state a visitor could not have been handed: the light in the staff room.
    index = papers.Recording(papers.index())
    turn = ask(
        "visitor",
        "What does its manual say?",
        papers.its_manual("ballast"),
        None,
        retriever=Retriever(index, papers.Embedder()),
        state=held("visitor", papers.LIGHT_IN_108),
    )

    assert (turn["kind"], turn["cause"]) == ("abstention", "nothing-referred-to")
    assert index.queries == []
    assert papers.LIGHT_IN_108 not in json.dumps(turn)


def test_a_place_document_plan_cannot_be_anchored_by_an_asset() -> None:
    plan = json.loads(papers.its_manual("access")) | {"shape": "place-document"}

    turn = ask("visitor", "?", json.dumps(plan), None, state=held("visitor", LIGHT_NORD))

    assert turn["cause"] == "plan-rejected"
    assert turn["trace"]["plans"][0]["rejection"] == [
        "a place-document plan retrieves by places, but step s2 reads asset"
    ]


# --- the search ---------------------------------------------------------------------------


def test_the_index_is_asked_for_both_rankings_under_the_whole_filter() -> None:
    index = papers.Recording(papers.index())
    embedder = papers.Embedder()
    plan = manual("servicing", "Audimax")
    ask("visitor", "?", plan, retriever=Retriever(index, embedder))

    [query] = index.queries
    assert query.text == "servicing"
    assert query.vector == papers.vector("servicing")
    assert embedder.asked == [(papers.EMBEDDING, "servicing")]
    # The assets found, and the types of those assets. No panelboard for a visitor.
    assert query.filter.clearance_classes == ("visitor",)
    assert query.filter.asset_ids == {
        "seed:asset:L1",
        "seed:asset:L2",
        "seed:asset:L3",
        "osm:asset:node/9001",
    }
    assert query.filter.asset_type_ids == {fixture.LIGHT_B1, fixture.EXTINGUISHER}
    assert query.filter.space_ids == frozenset()


def test_the_semantic_ranker_is_off_unless_switched_on() -> None:
    plan = page("access", "Audimax")
    off, on = papers.Recording(papers.index()), papers.Recording(papers.index())

    default = ask("visitor", "?", plan, retriever=Retriever(off, papers.Embedder()))
    switched = ask(
        "visitor", "?", plan, retriever=Retriever(on, papers.Embedder(), semantic_ranker=True)
    )

    assert [q.semantic_ranker for q in off.queries] == [False]
    assert [q.semantic_ranker for q in on.queries] == [True]
    assert default["trace"]["documents"]["semantic_ranker"] is False
    assert switched["trace"]["documents"]["semantic_ranker"] is True


def test_with_no_retrieval_bound_a_document_question_finds_nothing() -> None:
    llm = fixture.llm(lambda _p: page("access", "Audimax"), None, [])
    app = create_app(papers.store(), llm)
    turn = TestClient(app).post("/turn", json={"persona": "visitor", "question": "?"}).json()

    assert (turn["kind"], turn["cause"]) == ("abstention", "no-document")


def test_an_index_that_cannot_be_reached_answers_nothing() -> None:
    class Down:
        def search(self, name: str, query: Query) -> list[Hit]:
            raise RetrievalUnavailable("connection refused")

    turn = ask("visitor", "?", page("access", "Audimax"), retriever=Retriever(Down()), status=503)

    assert turn["kind"] == "unavailable"
    assert turn["trace"]["unavailable"] == "retrieval"
    assert [c["purpose"] for c in turn["llm_calls"]] == ["planning"]
    # The state goes back as it came, so the client can ask again.
    assert turn["state"]["turns"] == 0


# --- summaries ----------------------------------------------------------------------------


def _no_summary_text(turn: dict[str, Any], seen: list[str]) -> None:
    told = json.dumps(turn) + "\n".join(seen)
    assert "Precis" not in told
    assert "#summary" not in told
    for summary in papers.SUMMARIES.values():
        assert summary not in told


def test_a_summary_ranks_its_document_and_its_text_shows_up_nowhere() -> None:
    # Only the Q4 manual's summary speaks of fluorescent upkeep. Its passage does not.
    seen: list[str] = []
    plan = manual("fluorescent upkeep", "50.34 level 0", "lights")
    turn = ask("visitor", "Fluorescent upkeep on level 0?", plan, inputs=seen)

    assert turn["kind"] == "answer"
    assert retrieved(turn) == [Q4, B1]
    assert turn["trace"]["documents"]["retrieved"][0]["via_summary"] is True
    # Passages only: in the answer, the quotes, the cited passages, the steps and the trace.
    assert turn["trace"]["plans"][0]["steps"][-1]["ids"] == [Q4, B1]
    cited = [p for segment in turn["segments"] for p in segment["passages"]]
    assert [p["chunk_id"] for p in cited] == [Q4, B1]
    assert cited[0]["text"] == papers.Q4_TEXT
    _no_summary_text(turn, seen)


def test_a_composer_quoting_a_summary_it_was_never_shown_gets_that_segment_dropped() -> None:
    seen: list[str] = []
    turn = ask(
        "visitor",
        "?",
        manual("fluorescent upkeep", "Treppenhaus"),
        saying(
            {"text": 'It says "Replace the starter together with the tube."', "cites": ["p1"]},
            {"text": 'It says "fluorescent upkeep for the stairwell luminaire".', "cites": ["p1"]},
        ),
        inputs=seen,
    )

    assert len(turn["segments"]) == 2
    assert [d["segment"] for d in turn["trace"]["dropped_segments"]] == [3]
    _no_summary_text(turn, seen)


def test_a_document_found_only_by_its_summary_still_returns_its_passage() -> None:
    # Keywords only, so the Q4 passage, which shares no word with the query, is no hit itself.
    turn = ask(
        "visitor",
        "?",
        manual("fluorescent upkeep", "Treppenhaus"),
        retriever=Retriever(papers.index()),
    )

    assert retrieved(turn) == [Q4]
    assert turn["embedding_calls"] == []
    _no_summary_text(turn, [])


@pytest.mark.parametrize("persona", ["visitor", "staff"])
def test_no_summary_of_a_document_beyond_clearance_moves_anything(persona: str) -> None:
    # "switchgear inspection" is the panelboard manual's summary, a technician's.
    plan = manual("annual switchgear inspection routine", "Audimax")
    turn = ask(persona, "?", plan)

    assert retrieved(turn) == [B1]
    for word in ("switchgear", "manual:p1", "Panelboard"):
        assert word not in json.dumps({k: v for k, v in turn.items() if k != "trace"})
        assert word not in json.dumps(turn["trace"]["documents"]["retrieved"])
