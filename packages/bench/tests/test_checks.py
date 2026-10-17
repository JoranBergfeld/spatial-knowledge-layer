"""Tool correctness and the two leak checks, on the test bench and on the demo venue.

Everything goes through `POST /turn` against a real Postgres, with a scripted
model. These are the checks CI gates on: a leak here fails the build.

The test bench is the hand-made handover and the demo venue is the mini campus.
Neither is the real thing.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import prose
import route_fixture as rf
from spatial_bench import suite
from spatial_bench.bundle import Bundle
from spatial_bench.check import compare_all
from spatial_bench.cli import main
from spatial_bench.leaks import (
    Difference,
    Hidden,
    Leak,
    LeakReport,
    input_leaks,
    turn_differences,
)
from spatial_bench.questions import FORMAT, QuestionSet
from spatial_bench.run import Asked, Bench, Loaded, loaded
from spatial_bench.suite import Report
from spatial_bench.variants import (
    HIDDEN_TYPE_NAME,
    MARKER_ATTRIBUTE,
    with_restricted_added,
    without_restricted,
)
from spatial_core.circulation import CROSSABLE_DOORS
from spatial_core.records import Edge, Persona
from spatial_core.search import Filter, Query
from spatial_service import store
from spatial_service.places import Venue
from spatial_service.retrieval import Embedded
from spatial_service.routes import door_counts
from spatial_service.store import PostgresStore, circulation


class Embedder:
    """Embeds a query the way the stand-in model embedded the chunks."""

    def embed(self, deployment: str, text: str) -> Embedded:
        return Embedded(vector=tuple(prose.embedding(text)), deployment=deployment, input_tokens=1)


@pytest.fixture(scope="module")
def bench_report(
    bench_bundle: Path,
    bench_gold: QuestionSet,
    admin_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Report:
    work = tmp_path_factory.mktemp("bench-checks")
    return suite.run(bench_bundle, bench_gold, admin_url, work, embedder=Embedder())


@pytest.fixture(scope="module")
def campus_report(
    campus_bundle: Path,
    campus_gold: QuestionSet,
    admin_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Report:
    work = tmp_path_factory.mktemp("campus-checks")
    return suite.run(campus_bundle, campus_gold, admin_url, work, embedder=Embedder())


def wrong(report: Report) -> list[tuple[str, tuple[str, ...]]]:
    return [(v.question, v.mismatches) for v in report.verdicts if not v.ok]


# --- the test bench ------------------------------------------------------------------------


def test_on_the_bench_every_turn_matches_gold_in_kind_ids_and_counts(
    bench_report: Report, bench_gold: QuestionSet
) -> None:
    assert wrong(bench_report) == []
    assert len(bench_report.verdicts) == len(bench_gold.questions) > 100
    assert {verdict.got for verdict in bench_report.verdicts} == {
        question.expected.kind for question in bench_gold.questions
    }
    # And the document each question is after came back, on this little corpus.
    assert [v.question for v in bench_report.verdicts if v.missed_documents] == []


def test_on_the_bench_no_model_input_holds_anything_outside_the_visible_set(
    bench_report: Report,
) -> None:
    assert list(bench_report.leaks.inputs) == []
    assert bench_report.leaks.turns["input"] > 200


def test_the_input_leak_check_asks_as_every_persona_staff_included(
    bench_report: Report, bench_gold: QuestionSet
) -> None:
    # Gold is paired for the visitor and the technician. Staff get the same words and
    # no gold: a leak check needs none.
    pairs = len({question.pair for question in bench_gold.questions})
    said = bench_report.as_json()["leak_checks"]["input"]

    assert bench_gold.personas == [Persona.VISITOR, Persona.MAINTENANCE]
    assert bench_report.leaks.asked_as[Persona.STAFF] == pairs
    assert said["turns_by_persona"]["staff"] == pairs
    assert set(said["turns_by_persona"]) == {"visitor", "staff", "maintenance"}
    # And none of them are scored against gold they do not have.
    assert len(bench_report.verdicts) == len(bench_gold.questions)


def test_a_service_that_shows_staff_too_much_fails_the_input_leak_check(
    bench_loaded: Loaded, bench_bundle: Path, bench_gold: QuestionSet
) -> None:
    asking = suite.unpaired(bench_gold)
    assert {question.persona for question in asking} == {Persona.STAFF}
    leaky = Bench(Leaky(PostgresStore(bench_loaded.database_url)), bench_loaded.index)

    leaks = input_leaks(leaky.ask_all(asking), Hidden.of(Bundle.read(bench_bundle), Persona.STAFF))

    # Staff were shown the plant rooms only a technician gets.
    assert "ELEC. ROOM" in {leak.value for leak in leaks if leak.kind == "name"}
    assert {leak.persona for leak in leaks} == {Persona.STAFF}


def test_on_the_bench_adding_or_removing_restricted_things_changes_no_visitor_turn(
    bench_report: Report, bench_gold: QuestionSet
) -> None:
    assert list(bench_report.leaks.counterfactual) == []
    # Every visitor question, once in each copy.
    assert bench_report.leaks.turns["counterfactual"] == 2 * len(
        bench_gold.asked_as(Persona.VISITOR)
    )
    assert bench_report.passed


# --- the demo venue, as an internal regression -----------------------------------------------


def test_on_the_demo_venue_every_turn_matches_gold(campus_report: Report) -> None:
    assert wrong(campus_report) == []
    assert len(campus_report.verdicts) > 100


def test_on_the_demo_venue_both_leak_checks_pass(campus_report: Report) -> None:
    assert list(campus_report.leaks.inputs) == []
    assert list(campus_report.leaks.counterfactual) == []
    assert campus_report.leaks.turns["counterfactual"] > 50
    assert campus_report.venue == "mini-campus"


# --- the copies the counterfactual check asks ------------------------------------------------


def test_the_copies_differ_only_in_what_a_visitor_cannot_see(bench_bundle: Path) -> None:
    base = Bundle.read(bench_bundle)
    removed, added = without_restricted(base), with_restricted_added(base)

    assert len(base.assets) == 35
    assert {asset.id for asset in removed.assets} == set(base.visible(Persona.VISITOR).assets)
    # The closeout and the first floor notes are handover records on untagged spaces: staff.
    assert len(removed.documents) == 4
    assert len(added.assets) > 2 * 35 - 14
    assert len(added.documents) == 12 + 6 + 6
    for copy in (removed, added):
        for persona_sees in ("spaces", "assets", "types"):
            assert getattr(copy.visible(Persona.VISITOR), persona_sees) == getattr(
                base.visible(Persona.VISITOR), persona_sees
            )
    # A panelboard in every room a visitor walks through, and a type nobody had before.
    reception = [a for a in added.assets if a.space_id == "space:1g01"]
    assert "type:panelboard-p1" in {asset.type_id for asset in reception}
    assert HIDDEN_TYPE_NAME in {kind.name.value for kind in added.types}
    assert any(MARKER_ATTRIBUTE in asset.attributes for asset in added.assets)
    assert len(added.visible(Persona.MAINTENANCE).assets) == len(added.assets)


# --- a leak is caught ------------------------------------------------------------------------


class Leaky:
    """A store with a clearance defect: it hands a visitor what maintenance sees."""

    def __init__(self, behind: PostgresStore) -> None:
        self.behind = behind

    def served(self) -> tuple[str | None, str | None]:
        return self.behind.served()

    def venue(self, snapshot: str, persona: Persona) -> Venue:
        everything = self.behind.venue(snapshot, Persona.MAINTENANCE)
        return dataclasses.replace(everything, persona=persona)

    def sourced_clearance_share(self, snapshot: str) -> float | None:
        return self.behind.sourced_clearance_share(snapshot)


@pytest.fixture(scope="module")
def leaked(
    bench_loaded: Loaded,
    bench_bundle: Path,
    bench_gold: QuestionSet,
    admin_url: str,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[list[Asked], list[Asked]]]:
    """The visitor's turns from a leaky service, on the bench and on its copy with less in it."""
    asking = bench_gold.asked_as(Persona.VISITOR)
    work = tmp_path_factory.mktemp("leaky")
    copy = without_restricted(Bundle.read(bench_bundle)).write(work / "removed")
    with loaded(copy, admin_url, work / "index") as other:
        yield (
            Bench(Leaky(PostgresStore(bench_loaded.database_url)), bench_loaded.index).ask_all(
                asking
            ),
            Bench(Leaky(PostgresStore(other.database_url)), other.index).ask_all(asking),
        )


def test_a_service_that_shows_a_visitor_too_much_fails_tool_correctness(
    leaked: tuple[list[Asked], list[Asked]],
) -> None:
    assert [verdict for verdict in compare_all(leaked[0]) if not verdict.ok]


def test_a_service_that_shows_a_visitor_too_much_fails_the_input_leak_check(
    leaked: tuple[list[Asked], list[Asked]], bench_bundle: Path
) -> None:
    leaks = input_leaks(leaked[0], Hidden.of(Bundle.read(bench_bundle), Persona.VISITOR))

    assert leaks
    # The planner was shown rooms a visitor has no business knowing about.
    assert "ELEC. ROOM" in {leak.value for leak in leaks if leak.kind == "name"}
    assert not LeakReport(inputs=tuple(leaks)).passed


def test_a_service_that_shows_a_visitor_too_much_fails_the_counterfactual_check(
    leaked: tuple[list[Asked], list[Asked]],
) -> None:
    differences = turn_differences(leaked[0], leaked[1], "removed")

    assert differences
    assert not LeakReport(counterfactual=tuple(differences)).passed


def test_a_hidden_id_in_a_model_input_or_an_index_filter_is_a_leak(
    bench_loaded: Loaded, bench_bundle: Path, bench_gold: QuestionSet
) -> None:
    hidden = Hidden.of(Bundle.read(bench_bundle), Persona.VISITOR)
    [question] = [q for q in bench_gold.questions if q.id.endswith("no-such-room/001/visitor")]
    clean = bench_loaded.bench().ask(question)
    assert input_leaks([clean], hidden) == []

    def kinds(asked: Asked) -> list[tuple[str, str]]:
        return [(leak.kind, leak.value) for leak in input_leaks([asked], hidden)]

    payload = json.dumps({"facts": [{"ref": "f1", "id": "asset:panel-p1-03"}]})
    told = dataclasses.replace(clean, inputs=(*clean.inputs, payload))
    assert kinds(told) == [("id", "asset:panel-p1-03")]
    # An id a visitor can see is no leak, and neither is one that only looks alike.
    seen = json.dumps({"ids": ["asset:light-a-01", "asset:panel-p1-031"]})
    assert kinds(dataclasses.replace(clean, inputs=(seen,))) == []
    # A hidden chunk, and an attribute only hidden assets carry.
    assert kinds(dataclasses.replace(clean, inputs=("see doc:panelboard-om-pdf#0.",))) == [
        ("id", "doc:panelboard-om-pdf#0")
    ]
    chosen = Filter(
        clearance_classes=(Persona.VISITOR,), asset_type_ids=("type:ahu", "type:door-type-a")
    )
    asked = Query(text="x", filter=chosen)
    assert kinds(dataclasses.replace(clean, queries=(asked,))) == [("filter", "type:ahu")]
    assert kinds(dataclasses.replace(clean, embedded=("MECH. ROOM filters",))) == [
        ("name", "MECH. ROOM")
    ]


def test_what_the_asker_typed_themselves_is_not_a_leak(
    bench_loaded: Loaded, bench_bundle: Path, bench_gold: QuestionSet
) -> None:
    """A visitor may ask after a room they cannot see. Their own words going to the planner
    tell them nothing."""
    hidden = Hidden.of(Bundle.read(bench_bundle), Persona.VISITOR)
    [question] = [
        q for q in bench_gold.asked_as(Persona.VISITOR) if q.question == "Where is ELEC. ROOM?"
    ]
    asked = bench_loaded.bench().ask(question)

    assert "ELEC. ROOM" in hidden.names
    assert any("ELEC. ROOM" in sent for sent in asked.inputs)
    assert input_leaks([asked], hidden) == []


# --- restricted spaces and doors -----------------------------------------------------------
#
# The routes building, with room 203 private. Room 201 then has two doors a visitor may
# take and a third into a room that does not exist for them.

NO_QUESTIONS = QuestionSet(
    format=FORMAT,
    source={},
    per_template=0,
    personas=[Persona.VISITOR],
    unmeasured=[],
    world={},
    questions=[],
)


@pytest.fixture(scope="module")
def routes_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return rf.bundle(tmp_path_factory.mktemp("routes"), private=(rf.ROOM_203,))


def test_the_copies_vary_the_restricted_spaces_and_the_doors_into_them(
    routes_bundle: Path,
) -> None:
    base = Bundle.read(routes_bundle)
    removed, added = without_restricted(base), with_restricted_added(base)
    seen = base.visible(Persona.VISITOR).spaces

    assert rf.ROOM_203 not in seen and rf.STAFF_CORRIDOR not in seen
    # Gone: every space a visitor cannot see, and every edge they may not take.
    assert {space.id for space in removed.spaces} == set(seen)
    assert removed.edges
    for edge in removed.edges:
        assert edge.to_id in seen and (edge.from_id is None or edge.from_id in seen)
        assert Persona.VISITOR.clears(edge.clearance)
    # Added: a hidden room behind three more doors on rooms a visitor walks through.
    extra = [space for space in added.spaces if space.id not in {s.id for s in base.spaces}]
    assert extra and not any(Persona.VISITOR.clears(s.clearance) for s in extra if s.clearance)
    doors = {edge.door for edge in added.edges if edge.from_id == rf.ROOM_201}
    assert len(doors) == 3 + CROSSABLE_DOORS
    for copy in (removed, added):
        assert copy.visible(Persona.VISITOR).spaces == seen


def test_on_a_venue_with_doors_the_counterfactual_check_routes_to_what_a_visitor_sees(
    routes_bundle: Path, admin_url: str, tmp_path: Path
) -> None:
    report = suite.run(routes_bundle, NO_QUESTIONS, admin_url, tmp_path)

    assert list(report.leaks.counterfactual) == []
    assert list(report.leaks.inputs) == []
    # Every labelled room a visitor sees, once on the bundle and once on each copy.
    assert report.leaks.turns["route_probes"] == 2 * 12
    assert report.as_json()["leak_checks"]["counterfactual"]["route_probes"] == 24


def test_a_door_count_that_reads_hidden_doors_fails_the_counterfactual_check(
    routes_bundle: Path, admin_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect this check once missed: a door into a hidden room counted towards
    making a visible room a way through."""

    def every_door(edges: list[Edge], visible: set[str], persona: Persona, origin: Any) -> Any:
        held = circulation(edges, visible, persona, origin)
        return {**held, "doors": door_counts(e for e in edges if e.door is not None)}

    monkeypatch.setattr(store, "circulation", every_door)

    report = suite.run(routes_bundle, NO_QUESTIONS, admin_url, tmp_path)

    changed = {(d.question, d.variant) for d in report.leaks.counterfactual}
    # With 203 there a visitor is walked through 201 to 202. With it gone they are not.
    assert ("probe/route/20.20 room 202/visitor", "removed") in changed
    assert not report.passed


# --- the command -----------------------------------------------------------------------------


def test_bench_check_passes_on_the_bench_and_writes_its_report(
    bench_bundle: Path,
    bench_gold: QuestionSet,
    admin_url: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    gold = tmp_path / "gold.json"
    gold.write_text(bench_gold.model_dump_json())
    report = tmp_path / "report.json"
    check = ["check", "--bundle", str(bench_bundle), "--gold", str(gold), "--admin-url", admin_url]

    assert main([*check, "--report", str(report)]) == 0

    said = json.loads(report.read_text())
    assert said["venue"] == "mini-bench"
    assert said["model"] == "scripted"
    assert said["tool_correctness"]["passed"] is True
    assert said["leak_checks"]["input"] == {
        "passed": True,
        "turns": said["leak_checks"]["input"]["turns"],
        "turns_by_persona": {"visitor": 83, "staff": 83, "maintenance": 83},
        "leaks": [],
    }
    assert said["leak_checks"]["counterfactual"]["passed"] is True
    assert "counterfactual leak check: pass" in capsys.readouterr().out


def test_bench_check_exits_nonzero_when_a_turn_does_not_match_gold(
    bench_bundle: Path, bench_gold: QuestionSet, admin_url: str, tmp_path: Path
) -> None:
    tampered = json.loads(bench_gold.model_dump_json())
    for question in tampered["questions"]:
        if question["expected"]["kind"] == "clarification":
            question["expected"]["kind"] = "answer"
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps(tampered))
    check = ["check", "--bundle", str(bench_bundle), "--gold", str(gold), "--admin-url", admin_url]

    assert main([*check, "--no-counterfactual"]) == 1


def test_a_report_with_a_leak_in_it_has_not_passed(bench_report: Report) -> None:
    [first, *_] = bench_report.verdicts
    with_input = dataclasses.replace(
        bench_report,
        leaks=LeakReport(inputs=(Leak(first.question, Persona.VISITOR, "id", "x", "0"),)),
    )
    with_difference = dataclasses.replace(
        bench_report,
        leaks=LeakReport(counterfactual=(Difference(first.question, "added", ("/kind",)),)),
    )

    assert bench_report.passed
    assert not with_input.passed
    assert not with_difference.passed
    assert with_input.as_json()["leak_checks"]["input"]["passed"] is False
