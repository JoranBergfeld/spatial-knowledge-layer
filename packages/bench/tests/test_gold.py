"""Gold: computed from the handover workbook by scripts that share nothing with the layer.

The expected values here are counted by hand off the tables in mini_handover.py.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

import mini_handover
from spatial_bench import questions
from spatial_bench.questions import Expected, ExpectedKind, QuestionSet
from spatial_core.records import Persona

GOLD = questions.gold_dir()
LAYER = ("spatial_core", "spatial_ingest", "spatial_service", "spatial_bench")
SHAPES = {
    "place-location",
    "asset-containment",
    "asset-aggregation",
    "attribute-filter",
    "asset-document",
    "place-document",
    "step-dependent",
}


@pytest.fixture(scope="module")
def everything(tmp_path_factory: pytest.TempPathFactory) -> QuestionSet:
    """Every instance of every template, not a sample."""
    return questions.from_cobie(
        mini_handover.DIR / mini_handover.WORKBOOK,
        tmp_path_factory.mktemp("gold") / "all.json",
        ocr=mini_handover.DIR / mini_handover.OCR_DIR,
        per_template=50,
    )


def expected(gold: QuestionSet, question: str, persona: str) -> Expected:
    [found] = [
        q for q in gold.questions if q.question == question and q.persona is Persona(persona)
    ]
    return found.expected


def ids(prefix: str, *names: str) -> list[str]:
    return sorted(f"{prefix}:{name}" for name in names)


# --- no code shared with the layer ----------------------------------------------------------


def test_gold_imports_nothing_but_the_standard_library_and_itself() -> None:
    own = {path.stem for path in GOLD.glob("*.py")}
    for path in GOLD.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                assert node.level == 0, f"{path.name} has a relative import"
                names = [node.module or ""]
            for name in names:
                top = name.split(".")[0]
                assert top not in LAYER, f"{path.name} imports {name}"
                assert top in own or top in sys.stdlib_module_names, f"{path.name} imports {name}"


def test_where_gold_runs_the_layer_cannot_be_imported() -> None:
    """Gold is run with -I -S. Over there the layer's packages do not exist."""
    for package in LAYER:
        found = subprocess.run(
            [sys.executable, "-I", "-S", "-c", f"import {package}"],
            capture_output=True,
            check=False,
        )
        assert found.returncode != 0, f"{package} can be imported where gold runs"
    # And here they do, so the check above is not passing for some other reason.
    assert (
        subprocess.run([sys.executable, "-c", "import spatial_core"], check=False).returncode == 0
    )


def test_gold_is_not_a_package_of_the_workspace_and_the_layer_never_imports_it() -> None:
    assert not (GOLD / "__init__.py").exists()
    assert not (GOLD / "pyproject.toml").exists()
    modules = {path.stem for path in GOLD.glob("*.py")} - {"__main__"}
    for path in Path(GOLD.parent / "packages").glob("*/src/**/*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                assert not {alias.name for alias in node.names} & modules, path
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert node.module not in modules, path


def test_the_same_workbook_gives_the_same_gold(tmp_path: Path) -> None:
    workbook = mini_handover.DIR / mini_handover.WORKBOOK
    questions.from_cobie(workbook, tmp_path / "first.json")
    questions.from_cobie(workbook, tmp_path / "second.json")

    assert (tmp_path / "first.json").read_bytes() == (tmp_path / "second.json").read_bytes()


# --- the question set ----------------------------------------------------------------------


def test_every_shape_has_template_questions_tagged_with_kind_and_persona(
    everything: QuestionSet,
) -> None:
    assert {question.shape for question in everything.questions} == SHAPES
    assert everything.unmeasured == ["proximity", "route"]
    assert {question.expected.kind for question in everything.questions} == set(ExpectedKind)
    assert {question.persona for question in everything.questions} == {
        Persona.VISITOR,
        Persona.MAINTENANCE,
    }
    assert everything.source["kind"] == "cobie"
    assert len(everything.source["sha256"]) == 64


def test_every_question_is_asked_as_a_visitor_and_as_maintenance(everything: QuestionSet) -> None:
    pairs: dict[str, list[tuple[Persona, str]]] = {}
    for question in everything.questions:
        pairs.setdefault(question.pair, []).append((question.persona, question.question))

    for asked in pairs.values():
        assert sorted(persona for persona, _ in asked) == [Persona.MAINTENANCE, Persona.VISITOR]
        assert len({text for _, text in asked}) == 1
    # And the two do get different gold where clearance bites.
    differing = [
        pair
        for pair in pairs
        if len({q.expected.model_dump_json() for q in everything.questions if q.pair == pair}) == 2
    ]
    assert len(differing) > len(pairs) // 2


def test_gold_sees_what_each_persona_sees(everything: QuestionSet) -> None:
    seen = everything.world["visible"]

    assert seen["visitor"] == {"spaces": 9, "asset_types": 4, "assets": 14, "documents": 4}
    assert seen["staff"]["assets"] == 23
    assert seen["maintenance"] == {"spaces": 17, "asset_types": 10, "assets": 35, "documents": 12}


# --- gold itself, counted by hand ------------------------------------------------------------


def test_a_restricted_asset_in_a_visible_room_is_in_one_personas_gold_only(
    everything: QuestionSet,
) -> None:
    asked = "What is installed in RESTROOM?"
    visitor, technician = (expected(everything, asked, p) for p in ("visitor", "maintenance"))

    assert visitor.kind is ExpectedKind.ANSWER
    assert visitor.ids == ids("asset", "door-d-g03", "lavatory-l1-01", "light-a-05")
    assert visitor.counts == {
        "type:door-type-a": 1,
        "type:lavatory-l1": 1,
        "type:light-fixture-a": 1,
    }
    # The ball valve in the restroom wall is a maintenance type.
    assert technician.ids == ids(
        "asset", "ball-valve-bv-01", "door-d-g03", "lavatory-l1-01", "light-a-05"
    )
    assert technician.total == 4


def test_a_room_beyond_clearance_is_an_abstention_for_the_visitor(everything: QuestionSet) -> None:
    asked = "Where is ELEC. ROOM?"

    assert expected(everything, asked, "visitor").kind is ExpectedKind.ABSTENTION
    technician = expected(everything, asked, "maintenance")
    assert technician.ids == ["space:1g05"]
    assert technician.within == ["building:b-12", "floor:ground-floor"]


def test_clarification_candidates_are_golds_filtered_by_clearance(everything: QuestionSet) -> None:
    corridor = expected(everything, "Where is CORRIDOR?", "visitor")
    assert corridor.kind is ExpectedKind.CLARIFICATION
    assert corridor.candidates == ["space:1g02", "space:2f01"]
    # Both offices are staff rooms.
    assert expected(everything, "Where is OFFICE?", "visitor").kind is ExpectedKind.ABSTENTION
    office = expected(everything, "Where is OFFICE?", "maintenance")
    assert office.candidates == ["space:1g04", "space:2f03"]
    # A room number alone is asked about even when only one room has it.
    bare = expected(everything, "Where is room 1G03?", "visitor")
    assert (bare.kind, bare.candidates) == (ExpectedKind.CLARIFICATION, ["space:1g03"])


def test_a_count_per_floor_covers_only_what_the_persona_can_see(everything: QuestionSet) -> None:
    asked = "Which floor has the most Light Fixture A?"
    visitor, technician = (expected(everything, asked, p) for p in ("visitor", "maintenance"))

    assert visitor.counts == {"floor:ground-floor": 6, "floor:first-floor": 3}
    assert visitor.most == ["floor:ground-floor"]
    assert visitor.total == 9
    # The roof has an air handler and no lights: a zero, and ranked.
    assert technician.counts == {
        "floor:ground-floor": 9,
        "floor:first-floor": 6,
        "floor:roof": 0,
    }
    assert technician.not_recorded == []


def test_a_room_with_nothing_recorded_is_listed_apart_and_not_counted_as_zero(
    everything: QuestionSet,
) -> None:
    rooms = expected(everything, "Which room on First Floor has the most AHU?", "maintenance")

    assert rooms.not_recorded == ["space:2f06"]
    assert rooms.most == ["space:2f04"]
    assert rooms.counts is not None and "space:2f06" not in rooms.counts


def test_a_filter_carries_its_fill_rate_and_a_fill_rate_of_zero_is_a_coverage_gap(
    everything: QuestionSet,
) -> None:
    asked = "Which Light Fixture A in Harbour Clinic have no serial number recorded?"
    lights = expected(everything, asked, "visitor")
    assert lights.ids == ids(
        "asset", "light-a-02", "light-a-04", "light-a-05", "light-a-08", "light-a-10", "light-a-13"
    )
    assert lights.fill is not None
    assert (lights.fill.carried, lights.fill.of, lights.fill.not_recorded) == (3, 9, 6)

    doors = "Which Door Type A in Harbour Clinic have a serial number recorded?"
    gap = expected(everything, doors, "visitor")
    assert gap.kind is ExpectedKind.COVERAGE_GAP
    assert gap.fill is not None and (gap.fill.carried, gap.fill.of) == (0, 1)


def test_service_due_is_the_installation_date_plus_a_year_where_nothing_was_serviced(
    everything: QuestionSet,
) -> None:
    # Fifteen lights. Fourteen carry an installation date. The median due date is
    # 2019-03-05, and four were installed on the first and second of March 2018.
    asked = "Which Light Fixture A in Harbour Clinic fall due for service before 2019-03-05?"
    due = expected(everything, asked, "maintenance")

    assert due.ids == ids("asset", "light-a-01", "light-a-02", "light-a-03", "light-a-04")
    assert due.fill is not None
    assert (due.fill.attribute, due.fill.carried, due.fill.of) == ("service_due", 14, 15)
    assert due.fill.not_recorded == 1


def test_a_document_is_reachable_only_through_what_the_persona_sees(
    everything: QuestionSet,
) -> None:
    panel = "What does Panelboard Manual say about the Panelboard P1 in Harbour Clinic?"
    assert expected(everything, panel, "visitor").kind is ExpectedKind.ABSTENTION
    technician = expected(everything, panel, "maintenance")
    assert technician.documents == ["doc:panelboard-om-pdf"]
    assert technician.retrievable == ["doc:panelboard-om-pdf", "doc:warranty-general-pdf"]

    # The closeout is a handover record on an untagged building, so staff. The visitor
    # can see the building, so either an answer from something else or no document.
    closeout = "What does Building Closeout say about Harbour Clinic?"
    visitor = expected(everything, closeout, "visitor")
    assert (visitor.kind, visitor.documents, visitor.or_no_document) == (
        ExpectedKind.ANSWER,
        [],
        True,
    )
    assert expected(everything, closeout, "maintenance").documents == ["doc:closeout-pdf"]
    # Everything a visitor may read at all: nothing keyed to plant, staff equipment or a
    # system with a maintenance type in it, and no handover record on untagged spaces.
    assert visitor.retrievable == [
        "doc:lighta-productdata-pdf",
        "doc:reception-finishes-pdf",
        "doc:sprinkler-test-pdf",
        "doc:warranty-general-pdf",
    ]


def test_a_room_reaches_its_buildings_documents_and_no_further(everything: QuestionSet) -> None:
    # The closeout hangs on Harbour Clinic. Asked about a room in it, it is in reach,
    # for whoever may read it at all. The visitor may not.
    reception = "What does Reception Finish Schedule say about RECEPTION?"
    assert "doc:closeout-pdf" in (expected(everything, reception, "maintenance").retrievable or [])
    assert "doc:closeout-pdf" not in (expected(everything, reception, "visitor").retrievable or [])


def test_a_document_with_no_ocr_text_is_not_in_gold(everything: QuestionSet) -> None:
    assert everything.world["documents"] == 12
    assert not any("Lost Scan" in question.question for question in everything.questions)


def test_a_question_that_needs_one_answer_to_ask_the_next_is_expected_to_be_declined(
    everything: QuestionSet,
) -> None:
    declined = [q for q in everything.questions if q.shape == "step-dependent"]

    assert {q.expected.kind for q in declined} == {ExpectedKind.ABSTENTION}
    assert len(declined) == 2


def test_the_demo_venue_gets_gold_read_off_its_bundle(campus_gold: QuestionSet) -> None:
    assert campus_gold.source["kind"] == "bundle"
    assert {question.shape for question in campus_gold.questions} == SHAPES
    assert (
        campus_gold.world["visible"]["visitor"]["assets"]
        < (campus_gold.world["visible"]["maintenance"]["assets"])
    )
