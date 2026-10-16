"""The arithmetic under a run's figures: retrieval scores, intervals, the two adoption rules.

These are pure and their mistakes are silent, so they get tests of their own
beneath `bench run`. Expected values are worked by hand or come from a textbook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from spatial_bench import questions, stats
from spatial_bench.questions import GoldError
from spatial_bench.score import numbers_are_facts, recall_at, reciprocal_rank


def test_recall_at_five_is_the_share_of_wanted_documents_in_the_top_five() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]

    assert recall_at(ranked, ["a"]) == 1.0
    assert recall_at(ranked, ["c", "f"]) == 0.5
    assert recall_at(ranked, ["f"]) == 0.0
    assert recall_at([], ["a"]) == 0.0


def test_reciprocal_rank_is_one_over_the_place_of_the_first_wanted_document() -> None:
    ranked = ["a", "b", "c", "d", "e", "f"]

    assert reciprocal_rank(ranked, ["a"]) == 1.0
    assert reciprocal_rank(ranked, ["d", "b"]) == 0.5
    assert reciprocal_rank(ranked, ["f"]) == 0.0


def test_a_share_gets_its_wilson_interval() -> None:
    # The usual worked example: 15 of 20 gives 0.531 to 0.888.
    figure = stats.share(15, 20)

    assert (figure.value, figure.n) == (0.75, 20)
    assert figure.ci95 == pytest.approx((0.5313, 0.8881), abs=1e-4)
    assert stats.share(0, 10).ci95 == pytest.approx((0.0, 0.2775), abs=1e-4)


def test_a_figure_over_nothing_has_no_value_and_no_interval() -> None:
    for figure in (stats.share(0, 0), stats.mean([])):
        assert (figure.value, figure.n, figure.ci95) == (None, 0, None)


def test_a_mean_gets_the_same_interval_every_time() -> None:
    scores = [1.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0]

    first, second = stats.mean(scores), stats.mean(scores)

    assert first == second
    assert first.value == 0.6875
    assert first.ci95 is not None
    assert 0.3 < first.ci95[0] < 0.6875 < first.ci95[1] <= 1.0


def test_the_ranker_is_adopted_only_when_the_gains_interval_excludes_zero() -> None:
    clear = stats.ranker_rule([0.5, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5])
    noisy = stats.ranker_rule([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0])
    worse = stats.ranker_rule([-0.5, -0.5, -1.0, -0.5, -0.5, -0.5, -1.0, -0.5, -0.5, -0.5])

    assert clear.met is True
    assert noisy.met is False
    # An interval wholly below zero excludes zero as well. It adopts nothing.
    assert worse.met is False
    assert stats.ranker_rule(None).met is None


def test_a_cheaper_pair_may_replace_the_baseline_when_its_lower_bound_is_above_minus_five() -> None:
    same = [0.0] * 100
    # Wrong on 20 more questions in 100: twenty points down.
    worse = [-1.0] * 20 + [0.0] * 80

    assert stats.pair_rule(same, cheaper=True).met is True
    assert stats.pair_rule(worse, cheaper=True).met is False
    # The rule is about a cheaper pair. A dearer one never displaces the baseline.
    assert stats.pair_rule(same, cheaper=False).met is False
    assert stats.pair_rule(None).met is None


def segment(text: str, *citations: dict[str, Any], quotes: tuple[str, ...] = ()) -> dict[str, Any]:
    passages = [{"quotes": list(quotes), "text": "whatever it says"}] if quotes else []
    return {"text": text, "citations": list(citations), "passages": passages}


def test_every_number_in_an_answer_has_to_be_a_count_a_citation_carries() -> None:
    room = {"label": "50.34 room 107", "count": None}
    four = {"label": "4", "count": 4}
    told = {"segments": [segment("50.34 room 107 holds 4.", room, four)]}
    made_up = {"segments": [segment("50.34 room 107 holds 4, or maybe 5.", room, four)]}
    mislabelled = {"segments": [segment("It holds 4.", {"label": "4", "count": 3})]}
    quoted = {"segments": [segment('It says "torque to 12 Nm".', quotes=("torque to 12 Nm",))]}

    assert numbers_are_facts(told)
    assert not numbers_are_facts(made_up)
    assert not numbers_are_facts(mislabelled)
    assert numbers_are_facts(quoted)


def test_a_number_is_held_to_the_fact_it_cites_and_not_to_how_the_fact_is_worded() -> None:
    # A generated count fills in as "3 generated", and a route's as "41 seconds (modelled)".
    generated = {"label": "3 generated", "count": 3}
    sourced = {"label": "1 sourced", "count": 1}
    duration = {"label": "41 seconds (modelled)", "count": None, "seconds": 41, "metres": None}
    distance = {"label": "39 metres", "count": None, "seconds": None, "metres": 39}
    mixed = {"segments": [segment("It holds 3 generated and 1 sourced.", generated, sourced)]}
    walked = {"segments": [segment("41 seconds (modelled), 39 metres.", duration, distance)]}
    # The label says one number and the fact behind it is another.
    drifted = {"segments": [segment("It holds 3 generated.", {"label": "3 generated", "count": 2})]}
    slower = {"segments": [segment("41 seconds (modelled).", {**duration, "seconds": 45})]}
    # A fact of another segment is not this segment's to state.
    elsewhere = {
        "segments": [segment("It holds 3 generated.", generated), segment("Or 3.", sourced)]
    }

    assert numbers_are_facts(mixed)
    assert numbers_are_facts(walked)
    assert not numbers_are_facts(drifted)
    assert not numbers_are_facts(slower)
    assert not numbers_are_facts(elsewhere)


def test_a_type_name_after_a_count_has_digits_that_count_nothing() -> None:
    # The host puts the type's name after a count, and cites the type with it.
    vav = {"label": "VAV 200 mm", "count": None}
    told = {"segments": [segment("It holds 25 VAV 200 mm.", vav, {"label": "25", "count": 25})]}
    uncited = {"segments": [segment("It holds 25 VAV 200 mm.", {"label": "25", "count": 25})]}

    assert numbers_are_facts(told)
    assert not numbers_are_facts(uncited)


def test_the_committed_hand_written_file_is_readable() -> None:
    committed = Path(__file__).resolve().parents[3] / questions.HAND_WRITTEN_FILE

    found = questions.load_hand_written(committed)

    assert found
    assert all(question.id.startswith(f"{questions.HAND_WRITTEN}/") for question in found)


def test_a_hand_written_question_becomes_a_question_with_no_slots(tmp_path: Path) -> None:
    path = tmp_path / "hand-written.json"
    path.write_text(
        '{"format": "spatial-hand-written@1", "venue": "test-bench", "questions": ['
        '{"id": "boiler-room", "persona": "visitor", "shape": "place-location",'
        ' "question": "Where would I find the boiler room?",'
        ' "expected": {"kind": "abstention"}}]}'
    )

    (question,) = questions.load_hand_written(path)

    assert question.id == "hand-written/boiler-room"
    assert question.template == "hand-written"
    assert question.slots == questions.Slots()
    assert question.expected.kind is questions.ExpectedKind.ABSTENTION


def test_a_hand_written_file_in_another_format_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "hand-written.json"
    path.write_text('{"format": "spatial-gold@1", "venue": "test-bench", "questions": []}')

    with pytest.raises(GoldError, match="spatial-hand-written@1"):
        questions.load_hand_written(path)
