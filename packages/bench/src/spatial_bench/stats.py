"""The intervals behind every figure, and the two adoption rules.

A share (so many of n) gets a Wilson interval. A mean of per-question scores
(Recall@5, MRR, a paired difference) gets a percentile bootstrap with a fixed
seed, so the same answers give the same interval twice.

The adoption rules are worked out here and printed by the run. Nothing applies
them: switching the ranker on or swapping a deployment stays a decision somebody
makes with the numbers in front of them.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

from spatial_service.evaluation import Figure

# 95%, two-sided.
Z = 1.959963984540054
RESAMPLES = 2000
SEED = 73
# The cheaper model pair may be worse by less than this, in correctness, at the lower bound.
PAIR_MARGIN = -0.05


def _rounded(value: float) -> float:
    return round(value, 4)


def share(hits: int, n: int) -> Figure:
    """``hits`` of ``n`` with its Wilson interval."""
    if n == 0:
        return Figure(value=None, n=0, ci95=None)
    p = hits / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    low, high = max(0.0, centre - half), min(1.0, centre + half)
    return Figure(value=_rounded(p), n=n, ci95=(_rounded(low), _rounded(high)))


def mean(scores: Sequence[float]) -> Figure:
    """The mean of per-question scores with a bootstrap interval."""
    n = len(scores)
    if n == 0:
        return Figure(value=None, n=0, ci95=None)
    drawn = random.Random(SEED)
    means = sorted(sum(drawn.choices(scores, k=n)) / n for _ in range(RESAMPLES))
    low = means[int(0.025 * RESAMPLES)]
    high = means[min(RESAMPLES - 1, int(0.975 * RESAMPLES))]
    return Figure(value=_rounded(sum(scores) / n), n=n, ci95=(_rounded(low), _rounded(high)))


@dataclass(frozen=True)
class Rule:
    """One adoption rule, worked out. ``met`` is None when there was nothing to compare."""

    name: str
    difference: Figure
    met: bool | None
    said: str


def ranker_rule(gains: Sequence[float] | None) -> Rule:
    """The semantic ranker is adopted when the interval of its paired Recall@5 gain excludes zero.

    ``gains`` is, per question, Recall@5 with the ranker minus Recall@5 without.
    An interval wholly below zero excludes zero too, and adopts nothing.
    """
    name = "semantic ranker"
    if gains is None:
        nothing = Figure(value=None, n=0, ci95=None)
        return Rule(name, nothing, None, "not compared in this run (--compare-ranker)")
    difference = mean(gains)
    if difference.ci95 is None:
        return Rule(name, difference, None, "no question with a document to find")
    met = difference.ci95[0] > 0
    said = "the interval excludes zero: adopt" if met else "no gain the interval can show: keep off"
    return Rule(name, difference, met, said)


def pair_rule(differences: Sequence[float] | None, cheaper: bool | None = None) -> Rule:
    """The model pair stays unless a cheaper pair's paired correctness difference
    has a lower bound above minus five points.

    ``differences`` is, per question, this run's correctness minus the baseline
    run's, as one or zero each. ``cheaper`` is whether this run spent less.
    """
    name = "model pair"
    if differences is None:
        nothing = Figure(value=None, n=0, ci95=None)
        return Rule(name, nothing, None, "no baseline run to compare with (--baseline)")
    difference = mean(differences)
    if difference.ci95 is None:
        return Rule(name, difference, None, "no question in both runs")
    if not cheaper:
        return Rule(name, difference, False, "this pair is not the cheaper one: the baseline stays")
    met = difference.ci95[0] > PAIR_MARGIN
    said = (
        "cheaper, and the lower bound is above minus five points: it may replace the baseline"
        if met
        else "cheaper, but the lower bound is at or below minus five points: the baseline stays"
    )
    return Rule(name, difference, met, said)
