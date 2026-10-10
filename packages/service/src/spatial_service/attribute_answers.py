"""What an attribute filter adds to an answer, and how a gap in an attribute is worded.

A filtered answer is an asset answer (compose.py) over the assets that passed, with
one more fact per filter: the condition, in the host's words. "no serial number
recorded", "service due before 2026-01-01". The model places it and cannot reword
it, and the answer has to name it, or a filtered count would read as the whole.

Where a filter tests a value, the assets that carry no value did not fail it. They
are not recorded, so they get a count of their own that the answer has to name too.
Leaving them out would let "two are due" pass for "the rest are not".

The model still writes the words around the placeholders, so the instructions tell
it what a gap is, and an answer saying "never" is thrown out. That one word is all
that is checked. Mostly works.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from spatial_service import attributes
from spatial_service.assets import ScopeResult, Tally
from spatial_service.compose import Answer, Citation, FactKind, Facts
from spatial_service.places import Venue
from spatial_service.plan import Filter, FilterTest

SERVICE_DUE_BASIS = (
    "the latest service event, or the installation date where there is none, plus a default "
    f"interval of {attributes.SERVICE_INTERVAL_YEARS} year, the same for every asset type"
)
BANNED = ("never",)

RULES = """
"filters" lists what the assets counted were tested for. Each "condition" is a
placeholder that fills in as the test in words: name every one of them, so the count
reads as the count of assets it holds for. "not_recorded", where present, are assets
in scope that carry no value to test: "counts" is how many and "as" fills in as what
they lack. Say that it is not recorded for them and nothing more. It is a gap in the
data. Do not say they passed or failed the test, that something was never done, or
anything else about the assets themselves.
"""

GAP = "No {what} is recorded for the assets asked about in {scope}."
GAP_ANYWHERE = "No {what} is recorded for the assets asked about."
GAP_SERVICE_DUE = " Service due is counted from one of those, so it cannot be worked out."


def _citation(one: Filter, text: str) -> Citation:
    due = one.attribute == attributes.SERVICE_DUE
    return Citation(
        kind=FactKind.CONDITION,
        label=text,
        attribute=one.attribute,
        provenance="derived" if due else None,
        rule=attributes.SERVICE_DUE_RULE if due else None,
        basis=SERVICE_DUE_BASIS if due else None,
    )


def annotate(venue: Venue, answer: Answer, found: ScopeResult) -> Answer:
    """``answer``, with each of ``found``'s filters as a condition it has to name."""
    if not found.filters:
        return answer
    facts = Facts(answer.facts)
    required = list(answer.required)
    together = list(answer.together)
    told: list[dict[str, Any]] = []
    for rate in found.filters:
        one = rate.filter
        text = attributes.condition(one)
        entry: dict[str, Any] = {
            "condition": facts.condition(text, _citation(one, text), test=one.test.value)
        }
        required.append(entry["condition"])
        if one.test is not FilterTest.NOT_RECORDED and rate.lacking:
            lacking = attributes.not_recorded(one.attribute)
            counts = facts.counts(
                Tally.of(venue.assets[i] for i in rate.lacking), space_id=found.scope
            )
            lacked = facts.condition(
                lacking, _citation(one, lacking), test=FilterTest.NOT_RECORDED.value
            )
            entry["not_recorded"] = {"counts": counts, "as": lacked}
            required += [*counts, lacked]
            together.append(counts)
        told.append(entry)
    payload = {**answer.payload, "filters": told, "required": required, "facts": facts.told()}
    return replace(
        answer,
        facts=facts.items,
        payload=payload,
        required=required,
        together=together,
        banned=BANNED,
    )


def gap(found: ScopeResult) -> Filter | None:
    """The first filter whose attribute no asset asked about carries, if there is one."""
    return next((rate.filter for rate in found.filters if rate.gap), None)


def gap_text(one: Filter, scope: str | None) -> str:
    """The coverage gap in words, naming the attribute. The host's, with no model call."""
    what = attributes.not_recorded(one.attribute).removeprefix("no ").removesuffix(" recorded")
    text = GAP.format(what=what, scope=scope) if scope else GAP_ANYWHERE.format(what=what)
    return text + (GAP_SERVICE_DUE if one.attribute == attributes.SERVICE_DUE else "")
