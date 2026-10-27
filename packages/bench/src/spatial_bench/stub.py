"""A scripted model: the planner and the composer a turn gets when no real one is called.

The planner writes the plan a question's slots spell out, and nothing cleverer.
The composer names every fact it is told to and cites every passage, by
placeholder only. Between them a turn runs end to end for free, which is what CI
needs: the tools, the clearance filter and the host's checks are all real, and
only the two things a model would choose are fixed.

Both clients keep everything they were sent. That record is what the input leak
check reads.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from agent_framework import BaseChatClient, ChatResponse, Message, UsageDetails

from spatial_bench.questions import Question
from spatial_service.llm import Deployment, Llm

PLANNER = "scripted-planner"
COMPOSER = "scripted-composer"

Reply = Callable[[dict[str, Any]], str]


def plan_for(question: Question) -> dict[str, Any]:
    """The plan a question's slots spell out, for its shape."""
    slots = question.slots
    if question.shape in ("step-dependent", "unsupported"):
        return {"shape": question.shape, "mentions": [], "steps": []}

    mentions: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def resolve(text: str | None, kind: str, tool: str) -> dict[str, str] | None:
        if text is None:
            return None
        mentions.append({"id": f"m{len(mentions) + 1}", "text": text, "kind": kind})
        steps.append({"id": f"s{len(steps) + 1}", "tool": tool, "mention": mentions[-1]["id"]})
        return {"step": steps[-1]["id"]}

    def add(step: dict[str, Any]) -> dict[str, str]:
        steps.append({"id": f"s{len(steps) + 1}", **step})
        return {"step": steps[-1]["id"]}

    place = resolve(slots.place, "place", "resolve_place")
    if question.shape == "place-location":
        return {"shape": question.shape, "mentions": mentions, "steps": steps}
    if question.shape == "route":
        # From the default origin, to the one place named. Only the counterfactual
        # check asks these: routes have no gold.
        add({"tool": "route", "destination": place})
        return {"shape": question.shape, "mentions": mentions, "steps": steps}
    if question.shape == "place-document":
        add({"tool": "retrieve_documents", "anchors": place, "query": slots.query})
        return {"shape": question.shape, "mentions": mentions, "steps": steps}

    types = resolve(slots.asset_type, "asset-type", "resolve_asset_type")
    scope: dict[str, Any] = {"tool": "assets_in_scope"}
    if place is not None:
        scope["scope"] = place
    if types is not None:
        scope["types"] = types
    if slots.filter is not None:
        scope["filters"] = [slots.filter.model_dump(mode="json")]
    found = add(scope)
    if question.shape == "asset-aggregation":
        add({"tool": "aggregate", "assets": found, "by": slots.by})
    elif question.shape == "asset-document":
        add({"tool": "retrieve_documents", "anchors": found, "query": slots.query})
    elif question.shape == "asset-maintenance":
        add({"tool": "maintenance_tasks", "assets": found})
    return {"shape": question.shape, "mentions": mentions, "steps": steps}


def planner(questions: Sequence[Question]) -> Reply:
    """A planner that knows these questions by their words. Asked twice, it says the same twice."""
    plans = {question.question: json.dumps(plan_for(question)) for question in questions}
    declined = json.dumps({"shape": "unsupported", "mentions": [], "steps": []})

    def reply(payload: dict[str, Any]) -> str:
        return plans.get(payload["question"], declined)

    return reply


def _named(refs: Sequence[str]) -> str:
    return " and ".join(f"{{{ref}}}" for ref in refs)


def compose(payload: dict[str, Any]) -> str:
    """Segments that name what the payload says to name, and cite each passage once.

    It writes no word of its own that could be wrong: every claim is a placeholder
    the host fills from a fact. A passage is cited and not quoted, since a real
    passage has digits in it and nothing a model writes may.
    """
    segments: list[dict[str, Any]] = []
    said: set[str] = set()

    def say(text: str, cites: Sequence[str]) -> None:
        segments.append({"text": text, "cites": list(cites)})
        said.update(cites)

    if "subject" in payload:
        say(f"About {{{payload['subject']}}}.", [payload["subject"]])
    elif "total" in payload:
        scope = [payload["scope"]] if payload.get("scope") else []
        opening = f"{{{scope[0]}}} holds" if scope else "The venue holds"
        say(f"{opening} {_named(payload['total'])}.", [*scope, *payload["total"]])
        for row in payload.get("rows", []):
            say(f"{{{row['of']}}}: {_named(row['counts'])}.", [row["of"], *row["counts"]])
        for told in payload.get("filters", []):
            say(f"Counted: {{{told['condition']}}}.", [told["condition"]])
            if "not_recorded" in told:
                counts, lacking = told["not_recorded"]["counts"], told["not_recorded"]["as"]
                say(f"{_named(counts)} with {{{lacking}}}.", [*counts, lacking])
    else:
        facts = [fact["ref"] for fact in payload.get("facts", [])]
        if facts:
            inside = ", ".join(f"{{{ref}}}" for ref in facts[1:])
            say(f"{{{facts[0]}}} is in {inside}." if inside else f"{{{facts[0]}}}.", facts)

    # Whatever else has to be named, and counts that only make sense said together.
    for key in ("no_recorded_installations", "in_spaces_with_no_label"):
        refs = [ref for ref in _refs(payload.get(key)) if ref not in said]
        if refs:
            say(f"Also: {_named(refs)}.", refs)
    for ref in _refs(payload.get("required")):
        if ref not in said:
            say(f"Also: {{{ref}}}.", [ref])
    for passage in payload.get("passages", []):
        say("The cited passage covers this.", [passage["ref"]])
    return json.dumps({"segments": segments})


def _refs(told: Any) -> list[str]:
    """The fact refs in whatever shape a payload entry has them: one, a list, or nested."""
    if told is None:
        return []
    if isinstance(told, str):
        return [told]
    if isinstance(told, Mapping):
        return [ref for value in told.values() for ref in _refs(value)]
    return [ref for value in told for ref in _refs(value)]


class ScriptedClient(BaseChatClient):
    """A chat client that answers from a function of the payload and keeps every input."""

    def __init__(self, reply: Reply, inputs: list[str], usage: tuple[int, int] = (0, 0)) -> None:
        super().__init__()
        self.reply = reply
        self.inputs = inputs
        # Input and output tokens it reports per call. Zero, unless a test of spend says otherwise.
        self.usage = usage

    async def _inner_get_response(  # type: ignore[override]
        self, *, messages: Sequence[Message], stream: bool, options: dict[str, Any], **_: Any
    ) -> ChatResponse:
        self.inputs.append(str(options.get("instructions", "")))
        self.inputs.extend(message.text for message in messages)
        answer = self.reply(json.loads(messages[-1].text))
        return ChatResponse(
            messages=[Message(role="assistant", contents=[answer])],
            usage_details=UsageDetails(
                input_token_count=self.usage[0],
                output_token_count=self.usage[1],
                total_token_count=sum(self.usage),
            ),
        )


def scripted_llm(
    questions: Sequence[Question],
    inputs: list[str],
    *,
    usage: tuple[int, int] = (0, 0),
    composer: Reply = compose,
) -> Llm:
    """The scripted planner and composer as the service's two deployments.

    Everything either is sent, instructions included, is appended to ``inputs``.
    ``usage`` is the input and output tokens each call reports.
    """
    return Llm(
        planning=Deployment(PLANNER, ScriptedClient(planner(questions), inputs, usage)),
        composition=Deployment(COMPOSER, ScriptedClient(composer, inputs, usage)),
    )
