"""The turn: a persona, a snapshot, prior state and a question in; a turn with its trace out.

    plan        the planning model reads the question and the collapsed tree
                (tree.py) and writes a plan, which is checked in code before any
                tool runs
    execute     the steps run in order with the venue the host bound: what this
                persona can see in this snapshot. No model call between them.
    recover     only on a typed trigger (recovery.py): the planning deployment gets
                its plan back with the reasons and writes a second, which is checked
                and run the same way. A second trigger ends the turn.
    compose     the composing model words the answer over facts by reference; the
                host checks it and fills in every label and number

So a turn makes at most three model calls, however many steps run: CALL_CAP, which
the host enforces rather than trusts.

Every turn is one of four kinds. An answer. A clarification, when a mention names
more than one place the persona can see, or a follow-up reference points at more
than one asset. An abstention, with its cause. Or unavailable, when a model could
not be reached, which answers nothing at all.

A later turn can lean on an earlier one. The prior state comes in with the request,
is cut to what the persona can see before anything reads it, and the new state goes
back out (conversation.py). The trace shows each follow-up reference, what it
resolved to and from which turn.

A route that cannot be found is one abstention, whatever stopped it: clearance,
the crossing rule or a gap in the mapping. Told apart, they would say where the
restricted corridors are.

Nothing the persona cannot see reaches a model: the planner gets the question,
the collapsed tree of visible spaces, the earlier questions and the names of what
the state still holds once it is cut down, and the composer gets references and
kinds only. Nothing counts what was left out, so
nothing can say so.

A document answer adds a second list to composition, the passages retrieved this
turn (retrieval.py), which the composer does get the text of: they passed the
clearance filter. It is settled apart (_settle_documents), since a segment about a
passage can be dropped on its own where a wrong fact rejects the whole answer.
Each segment goes out with the full text of the passages it cites, for the
entailment check, which is not done here. A query embedding is a paid call but
not a model call of the turn, so it is reported in a list of its own.

The pieces live apart: the models in turn_models.py, what the models are told
in instructions.py, the trace's parts in traces.py, and the turn being built with
the ways it ends early in outcomes.py.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from spatial_service import attribute_answers, attributes, compose, outcomes, route_answer, traces
from spatial_service.assets import ScopeResult
from spatial_service.conversation import ASSET_CARRY_CAP, offered, told
from spatial_service.executor import Execution, StepResult, execute
from spatial_service.facts import Answer
from spatial_service.grounding import CompositionRejected, Draft, Segment, ground
from spatial_service.instructions import (
    COMPOSITION_INSTRUCTIONS,
    PLANNING_INSTRUCTIONS,
    RECOVERY_INSTRUCTIONS,
)
from spatial_service.llm import Llm, LlmUnavailable, Purpose
from spatial_service.outcomes import OpenTurn
from spatial_service.places import Outcome, Venue
from spatial_service.plan import DECLINED, Plan, QuestionShape
from spatial_service.plan_check import PlanRejected, parse_plan
from spatial_service.recovery import RecoveryOutcome, Trigger, diff
from spatial_service.retrieval import (
    Documents,
    DroppedSegment,
    EmbeddingCall,
    RetrievalUnavailable,
)
from spatial_service.retrieval import trace as documents_trace
from spatial_service.store import Store
from spatial_service.tree import NODE_CAP, Collapsed, collapse
from spatial_service.turn_models import (
    CALL_CAP,
    SNAPSHOT_DROPPED,
    UNPARSED_CAP,
    AbstentionCause,
    PlanVersion,
    Trace,
    TreeMeasure,
    Turn,
    TurnKind,
    TurnRefused,
    TurnRequest,
    Verdict,
)

# What app.py, metrics.py and the tests have always taken from here.
__all__ = [
    "CALL_CAP",
    "AbstentionCause",
    "Turn",
    "TurnKind",
    "TurnRefused",
    "TurnRequest",
    "TurnService",
]


class TurnService:
    def __init__(
        self,
        store: Store,
        llm: Llm,
        node_cap: int = NODE_CAP,
        *,
        documents: Documents | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.node_cap = node_cap
        # Document retrieval, bound here and not by any plan. None retrieves nothing.
        self.documents = documents

    async def take(self, request: TurnRequest) -> Turn:
        snapshot = await asyncio.to_thread(self._snapshot, request)
        venue = await asyncio.to_thread(self.store.venue, snapshot, request.persona)
        state = outcomes.own_state(request, snapshot, venue)
        shown = collapse(venue, self.node_cap)
        measure = TreeMeasure(
            depth=shown.depth,
            full_depth=shown.full_depth,
            nodes=shown.nodes,
            full_nodes=shown.full_nodes,
            node_cap=shown.node_cap,
        )
        trace = Trace(snapshot=snapshot, persona=request.persona, tree=measure)
        turn = OpenTurn(state, trace, [], question=request.question)
        try:
            first = await self._attempt(turn, request.question, venue, shown, None)
            if first.trigger is None:
                return await self._settle(turn, request.question, venue, first)
            # Unrepaired until the second plan runs, so a failed recovery call counts.
            turn.recovery = RecoveryOutcome.UNREPAIRED
            second = await self._attempt(turn, request.question, venue, shown, first)
            if second.trigger is None:
                # A second plan with nothing to run is the model giving up, not a repair.
                if second.execution is not None:
                    turn.recovery = RecoveryOutcome.REPAIRED
                return await self._settle(turn, request.question, venue, second)
        except (LlmUnavailable, RetrievalUnavailable) as error:
            return turn.unavailable(error)
        return outcomes.give_up(turn, venue, second.trigger)

    async def _attempt(
        self,
        turn: OpenTurn,
        question: str,
        venue: Venue,
        shown: Collapsed,
        previous: _Attempt | None,
    ) -> _Attempt:
        """One plan from the model, checked and, when it may, run."""
        payload: dict[str, Any] = {
            "question": question,
            "venue": {
                "tiers": list(shown.tiers),
                "tree": shown.tree,
                "attributes": list(attributes.catalogue(venue).attributes),
            },
        }
        conversation = told(turn.state, venue)
        filters = attribute_answers.in_force(turn.state, venue)
        if filters:
            conversation["filters"] = filters
        if turn.state.route is not None:
            conversation["last_route"] = outcomes.told_last_route(turn.state.route, venue)
        if conversation:
            payload["conversation"] = conversation
        if turn.state.pending is not None:
            payload["pending"] = {
                "mention": turn.state.pending.mention,
                "kind": turn.state.pending.kind.value,
                "candidates": offered(venue, turn.state.pending),
            }
        if previous is not None:
            assert previous.trigger is not None
            payload["previous"] = {
                "plan": previous.written,
                "trigger": previous.trigger.model_dump(mode="json", exclude_none=True),
            }
        text = await turn.call(
            self.llm,
            Purpose.PLANNING if previous is None else Purpose.RECOVERY,
            instructions=PLANNING_INSTRUCTIONS if previous is None else RECOVERY_INSTRUCTIONS,
            payload=payload,
            response_format=Plan,
        )
        number = 1 if previous is None else 2
        trigger: Trigger | None
        try:
            plan = parse_plan(text)
        except PlanRejected as rejected:
            written: Any = _json_object(text)
            if written is None:
                written = text[:UNPARSED_CAP]
            trigger = Trigger(kind=rejected.trigger, reasons=rejected.reasons)
            version = PlanVersion(
                version=number,
                plan=written if isinstance(written, dict) else None,
                unparsed=written if isinstance(written, str) else None,
                verdict=Verdict.REJECTED,
                rejection=rejected.reasons,
                trigger=trigger,
            )
            attempt = _Attempt(written, None, None, trigger)
        else:
            written = plan.model_dump(mode="json", exclude_none=True)
            execution = None
            if plan.shape not in DECLINED:
                # Off the loop: a retrieval step waits on the index.
                execution = await asyncio.to_thread(
                    execute, plan, venue, turn.state, self.documents
                )
                turn.embeddings.extend(
                    EmbeddingCall(
                        deployment=r.retrieval.embedding.deployment,
                        input_tokens=r.retrieval.embedding.input_tokens,
                    )
                    for r in execution.results
                    if r.retrieval is not None and r.retrieval.embedding is not None
                )
            trigger = execution.trigger if execution is not None else None
            version = PlanVersion(
                version=number,
                plan=written,
                verdict=Verdict.ACCEPTED,
                steps=traces.steps(execution) if execution is not None else [],
                trigger=trigger,
            )
            attempt = _Attempt(written, plan, execution, trigger)
        if previous is not None:
            version = version.model_copy(update={"changes": diff(previous.written, written)})
        turn.add_version(version)
        return attempt

    async def _settle(self, turn: OpenTurn, question: str, venue: Venue, attempt: _Attempt) -> Turn:
        """The turn a plan that raised no trigger ends in."""
        assert attempt.plan is not None
        shape = attempt.plan.shape
        turn.note(shape=shape)
        if attempt.execution is None:
            return turn.abstain(outcomes.DECLINED_CAUSE[shape])

        execution = attempt.execution
        turn.note(
            resolved=traces.resolved(execution, venue),
            resolved_types=traces.resolved_types(execution, venue),
            references=traces.references(execution, venue),
        )
        turn.note(route=route_answer.trace(venue, execution))
        last = execution.last
        if last.referred is not None and last.outcome is not Outcome.RESOLVED:
            return outcomes.not_referred(turn, venue, last)
        if last.outcome is Outcome.NONE:
            return turn.abstain(outcomes.NOTHING_FOUND[last.tool])
        if last.outcome is Outcome.AMBIGUOUS:
            assert last.text is not None
            return outcomes.clarify(turn, venue, last.text, last.ids)
        if last.retrieval is not None:
            return await self._settle_documents(turn, question, venue, shape, last)

        found: ScopeResult | None = None
        if shape is QuestionShape.PLACE_LOCATION:
            subject = venue.spaces[last.ids[0]]
            answer = compose.location(venue, question, subject)
            state = turn.next(place=subject.id)
        elif shape is QuestionShape.ASSET_LOCATION:
            asset = venue.assets[last.ids[0]]
            answer = compose.asset_location(venue, question, asset)
            turn.note(
                asset_types=[traces.type_trace(venue.asset_types[asset.type_id])],
                assets=traces.one_asset_trace(venue, asset),
            )
            state = turn.next(place=asset.space_id, assets=[asset.id])
        elif shape is QuestionShape.ROUTE:
            runs = [r.routed for r in execution.by_tool("route") if r.routed is not None]
            answer = compose.route(venue, question, runs)
            turn.routes = route_answer.routes_of(venue, execution)
            # Where the last route ended, and the route itself as the last route.
            state = turn.next(place=last.ids[0], route=outcomes.last_route(turn, turn.routes[-1]))
        elif shape is QuestionShape.PROXIMITY:
            assert last.near is not None and last.found is not None
            answer = compose.proximity(venue, question, last.near)
            turn.routes = route_answer.routes_of(venue, execution)
            near = [venue.assets[i] for i in last.near.asset_ids]
            types = [venue.asset_types[i] for i in sorted({asset.type_id for asset in near})]
            turn.note(asset_types=[traces.type_trace(kind) for kind in types])
            state = turn.next(
                place=near[0].space_id,
                asset_types=[kind.id for kind in types],
                assets=[asset.id for asset in near][:ASSET_CARRY_CAP],
            )
        else:
            found = last.found
            assert found is not None
            if last.aggregation is not None:
                answer = compose.aggregation(venue, question, found, last.aggregation)
            else:
                answer = compose.containment(venue, question, found)
            answer = attribute_answers.annotate(venue, answer, found)
            types = [venue.asset_types[i] for i in found.types or ()]
            turn.note(
                asset_types=[traces.type_trace(kind) for kind in types],
                assets=traces.assets_trace(venue, last, answer),
            )
            state = turn.next(
                **outcomes.carried(venue, last, found),
                constraints=attribute_answers.carried(found, turn.number),
            )
        places = [fact.space for fact in answer.facts if fact.space is not None]
        turn.note(
            provenance=[traces.provenance(s) for s in places],
            authored_defaults=traces.defaults(places),
        )

        instructions = COMPOSITION_INSTRUCTIONS[shape]
        if found is not None:
            if found.installed == 0:
                return outcomes.nothing_installed(turn, venue, found, state)
            unfilled = attribute_answers.gap(found)
            if unfilled is not None:
                return outcomes.attribute_gap(turn, venue, found, unfilled, state)
            if found.filters:
                instructions += attribute_answers.RULES

        composed = await self._compose(turn, instructions, answer)
        if composed is None:
            return turn.abstain(AbstentionCause.COMPOSITION_REJECTED)
        # No passages, so nothing was dropped.
        segments, _ = composed
        return turn.finish(TurnKind.ANSWER, segments, state=state)

    async def _compose(
        self, turn: OpenTurn, instructions: str, answer: Answer
    ) -> tuple[list[Segment], list[DroppedSegment]] | None:
        """The composition call, checked. None for a rejected draft, and the trace says why."""
        text = await turn.call(
            self.llm,
            Purpose.COMPOSITION,
            instructions=instructions,
            payload=answer.payload,
            response_format=Draft,
        )
        try:
            return ground(
                text, answer.facts, answer.required, answer.together, answer.passages, answer.banned
            )
        except CompositionRejected as rejected:
            turn.note(composition_rejection=rejected.reasons)
            return None

    async def _settle_documents(
        self, turn: OpenTurn, question: str, venue: Venue, shape: QuestionShape, last: StepResult
    ) -> Turn:
        """An answer over two lists: the facts as before, and the passages retrieved."""
        retrieved = last.retrieval
        assert retrieved is not None
        found = last.found
        if found is not None:
            answer = compose.asset_documents(venue, question, found, retrieved)
            # The types the anchored assets are of, whether or not the question named any.
            types = [venue.asset_types[i] for i in retrieved.anchors.asset_type_ids]
            types.sort(key=lambda kind: kind.id)
            turn.note(
                asset_types=[traces.type_trace(kind) for kind in types],
                assets=traces.assets_trace(venue, last, answer),
            )
            state = turn.next(**outcomes.carried(venue, last, found))
        else:
            assert retrieved.anchors.scope is not None
            subject = venue.spaces[retrieved.anchors.scope]
            answer = compose.place_documents(venue, question, subject, retrieved)
            state = turn.next(place=subject.id)
        places = [fact.space for fact in answer.facts if fact.space is not None]
        turn.note(
            documents=documents_trace(last.step, retrieved),
            provenance=[traces.provenance(s) for s in places],
            authored_defaults=traces.defaults(places),
        )

        composed = await self._compose(turn, COMPOSITION_INSTRUCTIONS[shape], answer)
        if composed is None:
            return turn.abstain(AbstentionCause.COMPOSITION_REJECTED)
        segments, dropped = composed
        turn.note(dropped_segments=dropped)
        # A document answer with no document segment left is not one.
        if not any(segment.passages for segment in segments):
            return turn.abstain(AbstentionCause.UNGROUNDED)
        return turn.finish(TurnKind.ANSWER, segments, state=state)

    def _snapshot(self, request: TurnRequest) -> str:
        current, previous = self.store.served()
        held = request.state.snapshot if request.state else None
        if request.snapshot and held and request.snapshot != held:
            raise TurnRefused(409, f"the conversation is held on {held}, not {request.snapshot}")
        snapshot = request.snapshot or held or current
        if snapshot is None:
            raise TurnRefused(503, "no snapshot is current")
        if snapshot not in (current, previous):
            # Retired, or never there at all. Told apart nowhere.
            raise TurnRefused(
                409, f"{snapshot} is not served; start a new conversation", SNAPSHOT_DROPPED
            )
        if request.state and request.state.persona is not request.persona:
            raise TurnRefused(409, "a conversation keeps the persona it began with")
        return snapshot


@dataclass(frozen=True)
class _Attempt:
    # The plan as written: a checked plan's dump, else the model's JSON or its text.
    written: Any
    plan: Plan | None
    execution: Execution | None
    trigger: Trigger | None


def _json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None
