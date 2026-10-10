"""Conversation state, and the follow-up references that resolve against it.

The service keeps nothing between turns. It is handed the prior state with each turn
and hands back the new one. What the state holds:

    questions     the text of earlier questions, oldest first. Never an answer: a
                  composed answer comes from facts the state already holds by id
    place         the last place a turn resolved
    asset_types   the last asset types a turn asked for
    assets        the assets the last asset answer rested on, when they are few
                  enough to point at (ASSET_CARRY_CAP). An answer over more holds none
    since         for each of those three, the turn that set it
    constraints   the constraints in force. Nothing sets one yet
    route         the last route. Nothing sets one yet
    pending       the candidates a clarification offered, and what kind they are

The state comes back from a client, so none of it is trusted. `own` runs first every
turn and keeps only ids the persona can see in this snapshot. A tampered id, a stale
one and one that never existed all go the same way: dropped, with nothing said.

A follow-up reference is a mention the planner marked as pointing back ("that one",
"like it", "the second one"). The model only says where it points and what kind of
thing it is. `resolve` finds what, from the state `own` already cut down:

    a candidate, by ordinal    the n-th candidate the pending clarification offered.
                               Read as an asset type, it is that asset's type
    an earlier turn, a place   the place in state
    an earlier turn, an asset  the asset in state. Several is ambiguous: a clarification
    an earlier turn, a type    the type of the asset in state ("like it"). Assets of
                               several types is ambiguous, and asks which asset. With
                               no asset held, the types in state

Nothing held, or an ordinal past the last candidate, is nothing referred to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from spatial_core.records import Persona
from spatial_service.places import CLARIFY_CAP, Outcome, Venue
from spatial_service.plan import FollowUpTarget, Mention, MentionKind

# How many assets an answer may rest on and still leave them in state. More than a
# clarification can name could never be told apart by "that one".
ASSET_CARRY_CAP = CLARIFY_CAP
# How many earlier questions are kept. A conversation is three turns.
QUESTIONS_KEPT = 3
# The slots a reference to an earlier turn reads, each stamped in `since`.
SLOTS = ("place", "asset_types", "assets")

# What a resolving step's `how` says when it read state and looked nothing up.
HOW = "follow-up"


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Pending(_Model):
    """A clarification asked and not yet answered."""

    mention: str
    candidates: list[str]
    # What the candidates are: places or assets.
    kind: MentionKind = MentionKind.PLACE
    # The turn that offered them.
    turn: int | None = None


class Carried(_Model):
    """Something a turn left in force: a constraint, or the last route.

    Loose on purpose, the tickets that fill these decide what goes in. The one rule:
    every id goes in the id list of its kind and nowhere else, since those lists are
    what gets re-filtered. If one id in it is not visible the whole thing is dropped,
    because a constraint or a route with a piece missing means something else.
    """

    kind: str
    # The turn that set it.
    turn: int
    space_ids: list[str] = []
    asset_ids: list[str] = []
    asset_type_ids: list[str] = []
    # Everything that is not an id: an attribute, a date, a duration.
    args: dict[str, Any] = {}


class ConversationState(_Model):
    """What a conversation carries from one turn to the next. The client holds it."""

    snapshot: str
    persona: Persona
    turns: int = 0
    questions: list[Annotated[str, Field(max_length=500)]] = Field(default=[], max_length=20)
    place: str | None = None
    asset_types: list[str] = []
    assets: list[str] = []
    since: dict[str, int] = {}
    constraints: list[Carried] = []
    route: Carried | None = None
    pending: Pending | None = None


def _labelled(venue: Venue, space_id: str | None) -> bool:
    # A space with no label cannot be named back, so it is no use in state.
    space = venue.spaces.get(space_id) if space_id is not None else None
    return space is not None and space.label is not None


def _seen(venue: Venue, kind: MentionKind, item: str) -> bool:
    if kind is MentionKind.PLACE:
        return _labelled(venue, item)
    return item in (venue.assets if kind is MentionKind.ASSET else venue.asset_types)


def _all_seen(venue: Venue, carried: Carried) -> bool:
    return (
        all(i in venue.spaces for i in carried.space_ids)
        and all(i in venue.assets for i in carried.asset_ids)
        and all(i in venue.asset_types for i in carried.asset_type_ids)
    )


def own(state: ConversationState, venue: Venue) -> ConversationState:
    """The prior state, keeping only the ids the persona can see. Says nothing of the rest."""
    pending = state.pending
    if pending is not None:
        kept = [i for i in dict.fromkeys(pending.candidates) if _seen(venue, pending.kind, i)]
        pending = pending.model_copy(update={"candidates": kept[:CLARIFY_CAP]}) if kept else None
    route = state.route
    held: dict[str, Any] = {
        "place": state.place if _labelled(venue, state.place) else None,
        "asset_types": [i for i in state.asset_types if i in venue.asset_types],
        "assets": [i for i in dict.fromkeys(state.assets) if i in venue.assets][:ASSET_CARRY_CAP],
    }
    return state.model_copy(
        update={
            **held,
            "questions": state.questions[-QUESTIONS_KEPT:],
            "since": {slot: n for slot, n in state.since.items() if held.get(slot)},
            "constraints": [c for c in state.constraints if _all_seen(venue, c)],
            "route": route if route is not None and _all_seen(venue, route) else None,
            "pending": pending,
        }
    )


def advance(state: ConversationState, question: str, **update: Any) -> ConversationState:
    """The state after one more turn. A slot named in ``update`` is stamped with this turn.

    A slot left out stays as it was, with the turn that set it. Nothing pending
    carries over unless ``update`` says so.
    """
    turn = state.turns + 1
    since = dict(state.since)
    for slot in SLOTS:
        if slot in update:
            since.pop(slot, None)
            if update[slot]:
                since[slot] = turn
    return state.model_copy(
        update={
            "turns": turn,
            "questions": [*state.questions, question][-QUESTIONS_KEPT:],
            "pending": None,
            **update,
            "since": since,
        }
    )


def name(venue: Venue, kind: MentionKind, item: str) -> str:
    """What one id is called. Only ever asked about ids the venue holds."""
    if kind is MentionKind.PLACE:
        label = venue.spaces[item].label
        assert label is not None
        return label.value
    if kind is MentionKind.ASSET_TYPE:
        return venue.asset_types[item].name.value
    return venue.assets[item].name.value


def offered(venue: Venue, pending: Pending) -> list[str]:
    """The pending candidates as the planner reads them, in the order they were offered."""
    if pending.kind is not MentionKind.ASSET:
        return [name(venue, pending.kind, i) for i in pending.candidates]
    told = []
    for asset in (venue.assets[i] for i in pending.candidates):
        place = venue.spaces[asset.space_id].label
        told.append(f"{asset.name.value}, in {place.value}" if place else asset.name.value)
    return told


def told(state: ConversationState, venue: Venue) -> dict[str, Any]:
    """What the planner hears of the conversation so far. Names, never ids. Empty at the start."""
    said: dict[str, Any] = {}
    if state.questions:
        said["earlier_questions"] = state.questions
    if state.place is not None:
        said["place"] = name(venue, MentionKind.PLACE, state.place)
    if state.assets:
        said["assets"] = [name(venue, MentionKind.ASSET, i) for i in state.assets]
    if state.asset_types:
        said["asset_types"] = [name(venue, MentionKind.ASSET_TYPE, i) for i in state.asset_types]
    return said


@dataclass(frozen=True)
class Referred:
    """What a follow-up reference came to."""

    outcome: Outcome
    # What the ids are. Assets, where a type was wanted and it has to ask which asset.
    kind: MentionKind
    # The one resolved, or the candidates to ask about. Empty for none.
    ids: tuple[str, ...]
    to: FollowUpTarget
    ordinal: int | None = None
    # The turn that resolved what it points at, or that offered the candidates.
    turn: int | None = None
    # The assets whose type it took, for "like it".
    through: tuple[str, ...] = ()


def resolve(
    venue: Venue, state: ConversationState, mention: Mention, wanted: MentionKind
) -> Referred:
    """What ``mention`` points back at, as a ``wanted``. Reads state that `own` cut down."""
    refers = mention.refers
    assert refers is not None
    if refers.to is FollowUpTarget.CANDIDATE:
        return _candidate(venue, state.pending, refers.ordinal or 0, wanted)

    def came_to(
        outcome: Outcome,
        ids: tuple[str, ...] = (),
        slot: str | None = None,
        kind: MentionKind = wanted,
        through: tuple[str, ...] = (),
    ) -> Referred:
        turn = state.since.get(slot) if slot else None
        return Referred(outcome, kind, ids, refers.to, turn=turn, through=through)

    if wanted is MentionKind.PLACE:
        if state.place is None:
            return came_to(Outcome.NONE)
        return came_to(Outcome.RESOLVED, (state.place,), "place")

    assets = tuple(state.assets)
    if wanted is MentionKind.ASSET:
        if not assets:
            return came_to(Outcome.NONE)
        outcome = Outcome.RESOLVED if len(assets) == 1 else Outcome.AMBIGUOUS
        return came_to(outcome, assets, "assets")

    types = tuple(sorted({venue.assets[i].type_id for i in assets}))
    if len(types) == 1:
        return came_to(Outcome.RESOLVED, types, "assets", through=assets)
    if types:
        return came_to(Outcome.AMBIGUOUS, assets, "assets", kind=MentionKind.ASSET)
    if state.asset_types:
        return came_to(Outcome.RESOLVED, tuple(state.asset_types), "asset_types")
    return came_to(Outcome.NONE)


def _candidate(
    venue: Venue, pending: Pending | None, ordinal: int, wanted: MentionKind
) -> Referred:
    to = FollowUpTarget.CANDIDATE
    if pending is None or not 1 <= ordinal <= len(pending.candidates):
        return Referred(Outcome.NONE, wanted, (), to, ordinal)
    picked = pending.candidates[ordinal - 1]
    if pending.kind is wanted:
        return Referred(Outcome.RESOLVED, wanted, (picked,), to, ordinal, pending.turn)
    if pending.kind is MentionKind.ASSET and wanted is MentionKind.ASSET_TYPE:
        kind = venue.assets[picked].type_id
        return Referred(Outcome.RESOLVED, wanted, (kind,), to, ordinal, pending.turn, (picked,))
    # A candidate of another kind than the step wants: "the second one" was a room,
    # and the plan read it as an asset.
    return Referred(Outcome.NONE, wanted, (), to, ordinal)
