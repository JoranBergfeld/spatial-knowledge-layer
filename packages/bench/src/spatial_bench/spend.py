"""What an evaluation run costs: the estimate before it starts and the ledger while it runs.

Money is tokens times a dated rate, the way the site's ledger works it out. The
rates come from a file the owner keeps, since this repository does not know
what a deployment costs today:

    {"currency": "USD", "as_of": "2026-10-01",
     "per_million_tokens": {"<deployment>": {"input": 0.0, "output": 0.0}, ...}}

Every deployment the run will call has to be in it, or the run does not start.
One that turns up mid-run with no rate stops the run there (evalrun.execute): its
calls are counted, their cost is not, and a run that cannot be costed has no
business going on.

The estimate is a worst case over assumed call sizes: every turn plans twice and
composes once. The assumptions are guesses until a real run has reported its
usage. The run prints both, so the next estimate can be better. What actually
keeps the bill down is the ceiling, not the estimate.

Evaluation spend is a ledger category of its own, "evaluation". The site's gate
reads visitor spend only and never this. The ledger itself is the backend's.
What comes out of here is the run's report of what it spent, in that category.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

# The ledger category. The backend's LedgerCategory.EVALUATION, on the wire.
CATEGORY = "evaluation"
MILLION = 1_000_000


class SpendError(Exception):
    pass


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Rate(_Model):
    input: float
    output: float = 0.0


class Rates(_Model):
    currency: str
    as_of: str
    per_million_tokens: dict[str, Rate]

    @classmethod
    def load(cls, path: Path) -> Rates:
        try:
            return cls.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise SpendError(f"cannot read the rates in {path}: {error}") from error

    def cost(self, deployment: str, input_tokens: int, output_tokens: int = 0) -> float:
        rate = self.per_million_tokens.get(deployment)
        if rate is None:
            raise SpendError(f"no rate for {deployment}, so the run cannot be costed")
        return (input_tokens * rate.input + output_tokens * rate.output) / MILLION


@dataclass(frozen=True)
class Assumed:
    """Tokens per call, in and out, for the estimate. Guesses. Nothing has measured them."""

    planning: tuple[int, int] = (8000, 600)
    composition: tuple[int, int] = (3000, 400)
    embedding: int = 40

    def of(self, purpose: str) -> tuple[int, int]:
        return self.composition if purpose == "composition" else self.planning


@dataclass(frozen=True)
class Models:
    """The deployments a run calls."""

    planning: str
    composition: str
    embedding: str | None
    # How many times a turn embeds its query: twice when the ranker is compared.
    embeddings_per_turn: int = 1


@dataclass(frozen=True)
class Estimate:
    turns: int
    # The most model calls the run can make: three a turn.
    calls: int
    per_turn: float
    amount: float


def estimate(turns: int, models: Models, rates: Rates, assumed: Assumed) -> Estimate:
    """The worst case: two planning calls and a composition call on every turn."""
    per_turn = 2 * rates.cost(models.planning, *assumed.planning)
    per_turn += rates.cost(models.composition, *assumed.composition)
    if models.embedding is not None:
        per_turn += models.embeddings_per_turn * rates.cost(models.embedding, assumed.embedding)
    return Estimate(turns=turns, calls=3 * turns, per_turn=per_turn, amount=turns * per_turn)


@dataclass
class _Entry:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0


@dataclass
class Ledger:
    """What the run has spent so far, by deployment and purpose."""

    rates: Rates
    assumed: Assumed
    entries: dict[tuple[str, str], _Entry] = field(default_factory=dict)
    # Calls that reported no usage. Costed at the assumed size, and counted here.
    unmetered: int = 0
    # Deployments that were called and are not in the rates. Their cost is not counted,
    # and the run stops after the turn that met one.
    unpriced: set[str] = field(default_factory=set)
    largest_turn: float = 0.0

    @property
    def amount(self) -> float:
        return sum(entry.cost for entry in self.entries.values())

    def _add(self, deployment: str, purpose: str, tokens_in: int | None, tokens_out: int) -> float:
        if tokens_in is None:
            self.unmetered += 1
            tokens_in, tokens_out = (
                (self.assumed.embedding, 0) if purpose == "embedding" else self.assumed.of(purpose)
            )
        if deployment not in self.rates.per_million_tokens:
            # prepare checks every deployment it knows of, so this is one it did not
            # expect. The rest of the turn is still counted, then the run stops.
            self.unpriced.add(deployment)
            cost = 0.0
        else:
            cost = self.rates.cost(deployment, tokens_in, tokens_out)
        entry = self.entries.setdefault((deployment, purpose), _Entry())
        entry.calls += 1
        entry.input_tokens += tokens_in
        entry.output_tokens += tokens_out
        entry.cost += cost
        return cost

    def add_turn(self, turn: dict[str, Any], extra_embeddings: int = 0) -> float:
        """Count one turn's calls, as the service reported them. Returns what the turn cost.

        ``extra_embeddings`` is how many more times the query was embedded outside
        the turn, for the ranker comparison.
        """
        cost = 0.0
        for call in turn.get("llm_calls", []):
            cost += self._add(
                call["deployment"],
                call["purpose"],
                call["input_tokens"],
                call["output_tokens"] or 0,
            )
        embeddings = turn.get("embedding_calls", [])
        for call in embeddings:
            cost += self._add(call["deployment"], "embedding", call["input_tokens"], 0)
        for call in embeddings[:1] * extra_embeddings:
            cost += self._add(call["deployment"], "embedding", call["input_tokens"], 0)
        self.largest_turn = max(self.largest_turn, cost)
        return cost

    def as_json(self) -> dict[str, Any]:
        return {
            "category": CATEGORY,
            "currency": self.rates.currency,
            "rates_as_of": self.rates.as_of,
            "amount": round(self.amount, 6),
            "unmetered_calls": self.unmetered,
            "unpriced": sorted(self.unpriced),
            "entries": [
                {
                    "deployment": deployment,
                    "purpose": purpose,
                    "calls": entry.calls,
                    "input_tokens": entry.input_tokens,
                    "output_tokens": entry.output_tokens,
                    "cost": round(entry.cost, 6),
                }
                for (deployment, purpose), entry in sorted(self.entries.items())
            ],
        }
