"""The question set against a snapshot that was just loaded and is not current yet.

Load runs this and reports it. It never blocks: a mismatch here is something to
look at before promoting, not an invariant. The invariants are load's own.

The model is the scripted one, so it costs nothing and says how the tools do
against gold on the new data. It is not an evaluation run and writes no figures.
Retrieval is by keyword only, since nothing here embeds a query.

The service only answers from the current snapshot and the one before it. A
freshly loaded one is neither, so the store is wrapped to say it is current.
Nothing is written anywhere and the pointer does not move.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from spatial_bench import questions
from spatial_bench.check import Verdict, compare_all
from spatial_bench.run import Bench
from spatial_core.records import Persona
from spatial_core.search import SearchQueries
from spatial_service import search
from spatial_service.places import Venue
from spatial_service.store import PostgresStore, Store


@dataclass(frozen=True)
class _AsIfCurrent:
    """A store that serves one loaded snapshot as the current one."""

    behind: Store
    snapshot: str

    def served(self) -> tuple[str | None, str | None]:
        return self.snapshot, None

    def venue(self, snapshot: str, persona: Persona) -> Venue:
        return self.behind.venue(snapshot, persona)

    def sourced_clearance_share(self, snapshot: str) -> float | None:
        return self.behind.sourced_clearance_share(snapshot)


def ask(
    gold: questions.QuestionSet, store: Store, snapshot: str, index: SearchQueries | None
) -> list[Verdict]:
    bench = Bench(_AsIfCurrent(store, snapshot), index)
    return compare_all(bench.ask_all(gold.questions))


def report(
    gold: Path,
    database_url: str,
    snapshot: str,
    index: SearchQueries | None = None,
    say: Callable[[str], None] = print,
) -> int:
    """Ask, and say how it went. Returns how many turns did not match gold."""
    asked = questions.load(gold)
    if index is None:
        index = search.from_env(os.environ, database_url)
    # The framework warns on every turn that a scripted client calls no functions.
    logging.getLogger("agent_framework").setLevel(logging.ERROR)
    verdicts = ask(asked, PostgresStore(database_url), snapshot, index)
    wrong = [verdict for verdict in verdicts if not verdict.ok]
    say(
        f"question set on {snapshot}, scripted model: {len(verdicts) - len(wrong)} of"
        f" {len(verdicts)} turns match gold. Reported only, this blocks nothing."
    )
    for verdict in wrong[:10]:
        say(f"  {verdict.question}: {'; '.join(verdict.mismatches)}")
    if len(wrong) > 10:
        say(f"  and {len(wrong) - 10} more")
    return len(wrong)
