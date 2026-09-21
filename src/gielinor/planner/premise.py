"""Bind an escalation to the facts that justified it, and drop it when they go.

An escalation is a question about a particular world. If that world stops
existing while the answer is in flight, the answer is not late — it is wrong,
and waiting for it is worse than not having asked.

## What this saves, and what it does not

It frees the concurrency slot, it stops the supervisor blocking on an answer it
will discard, and it lets the reflex layer proceed without a doomed request
hanging off the side of the loop.

It does **not** refund the request. OpenRouter counts a request when it is
issued; a cancelled request is a spent request. Anything in this codebase that
implies otherwise is wrong, which is why the counter here is named
`requests_issued` and is never decremented on cancellation.

The corollary matters more than the mechanism: since cancellation cannot recover
budget, the budget has to be protected at the point of *asking*. Escalate only
on premises whose expected lifetime exceeds the p95 round trip — goal-level
decisions at goal boundaries — and cancellation becomes a rare safety net rather
than a routine event. A high `cancelled` count is a signal that escalation
policy is wrong, not that this class is working well. See ADR 0005.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

log = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Premise:
    """The facts an escalation assumed.

    ``oids`` is the set of objects the planner was shown and may refer to; if any
    of them is gone, the answer can at best name something that no longer exists.
    ``still_true`` covers everything else — the goal has not changed, the
    dialogue is still open, the player is still where the question assumed.
    """

    oids: frozenset[int] = frozenset()
    still_true: Callable[[], bool] = lambda: True
    label: str = ""

    def holds(self, live_oids: set[int]) -> str | None:
        """Returns None if the premise still holds, else why it does not."""
        missing = self.oids - live_oids
        if missing:
            return f"oids gone: {sorted(missing)}"
        try:
            if not self.still_true():
                return "premise predicate false"
        except Exception as exc:
            return f"premise predicate raised: {exc!r}"
        return None


@dataclass
class PremiseScopedEscalation(Generic[T]):
    """Runs one request, watching the premise, cancelling on invalidation."""

    #: Returns the currently live oids. Closes over the supervisor's registry.
    live_oids: Callable[[], set[int]]
    poll_s: float = 1.0 / 15.0
    requests_issued: int = field(default=0, init=False)
    cancelled: int = field(default=0, init=False)
    completed: int = field(default=0, init=False)

    async def ask(
        self, request: Callable[[], Coroutine[Any, Any, T]], premise: Premise
    ) -> T | None:
        """Returns the answer, or None if the premise died first.

        ``request`` is a zero-arg factory rather than a coroutine so that nothing
        is created — and nothing is counted — unless the premise holds at entry.
        """
        reason = premise.holds(self.live_oids())
        if reason is not None:
            log.debug("escalation %s not issued: %s", premise.label, reason)
            return None

        self.requests_issued += 1  # counted at issue; cancellation does not refund
        task: asyncio.Task[T] = asyncio.create_task(request())
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.poll_s)
                if done:
                    self.completed += 1
                    return task.result()
                reason = premise.holds(self.live_oids())
                if reason is not None:
                    task.cancel()
                    self.cancelled += 1
                    log.info("escalation %s cancelled in flight: %s", premise.label, reason)
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task  # let the teardown finish before we return
                    return None
        except asyncio.CancelledError:
            task.cancel()
            raise
