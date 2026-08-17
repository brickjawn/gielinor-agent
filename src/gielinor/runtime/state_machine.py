"""A small, deterministic finite state machine.

The reflex planner is a state machine and it would be perfectly possible to
write it as an ``if/elif`` chain. This exists because three properties are hard
to get from a chain and cheap to get from a table:

* **Time in state.** Escalation trigger 3 in SPEC section 2 ("stuck") is defined
  as "in one state past its expected duration", which needs somewhere to record
  when the state was entered. Note that a self-transition deliberately does
  *not* reset that clock — otherwise a machine that keeps re-deciding to wait
  would never look stuck.
* **Edge vs level.** "Click the tree" must happen on *entering* CLICK_TREE, not
  on every frame spent there; ``just_entered`` makes that distinction explicit
  rather than implicit in a flag someone forgets to clear.
* **Inspectability.** ``history`` gives the trace file a state timeline, which
  is most of what makes a recorded run readable afterwards.

Determinism: transitions are evaluated in declaration order and the first whose
source matches and whose guard passes wins. Guards must be pure and cheap; they
run every frame inside the 50 ms budget. Guard exceptions are *not* swallowed —
a broken guard is a bug and should be loud, not silently false.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

S = TypeVar("S")
C = TypeVar("C")


@dataclass(frozen=True, slots=True)
class Transition(Generic[S, C]):
    """One edge. ``sources=None`` means "from any state"."""

    target: S
    guard: Callable[[C], bool]
    sources: tuple[S, ...] | None = None
    name: str = ""

    def matches(self, state: S) -> bool:
        return self.sources is None or state in self.sources


def always(_: C) -> bool:
    """Guard that always passes; use as the last transition to make it total."""
    return True


@dataclass
class StateMachine(Generic[S, C]):
    """Table-driven FSM with entry-edge detection and a state clock."""

    initial: S
    transitions: Sequence[Transition[S, C]]
    history_size: int = 64

    state: S = field(init=False)
    previous: S | None = field(default=None, init=False)
    entered_at: float = field(default=0.0, init=False)
    just_entered: bool = field(default=True, init=False)
    steps: int = field(default=0, init=False)
    last_transition: str = field(default="", init=False)
    history: deque[tuple[S, float]] = field(default_factory=deque, init=False)

    def __post_init__(self) -> None:
        self.state = self.initial
        self.history = deque(maxlen=self.history_size)
        self.history.append((self.state, self.entered_at))

    def step(self, ctx: C, *, now: float) -> S:
        """Evaluate the table once and move if a guard passes. Returns the new state."""
        self.steps += 1
        for transition in self.transitions:
            if not transition.matches(self.state):
                continue
            if not transition.guard(ctx):
                continue
            self.last_transition = transition.name or f"->{transition.target}"
            self._enter(transition.target, now)
            return self.state
        # No transition matched: staying put is a self-transition, and the state
        # clock keeps running so `stuck` detection still fires.
        self.just_entered = False
        return self.state

    def _enter(self, target: S, now: float) -> None:
        if target == self.state:
            self.just_entered = False
            return
        logger.debug("fsm %s -> %s via %s", self.state, target, self.last_transition)
        self.previous = self.state
        self.state = target
        self.entered_at = now
        self.just_entered = True
        self.history.append((target, now))

    def force(self, state: S, *, now: float) -> None:
        """Jump without consulting the table. For recovery paths and tests."""
        self.last_transition = "forced"
        self._enter(state, now)

    def reset(self, *, now: float = 0.0) -> None:
        self.state = self.initial
        self.previous = None
        self.entered_at = now
        self.just_entered = True
        self.steps = 0
        self.last_transition = ""
        self.history.clear()
        self.history.append((self.state, now))

    def time_in_state(self, now: float) -> float:
        return max(0.0, now - self.entered_at)

    def timeline(self) -> list[tuple[S, float]]:
        return list(self.history)


__all__ = ["StateMachine", "Transition", "always"]
