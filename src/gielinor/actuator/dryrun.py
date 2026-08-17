"""The default actuator: validate, resolve, log, execute nothing.

This is not a stub. It runs the entire dispatch path a real actuator runs —
validation, coordinate resolution, humanised path generation, timing — and stops
one call short of the syscall. That has three consequences worth having:

* The safe default is also the *well-tested* default, because every test in the
  suite drives this class, so the validator and resolver are never a code path
  that only runs in anger.
* A dry-run trace is directly comparable to a live trace. If the agent misclicks
  live, you can replay the same frames offline and see the same resolved points.
* Enabling real input is an explicit, greppable change (a different backend name
  plus ``allow_real_input=True``), not a config flag someone flips by accident.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gielinor.actuator.humanize import (
    click_delay_ms,
    mouse_path,
    sample_reaction_time_ms,
    typing_delays_ms,
)
from gielinor.actuator.resolve import Resolution, ResolutionError, resolve_action
from gielinor.actuator.validate import (
    RejectionCode,
    RejectionReason,
    ValidationPolicy,
    validate_action,
)
from gielinor.contracts import Action, ActionKind, ActionResult, GameState, Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from gielinor.config import Profile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlannedGesture:
    """What a real actuator would have done, in full, without doing it."""

    action: Action
    resolution: Resolution
    waypoints: int
    reaction_ms: float
    travel_ms: float
    hold_ms: float

    @property
    def total_ms(self) -> float:
        return self.reaction_ms + self.travel_ms + self.hold_ms


@dataclass
class DryRunActuator:
    """An `Actuator` that resolves and logs but never touches the OS."""

    profile: Profile | None = None
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)
    seed: int | None = 0
    move_speed: float = 1.0
    name: str = "dryrun"
    # Optional sink so the runtime can push every result into the trace without
    # this class importing the tracer (which would import the runtime).
    on_result: Callable[[ActionResult], None] | None = None
    now: Callable[[], float] = time.monotonic

    _rng: random.Random = field(init=False)
    # Resolved in __post_init__ from the profile rather than defaulted to a
    # literal. It previously read Point(382, 251) — the centre of a 765x503
    # client — which made this the one place a client dimension was baked into a
    # module instead of coming from calibration. Harmless today because it only
    # seeds travel-distance estimates, but it would have silently produced wrong
    # Fitts'-law timings the moment the profile described a different client.
    cursor: Point | None = None
    history: list[ActionResult] = field(default_factory=list, init=False)
    gestures: list[PlannedGesture] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        # Seeded by default: two runs over the same replay produce the same
        # click points, which is what makes a regression in perception visible
        # as a diff rather than as noise.
        self._rng = random.Random(self.seed)
        if self.cursor is None:
            # Where a real cursor would plausibly start. Origin rather than a
            # literal centre when no profile is available: an obviously-wrong
            # (0, 0) that shows up in the first travel-time number beats a
            # plausible-looking guess that silently encodes the wrong client.
            self.cursor = (
                self.profile.viewport_center() if self.profile is not None else Point(0, 0)
            )

    def execute(self, action: Action, state: GameState) -> ActionResult:
        rejection = validate_action(action, state, policy=self.policy, profile=self.profile)
        if rejection is not None:
            return self._finish(
                ActionResult(action=action, accepted=False, rejected_reason=str(rejection))
            )

        try:
            resolution = resolve_action(action, state, profile=self.profile, rng=self._rng)
        except ResolutionError as exc:
            logger.warning("resolution failed for %s: %s", action.kind.value, exc)
            reason = RejectionReason(RejectionCode.NO_SUCH_OBJECT, str(exc))
            return self._finish(
                ActionResult(action=action, accepted=False, rejected_reason=str(reason))
            )

        gesture = self._plan_gesture(action, resolution)
        self.gestures.append(gesture)
        if resolution.point is not None:
            self.cursor = resolution.point

        logger.info(
            "DRY-RUN %-17s %-22s point=%s in %.0fms (reaction %.0f + travel %.0f + hold %.0f) "
            "seq=%d reason=%r",
            action.kind.value,
            _target_repr(action, resolution),
            resolution.point,
            gesture.total_ms,
            gesture.reaction_ms,
            gesture.travel_ms,
            gesture.hold_ms,
            state.seq,
            action.reason[:64],
        )
        return self._finish(
            ActionResult(
                action=action,
                accepted=True,
                dispatched_at=self.now(),
                resolved_point=resolution.point,
            )
        )

    def _plan_gesture(self, action: Action, resolution: Resolution) -> PlannedGesture:
        reaction = sample_reaction_time_ms(self._rng)
        travel = 0.0
        hold = 0.0
        waypoints = 0
        # __post_init__ always resolves this; the check is here so the type
        # checker knows that too, and so a future refactor that removes the
        # resolution fails loudly rather than passing None into path planning.
        # An explicit raise rather than an assert, because asserts vanish under
        # `python -O` and this is library code that should not change behaviour
        # based on an interpreter flag.
        if self.cursor is None:  # pragma: no cover - unreachable via __post_init__
            raise RuntimeError("cursor was not resolved in __post_init__")
        if resolution.point is not None and resolution.bbox is not None:
            steps, _ = mouse_path(
                self.cursor,
                resolution.bbox,
                self._rng,
                speed=self.move_speed,
                click_point=resolution.point,
            )
            waypoints = len(steps)
            travel = steps[-1].t_ms
            hold = click_delay_ms(self._rng)
        elif resolution.text:
            delays = typing_delays_ms(resolution.text, self._rng)
            travel = sum(delays)
            hold = click_delay_ms(self._rng)
            waypoints = len(delays)
        elif action.kind in (ActionKind.WAIT, ActionKind.NOOP):
            reaction = 0.0
        return PlannedGesture(
            action=action,
            resolution=resolution,
            waypoints=waypoints,
            reaction_ms=reaction,
            travel_ms=travel,
            hold_ms=hold,
        )

    def _finish(self, result: ActionResult) -> ActionResult:
        self.history.append(result)
        if self.on_result is not None:
            self.on_result(result)
        return result

    # -- introspection used by tests and the CLI ---------------------------

    @property
    def accepted(self) -> list[ActionResult]:
        return [r for r in self.history if r.accepted]

    @property
    def rejected(self) -> list[ActionResult]:
        return [r for r in self.history if not r.accepted]

    def reset(self) -> None:
        self.history.clear()
        self.gestures.clear()
        self._rng = random.Random(self.seed)


def _target_repr(action: Action, resolution: Resolution) -> str:
    if action.target_oid is not None:
        label = resolution.notes.get("label", "?")
        return f"oid={action.target_oid}({label})"
    if action.target_ui:
        return f"ui={action.target_ui}"
    if action.option_index is not None:
        return f"option={action.option_index}"
    if action.text is not None:
        return f"text={action.text[:24]!r}"
    return f"ticks={action.ticks}"


__all__ = ["DryRunActuator", "PlannedGesture"]
