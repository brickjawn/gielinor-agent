"""Trajectory execution that can be withdrawn while it is happening.

## The window this closes

`humanize.py` spends 200-1000 ms moving the cursor before it clicks. Every
staleness gate in the system runs *before* that movement starts. So an action
can be validated against a perfectly fresh GameState and still land 800 ms later
on a world that has changed — the tree was chopped by someone else, an NPC
walked over the tile, the camera rotated. Nothing downstream notices, because
nothing downstream looks again.

The fix has two halves and both are needed.

**The trajectory is interruptible; the click is not.** Movement is divided into
waypoints, and before each one the executor checks whether the supervisor has
withdrawn the command. Dwells are implemented as a race against the abort event
rather than a sleep, so a withdrawal during a 400 ms pause is honoured in
microseconds instead of waiting the pause out. The click itself is atomic: there
is no useful state between "press" and "release", so the last abort check
happens immediately before the press and the click, once begun, completes.

That makes the residual exposure equal to the duration of a single
:meth:`MouseDriver.click` call — tens of microseconds for an X11/Wayland
synthetic event — instead of the full trajectory. Not zero. Bounded, small, and
measurable, which is the most an asynchronous world allows.

**The last check is not a clock, it is the screen.** An abort can only fire for
changes the supervisor's perception loop noticed. The failure it cannot catch is
the one where GameState is still valid but the *cursor is not where the model
of the world says it is* — a stale bbox, a camera nudge, a one-tile mis-projection.

So before clicking, the executor reads the client's own hover text (top-left
"Chop down Tree") and requires it to match what the action intended. This is the
client telling us, in its own words, what it believes is under the cursor. It
costs one small OCR of a fixed rect — sub-millisecond with the glyph matcher
ADR 0002 already specifies — and it is the only check in the system that
verifies the actual click target rather than a model of it.

Note what this does *not* require: the actuator does not get a GameState, does
not get a Perceiver, and does not gain judgement. It gets one rect and a string
comparison. ADR 0004's boundary survives.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence


class ExecStatus(StrEnum):
    COMPLETED = "completed"  # click issued
    ABORTED = "aborted"  # supervisor withdrew it; no click issued
    REJECTED_HOVER = "rejected_hover"  # client disagreed about the target; no click issued
    FAILED = "failed"  # driver error; click may or may not have landed


@dataclass(frozen=True, slots=True)
class Waypoint:
    x: int
    y: int
    #: Seconds to dwell *before* moving here. Produced by `humanize.py`.
    delay: float = 0.0


@dataclass(frozen=True, slots=True)
class Outcome:
    status: ExecStatus
    #: The single most important field in this module. Assertions about abort
    #: correctness are assertions about this boolean, not about the status.
    clicked: bool
    seq: int
    waypoints_travelled: int
    elapsed_s: float
    detail: str = ""
    hover_text: str | None = None


@runtime_checkable
class MouseDriver(Protocol):
    async def move_to(self, x: int, y: int) -> None: ...
    async def click(self, x: int, y: int, button: str = "left") -> None: ...


@runtime_checkable
class HoverVerifier(Protocol):
    """Reads the client's own description of what is under the cursor.

    Returns ``None`` when the hover region is empty, which for RS2 means the
    cursor is over nothing interactive. That is a rejection, not an unknown.
    """

    def read_hover_text(self) -> str | None: ...


_TAG = re.compile(r"<[^>]*>")
_WS = re.compile(r"\s+")


def normalise_hover(text: str | None) -> str:
    """Strip colour tags, drop the 'N more options' tail, collapse whitespace.

    RS2 renders e.g. ``Chop down Tree<col=ffffff> / 2 more options``. Only the
    leading option is under the cursor for a left click, so only it is compared.
    """
    if not text:
        return ""
    cleaned = _TAG.sub("", text)
    cleaned = cleaned.split(" / ")[0]
    return _WS.sub(" ", cleaned).strip().lower()


def hover_matches(actual: str | None, expected: str) -> bool:
    """Exact match after normalisation.

    Deliberately strict. A failsafe that accepts near-misses is not a failsafe,
    and the cost of a false rejection is one dropped action while the reflex
    layer keeps playing.
    """
    want = normalise_hover(expected)
    return bool(want) and normalise_hover(actual) == want


@dataclass
class AbortableExecutor:
    mouse: MouseDriver
    hover: HoverVerifier | None = None
    #: Injectable so tests run in zero wall-clock and so a future replay harness
    #: can drive the trajectory deterministically.
    sleeper: Sleeper = field(default_factory=lambda: _RealSleeper())

    async def run(
        self,
        seq: int,
        path: Sequence[Waypoint],
        abort: asyncio.Event,
        expect_hover: str | None = None,
        button: str = "left",
    ) -> Outcome:
        started = time.perf_counter()
        travelled = 0

        def done(
            status: ExecStatus, clicked: bool, detail: str = "", hov: str | None = None
        ) -> Outcome:
            return Outcome(
                status=status,
                clicked=clicked,
                seq=seq,
                waypoints_travelled=travelled,
                elapsed_s=time.perf_counter() - started,
                detail=detail,
                hover_text=hov,
            )

        if not path:
            return done(ExecStatus.FAILED, False, "empty trajectory")

        for wp in path:
            if abort.is_set():
                return done(ExecStatus.ABORTED, False, "withdrawn during travel")
            if wp.delay > 0.0 and await self.sleeper.sleep_or_abort(wp.delay, abort):
                return done(ExecStatus.ABORTED, False, "withdrawn during dwell")
            try:
                await self.mouse.move_to(wp.x, wp.y)
            except Exception as exc:
                return done(ExecStatus.FAILED, False, f"move_to failed: {exc!r}")
            travelled += 1

        # --- terminal gate: last chance to not click -------------------------
        if abort.is_set():
            return done(ExecStatus.ABORTED, False, "withdrawn before click")

        seen: str | None = None
        if expect_hover is not None:
            if self.hover is None:
                return done(
                    ExecStatus.FAILED, False, "action requires hover verification, none configured"
                )
            try:
                seen = self.hover.read_hover_text()
            except Exception as exc:
                return done(ExecStatus.FAILED, False, f"hover read failed: {exc!r}")
            if not hover_matches(seen, expect_hover):
                return done(
                    ExecStatus.REJECTED_HOVER,
                    False,
                    f"client says {normalise_hover(seen)!r}, "
                    f"action wanted {normalise_hover(expect_hover)!r}",
                    seen,
                )

        # Hover verification is itself an inference window, however short. Check
        # once more so a withdrawal that arrived during the read is still honoured.
        if abort.is_set():
            return done(ExecStatus.ABORTED, False, "withdrawn after hover check", seen)

        try:
            await self.mouse.click(path[-1].x, path[-1].y, button)
        except Exception as exc:
            # `clicked` is True here on purpose: the press may have landed. An
            # action whose effect is unknown must be treated as having happened,
            # or the supervisor will retry something that already took effect.
            return done(ExecStatus.FAILED, True, f"click failed: {exc!r}", seen)

        return done(ExecStatus.COMPLETED, True, "", seen)


# --- clock -------------------------------------------------------------------


class Sleeper(Protocol):
    async def sleep_or_abort(self, seconds: float, abort: asyncio.Event) -> bool:
        """Sleep, returning True if aborted first."""
        ...


class _RealSleeper:
    async def sleep_or_abort(self, seconds: float, abort: asyncio.Event) -> bool:
        try:
            await asyncio.wait_for(abort.wait(), timeout=seconds)
        except TimeoutError:
            return False
        return True
