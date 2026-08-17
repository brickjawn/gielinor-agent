"""Frame timing instrumentation.

SPEC section 2 puts a 50 ms budget on the fast loop, which is only meaningful if
something measures it. A nominal FPS setting is not evidence: PipeWire will
happily negotiate 30 fps and then deliver 11 because the compositor is busy, and
the failure looks like "the agent is bad at clicking trees" rather than "frames
are 90 ms old by the time they are perceived".

So FrameClock tracks two different things that are easy to conflate:

* **interval** — wall time between successive frames arriving. This is achieved
  FPS, i.e. how fast the source is.
* **age** — ``now - frame.captured_at``. This is latency, i.e. how stale the
  pixels are that the agent is about to act on. A source can be fast and stale
  at the same time (deep buffering), and only age predicts misclicks.

The clock is injectable so tests assert exact numbers instead of sleeping.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from gielinor.contracts import Frame


@dataclass(frozen=True, slots=True)
class ClockStats:
    frames: int
    achieved_fps: float
    mean_interval_ms: float
    p95_interval_ms: float
    max_interval_ms: float
    mean_age_ms: float
    p95_age_ms: float
    max_age_ms: float
    late_frames: int
    duplicate_frames: int


def _percentile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank percentile. No numpy on a path that runs every frame."""
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, round(q * (len(sorted_values) - 1))))
    return sorted_values[k]


@dataclass
class FrameClock:
    """Rolling window of frame arrival intervals and frame ages."""

    window: int = 120
    budget_ms: float = 50.0
    now: Callable[[], float] = time.monotonic

    _intervals: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False)
    _ages: deque[float] = field(default_factory=lambda: deque(maxlen=120), init=False)
    _last_arrival: float | None = field(default=None, init=False)
    _last_seq: int | None = field(default=None, init=False)
    frames: int = field(default=0, init=False)
    late_frames: int = field(default=0, init=False)
    duplicate_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._intervals = deque(maxlen=self.window)
        self._ages = deque(maxlen=self.window)

    def tick(self, frame: Frame, *, arrived_at: float | None = None) -> float:
        """Record a frame's arrival. Returns its age in ms.

        ``arrived_at`` defaults to now; passing it lets the loop use the same
        timestamp it already sampled rather than calling the clock twice.
        """
        t = self.now() if arrived_at is None else arrived_at
        age_ms = (t - frame.captured_at) * 1000.0

        if self._last_seq is not None and frame.seq == self._last_seq:
            # Backends that return the newest frame will repeat themselves when
            # the loop outruns the source. That is not a dropped frame and must
            # not be counted as one, but it is worth knowing about.
            self.duplicate_frames += 1
        else:
            if self._last_arrival is not None:
                self._intervals.append((t - self._last_arrival) * 1000.0)
            self._last_arrival = t
            self._last_seq = frame.seq
            self.frames += 1
            self._ages.append(age_ms)
            if age_ms > self.budget_ms:
                self.late_frames += 1
        return age_ms

    @property
    def achieved_fps(self) -> float:
        mean = self.mean_interval_ms
        return 1000.0 / mean if mean > 0 else 0.0

    @property
    def mean_interval_ms(self) -> float:
        return sum(self._intervals) / len(self._intervals) if self._intervals else 0.0

    @property
    def mean_age_ms(self) -> float:
        return sum(self._ages) / len(self._ages) if self._ages else 0.0

    def stats(self) -> ClockStats:
        intervals = sorted(self._intervals)
        ages = sorted(self._ages)
        return ClockStats(
            frames=self.frames,
            achieved_fps=self.achieved_fps,
            mean_interval_ms=self.mean_interval_ms,
            p95_interval_ms=_percentile(intervals, 0.95),
            max_interval_ms=intervals[-1] if intervals else 0.0,
            mean_age_ms=self.mean_age_ms,
            p95_age_ms=_percentile(ages, 0.95),
            max_age_ms=ages[-1] if ages else 0.0,
            late_frames=self.late_frames,
            duplicate_frames=self.duplicate_frames,
        )

    def reset(self) -> None:
        self._intervals.clear()
        self._ages.clear()
        self._last_arrival = None
        self._last_seq = None
        self.frames = 0
        self.late_frames = 0
        self.duplicate_frames = 0


__all__ = ["ClockStats", "FrameClock"]
