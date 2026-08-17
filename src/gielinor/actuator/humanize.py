"""Human-plausible mouse paths, click points and reaction times.

Why bother, when the game cannot see the cursor between frames? Two reasons,
and only one of them is about detection.

The honest engineering reason: a teleporting cursor plus a zero-latency click is
a *test oracle problem*. It hides bugs. Real input has travel time, so an action
issued at frame N lands at frame N+3, and any bug in the runtime's pending-click
tracking or staleness logic only shows up once that gap exists. Modelling the
gap makes the fast loop's timing assumptions falsifiable offline.

The other reason: clicking the exact geometric centre of a bounding box, every
time, to the pixel, is not a thing a hand does — and the whole project is
premised on interacting with the client the way a person does, through the
screen, rather than through a socket.

Everything here is a **pure function of (arguments, random.Random)**. No clocks,
no sleeps, no I/O. Seed the RNG and the output is reproducible, which is what
makes mouse behaviour unit-testable at all.

Models used:

* **Fitts' law** for movement duration: ``MT = a + b * log2(2D / W)``. Bigger
  targets and shorter distances are faster; the log term is why a 3 px item slot
  costs much more than a 200 px tree.
* **Lognormal** for reaction time. Human latency distributions are right-skewed
  — you cannot react faster than ~150 ms but you can always be slower — and a
  Gaussian would produce impossible negative tails.
* **Cubic Bezier + min-jerk velocity** for the path. Real pointing motion is
  smooth, accelerates out and decelerates in, and rarely travels a straight line.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from gielinor.contracts import Point, Rect

if TYPE_CHECKING:  # pragma: no cover - typing only
    import random

# Fitts' law constants in ms. Typical fitted values for a mouse land near
# a=50-150, b=100-150; these produce ~200 ms for a comfortable target at 300 px,
# which matches casual play rather than a speedrunner.
FITTS_A_MS = 85.0
FITTS_B_MS = 115.0

REACTION_MEDIAN_MS = 235.0
REACTION_SIGMA = 0.38
REACTION_FLOOR_MS = 110.0
REACTION_CEIL_MS = 1400.0


@dataclass(frozen=True, slots=True)
class PathStep:
    """One cursor waypoint. ``t_ms`` is cumulative time since the move started."""

    point: Point
    t_ms: float


def sample_reaction_time_ms(
    rng: random.Random,
    *,
    median_ms: float = REACTION_MEDIAN_MS,
    sigma: float = REACTION_SIGMA,
    floor_ms: float = REACTION_FLOOR_MS,
    ceil_ms: float = REACTION_CEIL_MS,
) -> float:
    """Sample a reaction delay from a lognormal, clamped to plausible bounds.

    A constant here would be the tell: perfectly periodic input is the one
    signal no human ever produces.
    """
    value = median_ms * math.exp(rng.gauss(0.0, sigma))
    return min(max(value, floor_ms), ceil_ms)


def fitts_movement_ms(
    distance_px: float,
    target_w_px: float,
    *,
    a_ms: float = FITTS_A_MS,
    b_ms: float = FITTS_B_MS,
    rng: random.Random | None = None,
    noise: float = 0.14,
    speed: float = 1.0,
) -> float:
    """Movement time for a pointing gesture, per Fitts' law, optionally jittered.

    ``speed`` > 1 makes the agent faster than the fitted human baseline; it
    exists so a demo can run quickly without pretending the model changed.
    """
    width = max(1.0, target_w_px)
    distance = max(1.0, distance_px)
    index_of_difficulty = math.log2((2.0 * distance) / width)
    base = a_ms + b_ms * max(0.0, index_of_difficulty)
    if rng is not None and noise > 0:
        base *= math.exp(rng.gauss(0.0, noise))
    return max(20.0, base / max(0.05, speed))


def jitter_point_in(
    bbox: Rect,
    rng: random.Random,
    *,
    bias: Point | None = None,
    sigma_frac: float = 0.22,
) -> Point:
    """Pick a click point inside ``bbox``, clustered near ``bias`` (default centre).

    Gaussian rather than uniform because a hand aims at the middle of a thing
    and misses by a little. Guaranteed to land inside the box: the result is
    clamped, so a wide sigma degrades to "hugs the edges", never to "misses".
    """
    if bbox.w <= 0 or bbox.h <= 0:
        raise ValueError(f"cannot pick a point inside a degenerate rect {bbox}")
    target = bias if bias is not None else bbox.center
    sx = max(0.5, bbox.w * sigma_frac)
    sy = max(0.5, bbox.h * sigma_frac)
    x = round(rng.gauss(target.x, sx))
    y = round(rng.gauss(target.y, sy))
    x = min(max(x, bbox.x), bbox.x + bbox.w - 1)
    y = min(max(y, bbox.y), bbox.y + bbox.h - 1)
    return Point(x, y)


def _ease_min_jerk(t: float) -> float:
    """Min-jerk position profile: zero velocity and acceleration at both ends."""
    return 10 * t**3 - 15 * t**4 + 6 * t**5


def _cubic_bezier(
    p0: Point, p1: tuple[float, float], p2: tuple[float, float], p3: Point, t: float
) -> tuple[float, float]:
    u = 1.0 - t
    x = u**3 * p0.x + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3.x
    y = u**3 * p0.y + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3.y
    return x, y


def bezier_path(
    start: Point,
    end: Point,
    rng: random.Random,
    *,
    steps: int | None = None,
    curvature: float = 0.16,
) -> list[Point]:
    """A curved, evenly-sampled path from ``start`` to ``end`` inclusive.

    Control points are pushed perpendicular to the straight line by a fraction
    of the travel distance, with independent signs, which gives the gentle S
    that a wrist produces. Endpoints are exact — the caller must be able to rely
    on ``path[-1]`` being the point that will be clicked.
    """
    dx, dy = end.x - start.x, end.y - start.y
    distance = math.hypot(dx, dy)
    if steps is None:
        # Enough samples that consecutive points are a few px apart, bounded so
        # a cross-screen move does not become a thousand events.
        steps = int(min(96, max(2, distance / 6.0)))
    if distance < 1.0:
        return [start, end]

    # Unit perpendicular.
    px, py = -dy / distance, dx / distance
    amp = distance * curvature
    off1 = rng.gauss(0.0, amp)
    off2 = rng.gauss(0.0, amp)
    c1 = (start.x + dx * 0.33 + px * off1, start.y + dy * 0.33 + py * off1)
    c2 = (start.x + dx * 0.66 + px * off2, start.y + dy * 0.66 + py * off2)

    points: list[Point] = [start]
    for i in range(1, steps):
        t = _ease_min_jerk(i / steps)
        x, y = _cubic_bezier(start, c1, c2, end, t)
        candidate = Point(round(x), round(y))
        if candidate != points[-1]:
            points.append(candidate)
    if points[-1] != end:
        points.append(end)
    return points


def mouse_path(
    start: Point,
    target: Rect,
    rng: random.Random,
    *,
    speed: float = 1.0,
    curvature: float = 0.16,
    overshoot_chance: float = 0.15,
    click_point: Point | None = None,
) -> tuple[list[PathStep], Point]:
    """Full gesture: curved travel to a jittered point inside ``target``.

    Returns the timed waypoints and the final click point. Long moves sometimes
    overshoot and correct, which is both realistic and a useful stress case for
    the runtime — the correction is extra frames between "decided" and "clicked".
    """
    end = click_point if click_point is not None else jitter_point_in(target, rng)
    distance = math.hypot(end.x - start.x, end.y - start.y)
    duration = fitts_movement_ms(distance, min(target.w, target.h), rng=rng, speed=speed)

    points = bezier_path(start, end, rng, curvature=curvature)
    if distance > 220 and rng.random() < overshoot_chance:
        beyond = Point(
            end.x + round(rng.gauss(0, 6)) + (2 if end.x >= start.x else -2),
            end.y + round(rng.gauss(0, 6)) + (2 if end.y >= start.y else -2),
        )
        correction = bezier_path(beyond, end, rng, steps=4, curvature=0.05)
        points = [*points[:-1], beyond, *correction[1:]]
        duration *= 1.18

    n = len(points)
    steps: list[PathStep] = []
    for i, p in enumerate(points):
        # Time is distributed along the min-jerk profile, so waypoints bunch up
        # at the ends (slow start, slow settle) and spread through the middle.
        frac = _ease_min_jerk(i / (n - 1)) if n > 1 else 1.0
        steps.append(PathStep(point=p, t_ms=frac * duration))
    return steps, end


def click_delay_ms(rng: random.Random, *, median_ms: float = 62.0, sigma: float = 0.32) -> float:
    """Dwell between mouse-down and mouse-up. Shorter and tighter than reaction time."""
    return min(max(median_ms * math.exp(rng.gauss(0.0, sigma)), 18.0), 400.0)


def typing_delays_ms(text: str, rng: random.Random, *, wpm: float = 55.0) -> list[float]:
    """Per-character inter-key delays for ``text``.

    Space gets a small extra pause and repeated characters a small deduction,
    matching the coarse shape of real typing without pretending to model
    digraph frequencies.
    """
    base = 60_000.0 / (wpm * 5.0)
    delays: list[float] = []
    previous = ""
    for ch in text:
        d = base * math.exp(rng.gauss(0.0, 0.33))
        if ch == " ":
            d *= 1.25
        elif ch == previous:
            d *= 0.85
        delays.append(min(max(d, 25.0), 600.0))
        previous = ch
    return delays


__all__ = [
    "FITTS_A_MS",
    "FITTS_B_MS",
    "REACTION_MEDIAN_MS",
    "PathStep",
    "bezier_path",
    "click_delay_ms",
    "fitts_movement_ms",
    "jitter_point_in",
    "mouse_path",
    "sample_reaction_time_ms",
    "typing_delays_ms",
]
