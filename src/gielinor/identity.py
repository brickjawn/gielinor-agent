"""Stable identity for world objects across frames.

## Why this is a safety component and not a convenience

The planner never emits coordinates. It emits ``oid``, and the actuator clicks
wherever ``oid`` currently is. That indirection is what makes
:func:`validate_action` meaningful: an action naming an object that is no longer
on screen fails, and gets dropped.

It also means the binding from ``oid`` to pixels is a trusted link. If oid 7 is
a tree when the planner is asked, and oid 7 is a *different* tree — or a player,
or an NPC — by the time the click lands, every check in the pipeline passes and
the agent clicks the wrong thing with full confidence. Identity churn does not
present as an error; it presents as the agent doing something inexplicable.

Two rules follow, and this module enforces both.

**Never recycle.** A detector that numbers objects by their index in this
frame's detection list reuses numbers constantly and silently. Here an oid comes
from a :class:`~gielinor.monotonic.MonotonicCounter`; when a track dies its oid
dies with it. A stale reference then names a *retired* oid, which fails
validation loudly instead of resolving to a stranger.

**Prefer a new identity to a wrong one.** When association is uncertain — the
object moved further than expected, or it no longer looks the same — this
allocates a fresh oid rather than stretching the match. The cost of a spurious
new oid is one dropped escalation. The cost of a wrong match is a wrong click.
The asymmetry is large and it points one way.

## Association

Geometry alone is not enough. Two trees of the same species a few tiles apart,
or one tree behind a passing player, both produce the case where IoU says "maybe"
and is wrong. So the cost combines three cheap signals:

* **IoU** — where it is.
* **Appearance** — a 4x4x4 quantised RGB histogram, compared by intersection.
  Coarse on purpose: RS2 sprites are flat-shaded with a tiny palette, so a coarse
  histogram is both discriminative between object *types* and stable under the
  1-2px jitter of a rotating camera. It is ~40us for a 64x64 patch.
* **Shape** — bounding-box aspect and area ratio, which separates a tree from a
  player standing in front of it even when their palettes overlap.

Matching is greedy over the sorted cost list rather than Hungarian. With the
dozens of objects an RS2 viewport holds, greedy and optimal agree almost always,
and when they disagree the gate below rejects the marginal pair anyway. Adding
scipy for the difference is not worth the dependency.

Every candidate pair must pass a **gate** before it can be matched at all. The
gate, not the cost, is what makes "prefer a new identity" true: a pair that is
merely the best available still loses if it is not also good.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .monotonic import MonotonicCounter

if TYPE_CHECKING:
    from collections.abc import Sequence

# --- Tuning ------------------------------------------------------------------
# These are the honest starting values, not measured ones. Phase 1 replaces them
# with numbers derived from labelled frames; until then treat the gate as
# deliberately tight, because a tight gate fails toward "new oid" and a loose one
# fails toward "wrong oid".

DESC_BINS = 4  # per channel -> 64-bin histogram
MAX_CENTROID_PX = 48.0  # a real object cannot cross the viewport between frames
MIN_IOU = 0.20
MAX_APPEARANCE_DIST = 0.45
MAX_COST = 0.60

W_IOU, W_APPEARANCE, W_SHAPE = 0.45, 0.40, 0.15

#: Frames a track may go unseen before it is retired. At 15 Hz this is ~0.5 s,
#: enough to survive a player walking across it, short enough that a genuinely
#: despawned object does not linger as a clickable ghost.
DEFAULT_MAX_MISSES = 8

BBox = tuple[int, int, int, int]  # (x0, y0, x1, y1), half-open, screen pixels


class RectLike(Protocol):
    """Structural stand-in for ``contracts.Rect`` so this module does not import
    it and does not define a second geometry type.

    Mirrors ``contracts.Rect``'s (x, y, w, h) convention. Note this module's
    internal :data:`BBox` is corner-format (x0, y0, x1, y1) because that is what
    :func:`iou` needs; :meth:`Observation.from_rect` is where the two meet."""

    x: int
    y: int
    w: int
    h: int


# --- Input -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Observation:
    """One detection in one frame, before it has an identity.

    ``bbox`` is the single place geometry crosses into this module. Everything
    upstream speaks :class:`~gielinor.contracts.Rect`; use :meth:`from_rect` so
    that conversion happens once, here, rather than at each call site. That is
    the same rule the profile bridge applies to screen geometry, for the same
    reason.
    """

    bbox: BBox
    label: str
    descriptor: np.ndarray
    confidence: float = 1.0

    @classmethod
    def from_rect(
        cls, rect: RectLike, label: str, frame_rgb: np.ndarray, confidence: float = 1.0
    ) -> Observation:
        """Build from a ``contracts.Rect``.

        The one line below is the entire coupling to the Rect field convention.
        ``contracts.Rect`` is (x, y, w, h); :data:`BBox` is (x0, y0, x1, y1), so
        the width and height are added here and nowhere else.
        """
        bbox = (int(rect.x), int(rect.y), int(rect.x) + int(rect.w), int(rect.y) + int(rect.h))
        return cls(
            bbox=bbox, label=label, descriptor=appearance(frame_rgb, bbox), confidence=confidence
        )


@dataclass(slots=True)
class Track:
    """An object with a durable identity."""

    oid: int
    bbox: BBox
    label: str
    descriptor: np.ndarray
    first_seen: int
    last_seen: int
    misses: int = 0
    #: How many times this track survived a frame in which it was not detected.
    #: High values mean the object is frequently occluded, which is worth seeing
    #: in a trace when a click misses.
    occlusions: int = 0

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# --- Feature extraction ------------------------------------------------------


def appearance(frame_rgb: np.ndarray, bbox: BBox) -> np.ndarray:
    """Quantised RGB histogram over the bbox, L1-normalised.

    Returns a zero vector for an empty or out-of-bounds box, which the distance
    function scores as maximally distant — so a degenerate detection can never
    accidentally match something.
    """
    x0, y0, x1, y1 = bbox
    h, w = frame_rgb.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return np.zeros(DESC_BINS**3, dtype=np.float32)

    patch = frame_rgb[y0:y1, x0:x1, :3].reshape(-1, 3)
    q = (patch.astype(np.uint16) * DESC_BINS) >> 8
    idx = q[:, 0] * (DESC_BINS * DESC_BINS) + q[:, 1] * DESC_BINS + q[:, 2]
    hist = np.bincount(idx, minlength=DESC_BINS**3).astype(np.float32)
    total = hist.sum()
    return hist / total if total else hist


def appearance_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - histogram intersection. 0.0 identical, 1.0 disjoint."""
    if a.sum() == 0.0 or b.sum() == 0.0:
        return 1.0
    return float(1.0 - np.minimum(a, b).sum())


def iou(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def _shape_distance(a: BBox, b: BBox) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    aw, ah = max(1, ax1 - ax0), max(1, ay1 - ay0)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)
    area_ratio = min(aw * ah, bw * bh) / max(aw * ah, bw * bh)
    aspect_ratio = min(aw / ah, bw / bh) / max(aw / ah, bw / bh)
    return float(1.0 - 0.5 * (area_ratio + aspect_ratio))


def _centroid_distance(a: BBox, b: BBox) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return float(np.hypot(acx - bcx, acy - bcy))


def match_cost(track: Track, obs: Observation) -> float | None:
    """Cost of binding ``obs`` to ``track``, or ``None`` if the pair is gated out.

    ``None`` is not "expensive". It means the pair is not eligible at any price,
    which is what stops a best-available match from becoming a wrong match when
    nothing good is available.
    """
    if track.label != obs.label:
        return None
    if _centroid_distance(track.bbox, obs.bbox) > MAX_CENTROID_PX:
        return None

    overlap = iou(track.bbox, obs.bbox)
    app = appearance_distance(track.descriptor, obs.descriptor)
    if overlap < MIN_IOU:
        return None
    if app > MAX_APPEARANCE_DIST:
        return None

    cost = (
        W_IOU * (1.0 - overlap)
        + W_APPEARANCE * app
        + W_SHAPE * _shape_distance(track.bbox, obs.bbox)
    )
    return cost if cost <= MAX_COST else None


# --- Tracker -----------------------------------------------------------------


@dataclass
class IdentityRegistry:
    """Assigns and maintains ``oid`` across frames.

    Call :meth:`update` once per perceived frame with every detection in it. The
    returned tracks are the objects the planner may be shown; nothing else has
    an identity, and therefore nothing else is addressable.
    """

    counter: MonotonicCounter = field(default_factory=MonotonicCounter)
    max_misses: int = DEFAULT_MAX_MISSES
    _tracks: dict[int, Track] = field(default_factory=dict, init=False)
    _retired: dict[int, int] = field(default_factory=dict, init=False)  # oid -> frame retired

    # -- queries the validator needs ------------------------------------------

    @property
    def live(self) -> dict[int, Track]:
        return dict(self._tracks)

    def get(self, oid: int) -> Track | None:
        return self._tracks.get(oid)

    def is_live(self, oid: int) -> bool:
        return oid in self._tracks

    def was_retired(self, oid: int) -> bool:
        """True if this oid existed and is gone.

        Worth distinguishing from "never existed" at the validation boundary.
        A retired oid means the world moved during the inference window, which is
        expected and should be counted. An unknown oid means the planner invented
        a reference, which is a model-behaviour signal and should be logged.
        """
        return oid in self._retired

    # -- the loop --------------------------------------------------------------

    def update(self, frame_index: int, observations: Sequence[Observation]) -> list[Track]:
        pairs: list[tuple[float, int, int]] = []
        track_ids = list(self._tracks)
        for ti, oid in enumerate(track_ids):
            for di, obs in enumerate(observations):
                cost = match_cost(self._tracks[oid], obs)
                if cost is not None:
                    pairs.append((cost, ti, di))
        pairs.sort(key=lambda p: (p[0], p[1], p[2]))  # ties broken deterministically

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _cost, ti, di in pairs:
            if ti in used_tracks or di in used_dets:
                continue
            used_tracks.add(ti)
            used_dets.add(di)
            trk = self._tracks[track_ids[ti]]
            obs = observations[di]
            if trk.misses:
                trk.occlusions += 1
            trk.bbox = obs.bbox
            trk.descriptor = obs.descriptor
            trk.last_seen = frame_index
            trk.misses = 0

        # Unmatched detections become new identities. This is the branch that
        # makes an uncertain association safe: it costs an oid, not a mis-click.
        for di, obs in enumerate(observations):
            if di in used_dets:
                continue
            oid = self.counter.next()
            self._tracks[oid] = Track(
                oid=oid,
                bbox=obs.bbox,
                label=obs.label,
                descriptor=obs.descriptor,
                first_seen=frame_index,
                last_seen=frame_index,
            )

        # Unmatched tracks age out. A retired oid is never re-issued, so a
        # reference to it fails validation rather than resolving to a stranger.
        for ti, oid in enumerate(track_ids):
            if ti in used_tracks:
                continue
            trk = self._tracks[oid]
            trk.misses += 1
            if trk.misses > self.max_misses:
                del self._tracks[oid]
                self._retired[oid] = frame_index

        return sorted(self._tracks.values(), key=lambda t: t.oid)
