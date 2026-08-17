"""Viewport object detection: colour signatures, tracking, click points.

Three separable jobs live here, and they fail in different ways:

  * `ViewportDetector` finds *candidate regions* by HSV signature. Cheap, dumb,
    and tuned per scenery type. It has no memory.
  * `ObjectTracker` assigns the `oid`s. This is the part that matters most and
    looks least impressive: `oid` is the only handle the planner ever gets
    (contracts.py), so if oids shuffle between frames, a planner decision made
    on frame N clicks a different tree on frame N+1. Stability is a
    correctness property, not a nicety.
  * `click_point_for` picks where to click. The bbox centre is wrong — a
    forked tree, an L-shaped rock or any concave silhouette can have a centre
    that lands on the grass behind it, which in RS means a walk-here instead of
    a chop. We erode the mask and pick from what survives, which is inside the
    object by construction.

A template-matching path exists for UI sprites, where colour signatures are
useless because the sprite is the identity.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..contracts import DetectedObject, Point, Rect
from .layout import DEFAULT_LAYOUT, LayoutProfile

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Signatures
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ObjectSignature:
    """A colour+shape recipe for one kind of scenery.

    CALIBRATION WARNING: the HSV ranges below are plausible defaults for RS2's
    palette, not measurements from a running client. Lighting, the client's
    brightness setting and the ground texture underneath all shift them. The
    registry is pluggable exactly so a user can replace these with values
    sampled from their own frames instead of patching code.
    """

    label: str
    hsv_lo: tuple[int, int, int]
    hsv_hi: tuple[int, int, int]
    min_area: int = 150
    max_area: int = 60_000
    aspect_lo: float = 0.15  # w/h
    aspect_hi: float = 6.0
    min_fill: float = 0.18  # component pixels / bbox area: rejects speckle
    open_k: int = 3
    close_k: int = 5
    erode_iters: int = 2
    priority: int = 0

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        """Binary mask for this signature, handling hue wraparound."""
        lo, hi = self.hsv_lo, self.hsv_hi
        if lo[0] <= hi[0]:
            m = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        else:  # e.g. red spans 170..179 and 0..10
            a = cv2.inRange(hsv, np.array(lo, np.uint8), np.array((179, hi[1], hi[2]), np.uint8))
            b = cv2.inRange(hsv, np.array((0, lo[1], lo[2]), np.uint8), np.array(hi, np.uint8))
            m = cv2.bitwise_or(a, b)
        if self.open_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.open_k, self.open_k))
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if self.close_k > 1:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.close_k, self.close_k))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        return m


# Note the value/saturation bounds on `tree`: grass shares the hue band, so hue
# alone selects the entire ground plane. Foliage is darker (lower V) and more
# saturated than grass, and that is what separates them — a fact worth stating
# because "just threshold the green" is the obvious approach and it does not work.
DEFAULT_SIGNATURES: tuple[ObjectSignature, ...] = (
    ObjectSignature(
        label="tree",
        hsv_lo=(40, 125, 40),
        hsv_hi=(80, 255, 132),
        min_area=200,
        aspect_lo=0.3,
        aspect_hi=3.0,
        priority=2,
    ),
    ObjectSignature(
        label="willow_tree",
        hsv_lo=(24, 130, 100),
        hsv_hi=(38, 255, 225),
        min_area=250,
        aspect_lo=0.3,
        aspect_hi=3.0,
        priority=3,
    ),
    ObjectSignature(
        label="bank_booth",
        hsv_lo=(8, 120, 60),
        hsv_hi=(22, 235, 210),
        min_area=400,
        aspect_lo=0.5,
        aspect_hi=4.0,
        priority=1,
    ),
)


@dataclass(frozen=True, slots=True)
class Detection:
    """A pre-tracking candidate: no oid yet, because oids come from the tracker."""

    label: str
    bbox: Rect  # absolute frame coordinates
    click_point: Point  # absolute, proven inside the mask
    confidence: float
    area: int
    mask: np.ndarray = field(
        repr=False, compare=False, default_factory=lambda: np.zeros((0, 0), bool)
    )

    @property
    def center(self) -> Point:
        return self.bbox.center


# --------------------------------------------------------------------------
# Click point selection
# --------------------------------------------------------------------------


def click_point_for(mask: np.ndarray, erode_iters: int = 2) -> tuple[int, int] | None:
    """Pick a point inside `mask`, preferring one well away from the edge.

    Returns local (x, y) or None for an empty mask. Two tiers:

      1. Erode by `erode_iters`; if anything survives, take the surviving pixel
         nearest the component centroid. Nearest-to-centroid rather than any
         pixel keeps the click visually sensible on a normal blob, while the
         "must survive erosion" constraint is what keeps it off the background
         on a concave one.
      2. If erosion empties the mask (thin objects — a rope, a fence post), fall
         back to the distance-transform maximum, which is by definition the
         pixel furthest from any background pixel, hence inside.
    """
    m = mask.astype(np.uint8)
    if not m.any():
        return None
    if erode_iters > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        eroded = cv2.erode(m, k, iterations=erode_iters)
    else:
        eroded = m
    if eroded.any():
        ys, xs = np.nonzero(eroded)
        cy, cx = float(np.mean(np.nonzero(m)[0])), float(np.mean(np.nonzero(m)[1]))
        d = (xs - cx) ** 2 + (ys - cy) ** 2
        i = int(np.argmin(d))
        return int(xs[i]), int(ys[i])
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    idx = int(np.argmax(dist))
    return int(idx % dist.shape[1]), int(idx // dist.shape[1])


# --------------------------------------------------------------------------
# Detector
# --------------------------------------------------------------------------


class ViewportDetector:
    """Colour-signature detection over a region of the frame."""

    def __init__(
        self,
        signatures: Sequence[ObjectSignature] = DEFAULT_SIGNATURES,
        *,
        layout: LayoutProfile = DEFAULT_LAYOUT,
        max_objects: int = 24,
        nms_iou: float = 0.55,
        max_coverage: float = 0.5,
    ) -> None:
        self.signatures = tuple(signatures)
        self.layout = layout
        self.max_objects = max_objects
        self.nms_iou = nms_iou
        self.max_coverage = max_coverage

    def detect(self, img: np.ndarray, rect: Rect | None = None) -> list[Detection]:
        r = rect if rect is not None else self.layout.viewport
        r = r.clip_to(Rect(0, 0, int(img.shape[1]), int(img.shape[0])))
        if r.w <= 0 or r.h <= 0:
            return []
        crop = img[r.y : r.y + r.h, r.x : r.x + r.w]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        found: list[Detection] = []
        for sig in self.signatures:
            found.extend(self._detect_one(hsv, sig, r))
        found.sort(key=lambda d: (d.confidence, d.area), reverse=True)
        kept = _suppress_overlaps(found, self.nms_iou)
        if len(kept) > self.max_objects:
            logger.debug("clipping %d detections to %d", len(kept), self.max_objects)
        return kept[: self.max_objects]

    def _detect_one(self, hsv: np.ndarray, sig: ObjectSignature, origin: Rect) -> list[Detection]:
        m = sig.mask(hsv)
        # A signature that selects most of the scene is mis-calibrated, not
        # lucky. Left unguarded it produces one giant "tree" covering the
        # viewport whose click point is somewhere in the grass — a failure that
        # looks like a tracking bug three layers downstream.
        coverage = float(np.count_nonzero(m)) / float(m.size)
        if coverage > self.max_coverage:
            logger.warning(
                "signature %r matched %.0f%% of the region; ignoring it as mis-calibrated",
                sig.label,
                coverage * 100,
            )
            return []
        n, labels, stats, _centroids = cv2.connectedComponentsWithStats(m, connectivity=8)
        out: list[Detection] = []
        for i in range(1, n):
            x, y, w, h, area = (int(v) for v in stats[i])
            if area < sig.min_area or area > sig.max_area:
                continue
            aspect = w / h if h else 0.0
            if not (sig.aspect_lo <= aspect <= sig.aspect_hi):
                continue
            fill = area / float(max(1, w * h))
            if fill < sig.min_fill:
                continue
            comp = labels[y : y + h, x : x + w] == i
            pt = click_point_for(comp, sig.erode_iters)
            if pt is None:  # pragma: no cover - area check already excludes this
                continue
            # Confidence blends "is it solid" with "is it a plausible size", so a
            # 3-pixel-over-threshold smear does not arrive claiming certainty.
            size_term = min(1.0, area / (4.0 * sig.min_area))
            conf = float(np.clip(0.5 * fill + 0.3 * size_term + 0.2, 0.0, 1.0))
            out.append(
                Detection(
                    label=sig.label,
                    bbox=Rect(origin.x + x, origin.y + y, w, h),
                    click_point=Point(origin.x + x + pt[0], origin.y + y + pt[1]),
                    confidence=conf,
                    area=area,
                    mask=comp,
                )
            )
        return out


def _iou(a: Rect, b: Rect) -> float:
    inter = a.clip_to(b)
    ia = inter.w * inter.h
    if ia <= 0:
        return 0.0
    union = a.w * a.h + b.w * b.h - ia
    return ia / union if union else 0.0


def _suppress_overlaps(dets: Sequence[Detection], thresh: float) -> list[Detection]:
    kept: list[Detection] = []
    for d in dets:
        if any(_iou(d.bbox, k.bbox) > thresh for k in kept):
            continue
        kept.append(d)
    return kept


# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Track:
    oid: int
    label: str
    bbox: Rect
    center: Point
    missing: int = 0
    hits: int = 1


class ObjectTracker:
    """Assigns stable `oid`s across frames by IoU + centroid association.

    Deliberately simple (greedy, no Kalman, no appearance model) because the
    input is 10-20 Hz video of near-static scenery; the failure mode worth
    engineering against is *identity swaps between two similar trees*, which a
    motion model would not help with anyway. Two guards do the real work:
    association requires a matching label, and a track survives `max_missing`
    frames of non-detection so one bad frame does not renumber the world.

    Note on purity: `contracts.Perceiver` says perception is side-effect free,
    and this object is the one exception — it must remember oids to hand out
    stable ones. `PixelPerceiver` keeps exactly one of these and nothing else.
    """

    def __init__(
        self,
        *,
        min_iou: float = 0.15,
        max_center_dist: float = 48.0,
        max_missing: int = 4,
        iou_weight: float = 0.7,
    ) -> None:
        self.min_iou = min_iou
        self.max_center_dist = max_center_dist
        self.max_missing = max_missing
        self.iou_weight = iou_weight
        self._tracks: dict[int, _Track] = {}
        self._next_oid = 1

    @property
    def active_oids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, dets: Sequence[Detection]) -> list[int]:
        """Return an oid per detection, in the same order as `dets`."""
        pairs: list[tuple[float, int, int]] = []  # (score, det_idx, oid)
        for di, d in enumerate(dets):
            for oid, t in self._tracks.items():
                if t.label != d.label:
                    continue
                score = self._score(d, t)
                if score > 0.0:
                    pairs.append((score, di, oid))
        pairs.sort(reverse=True)

        assigned: dict[int, int] = {}
        used_oids: set[int] = set()
        for _score, di, oid in pairs:
            if di in assigned or oid in used_oids:
                continue
            assigned[di] = oid
            used_oids.add(oid)

        out: list[int] = []
        for di, d in enumerate(dets):
            matched = assigned.get(di)
            if matched is None:
                new_oid = self._next_oid
                self._next_oid += 1
                self._tracks[new_oid] = _Track(new_oid, d.label, d.bbox, d.center)
                out.append(new_oid)
                continue
            track = self._tracks[matched]
            track.bbox = d.bbox
            track.center = d.center
            track.missing = 0
            track.hits += 1
            out.append(matched)

        for oid in list(self._tracks):
            if oid in used_oids or oid in out:
                continue
            t = self._tracks[oid]
            t.missing += 1
            if t.missing > self.max_missing:
                logger.debug(
                    "dropping track oid=%d label=%s after %d misses", oid, t.label, t.missing
                )
                del self._tracks[oid]
        return out

    def _score(self, d: Detection, t: _Track) -> float:
        iou = _iou(d.bbox, t.bbox)
        dist = float(np.hypot(d.center.x - t.center.x, d.center.y - t.center.y))
        if iou < self.min_iou and dist > self.max_center_dist:
            return 0.0
        prox = max(0.0, 1.0 - dist / max(1.0, self.max_center_dist))
        return self.iou_weight * iou + (1.0 - self.iou_weight) * prox


def to_detected_objects(
    dets: Sequence[Detection],
    oids: Sequence[int],
    *,
    space: str = "viewport",
) -> tuple[DetectedObject, ...]:
    """Zip detections with their tracker-assigned oids into the contract type."""
    if len(dets) != len(oids):
        raise ValueError(f"{len(dets)} detections but {len(oids)} oids")
    return tuple(
        DetectedObject(
            oid=int(oid),
            label=d.label,
            bbox=d.bbox,
            click_point=d.click_point,
            confidence=d.confidence,
            space=space,  # type: ignore[arg-type]
            attrs={"area": str(d.area)},
        )
        for d, oid in zip(dets, oids, strict=True)
    )


# --------------------------------------------------------------------------
# Template matching (UI sprites)
# --------------------------------------------------------------------------


class TemplateMatcher:
    """Normalised-correlation matching for fixed UI sprites.

    Colour signatures identify *scenery*; a sprite (the run-energy icon, a tab
    button, a specific item) is identified by its exact pixels, and template
    matching is both faster and stricter there. Optional by design: the
    woodcutting loop never needs it, so it stays out of the hot path unless
    templates are registered.
    """

    def __init__(self, *, method: int = cv2.TM_CCOEFF_NORMED) -> None:
        self.method = method
        self._templates: dict[str, np.ndarray] = {}

    def add_template(self, name: str, patch: np.ndarray) -> None:
        if patch.ndim == 3:
            patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        self._templates[name] = patch.astype(np.uint8)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._templates))

    def find(
        self,
        img: np.ndarray,
        rect: Rect | None = None,
        *,
        threshold: float = 0.85,
        max_hits: int = 8,
    ) -> list[Detection]:
        if not self._templates:
            return []
        full = Rect(0, 0, int(img.shape[1]), int(img.shape[0]))
        r = (rect or full).clip_to(full)
        if r.w <= 0 or r.h <= 0:
            return []
        crop = img[r.y : r.y + r.h, r.x : r.x + r.w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        out: list[Detection] = []
        for name, tpl in self._templates.items():
            th, tw = tpl.shape[:2]
            if th > gray.shape[0] or tw > gray.shape[1]:
                continue
            res = cv2.matchTemplate(gray, tpl, self.method)
            ys, xs = np.where(res >= threshold)
            hits = sorted(
                ((float(res[y, x]), int(x), int(y)) for y, x in zip(ys, xs, strict=True)),
                reverse=True,
            )
            for score, x, y in hits[:max_hits]:
                bbox = Rect(r.x + x, r.y + y, tw, th)
                if any(_iou(bbox, d.bbox) > 0.4 for d in out):
                    continue
                out.append(
                    Detection(
                        label=name,
                        bbox=bbox,
                        click_point=bbox.center,
                        confidence=score,
                        area=tw * th,
                        mask=np.ones((th, tw), dtype=bool),
                    )
                )
        return out


__all__ = [
    "DEFAULT_SIGNATURES",
    "Detection",
    "ObjectSignature",
    "ObjectTracker",
    "TemplateMatcher",
    "ViewportDetector",
    "click_point_for",
    "to_detected_objects",
]
