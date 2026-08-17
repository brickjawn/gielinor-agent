"""Read the 28-slot inventory grid.

`InventoryState.is_full` is the single most load-bearing boolean in the whole
agent: it is what ends a woodcutting trip. A false negative means chopping into
a full inventory forever; a false positive means banking a half-empty load. So
occupancy is decided by an explicit background model rather than "is this cell
bright".

The background model is estimated from the cells themselves, per frame. The
naive version — median across the 28 cells — breaks precisely when the answer
matters most, because at 28/28 the median cell *is* an item. Instead we pool
the outer border ring of every cell: item sprites are drawn inset, so the ring
is background even in a full inventory, and pooling 28 rings makes the estimate
robust to the few sprites that do bleed to the edge.

Item identification (which log? how many?) is deliberately out of scope for v1:
the reflex loop needs "how many slots are used", not "what is in slot 12", and
sprite classification is a much bigger, much more calibration-hungry problem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..contracts import InventorySlot, InventoryState, Rect
from .layout import DEFAULT_LAYOUT, LayoutProfile
from .ocr import channel_max, within_tol

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class InventoryReading:
    """Occupancy plus the evidence, so callers can judge how sure to be."""

    state: InventoryState
    fills: tuple[float, ...]  # per-slot fraction of non-background pixels
    background_rgb: tuple[int, int, int]
    confidence: float


class InventoryReader:
    """Occupancy detection for the fixed 4x7 inventory grid."""

    def __init__(
        self,
        layout: LayoutProfile = DEFAULT_LAYOUT,
        *,
        color_tol: int = 30,
        min_fill: float = 0.06,
        ring_px: int = 4,
        empty_reference: np.ndarray | None = None,
    ) -> None:
        self.layout = layout
        self.color_tol = color_tol
        self.min_fill = min_fill
        self.ring_px = ring_px
        # An explicit empty-inventory capture beats any estimate; offered for
        # users who calibrate, never required, because demanding a calibration
        # step before the agent runs at all is how projects die on the shelf.
        self.empty_reference = empty_reference

    def read(self, img: np.ndarray) -> InventoryReading:
        slots_rects = self.layout.inventory_slots()
        cells = [_crop(img, r) for r in slots_rects]
        valid = [c for c in cells if c.size]
        if not valid:
            logger.warning("inventory region %s is outside the frame", self.layout.inventory_rect)
            return InventoryReading(InventoryState(()), (), (0, 0, 0), 0.0)

        bg_bgr = self._background(valid)
        fills: list[float] = []
        slots: list[InventorySlot] = []
        for i, (rect, cell) in enumerate(zip(slots_rects, cells, strict=True)):
            fill = self._fill_fraction(cell, bg_bgr)
            fills.append(fill)
            slots.append(InventorySlot(index=i, occupied=fill > self.min_fill, bbox=rect))

        conf = self._confidence(fills)
        state = InventoryState(tuple(slots))
        logger.debug("inventory %d/28 used (conf %.2f)", state.used, conf)
        return InventoryReading(
            state=state,
            fills=tuple(fills),
            background_rgb=(int(bg_bgr[2]), int(bg_bgr[1]), int(bg_bgr[0])),
            confidence=conf,
        )

    # -- internals ----------------------------------------------------------

    def _background(self, cells: Sequence[np.ndarray]) -> np.ndarray:
        if self.empty_reference is not None:
            return np.median(self.empty_reference.reshape(-1, 3).astype(np.float32), axis=0).astype(
                np.int16
            )
        ring_px: list[np.ndarray] = []
        for c in cells:
            h, w = c.shape[:2]
            k = min(self.ring_px, h // 2, w // 2)
            if k <= 0:
                continue
            ring_px.append(c[:k].reshape(-1, 3))
            ring_px.append(c[-k:].reshape(-1, 3))
            ring_px.append(c[:, :k].reshape(-1, 3))
            ring_px.append(c[:, -k:].reshape(-1, 3))
        if not ring_px:
            return np.array([0, 0, 0], dtype=np.int16)
        stack = np.concatenate(ring_px, axis=0)
        return np.median(stack.astype(np.float32), axis=0).astype(np.int16)

    def _fill_fraction(self, cell: np.ndarray, bg: np.ndarray) -> float:
        if cell.size == 0:
            return 0.0
        if self.empty_reference is not None and self.empty_reference.shape == cell.shape:
            diff = cv2.absdiff(cell, self.empty_reference)
            return float((channel_max(diff) > self.color_tol).mean())
        rgb = (int(bg[2]), int(bg[1]), int(bg[0]))
        return float((~within_tol(cell, rgb, self.color_tol)).mean())

    def _confidence(self, fills: Sequence[float]) -> float:
        """How cleanly the fills separate into "empty" and "occupied".

        A frame where every slot sits near the threshold is a frame where the
        background estimate is wrong, and the perceiver should say so rather
        than report a crisp 14/28 it does not believe.
        """
        if not fills:
            return 0.0
        arr = np.asarray(fills, dtype=np.float32)
        margins = np.abs(arr - self.min_fill) / max(self.min_fill, 1e-6)
        return float(np.clip(np.mean(np.minimum(margins, 1.0)), 0.0, 1.0))


def _crop(img: np.ndarray, rect: Rect) -> np.ndarray:
    r = rect.clip_to(Rect(0, 0, int(img.shape[1]), int(img.shape[0])))
    if r.w <= 0 or r.h <= 0:
        return img[0:0, 0:0]
    return img[r.y : r.y + r.h, r.x : r.x + r.w]


__all__ = ["InventoryReader", "InventoryReading"]
