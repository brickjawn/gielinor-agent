"""Is the player doing something? Pixel churn, and its honest limits.

The reflex loop needs to know "am I still chopping" without any game state
access, and RS gives no visual indicator beyond the animation itself. So we
measure churn: how many pixels changed in a player-centred box over the last
few frames.

WHAT THIS GETS WRONG, stated plainly because the failure modes are not exotic:

  * **Camera motion** churns every pixel on screen. Partly handled: churn in
    the player box is compared against churn in the rest of the viewport, and a
    reading only counts as "animating" when the player box churns
    *substantially more* than the scene. A camera rotation during a chop is
    still indistinguishable, and reads as moving-not-animating.
  * **A passing NPC or player** walking through the box churns it exactly like
    an animation does. Nothing here fixes that; a mask of the player's own
    silhouette would, and would need a sprite model we do not have.
  * **Ambient animation** — fire, water, a flashing light behind the player —
    raises the floor. The scene-ratio test absorbs the uniform case; a fire
    directly behind the player defeats it.
  * **Idle animation** churns a little. The threshold is set above idle
    breathing but this is the calibration most likely to need tuning per
    machine, because it depends on the client's frame rate.

Because a chopping animation has brief still moments between swings, the window
takes the *maximum* pairwise churn over the buffer rather than the latest
value; a single quiet frame must not read as "stopped".
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from ..contracts import Rect
from .layout import DEFAULT_LAYOUT, LayoutProfile

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AnimationReading:
    is_animating: bool
    is_moving: bool
    player_churn: float  # fraction of pixels changed in the player box
    scene_churn: float  # ...and in the viewport as a whole
    samples: int
    reason: str = ""

    @property
    def confidence(self) -> float:
        """Low when the two churn signals are too close to separate."""
        if self.samples < 2:
            return 0.0
        gap = abs(self.player_churn - self.scene_churn)
        return float(np.clip(0.55 + 4.0 * gap, 0.0, 1.0))


class AnimationDetector:
    """Rolling-buffer pixel churn over a player-centred region."""

    def __init__(
        self,
        layout: LayoutProfile = DEFAULT_LAYOUT,
        *,
        window: int = 5,
        downscale: int = 2,
        pixel_delta: int = 12,
        churn_threshold: float = 0.020,
        scene_ratio: float = 1.8,
        move_threshold: float = 0.055,
    ) -> None:
        self.layout = layout
        self.window = max(2, window)
        self.downscale = max(1, downscale)
        self.pixel_delta = pixel_delta
        self.churn_threshold = churn_threshold
        self.scene_ratio = scene_ratio
        self.move_threshold = move_threshold
        self._player: deque[np.ndarray] = deque(maxlen=self.window)
        self._scene: deque[np.ndarray] = deque(maxlen=self.window)

    def reset(self) -> None:
        self._player.clear()
        self._scene.clear()

    @property
    def samples(self) -> int:
        return len(self._player)

    def update(self, img: np.ndarray) -> AnimationReading:
        # Both regions are downscaled by the *same* factor. Using a coarser one
        # for the scene (tempting, it is 25x the area) makes the two churn
        # numbers incomparable: block averaging suppresses fine motion, so a
        # camera drag reads as a quiet scene with a busy player box — exactly
        # the false positive the ratio test exists to prevent.
        self._player.append(self._prep(img, self.layout.player_box, self.downscale))
        self._scene.append(self._prep(img, self.layout.viewport, self.downscale))

        if len(self._player) < 2:
            return AnimationReading(False, False, 0.0, 0.0, len(self._player), "warming up")

        player = _max_churn(self._player, self.pixel_delta)
        scene = _median_churn(self._scene, self.pixel_delta)

        moving = scene > self.move_threshold
        animating = player > self.churn_threshold and player > self.scene_ratio * scene
        reason = (
            "player box churn dominates scene"
            if animating
            else ("camera/scene motion dominates" if moving else "below churn threshold")
        )
        return AnimationReading(animating, moving, player, scene, len(self._player), reason)

    @staticmethod
    def _prep(img: np.ndarray, rect: Rect, downscale: int) -> np.ndarray:
        r = rect.clip_to(Rect(0, 0, int(img.shape[1]), int(img.shape[0])))
        if r.w <= 0 or r.h <= 0:
            return np.zeros((1, 1), dtype=np.uint8)
        crop = img[r.y : r.y + r.h, r.x : r.x + r.w]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        if downscale > 1:
            gray = cv2.resize(
                gray,
                (max(1, gray.shape[1] // downscale), max(1, gray.shape[0] // downscale)),
                interpolation=cv2.INTER_AREA,
            )
        return gray


def _pair_churn(a: np.ndarray, b: np.ndarray, delta: int) -> float:
    if a.shape != b.shape:
        return 0.0
    return float((cv2.absdiff(a, b) > delta).mean())


def _max_churn(buf: deque[np.ndarray], delta: int) -> float:
    frames = list(buf)
    return max(_pair_churn(frames[i], frames[i + 1], delta) for i in range(len(frames) - 1))


def _median_churn(buf: deque[np.ndarray], delta: int) -> float:
    frames = list(buf)
    vals = [_pair_churn(frames[i], frames[i + 1], delta) for i in range(len(frames) - 1)]
    return float(np.median(vals))


__all__ = ["AnimationDetector", "AnimationReading"]
