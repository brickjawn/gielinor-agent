"""`PixelPerceiver`: Frame in, GameState out, inside 50 ms.

Composition only — every actual measurement lives in a submodule. What this
file owns is the two things a composed pipeline has to get right and usually
does not:

**Per-stage timing.** `GameState.perception_ms` is the total, but a total is
useless for optimisation. `stage_ms` keeps the breakdown on the perceiver so
`benchmarks/bench_perception.py` can say *which* stage ate the budget, and so a
regression shows up as "chat went from 2 ms to 19 ms" rather than "perception
got slower".

**Honest confidence.** `GameState.confidence` drives the SPEC section 2
escalation policy: low confidence is what wakes the LLM. Hardcoding 1.0 would
mean the slow loop never fires on novel state, which quietly deletes one of the
two loops. So confidence here is a weighted blend of what the sub-detectors
actually reported, with a hard cap when the layout profile does not match the
frame — because if the geometry is wrong, every number downstream is wrong and
no amount of sub-detector agreement fixes it.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..contracts import (
    DetectedObject,
    DialogueState,
    Frame,
    GameState,
    InventoryState,
    MinimapState,
    PlayerState,
    TextLine,
)
from .animation import AnimationDetector
from .chatbox import ChatboxReader
from .detect import ObjectTracker, ViewportDetector, to_detected_objects
from .inventory import InventoryReader
from .layout import DEFAULT_LAYOUT, LayoutProfile
from .ocr import GlyphOCR, ShadowFilter, classify_color

if TYPE_CHECKING:
    from .fonts import GlyphAtlas

logger = logging.getLogger(__name__)

_INT_RE = re.compile(r"\d{1,4}")


@dataclass(slots=True)
class _Confidence:
    """Sub-detector agreement, accumulated as (weight, score) pairs."""

    terms: list[tuple[float, float, str]]

    def add(self, name: str, score: float, weight: float = 1.0) -> None:
        self.terms.append((weight, float(np.clip(score, 0.0, 1.0)), name))

    def value(self) -> float:
        if not self.terms:
            return 0.0
        tw = sum(w for w, _, _ in self.terms)
        return float(sum(w * s for w, s, _ in self.terms) / tw) if tw else 0.0

    def breakdown(self) -> dict[str, float]:
        return {n: s for _, s, n in self.terms}


class PixelPerceiver:
    """The `contracts.Perceiver` implementation used by the fast loop."""

    def __init__(
        self,
        *,
        layout: LayoutProfile = DEFAULT_LAYOUT,
        atlas: GlyphAtlas | None = None,
        ocr: GlyphOCR | None = None,
        detector: ViewportDetector | None = None,
        tracker: ObjectTracker | None = None,
        inventory: InventoryReader | None = None,
        chatbox: ChatboxReader | None = None,
        animation: AnimationDetector | None = None,
        read_orbs: bool = True,
    ) -> None:
        self.layout = layout
        self.ocr = ocr or GlyphOCR(atlas)
        self.detector = detector or ViewportDetector(layout=layout)
        self.tracker = tracker or ObjectTracker()
        self.inventory = inventory or InventoryReader(layout)
        self.chatbox = chatbox or ChatboxReader(self.ocr, layout=layout)
        self.animation = animation or AnimationDetector(layout)
        self.read_orbs = read_orbs
        self._orb_filter = ShadowFilter()
        self.stage_ms: dict[str, float] = {}
        self.confidence_breakdown: dict[str, float] = {}
        self._layout_problems: list[str] | None = None

    # -- Perceiver protocol -------------------------------------------------

    def perceive(self, frame: Frame, previous: GameState | None = None) -> GameState:
        t_all = time.perf_counter()
        img = frame.image
        conf = _Confidence([])
        stage: dict[str, float] = {}

        problems = self._check_layout(img.shape)
        layout_ok = not problems

        with _Timed(stage, "chat"):
            chat = self.chatbox.read(img)
        with _Timed(stage, "tooltip"):
            tooltip = self.ocr.read_line(
                img, self.layout.tooltip, self._orb_filter, source="tooltip"
            )
        with _Timed(stage, "objects"):
            dets = self.detector.detect(img, self.layout.viewport)
            oids = self.tracker.update(dets)
            objects = to_detected_objects(dets, oids)
        with _Timed(stage, "inventory"):
            inv = self.inventory.read(img)
        with _Timed(stage, "animation"):
            anim = self.animation.update(img)
        with _Timed(stage, "orbs"):
            hp, prayer, run = self._read_orbs(img) if self.read_orbs else (None, None, None)

        text: list[TextLine] = list(chat.lines)
        if tooltip.text.strip():
            text.append(tooltip)

        dialogue: DialogueState | None = chat.dialogue if chat.dialogue.is_open else None

        # --- confidence -----------------------------------------------------
        if text:
            conf.add("ocr", float(np.mean([ln.confidence for ln in text])), weight=1.5)
            known = [1.0 if classify_color(ln.color) else 0.0 for ln in text]
            conf.add("ocr_palette", float(np.mean(known)), weight=0.5)
        if objects:
            conf.add("objects", float(np.mean([o.confidence for o in objects])), weight=1.0)
        conf.add("inventory", inv.confidence, weight=1.0)
        conf.add("animation", anim.confidence, weight=0.5)
        conf.add("layout", 1.0 if layout_ok else 0.0, weight=2.0)
        if previous is not None:
            conf.add("continuity", self._continuity(previous, objects, inv.state), weight=0.75)

        value = conf.value()
        if not layout_ok:
            # A geometry mismatch invalidates every region crop, so cap rather
            # than average it away; this is the signal that must reach the
            # escalation policy intact.
            value = min(value, 0.4)

        self.stage_ms = stage
        self.confidence_breakdown = conf.breakdown()
        total_ms = (time.perf_counter() - t_all) * 1000.0

        return GameState(
            seq=frame.seq,
            captured_at=frame.captured_at,
            player=PlayerState(
                hitpoints=hp,
                max_hitpoints=None,
                prayer=prayer,
                run_energy=run,
                is_animating=anim.is_animating,
                is_moving=anim.is_moving,
            ),
            inventory=inv.state,
            objects=objects,
            text=tuple(text),
            dialogue=dialogue,
            minimap=MinimapState(bbox=self.layout.minimap),
            confidence=float(np.clip(value, 0.0, 1.0)),
            perception_ms=total_ms,
        )

    # -- helpers ------------------------------------------------------------

    def _check_layout(self, shape: tuple[int, ...]) -> list[str]:
        problems = self.layout.validate(shape)
        if problems and problems != self._layout_problems:
            for p in problems:
                logger.warning("layout mismatch: %s", p)
        self._layout_problems = problems
        return problems

    def _read_orbs(self, img: np.ndarray) -> tuple[int | None, int | None, int | None]:
        out: list[int | None] = []
        for rect in (self.layout.hp_orb, self.layout.prayer_orb, self.layout.run_orb):
            line = self.ocr.read_line(img, rect, self._orb_filter, source="unknown")
            m = _INT_RE.search(line.text)
            out.append(int(m.group()) if m and line.confidence > 0.5 else None)
        return out[0], out[1], out[2]

    @staticmethod
    def _continuity(
        previous: GameState,
        objects: tuple[DetectedObject, ...],
        inventory: InventoryState,
    ) -> float:
        """Agreement with the previous frame — the cheapest novelty detector.

        Not a correctness check: the world does change. But an inventory that
        jumps by 10 slots or an object set that turns over completely in one
        frame is far more likely to be a perception failure than an event, and
        the escalation policy would rather look twice.
        """
        score = 1.0
        d_inv = abs(inventory.used - previous.inventory.used)
        if d_inv > 1:
            score -= min(0.5, 0.1 * d_inv)
        prev_oids = {o.oid for o in previous.objects}
        now_oids = {o.oid for o in objects}
        if prev_oids or now_oids:
            overlap = len(prev_oids & now_oids) / max(1, len(prev_oids | now_oids))
            score -= 0.4 * (1.0 - overlap)
        return max(0.0, score)

    def reset(self) -> None:
        """Forget cross-frame state. Call on a capture restart, not per frame."""
        self.tracker.reset()
        self.animation.reset()


class _Timed:
    """Tiny context manager; perf_counter twice is cheaper than a decorator."""

    __slots__ = ("key", "store", "t0")

    def __init__(self, store: dict[str, float], key: str) -> None:
        self.store = store
        self.key = key
        self.t0 = 0.0

    def __enter__(self) -> _Timed:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.store[self.key] = (time.perf_counter() - self.t0) * 1000.0


__all__ = ["PixelPerceiver"]
