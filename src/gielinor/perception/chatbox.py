"""Chat log and dialogue parsing.

Two different things share one screen region, and telling them apart is the
first job here:

  * the scrolling **chat log** — left-aligned lines, light text with a drop
    shadow over whatever is behind it, and *attacker-controlled*: another
    player typed some of it. Every line produced from it is tagged
    `source="chatbox"`, which makes `TextLine.is_untrusted` true, which is what
    the planner's injection defence keys off (SPEC section 4b).
  * a **dialogue interface** — a parchment panel with centred dark text, an
    optional speaker name, and either a continue prompt or a list of options.

The mode is decided by the region's dominant colour, not by trying both and
picking the nicer answer: one OCR pass is half the cost, and "the chatbox
background is parchment" is a much more stable signal than "the text parsed".

CALIBRATION WARNING: the panel colours and the English phrase patterns below
are 2009-era defaults. A different interface skin moves the colours; a
non-English client breaks the patterns. Both live at module scope so they can
be swapped without touching the parsing logic.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..contracts import DialogueState, Rect, TextLine
from .layout import DEFAULT_LAYOUT, LayoutProfile
from .ocr import ColorFilter, GlyphOCR, ShadowFilter, within_tol

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Parchment background of the chat/dialogue panel, and the dark text on it.
DIALOGUE_BG_RGB: tuple[int, int, int] = (194, 176, 143)
DIALOGUE_TEXT_RGB: tuple[int, int, int] = (0, 0, 0)

CONTINUE_RE = re.compile(r"click\s*here\s*to\s*continue", re.I)
OPTIONS_HEADER_RE = re.compile(r"(select\s*an?\s*option|choose\s*an?\s*option)", re.I)
LEVEL_UP_RE = re.compile(r"(congratulations.*advanc|you.*just\s*advanced|level\s*is\s*now)", re.I)
SPEAKER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9 '\-]{1,18}$")


@dataclass(frozen=True, slots=True)
class ChatReading:
    dialogue: DialogueState
    lines: tuple[TextLine, ...]
    mode: str  # "log" | "dialogue" | "empty"
    confidence: float


class ChatboxReader:
    """Reads the chatbox region into a `DialogueState` plus `TextLine`s."""

    def __init__(
        self,
        ocr: GlyphOCR | None = None,
        *,
        layout: LayoutProfile = DEFAULT_LAYOUT,
        panel_rgb: tuple[int, int, int] = DIALOGUE_BG_RGB,
        panel_tol: int = 40,
        panel_fraction: float = 0.45,
        player_name: str | None = None,
        center_tol: int = 60,
    ) -> None:
        self.ocr = ocr or GlyphOCR()
        self.layout = layout
        self.panel_rgb = panel_rgb
        self.panel_tol = panel_tol
        self.panel_fraction = panel_fraction
        self.player_name = player_name
        self.center_tol = center_tol
        self._log_filter = ShadowFilter()
        self._panel_filter = ColorFilter(
            rgb=panel_rgb, tol=panel_tol, invert=True, name="panel_text"
        )

    # -- public API ---------------------------------------------------------

    def read(self, img: np.ndarray) -> ChatReading:
        rect = self.layout.chatbox
        crop = _crop(img, rect)
        if crop.size == 0:
            return ChatReading(DialogueState(False), (), "empty", 0.0)

        panel_frac = self._panel_fraction(crop)
        if panel_frac >= self.panel_fraction:
            lines = self.ocr.read_region(img, rect, self._panel_filter, source="dialogue")
            if lines:
                dialogue = self._parse_dialogue(lines, rect)
                lines = self._retag(lines, dialogue)
                conf = float(np.clip(panel_frac, 0.0, 1.0) * _mean_conf(lines))
                return ChatReading(dialogue, tuple(lines), "dialogue", conf)
            logger.debug("panel background detected but no text parsed; falling back to log mode")

        lines = self.ocr.read_region(img, rect, self._log_filter, source="chatbox")
        if not lines:
            return ChatReading(DialogueState(False), (), "empty", 1.0 - panel_frac)
        return ChatReading(DialogueState(False), tuple(lines), "log", _mean_conf(lines))

    # -- internals ----------------------------------------------------------

    def _panel_fraction(self, crop: np.ndarray) -> float:
        return float(within_tol(crop, self.panel_rgb, self.panel_tol).mean())

    def _parse_dialogue(self, lines: Sequence[TextLine], rect: Rect) -> DialogueState:
        texts = [ln.text.strip() for ln in lines if ln.text.strip()]
        if not texts:
            return DialogueState(False)

        joined = " ".join(texts)
        centred = self._centred_fraction(lines, rect)

        if OPTIONS_HEADER_RE.search(joined):
            header_i = next(i for i, t in enumerate(texts) if OPTIONS_HEADER_RE.search(t))
            options = tuple(t for t in texts[header_i + 1 :] if t)
            return DialogueState(True, "options", None, (texts[header_i],), options)

        if LEVEL_UP_RE.search(joined):
            body = tuple(t for t in texts if not CONTINUE_RE.search(t))
            return DialogueState(True, "level_up", None, body, ())

        has_continue = any(CONTINUE_RE.search(t) for t in texts)
        body_texts = [t for t in texts if not CONTINUE_RE.search(t)]

        speaker: str | None = None
        if len(body_texts) >= 2 and SPEAKER_RE.match(body_texts[0]) and centred >= 0.5:
            speaker = body_texts[0]
            body_texts = body_texts[1:]

        if speaker is not None:
            # Pixels cannot tell an NPC chathead from the player's own without a
            # sprite library, so the only honest discriminator is the name.
            is_self = bool(self.player_name) and speaker.lower() == (self.player_name or "").lower()
            kind: Literal["npc", "player"] = "player" if is_self else "npc"
            return DialogueState(True, kind, speaker, tuple(body_texts), ())

        if has_continue:
            return DialogueState(True, "continue", None, tuple(body_texts), ())
        return DialogueState(True, "npc", None, tuple(body_texts), ())

    def _centred_fraction(self, lines: Sequence[TextLine], rect: Rect) -> float:
        if not lines:
            return 0.0
        cx = rect.x + rect.w / 2
        hits = sum(1 for ln in lines if abs((ln.bbox.x + ln.bbox.w / 2) - cx) <= self.center_tol)
        return hits / len(lines)

    @staticmethod
    def _retag(lines: Sequence[TextLine], dialogue: DialogueState) -> list[TextLine]:
        """Re-source option lines so downstream trust rules see the difference.

        Both are trusted (server-authored), but `dialogue_option` carries an
        index the planner may select, and conflating the two would let a body
        line be selected as if it were an option.
        """
        if dialogue.kind != "options":
            return list(lines)
        opts = set(dialogue.options)
        out: list[TextLine] = []
        for ln in lines:
            src = "dialogue_option" if ln.text.strip() in opts else "dialogue"
            out.append(
                TextLine(ln.text, ln.bbox, src, ln.color, ln.confidence, ln.speaker)  # type: ignore[arg-type]
            )
        return out


def _mean_conf(lines: Sequence[TextLine]) -> float:
    return float(np.mean([ln.confidence for ln in lines])) if lines else 0.0


def _crop(img: np.ndarray, rect: Rect) -> np.ndarray:
    r = rect.clip_to(Rect(0, 0, int(img.shape[1]), int(img.shape[0])))
    if r.w <= 0 or r.h <= 0:
        return img[0:0, 0:0]
    return img[r.y : r.y + r.h, r.x : r.x + r.w]


__all__ = ["DIALOGUE_BG_RGB", "DIALOGUE_TEXT_RGB", "ChatReading", "ChatboxReader"]
