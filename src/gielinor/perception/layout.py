"""Where things are on screen, as data rather than as literals in code.

SPEC section 1: "Every hardcoded coordinate lives in one calibration profile,
never inline in code, so retargeting is a data change." This module is
perception's half of that rule — a single immutable `LayoutProfile` that every
detector takes as a constructor argument, plus loaders so the values can come
from a profile TOML that workstream B owns without perception importing
`config.py` (which would invert the dependency and couple the two workstreams).

The defaults below describe RS2 fixed mode at 765x503. They are *derived from
the documented layout*, not measured against a live client, so treat every
number here as a starting point for calibration; `LayoutProfile.validate()`
exists to catch the case where they no longer fit the frame you handed it.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..contracts import Rect, TextSource

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegionSpec:
    """A named text region and the `TextSource` any text found in it carries.

    Binding source to region here (rather than at each call site) is what makes
    it structurally hard to read chatbox text and forget that it is untrusted.
    """

    name: str
    rect: Rect
    source: TextSource
    read_as_lines: bool = True


@dataclass(frozen=True, slots=True)
class LayoutProfile:
    """Immutable screen geometry for one client build/resolution."""

    # `field(default_factory=...)` rather than a bare `Rect(...)` default: Rect
    # is frozen so a shared instance would be safe, but the linter cannot know
    # that and the repo treats call-in-default as a bug class rather than a
    # case-by-case judgement. The values below are still meant to read as data.
    name: str = "rs2_fixed_765x503"
    screen_w: int = 765
    screen_h: int = 503

    viewport: Rect = field(default_factory=lambda: Rect(4, 4, 512, 334))
    minimap: Rect = field(default_factory=lambda: Rect(550, 8, 154, 154))
    side_panel: Rect = field(default_factory=lambda: Rect(516, 168, 249, 335))
    chatbox: Rect = field(default_factory=lambda: Rect(7, 345, 500, 120))
    chat_input: Rect = field(default_factory=lambda: Rect(7, 466, 500, 14))
    tooltip: Rect = field(default_factory=lambda: Rect(6, 6, 260, 24))

    # Stat orbs down the left edge of the minimap panel. Small, fixed, and the
    # only numeric player state readable from pixels without an interface tab.
    hp_orb: Rect = field(default_factory=lambda: Rect(519, 50, 28, 16))
    prayer_orb: Rect = field(default_factory=lambda: Rect(519, 85, 28, 16))
    run_orb: Rect = field(default_factory=lambda: Rect(523, 120, 28, 16))

    # Inventory: 4x7 grid of cells inside the side panel.
    inv_origin_x: int = 559
    inv_origin_y: int = 215
    inv_cell_w: int = 42
    inv_cell_h: int = 36
    inv_cols: int = 4
    inv_rows: int = 7

    # Player is drawn at the viewport centre in fixed mode; animation churn is
    # measured over this box rather than the whole viewport so a passing NPC at
    # the edge does not read as "still chopping".
    player_box_w: int = 70
    player_box_h: int = 96
    player_box_dy: int = 6  # feet sit slightly below centre

    # Dialogue panel occupies the chatbox area when open.
    dialogue: Rect = field(default_factory=lambda: Rect(7, 345, 500, 120))
    dialogue_speaker_h: int = 16

    # -- derived geometry ---------------------------------------------------

    @property
    def screen(self) -> Rect:
        return Rect(0, 0, self.screen_w, self.screen_h)

    @property
    def player_box(self) -> Rect:
        c = self.viewport.center
        return Rect(
            c.x - self.player_box_w // 2,
            c.y - self.player_box_h // 2 + self.player_box_dy,
            self.player_box_w,
            self.player_box_h,
        )

    def inventory_slots(self) -> tuple[Rect, ...]:
        return tuple(
            Rect(
                self.inv_origin_x + col * self.inv_cell_w,
                self.inv_origin_y + row * self.inv_cell_h,
                self.inv_cell_w,
                self.inv_cell_h,
            )
            for row in range(self.inv_rows)
            for col in range(self.inv_cols)
        )

    @property
    def inventory_rect(self) -> Rect:
        return Rect(
            self.inv_origin_x,
            self.inv_origin_y,
            self.inv_cols * self.inv_cell_w,
            self.inv_rows * self.inv_cell_h,
        )

    def text_regions(self) -> tuple[RegionSpec, ...]:
        """Every region perception OCRs, with its trust label attached."""
        return (
            RegionSpec("chatbox", self.chatbox, "chatbox"),
            RegionSpec("tooltip", self.tooltip, "tooltip", read_as_lines=False),
        )

    # -- validation and loading ---------------------------------------------

    def validate(self, frame_shape: tuple[int, ...] | None = None) -> list[str]:
        """Return human-readable problems; empty list means the profile is sane."""
        problems: list[str] = []
        if frame_shape is not None:
            h, w = int(frame_shape[0]), int(frame_shape[1])
            if (w, h) != (self.screen_w, self.screen_h):
                problems.append(
                    f"frame is {w}x{h} but profile {self.name} describes "
                    f"{self.screen_w}x{self.screen_h}"
                )
        for label, r in (
            ("viewport", self.viewport),
            ("minimap", self.minimap),
            ("chatbox", self.chatbox),
            ("inventory", self.inventory_rect),
        ):
            if r.clip_to(self.screen) != r:
                problems.append(
                    f"{label} {r} falls outside the {self.screen_w}x{self.screen_h} screen"
                )
        if self.inv_cols * self.inv_rows != 28:
            problems.append(f"inventory grid is {self.inv_cols}x{self.inv_rows}, expected 28 slots")
        return problems

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LayoutProfile:
        """Build from a plain dict; `Rect` fields accept [x, y, w, h] lists."""
        kwargs: dict[str, Any] = {}
        fields = {f: getattr(cls, f, None) for f in cls.__dataclass_fields__}
        for key, value in data.items():
            if key not in fields:
                logger.warning("layout profile: ignoring unknown key %r", key)
                continue
            if isinstance(value, (list, tuple)) and len(value) == 4:
                kwargs[key] = Rect(*(int(v) for v in value))
            else:
                kwargs[key] = value
        return replace(cls(), **kwargs)

    @classmethod
    def from_toml(cls, path: str | Path, section: str = "layout") -> LayoutProfile:
        """Load from `assets/profiles/*.toml` (workstream B's calibration data).

        Missing file is a hard error here on purpose: silently falling back to
        the built-in guesses would make a mis-typed profile path look like a
        perception bug at 3 a.m.
        """
        with Path(path).open("rb") as fh:
            doc = tomllib.load(fh)
        data = doc.get(section, doc)
        return cls.from_mapping(data)


DEFAULT_LAYOUT = LayoutProfile()

__all__ = ["DEFAULT_LAYOUT", "LayoutProfile", "RegionSpec"]
