"""Bridge between the calibration profile and the perception layout.

Two subsystems need screen geometry and they wanted it in different shapes.
The actuator needs click targets and a capability allowlist, so its natural
form is `config.Profile`: nested, validated, with a `ui_anchors` table that
doubles as the list of things a planner is permitted to click. Perception
needs flat rects it can slice a numpy array with on every frame, so its
natural form is `perception.LayoutProfile`: a frozen dataclass of `Rect`s with
derived helpers and no pydantic on the hot path.

Both shapes are reasonable. What is not reasonable is two independently
editable copies of the same numbers, which is what existed before this module:
the TOML said the minimap was at y=4 and perception believed y=8. Nothing
failed, because perception detects an object and reports an absolute
`click_point` that the actuator uses verbatim — the two tables never had to
agree for the demo to run. They would have had to agree the first time someone
calibrated against a real client, and the failure would have looked like a
perception bug rather than a stale constant.

So: the TOML is the source of truth, `config.Profile` is its validated
in-memory form, and this module projects that into the shape perception wants.
`LayoutProfile` keeps its own defaults, but they are now a fallback for code
constructed without a profile (mostly tests and the synthetic renderer), not a
second opinion about a real client.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import TYPE_CHECKING

from gielinor.config import DEFAULT_PROFILE, Profile, load_profile
from gielinor.perception.layout import LayoutProfile

if TYPE_CHECKING:
    from gielinor.config import RectSpec
    from gielinor.contracts import Rect

logger = logging.getLogger(__name__)

__all__ = ["layout_for", "layout_from_profile"]


def layout_from_profile(profile: Profile) -> LayoutProfile:
    """Project a calibration profile into the flat form perception consumes.

    One mapping is worth flagging. The TOML models the inventory grid the way
    the client actually draws it — a 36x32 item cell on a 42x36 pitch, so there
    is dead space between slots. `LayoutProfile` has a single `inv_cell_w/h`
    that it uses as *both* size and step, i.e. it assumes contiguous cells.
    Pitch is the one that has to be right, since it determines which pixels
    belong to slot 17; a slot rect that overshoots into the gutter costs
    nothing for occupancy detection, whereas a wrong step misreads the whole
    grid. So pitch wins and the cell size is discarded.
    """
    grid = profile.inventory_grid
    extras = profile.perception

    if grid.cols * grid.rows != 28:
        logger.warning(
            "profile %s declares a %dx%d inventory grid (%d slots); RS inventories are 28",
            profile.name,
            grid.cols,
            grid.rows,
            grid.cols * grid.rows,
        )

    base = LayoutProfile()

    def optional(name: str, spec: RectSpec | None) -> Rect:
        """A `[perception]` rect, or the built-in default if the TOML omits it."""
        if spec is not None:
            return spec.to_rect()
        fallback: Rect = getattr(base, name)
        logger.debug(
            "profile %s omits [perception].%s; using built-in default %r",
            profile.name,
            name,
            fallback,
        )
        return fallback

    layout = replace(
        base,
        name=profile.name,
        screen_w=profile.client_w,
        screen_h=profile.client_h,
        viewport=profile.viewport.to_rect(),
        minimap=profile.minimap.to_rect(),
        chatbox=profile.chatbox.to_rect(),
        chat_input=profile.chat_input.to_rect(),
        # Perception treats the dialogue frame as occupying the chatbox, which
        # is how the RS2 client draws it — the message log is replaced in place
        # rather than overlaid.
        dialogue=profile.chatbox.to_rect(),
        inv_origin_x=grid.origin_x,
        inv_origin_y=grid.origin_y,
        inv_cell_w=grid.pitch_x,  # pitch, deliberately — see docstring
        inv_cell_h=grid.pitch_y,
        inv_cols=grid.cols,
        inv_rows=grid.rows,
        dialogue_speaker_h=extras.dialogue_speaker_h,
        player_box_w=extras.player_box_w,
        player_box_h=extras.player_box_h,
        player_box_dy=extras.player_box_dy,
        side_panel=optional("side_panel", extras.side_panel),
        tooltip=optional("tooltip", extras.tooltip),
        hp_orb=optional("hp_orb", extras.hp_orb),
        prayer_orb=optional("prayer_orb", extras.prayer_orb),
        run_orb=optional("run_orb", extras.run_orb),
    )

    problems = layout.validate()
    if problems:
        logger.warning(
            "profile %s produced a layout with %d problem(s): %s",
            profile.name,
            len(problems),
            "; ".join(problems),
        )
    return layout


@lru_cache(maxsize=4)
def layout_for(name: str = DEFAULT_PROFILE) -> LayoutProfile:
    """Cached `layout_from_profile(load_profile(name))` for the hot path.

    Cached because perception asks for this once per construction and the TOML
    parse is not free; `load_profile` itself stays uncached so that editing a
    profile mid-session is picked up by the actuator. If you are calibrating
    and perception looks stale, call `layout_for.cache_clear()`.
    """
    return layout_from_profile(load_profile(name))
