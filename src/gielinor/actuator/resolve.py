"""Symbolic action -> concrete screen target.

This is the *only* place a coordinate is produced, and it deliberately sits
downstream of validation. The planner names a thing (``oid=41``,
``target_ui="inventory_tab"``); perception decided where that thing is; this
module turns the name into a rect and a point. A planner that wanted to click
(412, 233) would have to first convince perception that something is there.

Resolution is shared by all three actuators so that dry-run and real input agree
on where a click would land — otherwise the dry-run trace is a fiction and the
whole "default is safe" story is worth nothing.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gielinor.actuator.humanize import jitter_point_in
from gielinor.contracts import Action, ActionKind, GameState, Point, Rect

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gielinor.config import Profile

logger = logging.getLogger(__name__)


class ResolutionError(RuntimeError):
    """The action passed validation but its target cannot be located on screen.

    Distinct from a rejection: rejection means "not allowed", this means
    "allowed, but perception or the profile does not know where it is".
    """


@dataclass(frozen=True, slots=True)
class Resolution:
    """What an actuator needs to actually perform the action."""

    kind: ActionKind
    point: Point | None = None
    bbox: Rect | None = None
    button: str = "left"
    keys: tuple[str, ...] = ()
    text: str | None = None
    ticks: int = 0
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def is_input(self) -> bool:
        """False for WAIT/NOOP, which reach the OS as nothing at all."""
        return self.point is not None or bool(self.keys) or self.text is not None


def resolve_action(
    action: Action,
    state: GameState,
    *,
    profile: Profile | None = None,
    rng: random.Random | None = None,
    jitter: bool = True,
) -> Resolution:
    """Locate ``action``'s target. Assumes validation already passed."""
    rng = rng or random.Random()
    kind = action.kind

    if kind in (ActionKind.CLICK_OBJECT, ActionKind.WALK_MINIMAP):
        oid = action.target_oid
        obj = state.object_by_id(oid) if oid is not None else None
        if obj is None:
            raise ResolutionError(f"oid {oid} vanished between validation and resolution")
        # Perception already chose a point inside the clickable mask; jitter
        # around *that*, not around the bbox centre, because the centre of a
        # tree's bounding box is often sky.
        point = jitter_point_in(obj.bbox, rng, bias=obj.click_point) if jitter else obj.click_point
        return Resolution(
            kind=kind,
            point=point,
            bbox=obj.bbox,
            notes={"label": obj.label, "oid": str(obj.oid), "space": obj.space},
        )

    if kind == ActionKind.CLICK_UI:
        name = action.target_ui or ""
        rect = profile.anchor(name) if profile is not None else None
        if rect is None:
            raise ResolutionError(
                f"UI anchor {name!r} has no geometry; the active profile does not define it"
            )
        point = jitter_point_in(rect, rng) if jitter else rect.center
        return Resolution(kind=kind, point=point, bbox=rect, notes={"anchor": name})

    if kind == ActionKind.SELECT_DIALOGUE:
        index = action.option_index or 0
        rect = _dialogue_option_rect(state, index, profile)
        if rect is None:
            raise ResolutionError(f"cannot locate dialogue option {index} on screen")
        point = jitter_point_in(rect, rng) if jitter else rect.center
        return Resolution(kind=kind, point=point, bbox=rect, notes={"option": str(index)})

    if kind == ActionKind.CONTINUE_DIALOGUE:
        rect = profile.anchor("dialogue_continue") if profile is not None else None
        if rect is not None:
            point = jitter_point_in(rect, rng) if jitter else rect.center
            return Resolution(kind=kind, point=point, bbox=rect, keys=("space",))
        # Space works for every continue-style dialogue in this client, so a
        # missing anchor is degraded service rather than a failure.
        return Resolution(kind=kind, keys=("space",))

    if kind == ActionKind.TYPE_CHAT:
        return Resolution(kind=kind, text=action.text or "", keys=("enter",))

    if kind in (ActionKind.WAIT, ActionKind.NOOP):
        return Resolution(kind=kind, ticks=action.ticks)

    raise ResolutionError(f"no resolver for {kind!r}")


def _dialogue_option_rect(state: GameState, index: int, profile: Profile | None) -> Rect | None:
    """Prefer perception's real bbox; fall back to the profile's line grid."""
    option_lines = [t for t in state.text if t.source == "dialogue_option"]
    if index < len(option_lines):
        return option_lines[index].bbox
    if profile is not None:
        return profile.dialogue_option_rect(index)
    return None


__all__ = ["Resolution", "ResolutionError", "resolve_action"]
