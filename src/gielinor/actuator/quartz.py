"""macOS synthetic input via CoreGraphics event taps.

``CGEventPost`` at ``kCGHIDEventTap`` injects at the same level as the hardware,
so it works with any application and any window server mode — but it requires
**Accessibility** permission, which macOS grants per-binary and which is
effectively total control of the GUI. As with uinput, that means:

1. This class refuses to construct without ``allow_real_input=True``.
2. SPEC section 4 wants it in a separate process from the planner, so the
   process that reads untrusted chat never holds the tap.

One macOS-specific wrinkle worth knowing: Accessibility is granted to the
*hosting application* — Terminal, iTerm, your IDE — not to the script. Moving
the same code between terminals silently loses the permission, which is why
``probe()`` reports the trust state rather than letting the first event
disappear into nothing (CGEventPost fails silently when untrusted, which is a
miserable way to spend an afternoon).

Coordinates are in global display space with the origin at the top-left of the
main display, so client-space profile rects are offset by ``client_origin``.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Any

from gielinor.actuator.base import ActuatorUnavailable, require_explicit_consent
from gielinor.actuator.humanize import (
    click_delay_ms,
    mouse_path,
    sample_reaction_time_ms,
    typing_delays_ms,
)
from gielinor.actuator.resolve import Resolution, ResolutionError, resolve_action
from gielinor.actuator.validate import ValidationPolicy, validate_action
from gielinor.capture.base import current_platform
from gielinor.contracts import Action, ActionKind, ActionResult, GameState, Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from gielinor.config import Profile

logger = logging.getLogger(__name__)

CLIENT_ORIGIN = Point(0, 0)

INSTALL_HINT = (
    "pip install 'gielinor-agent[macos]', then add the application that launches python "
    "(Terminal.app, iTerm, your IDE) to System Settings > Privacy & Security > "
    "Accessibility, and restart it. The permission belongs to the host app, not to the "
    "script."
)

# Virtual key codes for the keys the agent actually uses. The full table is
# large and layout-dependent; chat text goes through CGEventKeyboardSetUnicodeString
# instead, which sidesteps layout entirely.
_VK = {"space": 49, "enter": 36, "return": 36, "escape": 53, "tab": 48}


class QuartzActuator:
    """An `Actuator` that posts real CoreGraphics events."""

    def __init__(
        self,
        *,
        allow_real_input: bool = False,
        profile: Profile | None = None,
        policy: ValidationPolicy | None = None,
        client_origin: Point = CLIENT_ORIGIN,
        seed: int | None = None,
        move_speed: float = 1.0,
        name: str = "quartz",
    ) -> None:
        require_explicit_consent(name, allow_real_input)
        self.name = name
        self.profile = profile
        self.policy = policy or ValidationPolicy()
        self.client_origin = client_origin
        self.move_speed = move_speed
        self._rng = random.Random(seed)
        self._quartz: Any = None
        self._started = False
        self.cursor = Point(0, 0)
        self.sleep: Callable[[float], None] = time.sleep

    # -- availability ------------------------------------------------------

    @staticmethod
    def probe() -> None:
        from gielinor.capture.base import missing_modules

        platform = current_platform()
        if platform != "darwin":
            raise ActuatorUnavailable(
                "quartz", f"platform is {platform!r}, not darwin", "use uinput on Linux"
            )
        missing = missing_modules(["Quartz"])
        if missing:
            raise ActuatorUnavailable("quartz", "pyobjc-framework-Quartz is missing", INSTALL_HINT)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.probe()
        try:
            import Quartz
        except ImportError as exc:  # pragma: no cover - probe() already checked
            raise ActuatorUnavailable("quartz", f"Quartz unusable: {exc}", INSTALL_HINT) from exc
        self._quartz = Quartz

        try:
            from ApplicationServices import AXIsProcessTrusted

            trusted = bool(AXIsProcessTrusted())
        except ImportError:
            trusted = True  # Cannot check; let the first event decide.
        if not trusted:
            raise ActuatorUnavailable(
                self.name,
                "this process is not trusted for Accessibility, so posted events would be "
                "silently discarded",
                INSTALL_HINT,
            )
        self._started = True
        logger.warning("quartz actuator armed: events will reach the focused application")

    def stop(self) -> None:
        self._started = False
        self._quartz = None

    # -- dispatch ----------------------------------------------------------

    def execute(self, action: Action, state: GameState) -> ActionResult:
        rejection = validate_action(action, state, policy=self.policy, profile=self.profile)
        if rejection is not None:
            return ActionResult(action=action, accepted=False, rejected_reason=str(rejection))
        if not self._started:
            self.start()
        try:
            resolution = resolve_action(action, state, profile=self.profile, rng=self._rng)
        except ResolutionError as exc:
            return ActionResult(action=action, accepted=False, rejected_reason=str(exc))

        self.sleep(sample_reaction_time_ms(self._rng) / 1000.0)
        self._perform(resolution)
        return ActionResult(
            action=action,
            accepted=True,
            dispatched_at=time.monotonic(),
            resolved_point=resolution.point,
        )

    def _perform(self, resolution: Resolution) -> None:
        q = self._quartz
        if resolution.point is not None and resolution.bbox is not None:
            steps, end = mouse_path(
                self.cursor,
                resolution.bbox,
                self._rng,
                speed=self.move_speed,
                click_point=resolution.point,
            )
            previous_t = 0.0
            for step in steps:
                self._post_mouse(q.kCGEventMouseMoved, step.point, 0)
                self.sleep(max(0.0, (step.t_ms - previous_t) / 1000.0))
                previous_t = step.t_ms
            self.cursor = end
            if resolution.kind in (
                ActionKind.CLICK_OBJECT,
                ActionKind.CLICK_UI,
                ActionKind.SELECT_DIALOGUE,
                ActionKind.WALK_MINIMAP,
                ActionKind.CONTINUE_DIALOGUE,
            ):
                self._post_mouse(q.kCGEventLeftMouseDown, end, q.kCGMouseButtonLeft)
                self.sleep(click_delay_ms(self._rng) / 1000.0)
                self._post_mouse(q.kCGEventLeftMouseUp, end, q.kCGMouseButtonLeft)
        if resolution.text:
            self._type(resolution.text)
        for key in resolution.keys:
            if resolution.text is None or key != "enter":
                self._tap(key)

    def _post_mouse(self, event_type: int, point: Point, button: int) -> None:
        q = self._quartz
        global_point = q.CGPointMake(
            float(self.client_origin.x + point.x), float(self.client_origin.y + point.y)
        )
        event = q.CGEventCreateMouseEvent(None, event_type, global_point, button)
        q.CGEventPost(q.kCGHIDEventTap, event)

    def _type(self, text: str) -> None:
        q = self._quartz
        for ch, delay in zip(text, typing_delays_ms(text, self._rng), strict=True):
            # Unicode string injection avoids a keycode table and works on any
            # keyboard layout, which a virtual-keycode map would not.
            for is_down in (True, False):
                event = q.CGEventCreateKeyboardEvent(None, 0, is_down)
                q.CGEventKeyboardSetUnicodeString(event, len(ch), ch)
                q.CGEventPost(q.kCGHIDEventTap, event)
            self.sleep(delay / 1000.0)
        self._tap("enter")

    def _tap(self, key: str) -> None:
        code = _VK.get(key)
        if code is None:
            logger.debug("no virtual key code for %r; skipping", key)
            return
        q = self._quartz
        for is_down in (True, False):
            event = q.CGEventCreateKeyboardEvent(None, code, is_down)
            q.CGEventPost(q.kCGHIDEventTap, event)


__all__ = ["INSTALL_HINT", "QuartzActuator"]
