"""Linux synthetic input via ``/dev/uinput``.

Why uinput and not XTest or ``ydotool``: under Wayland there is no XTest, and a
compositor will not let a client synthesise input for another client. uinput
works at the kernel level — it creates a *virtual input device* that the
compositor treats exactly like a real mouse — so it is the one approach that
works identically on X11, Wayland, and a bare console.

That power is the problem. A process holding ``/dev/uinput`` can type into
whatever has focus, including a terminal at a sudo prompt. Two mitigations, and
both matter:

1. Construction refuses without an explicit ``allow_real_input=True`` (see
   ``actuator.base``), so this cannot be reached by accident.
2. SPEC section 4 wants this class hosted in a **separate process** behind a
   Unix socket speaking only validated ``Action`` objects, so the planner
   process — the one that reads chat typed by strangers — never holds the
   device handle. This module is written to be that process's payload: it takes
   already-validated actions and does nothing else. ``scripts/actuator_server.py``
   is where the socket lives.

Grant access with a udev rule targeting the actuator binary, not by ``chmod 666
/dev/uinput`` and not by running the agent as root.

Uses ``evdev`` (extra ``[linux]``): a thin, well-maintained wrapper over the
uinput ioctls; hand-rolling them with ctypes is more code and more ways to leave
a wedged virtual keyboard behind.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gielinor.actuator.base import (
    ActuatorUnavailable,
    require_explicit_consent,
)
from gielinor.actuator.humanize import (
    click_delay_ms,
    mouse_path,
    sample_reaction_time_ms,
    typing_delays_ms,
)
from gielinor.actuator.resolve import Resolution, ResolutionError, resolve_action
from gielinor.actuator.validate import ValidationPolicy, validate_action
from gielinor.contracts import Action, ActionKind, ActionResult, GameState, Point

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from gielinor.config import Profile

logger = logging.getLogger(__name__)

# Frozen and shared: profile rects are client-space, and the default assumes
# the client window sits at the top-left of the display.
CLIENT_ORIGIN = Point(0, 0)
UINPUT_DEVICE = Path("/dev/uinput")
INSTALL_HINT = (
    "pip install 'gielinor-agent[linux]' for python-evdev, then grant the actuator "
    "access to /dev/uinput with a udev rule, e.g. "
    'SUBSYSTEM=="misc", KERNEL=="uinput", GROUP="input", MODE="0660" '
    "plus `usermod -aG input $USER` and a re-login. Do not chmod 666 it."
)

# Only the characters the chat validator already allows need a mapping; anything
# else has been rejected long before it reaches a key code.
_SHIFTED = {
    "!": "1",
    "@": "2",
    "#": "3",
    "$": "4",
    "%": "5",
    "^": "6",
    "&": "7",
    "*": "8",
    "(": "9",
    ")": "0",
    "_": "minus",
    "+": "equal",
    ":": "semicolon",
    '"': "apostrophe",
    "<": "comma",
    ">": "dot",
    "?": "slash",
}
_PUNCT = {
    " ": "space",
    ".": "dot",
    ",": "comma",
    "-": "minus",
    "=": "equal",
    "'": "apostrophe",
    ";": "semicolon",
    "/": "slash",
    "[": "leftbrace",
    "]": "rightbrace",
}


class UinputActuator:
    """An `Actuator` that emits real events through a virtual input device."""

    def __init__(
        self,
        *,
        allow_real_input: bool = False,
        profile: Profile | None = None,
        policy: ValidationPolicy | None = None,
        client_origin: Point = CLIENT_ORIGIN,
        screen_size: tuple[int, int] = (1920, 1080),
        device_name: str = "gielinor-agent virtual pointer",
        seed: int | None = None,
        move_speed: float = 1.0,
        name: str = "uinput",
    ) -> None:
        require_explicit_consent(name, allow_real_input)
        self.name = name
        self.profile = profile
        self.policy = policy or ValidationPolicy()
        # Profile rects are client-space; the kernel wants desktop pixels.
        self.client_origin = client_origin
        self.screen_size = screen_size
        self.device_name = device_name
        self.move_speed = move_speed
        self._rng = random.Random(seed)
        self._device: Any = None
        self._ecodes: Any = None
        self._started = False
        self.cursor = Point(0, 0)
        self.sleep: Callable[[float], None] = time.sleep

    # -- availability ------------------------------------------------------

    @staticmethod
    def probe() -> None:
        from gielinor.capture.base import current_platform, missing_modules

        platform = current_platform()
        if platform != "linux":
            raise ActuatorUnavailable(
                "uinput", f"platform is {platform!r}, not linux", "use quartz on macOS"
            )
        missing = missing_modules(["evdev"])
        if missing:
            raise ActuatorUnavailable("uinput", "python-evdev is not installed", INSTALL_HINT)
        if not UINPUT_DEVICE.exists():
            raise ActuatorUnavailable(
                "uinput",
                f"{UINPUT_DEVICE} does not exist",
                "load the module with `modprobe uinput` (and add it to /etc/modules-load.d)",
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.probe()
        try:
            from evdev import (
                AbsInfo,
                UInput,
                ecodes,
            )
        except ImportError as exc:  # pragma: no cover - probe() already checked
            raise ActuatorUnavailable("uinput", f"evdev unusable: {exc}", INSTALL_HINT) from exc

        self._ecodes = ecodes
        w, h = self.screen_size
        # An absolute pointer, not a relative one: the agent knows where it
        # wants to be, and relative deltas would accumulate error and interact
        # badly with pointer acceleration.
        capabilities = {
            ecodes.EV_KEY: [
                ecodes.BTN_LEFT,
                ecodes.BTN_RIGHT,
                *range(ecodes.KEY_ESC, ecodes.KEY_MAX),
            ],
            ecodes.EV_ABS: [
                (ecodes.ABS_X, AbsInfo(value=0, min=0, max=w, fuzz=0, flat=0, resolution=0)),
                (ecodes.ABS_Y, AbsInfo(value=0, min=0, max=h, fuzz=0, flat=0, resolution=0)),
            ],
        }
        try:
            self._device = UInput(capabilities, name=self.device_name, version=0x1)
        except PermissionError as exc:
            raise ActuatorUnavailable(
                self.name, f"cannot open {UINPUT_DEVICE}: {exc}", INSTALL_HINT
            ) from exc
        except OSError as exc:
            raise ActuatorUnavailable(self.name, f"uinput refused: {exc}", INSTALL_HINT) from exc
        self._started = True
        logger.warning("uinput virtual device created: %r", self.device_name)

    def stop(self) -> None:
        self._started = False
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                logger.debug("error closing uinput device", exc_info=True)
            self._device = None

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
                self._move_to(step.point)
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
                self._click(resolution.button)
        if resolution.text:
            self._type(resolution.text)
        for key in resolution.keys:
            if resolution.text is None or key != "enter":
                self._tap(key)

    def _move_to(self, point: Point) -> None:
        e = self._ecodes
        self._device.write(e.EV_ABS, e.ABS_X, self.client_origin.x + point.x)
        self._device.write(e.EV_ABS, e.ABS_Y, self.client_origin.y + point.y)
        self._device.syn()

    def _click(self, button: str) -> None:
        e = self._ecodes
        code = e.BTN_RIGHT if button == "right" else e.BTN_LEFT
        self._device.write(e.EV_KEY, code, 1)
        self._device.syn()
        self.sleep(click_delay_ms(self._rng) / 1000.0)
        self._device.write(e.EV_KEY, code, 0)
        self._device.syn()

    def _type(self, text: str) -> None:
        for ch, delay in zip(text, typing_delays_ms(text, self._rng), strict=True):
            self._tap_char(ch)
            self.sleep(delay / 1000.0)
        self._tap("enter")

    def _tap_char(self, ch: str) -> None:
        e = self._ecodes
        shifted = ch.isupper() or ch in _SHIFTED
        base = _SHIFTED.get(ch, _PUNCT.get(ch, ch.lower()))
        code = getattr(e, f"KEY_{base.upper()}", None)
        if code is None:
            logger.debug("no key code for %r; skipping", ch)
            return
        if shifted:
            self._device.write(e.EV_KEY, e.KEY_LEFTSHIFT, 1)
        self._device.write(e.EV_KEY, code, 1)
        self._device.write(e.EV_KEY, code, 0)
        if shifted:
            self._device.write(e.EV_KEY, e.KEY_LEFTSHIFT, 0)
        self._device.syn()

    def _tap(self, key: str) -> None:
        e = self._ecodes
        code = getattr(e, f"KEY_{key.upper()}", None)
        if code is None:
            logger.debug("no key code for %r; skipping", key)
            return
        self._device.write(e.EV_KEY, code, 1)
        self._device.write(e.EV_KEY, code, 0)
        self._device.syn()


__all__ = ["INSTALL_HINT", "UINPUT_DEVICE", "UinputActuator"]
