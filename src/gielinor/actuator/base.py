"""Errors and the shared guard for actuators that can really move the mouse.

SPEC section 4(a): synthetic input is a genuine privilege. ``/dev/uinput`` is
the ability to type into any focused window including a sudo prompt; macOS
Accessibility is functionally root for the GUI. So the two real actuators are
built around one rule, enforced here rather than restated in each of them:

**A real actuator refuses to *construct* without an explicit
``allow_real_input=True`` argument.** Not a config file, not an env var — an
argument, at the call site, that shows up in a diff. Nothing that gets imported,
autodiscovered, or defaulted into existence can move your cursor.

The env var ``GIELINOR_I_KNOW_WHAT_IM_DOING`` is deliberately *not* a way to
bypass this; there is no bypass.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ActuatorError(RuntimeError):
    """Base class for actuator failures."""


class ActuatorUnavailable(ActuatorError):
    """The backend cannot run here; the message says why and what to do."""

    def __init__(self, backend: str, problem: str, remedy: str = "") -> None:
        self.backend = backend
        self.problem = problem
        self.remedy = remedy
        message = f"[{backend}] {problem}"
        if remedy:
            message = f"{message}\n  try: {remedy}"
        super().__init__(message)


class RealInputRefused(ActuatorError):
    """Someone tried to build a real actuator without asking for real input."""


def require_explicit_consent(backend: str, allow_real_input: bool) -> None:
    """Guard called first in every real actuator's ``__init__``."""
    if not allow_real_input:
        raise RealInputRefused(
            f"{backend} moves the real mouse and keyboard and was constructed without "
            "allow_real_input=True. This is intentional: the default actuator is "
            "DryRunActuator, and enabling real input must be a visible, deliberate "
            "change at the call site. If you meant it: "
            f"open_actuator({backend!r}, allow_real_input=True)."
        )
    logger.warning(
        "%s constructed WITH REAL INPUT ENABLED - synthetic events will reach the "
        "focused window, whatever it is",
        backend,
    )


__all__ = [
    "ActuatorError",
    "ActuatorUnavailable",
    "RealInputRefused",
    "require_explicit_consent",
]
