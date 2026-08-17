"""Actuators: symbolic actions in, OS input events out (or, by default, nothing).

``open_actuator()`` defaults to ``"dryrun"`` and that default is load-bearing.
Everything else in this package — validation, resolution, humanised paths — runs
identically whichever backend is selected, so the only difference between safe
and dangerous is one explicit argument at one call site.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gielinor.actuator.base import (
    ActuatorError,
    ActuatorUnavailable,
    RealInputRefused,
)
from gielinor.actuator.dryrun import DryRunActuator
from gielinor.actuator.humanize import PathStep, mouse_path, sample_reaction_time_ms
from gielinor.actuator.quartz import QuartzActuator
from gielinor.actuator.resolve import Resolution, ResolutionError, resolve_action
from gielinor.actuator.uinput import UinputActuator
from gielinor.actuator.validate import (
    RejectionCode,
    RejectionReason,
    ValidationPolicy,
    is_stale,
    validate_action,
)
from gielinor.contracts import Actuator

if TYPE_CHECKING:  # pragma: no cover - typing only
    from gielinor.config import Profile

logger = logging.getLogger(__name__)

KNOWN_ACTUATORS: tuple[str, ...] = ("dryrun", "uinput", "quartz")


def open_actuator(
    name: str = "dryrun",
    *,
    profile: Profile | None = None,
    policy: ValidationPolicy | None = None,
    allow_real_input: bool = False,
    **kwargs: Any,
) -> Actuator:
    """Construct an actuator. Defaults to the one that cannot do any damage.

    Real backends additionally require ``allow_real_input=True``; passing the
    name alone is not enough, and asking for a real backend without it raises
    ``RealInputRefused`` rather than quietly downgrading to dry run. A silent
    downgrade would make "why is nothing happening" the failure mode, and that
    is worse than an exception.
    """
    if name not in KNOWN_ACTUATORS:
        raise ValueError(f"unknown actuator {name!r}; known: {', '.join(KNOWN_ACTUATORS)}")

    if name == "dryrun":
        if allow_real_input:
            logger.info("allow_real_input has no effect on the dry-run actuator")
        return DryRunActuator(profile=profile, policy=policy or ValidationPolicy(), **kwargs)

    logger.warning("constructing REAL actuator %r", name)
    if name == "uinput":
        return UinputActuator(
            allow_real_input=allow_real_input, profile=profile, policy=policy, **kwargs
        )
    return QuartzActuator(
        allow_real_input=allow_real_input, profile=profile, policy=policy, **kwargs
    )


__all__ = [
    "KNOWN_ACTUATORS",
    "Actuator",
    "ActuatorError",
    "ActuatorUnavailable",
    "DryRunActuator",
    "PathStep",
    "QuartzActuator",
    "RealInputRefused",
    "RejectionCode",
    "RejectionReason",
    "Resolution",
    "ResolutionError",
    "UinputActuator",
    "ValidationPolicy",
    "is_stale",
    "mouse_path",
    "open_actuator",
    "resolve_action",
    "sample_reaction_time_ms",
    "validate_action",
]
