"""Frame sources. One protocol, three implementations, one factory.

``open_capture()`` is the only place in the codebase that decides which backend
runs, and it has one opinionated rule: **"auto" may fall back to replay, an
explicit name never does.** Silently handing back a replay when the caller asked
for pipewire would turn "the compositor refused consent" into "the agent is
clicking on things that are not there", which is a much worse bug to debug than
an exception at startup.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from gielinor.capture.base import (
    CaptureError,
    CaptureExhausted,
    CaptureNotStarted,
    CaptureUnavailable,
    current_platform,
    has_display,
    missing_modules,
    to_bgr,
)
from gielinor.capture.clock import ClockStats, FrameClock
from gielinor.capture.pipewire import PipeWirePortalCapture
from gielinor.capture.replay import ReplayCapture
from gielinor.capture.screencapturekit import ScreenCaptureKitCapture
from gielinor.contracts import CaptureBackend

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__)

BackendName = str
KNOWN_BACKENDS: tuple[str, ...] = ("auto", "replay", "pipewire", "screencapturekit")


def platform_backend() -> str:
    """The backend that *should* work on this platform, ignoring whether it does."""
    platform = current_platform()
    if platform == "darwin":
        return "screencapturekit"
    if platform.startswith("linux"):
        return "pipewire"
    return "replay"


def open_capture(
    name: BackendName = "auto",
    *,
    replay_source: str | Path | Sequence[np.ndarray] | None = None,
    fps: float = 20.0,
    **kwargs: Any,
) -> CaptureBackend:
    """Construct a capture backend. Does not start it.

    Args:
        name: ``auto``, ``replay``, ``pipewire`` or ``screencapturekit``.
        replay_source: directory of PNGs or a list of BGR arrays. Required for
            the replay backend, including when ``auto`` falls back to it.
        fps: target (real backends) or synthetic (replay) frame rate.
        **kwargs: forwarded to the backend constructor.
    """
    if name not in KNOWN_BACKENDS:
        raise ValueError(f"unknown capture backend {name!r}; known: {', '.join(KNOWN_BACKENDS)}")

    if name == "replay":
        return _open_replay(replay_source, fps=fps, **kwargs)
    if name == "pipewire":
        return PipeWirePortalCapture(fps=fps, **kwargs)
    if name == "screencapturekit":
        return ScreenCaptureKitCapture(fps=fps, **kwargs)

    # auto
    chosen = platform_backend()
    if chosen == "replay":
        logger.info("no native capture backend for %s; using replay", current_platform())
        return _open_replay(replay_source, fps=fps, **kwargs)

    backend_cls: type[PipeWirePortalCapture] | type[ScreenCaptureKitCapture] = (
        ScreenCaptureKitCapture if chosen == "screencapturekit" else PipeWirePortalCapture
    )
    try:
        backend_cls.probe()
    except CaptureUnavailable as exc:
        logger.warning("%s unavailable, falling back to replay: %s", chosen, exc.problem)
        return _open_replay(replay_source, fps=fps, fallback_from=chosen, **kwargs)
    return backend_cls(fps=fps, **kwargs)


def _open_replay(
    source: str | Path | Sequence[np.ndarray] | None,
    *,
    fps: float,
    fallback_from: str | None = None,
    **kwargs: Any,
) -> ReplayCapture:
    if source is None:
        prefix = f"{fallback_from} is unavailable and " if fallback_from else ""
        raise CaptureUnavailable(
            "replay",
            f"{prefix}no replay source was given",
            "pass replay_source=<dir of PNGs>, or set GIELINOR_CAPTURE__REPLAY_DIR=/path/to/frames",
        )
    # A real backend takes fps/source-type kwargs the replay knows nothing
    # about; drop them rather than exploding on the fallback path.
    accepted = {
        "loop",
        "realtime",
        "epoch",
        "patterns",
        "limit",
        "stride",
        "cache_size",
        "now",
        "name",
    }
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    dropped = set(kwargs) - accepted
    if dropped:
        logger.debug("replay ignoring backend-specific kwargs: %s", sorted(dropped))
    return ReplayCapture(source, fps=fps, **filtered)


__all__ = [
    "KNOWN_BACKENDS",
    "CaptureBackend",
    "CaptureError",
    "CaptureExhausted",
    "CaptureNotStarted",
    "CaptureUnavailable",
    "ClockStats",
    "FrameClock",
    "PipeWirePortalCapture",
    "ReplayCapture",
    "ScreenCaptureKitCapture",
    "has_display",
    "missing_modules",
    "open_capture",
    "platform_backend",
    "to_bgr",
]
