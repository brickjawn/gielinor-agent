"""Errors and shared helpers for capture backends.

The rule every backend in this package follows, and the reason this module
exists at all: **a backend must import on any platform and fail only at
``start()``**. Import-time platform probing means a Linux CI box cannot even
collect the macOS tests, and a missing system library turns into an
``ImportError`` three frames up from anything that could explain it. So the
platform-specific imports happen inside ``start()``, and what comes out is a
``CaptureUnavailable`` whose message says what is missing and what to install.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """Base class for every capture failure."""


class CaptureUnavailable(CaptureError):
    """This backend cannot run here. The message must say why and what to do.

    Carries the structured pieces as well as the formatted message so a trace
    record can hold them without re-parsing English.
    """

    def __init__(self, backend: str, problem: str, remedy: str = "") -> None:
        self.backend = backend
        self.problem = problem
        self.remedy = remedy
        message = f"[{backend}] {problem}"
        if remedy:
            message = f"{message}\n  try: {remedy}"
        super().__init__(message)


class CaptureNotStarted(CaptureError):
    """grab() was called before start()."""


class CaptureExhausted(CaptureError):
    """A finite source (replay, non-looping) ran out of frames."""


def missing_modules(names: Sequence[str]) -> list[str]:
    """Return which of ``names`` cannot be imported, without importing them for real.

    ``importlib.util.find_spec`` is enough to answer "is it installed?" and
    avoids executing a module (PyGObject in particular has import side effects
    that can abort a process when GLib is half-installed).
    """
    import importlib.util

    missing: list[str] = []
    for name in names:
        try:
            if importlib.util.find_spec(name) is None:
                missing.append(name)
        except (ImportError, ValueError):  # namespace weirdness counts as missing
            missing.append(name)
    return missing


def current_platform() -> str:
    """``sys.platform``, behind a function.

    A type checker running on Linux narrows ``sys.platform`` to ``"linux"``
    and then calls every macOS branch unreachable, which is exactly backwards
    for code whose job is to work on both. The indirection keeps the checks
    honest without scattering ignores.
    """
    return sys.platform


def has_display() -> bool:
    """True if *some* graphical session looks reachable from this process."""
    return bool(
        os.environ.get("WAYLAND_DISPLAY")
        or os.environ.get("DISPLAY")
        or (sys.platform == "darwin" and not os.environ.get("GIELINOR_ASSUME_HEADLESS"))
    )


def to_bgr(image: np.ndarray) -> np.ndarray:
    """Coerce a decoded image to the contract's BGR uint8 (H, W, 3).

    Backends hand back whatever the compositor had: BGRA from PipeWire, RGBA
    from ScreenCaptureKit, greyscale from a PNG someone saved oddly. The Frame
    contract says BGR uint8 and the hot path should not have to check.
    """
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255)
        image = image.astype(np.uint8)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"expected a 2D or 3D array, got shape {image.shape}")
    channels = image.shape[2]
    if channels == 3:
        return np.ascontiguousarray(image)
    if channels == 4:
        return np.ascontiguousarray(image[:, :, :3])
    if channels == 1:
        return np.repeat(image, 3, axis=2)
    raise ValueError(f"cannot interpret {channels}-channel image as BGR")


__all__ = [
    "CaptureError",
    "CaptureExhausted",
    "CaptureNotStarted",
    "CaptureUnavailable",
    "current_platform",
    "has_display",
    "missing_modules",
    "to_bgr",
]
