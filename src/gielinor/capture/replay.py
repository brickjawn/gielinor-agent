"""Deterministic frame source backed by PNGs on disk or arrays in memory.

This is the backend CI, every test and every offline demo actually runs, so it
is the one worth getting right. Three properties it guarantees that a real
capture cannot:

**Deterministic.** Same source and same constructor arguments produce the same
sequence of ``(seq, captured_at, image)`` triples, byte for byte. Timestamps are
*synthetic* by default — ``captured_at`` advances by exactly ``1/fps`` per frame
from a fixed epoch — so latency assertions in tests are exact rather than flaky.
Pass ``realtime=True`` to pace against a real clock instead, which is what a
human watching a demo wants.

**Seekable.** ``seek()``/``rewind()``/``frame_at()`` make a captured bug
reproducible: park the replay on the frame where perception went wrong and
iterate on the matcher without touching the game.

**Loopable.** A 40-frame woodcutting capture loops forever so the agent loop can
run for as long as a test needs. Note that ``seq`` keeps counting across loop
boundaries even though the frame *index* wraps: staleness logic in the runtime
is defined in terms of seq deltas, and a seq that went backwards every 40 frames
would make stale actions look fresh.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from gielinor.capture.base import (
    CaptureNotStarted,
    CaptureUnavailable,
    to_bgr,
)
from gielinor.contracts import Frame

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PATTERNS: tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.bmp")

_DIGITS = re.compile(r"(\d+)")


def natural_key(path: Path) -> tuple[object, ...]:
    """Sort ``frame_2.png`` before ``frame_10.png``.

    Lexicographic ordering of a capture directory silently reorders the run,
    which turns "the agent chops a tree then banks" into "the agent banks then
    chops a tree" and wastes an afternoon.
    """
    parts = _DIGITS.split(path.name)
    return tuple(int(p) if p.isdigit() else p for p in parts)


class ReplayCapture:
    """A `CaptureBackend` that serves recorded frames at a synthetic frame rate."""

    def __init__(
        self,
        source: str | Path | Sequence[np.ndarray],
        *,
        fps: float = 20.0,
        loop: bool = True,
        realtime: bool = False,
        epoch: float = 0.0,
        patterns: Sequence[str] = DEFAULT_PATTERNS,
        limit: int | None = None,
        stride: int = 1,
        cache_size: int = 64,
        now: Callable[[], float] = time.monotonic,
        name: str = "replay",
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.name = name
        self.fps = fps
        self.loop = loop
        self.realtime = realtime
        self.epoch = epoch
        self.patterns = tuple(patterns)
        self.limit = limit
        self.stride = stride
        self._now = now

        self._paths: list[Path] = []
        self._arrays: list[np.ndarray] = []
        self._source_dir: Path | None = None
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_size = max(1, cache_size)

        if isinstance(source, (str, Path)):
            self._source_dir = Path(source)
        else:
            # Copy: the replay marks served frames read-only, and doing that to
            # an array the caller still owns is a nasty surprise.
            self._arrays = [to_bgr(np.asarray(a)).copy() for a in source]

        self._started = False
        self._index = 0
        self._served = 0
        self._loops = 0
        self._exhausted = False
        self._next_due: float | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        if self._source_dir is not None:
            self._paths = self._discover(self._source_dir)
        elif not self._arrays:
            raise CaptureUnavailable(
                self.name,
                "constructed with an empty in-memory frame list",
                "pass at least one HxWx3 uint8 array, or a directory of PNGs",
            )
        self._started = True
        self._next_due = self._now() if self.realtime else None
        logger.info(
            "replay started: %d frames, %.1f fps, loop=%s, realtime=%s",
            len(self),
            self.fps,
            self.loop,
            self.realtime,
        )

    def stop(self) -> None:
        self._started = False
        self._cache.clear()

    def __enter__(self) -> ReplayCapture:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def _discover(self, directory: Path) -> list[Path]:
        if not directory.exists():
            raise CaptureUnavailable(
                self.name,
                f"replay directory {directory} does not exist",
                "record one with `python scripts/record_replay.py --out captures/run1`, "
                "or point GIELINOR_CAPTURE__REPLAY_DIR at an existing directory",
            )
        if not directory.is_dir():
            raise CaptureUnavailable(
                self.name, f"replay source {directory} is not a directory", "pass a directory"
            )
        found: list[Path] = []
        for pattern in self.patterns:
            found.extend(directory.glob(pattern))
        paths = sorted(set(found), key=natural_key)[:: self.stride]
        if self.limit is not None:
            paths = paths[: self.limit]
        if not paths:
            raise CaptureUnavailable(
                self.name,
                f"no frames matching {list(self.patterns)} in {directory}",
                "check the pattern, or the directory really is empty",
            )
        return paths

    # -- introspection -----------------------------------------------------

    def __len__(self) -> int:
        return len(self._paths) if self._source_dir is not None else len(self._arrays)

    @property
    def started(self) -> bool:
        return self._started

    @property
    def position(self) -> int:
        """Index of the frame ``grab()`` will return next."""
        return self._index

    @property
    def frames_served(self) -> int:
        return self._served

    @property
    def loops_completed(self) -> int:
        return self._loops

    @property
    def exhausted(self) -> bool:
        return self._exhausted

    @property
    def source_paths(self) -> tuple[Path, ...]:
        return tuple(self._paths)

    # -- seeking -----------------------------------------------------------

    def seek(self, index: int) -> None:
        """Park the replay on ``index``. Negative indexes count from the end."""
        n = len(self)
        if n == 0:
            raise CaptureNotStarted("seek() before start()")
        if index < 0:
            index += n
        if not 0 <= index < n:
            raise IndexError(f"frame index {index} outside 0..{n - 1}")
        self._index = index
        self._exhausted = False

    def rewind(self) -> None:
        self._index = 0
        self._exhausted = False

    def frame_at(self, index: int, *, seq: int | None = None) -> Frame:
        """Build the Frame for ``index`` without moving the playhead.

        Used by debugging tools and by tests that want frame 37 specifically.
        """
        image = self._load(index)
        served = self._served if seq is None else seq
        return Frame(
            image=image,
            captured_at=self._synthetic_time(index if seq is None else served),
            seq=served,
            backend=self.name,
        )

    # -- the hot path ------------------------------------------------------

    def grab(self) -> Frame | None:
        if not self._started:
            raise CaptureNotStarted("grab() before start(); call start() first")
        if self._exhausted:
            return None

        n = len(self)
        if self._index >= n:
            if not self.loop:
                self._exhausted = True
                logger.debug("replay exhausted after %d frames", self._served)
                return None
            self._index = 0
            self._loops += 1

        if self.realtime:
            t = self._now()
            if self._next_due is None:
                self._next_due = t
            if t < self._next_due:
                return None  # Non-blocking, per the CaptureBackend contract.
            # Advance the schedule rather than resetting it, so a slow consumer
            # does not make the replay drift permanently late.
            self._next_due = max(self._next_due + 1.0 / self.fps, t - 1.0 / self.fps)
            captured_at = t
        else:
            captured_at = self._synthetic_time(self._served)

        image = self._load(self._index)
        frame = Frame(image=image, captured_at=captured_at, seq=self._served, backend=self.name)
        self._index += 1
        self._served += 1
        return frame

    def _synthetic_time(self, tick: int) -> float:
        return self.epoch + tick / self.fps

    def _load(self, index: int) -> np.ndarray:
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached

        if self._source_dir is None:
            image = self._arrays[index]
        else:
            path = self._paths[index]
            raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if raw is None:
                raise CaptureUnavailable(
                    self.name,
                    f"could not decode {path}",
                    "is it a valid image? opencv returned None",
                )
            image = to_bgr(raw)
            if self._cache and image.shape != next(iter(self._cache.values())).shape:
                logger.warning(
                    "frame %s is %s, earlier frames were %s; perception assumes a fixed "
                    "client size and will misbehave",
                    path.name,
                    image.shape,
                    next(iter(self._cache.values())).shape,
                )

        # Read-only: several frames may reference the same cached array, and a
        # perception bug that writes into the input must not corrupt the replay.
        image.flags.writeable = False
        self._cache[index] = image
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return image


__all__ = ["DEFAULT_PATTERNS", "ReplayCapture", "natural_key"]
