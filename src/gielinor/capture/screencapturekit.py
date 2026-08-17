"""macOS screen capture via ScreenCaptureKit.

ScreenCaptureKit (macOS 12.3+) replaced the deprecated ``CGDisplayStream`` and
``CGWindowListCreateImage`` paths, and it is the only one that still delivers
frames at a useful rate on Apple silicon. Like the Wayland portal it is consent
gated: the process must hold **Screen Recording** permission in System Settings
> Privacy & Security, granted per-binary. Note that this means granting it to
whatever launches Python — a terminal, an IDE — not to this file, which is worth
saying out loud because it is a common source of "it works in Terminal but not
in my editor".

Structure mirrors the PipeWire backend deliberately: ``probe()`` answers "could
this ever work here" without side effects, ``start()`` does the permission-gated
work and raises ``CaptureUnavailable`` with a remedy, and the delegate keeps
only the newest frame because a backlog of stale frames is worse than a gap.

Imports of the Objective-C bridge live inside methods, so this module imports
fine on Linux and the tests collect everywhere.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from gielinor.capture.base import (
    CaptureNotStarted,
    CaptureUnavailable,
    current_platform,
    missing_modules,
    to_bgr,
)
from gielinor.contracts import Frame

logger = logging.getLogger(__name__)

REQUIRED_MODULES: tuple[str, ...] = ("objc", "Quartz", "ScreenCaptureKit")
INSTALL_HINT = (
    "pip install 'gielinor-agent[macos]' (pyobjc-framework-ScreenCaptureKit), then grant "
    "Screen Recording to the app that launches python in System Settings > Privacy & "
    "Security > Screen Recording, and restart it"
)
MIN_MACOS = (12, 3)


class ScreenCaptureKitCapture:
    """A `CaptureBackend` for macOS. Newest-frame-wins, permission gated."""

    def __init__(
        self,
        *,
        fps: float = 20.0,
        window_title_contains: str | None = None,
        display_index: int = 0,
        include_cursor: bool = False,
        name: str = "screencapturekit",
    ) -> None:
        self.name = name
        self.fps = fps
        # Capturing the 2009scape window rather than the whole display keeps the
        # profile's client-space coordinates valid regardless of where the user
        # dragged the window.
        self.window_title_contains = window_title_contains
        self.display_index = display_index
        self.include_cursor = include_cursor

        self._stream: Any = None
        self._output: Any = None
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._seq = 0
        self._started = False

    # -- availability ------------------------------------------------------

    @staticmethod
    def probe() -> None:
        backend = "screencapturekit"
        platform = current_platform()
        if platform != "darwin":
            raise CaptureUnavailable(
                backend,
                f"platform is {platform!r}, not darwin",
                "use the pipewire backend on Wayland, or replay anywhere",
            )
        missing = missing_modules(REQUIRED_MODULES)
        if missing:
            raise CaptureUnavailable(
                backend, f"missing python modules: {', '.join(missing)}", INSTALL_HINT
            )
        version = tuple(int(p) for p in _macos_version().split(".")[:2])
        if version < MIN_MACOS:
            raise CaptureUnavailable(
                backend,
                f"ScreenCaptureKit needs macOS {MIN_MACOS[0]}.{MIN_MACOS[1]}+, found "
                f"{'.'.join(str(v) for v in version)}",
                "no remedy on this OS version; use replay",
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.probe()
        sc, quartz = self._import_frameworks()

        if not quartz.CGPreflightScreenCaptureAccess():
            # Ask once; the OS shows its own dialog and the answer only takes
            # effect on the next launch of the host app.
            quartz.CGRequestScreenCaptureAccess()
            raise CaptureUnavailable(
                self.name,
                "this process does not hold Screen Recording permission",
                INSTALL_HINT,
            )

        content = self._shareable_content(sc)
        filt = self._build_filter(sc, content)

        config = sc.SCStreamConfiguration.alloc().init()
        config.setMinimumFrameInterval_(_cmtime(1.0 / self.fps))
        config.setQueueDepth_(3)
        config.setShowsCursor_(self.include_cursor)
        # kCVPixelFormatType_32BGRA: BGRA maps to the contract's BGR by dropping
        # one channel, so there is no colour conversion on the hot path.
        config.setPixelFormat_(0x42475241)

        self._output = _StreamOutput(self._on_frame)
        self._stream = sc.SCStream.alloc().initWithFilter_configuration_delegate_(
            filt, config, None
        )
        error: Any = None
        ok = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
            self._output, 0, None, error
        )
        if not ok:
            raise CaptureUnavailable(
                self.name, f"SCStream.addStreamOutput failed: {error}", INSTALL_HINT
            )
        self._stream.startCaptureWithCompletionHandler_(lambda _err: None)
        self._started = True
        logger.info("screencapturekit started at %.1f fps", self.fps)

    def stop(self) -> None:
        self._started = False
        if self._stream is not None:
            try:
                self._stream.stopCaptureWithCompletionHandler_(lambda _err: None)
            except Exception:
                logger.debug("ignoring error while stopping SCStream", exc_info=True)
            self._stream = None
            self._output = None
        with self._lock:
            self._latest = None

    def grab(self) -> Frame | None:
        if not self._started:
            raise CaptureNotStarted("grab() before start(); call start() first")
        with self._lock:
            latest = self._latest
            self._latest = None
        if latest is None:
            return None
        image, captured_at = latest
        self._seq += 1
        return Frame(image=image, captured_at=captured_at, seq=self._seq, backend=self.name)

    # -- framework glue ----------------------------------------------------

    def _import_frameworks(self) -> tuple[Any, Any]:
        try:
            import Quartz
            import ScreenCaptureKit
        except ImportError as exc:
            raise CaptureUnavailable(
                self.name, f"pyobjc frameworks are unusable: {exc}", INSTALL_HINT
            ) from exc
        return ScreenCaptureKit, Quartz

    def _shareable_content(self, sc: Any) -> Any:
        """Fetch SCShareableContent, whose API is completion-handler only."""
        done = threading.Event()
        box: dict[str, Any] = {}

        def handler(content: Any, error: Any) -> None:
            box["content"], box["error"] = content, error
            done.set()

        sc.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
        if not done.wait(timeout=10.0):
            raise CaptureUnavailable(
                self.name,
                "SCShareableContent never called back",
                "usually means the permission prompt is waiting behind another window",
            )
        if box.get("error") is not None or box.get("content") is None:
            raise CaptureUnavailable(
                self.name, f"SCShareableContent failed: {box.get('error')}", INSTALL_HINT
            )
        return box["content"]

    def _build_filter(self, sc: Any, content: Any) -> Any:
        if self.window_title_contains:
            needle = self.window_title_contains.lower()
            for window in content.windows():
                title = window.title() or ""
                if needle in title.lower():
                    return sc.SCContentFilter.alloc().initWithDesktopIndependentWindow_(window)
            raise CaptureUnavailable(
                self.name,
                f"no on-screen window whose title contains {self.window_title_contains!r}",
                "start the 2009scape client first, or capture the display instead",
            )
        displays = content.displays()
        if not displays or self.display_index >= len(displays):
            raise CaptureUnavailable(
                self.name, f"no display at index {self.display_index}", "check display_index"
            )
        display = displays[self.display_index]
        return sc.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])

    def _on_frame(self, image: np.ndarray) -> None:
        with self._lock:
            self._latest = (to_bgr(image), time.monotonic())


class _StreamOutput:
    """SCStreamOutput delegate.

    Built with ``objc.python_method``-free plain callbacks and only instantiated
    on macOS; on any other platform this class is inert because ``start()``
    raises long before it is constructed.
    """

    def __init__(self, sink: Any) -> None:
        self._sink = sink

    def stream_didOutputSampleBuffer_ofType_(  # noqa: N802 - ObjC selector name
        self, stream: Any, sample_buffer: Any, output_type: int
    ) -> None:
        del stream
        if output_type != 0:  # SCStreamOutputTypeScreen
            return
        import Quartz

        pixel_buffer = Quartz.CMSampleBufferGetImageBuffer(sample_buffer)
        if pixel_buffer is None:
            return
        Quartz.CVPixelBufferLockBaseAddress(pixel_buffer, 1)
        try:
            width = Quartz.CVPixelBufferGetWidth(pixel_buffer)
            height = Quartz.CVPixelBufferGetHeight(pixel_buffer)
            stride = Quartz.CVPixelBufferGetBytesPerRow(pixel_buffer)
            base = Quartz.CVPixelBufferGetBaseAddress(pixel_buffer)
            buf = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8)
            # Rows are padded to the stride; slicing off the padding is cheaper
            # than asking CoreVideo for a tightly packed buffer.
            image = buf.reshape(height, stride // 4, 4)[:, :width, :3].copy()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(pixel_buffer, 1)
        self._sink(image)


def _macos_version() -> str:
    import platform

    return platform.mac_ver()[0] or "0.0"


def _cmtime(seconds: float) -> Any:
    import Quartz

    return Quartz.CMTimeMakeWithSeconds(seconds, 600)


def probe_screencapturekit() -> None:
    ScreenCaptureKitCapture.probe()


__all__ = ["INSTALL_HINT", "ScreenCaptureKitCapture", "probe_screencapturekit"]
