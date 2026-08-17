"""Wayland screen capture via xdg-desktop-portal ScreenCast + PipeWire.

Why this is not just "take a screenshot": under Wayland a client cannot read the
screen. There is no X11 ``XGetImage`` equivalent and that is the whole point of
the design. The only sanctioned path is to ask ``xdg-desktop-portal`` for a
ScreenCast session, have the *compositor* show the user a picker dialog, and
receive back a PipeWire node id plus a file descriptor to the PipeWire socket.
The user's consent is a hard gate, and it is granted per session — which means
this backend can never start silently in the background, and that is a feature
worth keeping rather than working around.

Flow implemented below:

1. ``org.freedesktop.portal.ScreenCast.CreateSession``
2. ``SelectSources`` (monitor or window, cursor embedded)
3. ``Start``  -> compositor shows the picker, user consents, we get node ids
4. ``OpenPipeWireRemote`` -> a file descriptor
5. GStreamer ``pipewiresrc fd=N path=NODE ! videoconvert ! video/x-raw,format=BGR
   ! appsink`` with ``max-buffers=1 drop=true``, because for an agent the newest
   frame is the only one that matters; a backlog is worse than a gap.

Dependencies (``pip install 'gielinor-agent[linux]'`` plus distro GStreamer
packages) are imported lazily inside ``start()``. Importing this module on macOS,
on a headless CI box, or with no GObject introspection installed is always safe.
"""

from __future__ import annotations

import functools
import logging
import sys
import threading
import time
from typing import Any, Literal

import numpy as np

from gielinor.capture.base import (
    CaptureNotStarted,
    CaptureUnavailable,
    current_platform,
    has_display,
    missing_modules,
    to_bgr,
)
from gielinor.contracts import Frame

logger = logging.getLogger(__name__)

REQUIRED_MODULES: tuple[str, ...] = ("gi",)
INSTALL_HINT = (
    "pip install 'gielinor-agent[linux]' and install the system GStreamer stack "
    "(Debian/Ubuntu: apt install gir1.2-gst-plugins-base-1.0 gstreamer1.0-pipewire "
    "gstreamer1.0-plugins-good; Fedora: dnf install gstreamer1-plugins-good "
    "pipewire-gstreamer python3-gobject)"
)

SourceType = Literal["monitor", "window", "any"]
_SOURCE_BITS: dict[SourceType, int] = {"monitor": 1, "window": 2, "any": 3}


class PipeWirePortalCapture:
    """A `CaptureBackend` for Wayland. Consent-gated, newest-frame-wins."""

    def __init__(
        self,
        *,
        fps: float = 20.0,
        source_type: SourceType = "window",
        include_cursor: bool = False,
        restore_token: str | None = None,
        portal_timeout_s: float = 60.0,
        name: str = "pipewire",
    ) -> None:
        self.name = name
        self.fps = fps
        self.source_type: SourceType = source_type
        self.include_cursor = include_cursor
        # The portal can hand back a token that lets a later run reuse the same
        # consent without re-prompting. Persisting it is the caller's business.
        self.restore_token = restore_token
        self.portal_timeout_s = portal_timeout_s

        self._pipeline: Any = None
        self._appsink: Any = None
        self._session_handle: str | None = None
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._seq = 0
        self._started = False

    # -- availability ------------------------------------------------------

    @staticmethod
    def probe() -> None:
        """Raise CaptureUnavailable if this machine cannot possibly run the backend.

        Cheap and side-effect free: no D-Bus traffic, no consent dialog. The
        factory uses it to decide whether to fall back to replay.
        """
        backend = "pipewire"
        platform = current_platform()
        if platform != "linux":
            raise CaptureUnavailable(
                backend,
                f"platform is {platform!r}, not linux",
                "use the screencapturekit backend on macOS, or replay anywhere",
            )
        missing = missing_modules(REQUIRED_MODULES)
        if missing:
            raise CaptureUnavailable(
                backend, f"missing python modules: {', '.join(missing)}", INSTALL_HINT
            )
        if not has_display():
            raise CaptureUnavailable(
                backend,
                "no WAYLAND_DISPLAY or DISPLAY in the environment",
                "run inside a graphical session; there is nothing to capture from a "
                "TTY or a container without a compositor socket",
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.probe()
        gst, glib = self._import_gstreamer()

        fd, node_id = self._negotiate_portal()
        pipeline_desc = (
            f"pipewiresrc fd={fd} path={node_id} do-timestamp=true keepalive-time=1000 "
            f"! videorate ! video/x-raw,framerate={int(self.fps)}/1 "
            f"! videoconvert ! video/x-raw,format=BGR "
            f"! appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        logger.debug("gst pipeline: %s", pipeline_desc)
        try:
            self._pipeline = gst.parse_launch(pipeline_desc)
        except glib.Error as exc:
            raise CaptureUnavailable(
                self.name, f"could not build the GStreamer pipeline: {exc}", INSTALL_HINT
            ) from exc

        self._appsink = self._pipeline.get_by_name("sink")
        self._appsink.connect("new-sample", self._on_sample)
        state = self._pipeline.set_state(gst.State.PLAYING)
        if state == gst.StateChangeReturn.FAILURE:
            self._pipeline.set_state(gst.State.NULL)
            raise CaptureUnavailable(
                self.name,
                "GStreamer refused to start the pipewiresrc pipeline",
                "check `gst-inspect-1.0 pipewiresrc` resolves; the pipewire plugin "
                "ships separately from gstreamer core on most distros",
            )
        self._started = True
        logger.info("pipewire capture started on node %s at %.1f fps", node_id, self.fps)

    def stop(self) -> None:
        self._started = False
        if self._pipeline is not None:
            try:
                gi_gst = sys.modules.get("gi.repository.Gst")
                if gi_gst is not None:
                    self._pipeline.set_state(gi_gst.State.NULL)
            except Exception:
                logger.debug("ignoring error while tearing down gst pipeline", exc_info=True)
            self._pipeline = None
            self._appsink = None
        self._close_portal_session()
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

    # -- GStreamer glue ----------------------------------------------------

    def _import_gstreamer(self) -> tuple[Any, Any]:
        try:
            import gi

            gi.require_version("Gst", "1.0")
            from gi.repository import GLib, Gst
        except (ImportError, ValueError) as exc:
            raise CaptureUnavailable(
                self.name, f"GObject introspection for GStreamer is unusable: {exc}", INSTALL_HINT
            ) from exc
        if not Gst.is_initialized():
            Gst.init(None)
        return Gst, GLib

    def _on_sample(self, sink: Any) -> Any:
        """appsink callback. Runs on a GStreamer thread — keep it short."""
        gst = sys.modules["gi.repository.Gst"]
        sample = sink.emit("pull-sample")
        if sample is None:
            return gst.FlowReturn.OK
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width, height = caps.get_value("width"), caps.get_value("height")
        ok, mapinfo = buf.map(gst.MapFlags.READ)
        if not ok:
            return gst.FlowReturn.OK
        try:
            # Copy out of the mapped buffer: it is unmapped the moment we return
            # and numpy would otherwise be pointing at reused memory.
            arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
            expected = height * width * 3
            if arr.size < expected:
                return gst.FlowReturn.OK
            image = to_bgr(arr[:expected].reshape(height, width, 3).copy())
        finally:
            buf.unmap(mapinfo)
        with self._lock:
            self._latest = (image, time.monotonic())
        return gst.FlowReturn.OK

    # -- portal glue -------------------------------------------------------

    def _negotiate_portal(self) -> tuple[int, int]:
        """Run the four-step portal handshake. Returns ``(pipewire_fd, node_id)``.

        The portal is request/response over D-Bus with results delivered on a
        ``Request`` object path signal, so each step means "call, then spin a
        main loop until the matching Response arrives or we time out".
        """
        try:
            import gi

            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib
        except (ImportError, ValueError) as exc:
            raise CaptureUnavailable(
                self.name, f"GObject introspection for Gio is unusable: {exc}", INSTALL_HINT
            ) from exc

        try:
            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            proxy = Gio.DBusProxy.new_sync(
                bus,
                Gio.DBusProxyFlags.NONE,
                None,
                "org.freedesktop.portal.Desktop",
                "/org/freedesktop/portal/desktop",
                "org.freedesktop.portal.ScreenCast",
                None,
            )
        except GLib.Error as exc:
            raise CaptureUnavailable(
                self.name,
                f"cannot reach xdg-desktop-portal on the session bus: {exc}",
                "install a portal backend for your compositor "
                "(xdg-desktop-portal-gnome / -kde / -wlr) and check "
                "`systemctl --user status xdg-desktop-portal`",
            ) from exc

        loop = GLib.MainLoop()
        results: dict[str, Any] = {}

        def on_response(*args: Any, _key: str = "") -> None:
            results[_key] = args[-1]
            loop.quit()

        def await_response(request_path: str, key: str) -> dict[str, Any]:
            sub_id = bus.signal_subscribe(
                "org.freedesktop.portal.Desktop",
                "org.freedesktop.portal.Request",
                "Response",
                request_path,
                None,
                Gio.DBusSignalFlags.NONE,
                functools.partial(on_response, _key=key),
            )
            timeout_id = GLib.timeout_add_seconds(int(self.portal_timeout_s), lambda: loop.quit())
            loop.run()
            GLib.source_remove(timeout_id)
            bus.signal_unsubscribe(sub_id)
            if key not in results:
                raise CaptureUnavailable(
                    self.name,
                    f"portal did not answer {key} within {self.portal_timeout_s:.0f}s",
                    "the consent dialog may not have appeared; check the portal backend "
                    "matches your compositor",
                )
            response_code, payload = results[key].unpack()
            if response_code != 0:
                raise CaptureUnavailable(
                    self.name,
                    f"portal {key} was refused (response code {response_code})",
                    "the user must approve the screen-share prompt for the agent to see "
                    "anything; nothing else can grant this",
                )
            return dict(payload)

        token = f"gielinor{int(time.time())}"
        handle = proxy.call_sync(
            "CreateSession",
            GLib.Variant("(a{sv})", ({"session_handle_token": GLib.Variant("s", token)},)),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        ).unpack()[0]
        session = await_response(handle, "create")["session_handle"]
        self._session_handle = session

        select_opts: dict[str, Any] = {
            "types": GLib.Variant("u", _SOURCE_BITS[self.source_type]),
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 2 if self.include_cursor else 1),
            "handle_token": GLib.Variant("s", f"{token}sel"),
        }
        if self.restore_token:
            select_opts["restore_token"] = GLib.Variant("s", self.restore_token)
            select_opts["persist_mode"] = GLib.Variant("u", 2)
        handle = proxy.call_sync(
            "SelectSources",
            GLib.Variant("(oa{sv})", (session, select_opts)),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        ).unpack()[0]
        await_response(handle, "select")

        handle = proxy.call_sync(
            "Start",
            GLib.Variant(
                "(osa{sv})",
                (session, "", {"handle_token": GLib.Variant("s", f"{token}start")}),
            ),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
        ).unpack()[0]
        started = await_response(handle, "start")
        streams = started.get("streams") or []
        if not streams:
            raise CaptureUnavailable(
                self.name, "portal returned no streams", "pick a window or monitor in the dialog"
            )
        node_id = int(streams[0][0])
        self.restore_token = started.get("restore_token", self.restore_token)

        fd_list_call = proxy.call_with_unix_fd_list_sync(
            "OpenPipeWireRemote",
            GLib.Variant("(oa{sv})", (session, {})),
            Gio.DBusCallFlags.NONE,
            -1,
            None,
            None,
        )
        fd_index = fd_list_call[0].unpack()[0]
        fd = fd_list_call[1].get(fd_index)
        return int(fd), node_id

    def _close_portal_session(self) -> None:
        if self._session_handle is None:
            return
        try:
            import gi

            gi.require_version("Gio", "2.0")
            from gi.repository import Gio, GLib

            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            bus.call_sync(
                "org.freedesktop.portal.Desktop",
                self._session_handle,
                "org.freedesktop.portal.Session",
                "Close",
                GLib.Variant("()", ()),
                None,
                Gio.DBusCallFlags.NONE,
                -1,
                None,
            )
        except Exception:
            logger.debug("could not close portal session cleanly", exc_info=True)
        finally:
            self._session_handle = None


def probe_pipewire() -> None:
    PipeWirePortalCapture.probe()


__all__ = ["INSTALL_HINT", "PipeWirePortalCapture", "probe_pipewire"]
