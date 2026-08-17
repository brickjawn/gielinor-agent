"""Capture backend tests. No display is used or required."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from conftest import synth_frame
from gielinor.capture import (
    KNOWN_BACKENDS,
    FrameClock,
    PipeWirePortalCapture,
    ReplayCapture,
    ScreenCaptureKitCapture,
    open_capture,
    platform_backend,
)
from gielinor.capture.base import (
    CaptureNotStarted,
    CaptureUnavailable,
    to_bgr,
)
from gielinor.capture.replay import natural_key
from gielinor.contracts import CaptureBackend, Frame

# --------------------------------------------------------------------------
# ReplayCapture
# --------------------------------------------------------------------------


class TestReplayFromMemory:
    def test_satisfies_the_protocol(self, frames: list[np.ndarray]) -> None:
        assert isinstance(ReplayCapture(frames), CaptureBackend)

    def test_serves_frames_in_order_with_synthetic_timestamps(
        self, frames: list[np.ndarray]
    ) -> None:
        cap = ReplayCapture(frames, fps=20.0, loop=False)
        cap.start()
        grabbed = [cap.grab() for _ in range(len(frames))]
        assert all(f is not None for f in grabbed)
        assert [f.seq for f in grabbed if f] == list(range(len(frames)))
        # 20 fps => exactly 50 ms apart, from epoch 0. Exactness is the point:
        # latency assertions elsewhere depend on it.
        assert [round(f.captured_at, 6) for f in grabbed if f] == [
            0.0,
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
        ]

    def test_pixel_content_matches_the_source(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        for expected in frames:
            frame = cap.grab()
            assert frame is not None
            np.testing.assert_array_equal(frame.image, expected)

    def test_is_deterministic_across_instances(self, frames: list[np.ndarray]) -> None:
        def run() -> list[tuple[int, float, int]]:
            cap = ReplayCapture(frames, fps=30.0)
            cap.start()
            out = []
            for _ in range(10):
                f = cap.grab()
                assert f is not None
                out.append((f.seq, f.captured_at, int(f.image[0, 0, 0])))
            return out

        assert run() == run()

    def test_loops_and_keeps_seq_monotonic(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames, loop=True)
        cap.start()
        seqs = []
        for _ in range(len(frames) * 2 + 1):
            f = cap.grab()
            assert f is not None
            seqs.append(f.seq)
        # The index wraps but seq does not: the runtime's staleness maths is
        # defined on seq deltas and would break if it went backwards.
        assert seqs == list(range(len(frames) * 2 + 1))
        assert cap.loops_completed == 2

    def test_non_looping_source_exhausts_and_returns_none(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames, loop=False)
        cap.start()
        for _ in frames:
            assert cap.grab() is not None
        assert cap.grab() is None
        assert cap.exhausted is True
        assert cap.grab() is None  # idempotent, not an error

    def test_seek_and_rewind(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        cap.grab()
        cap.seek(4)
        frame = cap.grab()
        assert frame is not None
        np.testing.assert_array_equal(frame.image, frames[4])
        cap.rewind()
        assert cap.position == 0
        cap.seek(-1)
        assert cap.position == len(frames) - 1

    def test_seek_out_of_range_raises(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        with pytest.raises(IndexError):
            cap.seek(len(frames))

    def test_frame_at_does_not_move_the_playhead(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        before = cap.position
        frame = cap.frame_at(3)
        np.testing.assert_array_equal(frame.image, frames[3])
        assert cap.position == before

    def test_grab_before_start_raises(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        with pytest.raises(CaptureNotStarted):
            cap.grab()

    def test_empty_in_memory_source_fails_at_start_not_construction(self) -> None:
        cap = ReplayCapture([])  # constructing is fine
        with pytest.raises(CaptureUnavailable) as exc:
            cap.start()
        assert "empty" in str(exc.value)

    def test_served_frames_are_read_only(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        frame = cap.grab()
        assert frame is not None
        with pytest.raises(ValueError, match="read-only"):
            frame.image[0, 0] = 255

    def test_does_not_freeze_the_callers_arrays(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames)
        cap.start()
        cap.grab()
        frames[0][0, 0] = 7  # must not raise: the replay copied

    def test_realtime_mode_returns_none_until_the_frame_is_due(
        self, frames: list[np.ndarray]
    ) -> None:
        clock = {"t": 100.0}
        cap = ReplayCapture(frames, fps=10.0, realtime=True, now=lambda: clock["t"])
        cap.start()
        assert cap.grab() is not None  # first is due immediately
        assert cap.grab() is None  # non-blocking: not yet
        clock["t"] += 0.1
        assert cap.grab() is not None
        assert cap.grab() is None

    def test_context_manager_stops(self, frames: list[np.ndarray]) -> None:
        with ReplayCapture(frames) as cap:
            assert cap.started
            assert cap.grab() is not None
        assert cap.started is False

    def test_stride_and_limit(self, frames: list[np.ndarray]) -> None:
        cap = ReplayCapture(frames, stride=1, limit=None)
        cap.start()
        assert len(cap) == len(frames)


class TestReplayFromDisk:
    def test_reads_pngs_in_natural_order(self, frames_dir: Path) -> None:
        cap = ReplayCapture(frames_dir, loop=False)
        cap.start()
        names = [p.name for p in cap.source_paths]
        # frame_9 before frame_10: lexicographic order would silently reorder
        # the run, which is a genuinely annoying bug to find.
        assert names == [
            "frame_0.png",
            "frame_1.png",
            "frame_2.png",
            "frame_9.png",
            "frame_10.png",
            "frame_11.png",
        ]
        first = cap.grab()
        assert first is not None
        np.testing.assert_array_equal(first.image, synth_frame(0))

    def test_missing_directory_is_actionable(self, tmp_path: Path) -> None:
        cap = ReplayCapture(tmp_path / "nope")
        with pytest.raises(CaptureUnavailable) as exc:
            cap.start()
        message = str(exc.value)
        assert "does not exist" in message
        assert "try:" in message  # the remedy, not just the complaint

    def test_empty_directory_is_actionable(self, tmp_path: Path) -> None:
        (tmp_path / "empty").mkdir()
        cap = ReplayCapture(tmp_path / "empty")
        with pytest.raises(CaptureUnavailable, match="no frames matching"):
            cap.start()

    def test_frames_are_bgr_uint8_three_channel(self, frames_dir: Path) -> None:
        cap = ReplayCapture(frames_dir)
        cap.start()
        frame = cap.grab()
        assert frame is not None
        assert frame.image.dtype == np.uint8
        assert frame.image.ndim == 3
        assert frame.image.shape[2] == 3

    def test_natural_key_orders_numerically(self) -> None:
        paths = [Path("f_10.png"), Path("f_2.png"), Path("f_1.png")]
        assert [p.name for p in sorted(paths, key=natural_key)] == [
            "f_1.png",
            "f_2.png",
            "f_10.png",
        ]


class TestToBgr:
    def test_greyscale_is_expanded(self) -> None:
        out = to_bgr(np.zeros((4, 5), dtype=np.uint8))
        assert out.shape == (4, 5, 3)

    def test_bgra_is_truncated(self) -> None:
        src = np.zeros((4, 5, 4), dtype=np.uint8)
        src[:, :, 3] = 255
        assert to_bgr(src).shape == (4, 5, 3)

    def test_float_is_converted(self) -> None:
        out = to_bgr(np.ones((2, 2, 3), dtype=np.float32))
        assert out.dtype == np.uint8
        assert out.max() == 255

    def test_absurd_channel_count_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot interpret"):
            to_bgr(np.zeros((2, 2, 7), dtype=np.uint8))


# --------------------------------------------------------------------------
# FrameClock
# --------------------------------------------------------------------------


class TestFrameClock:
    def _frame(self, seq: int, captured_at: float) -> Frame:
        return Frame(image=synth_frame(seq), captured_at=captured_at, seq=seq)

    def test_measures_achieved_fps_from_arrival_intervals(self) -> None:
        clock = FrameClock(now=lambda: 0.0)
        for i in range(11):
            clock.tick(self._frame(i, i * 0.05), arrived_at=i * 0.05)
        assert clock.frames == 11
        assert clock.achieved_fps == pytest.approx(20.0)
        assert clock.mean_interval_ms == pytest.approx(50.0)

    def test_age_is_measured_against_captured_at_not_arrival(self) -> None:
        clock = FrameClock(now=lambda: 0.0)
        age = clock.tick(self._frame(0, 10.0), arrived_at=10.037)
        assert age == pytest.approx(37.0, abs=1e-6)

    def test_late_frames_are_counted_against_the_budget(self) -> None:
        clock = FrameClock(budget_ms=50.0, now=lambda: 0.0)
        clock.tick(self._frame(0, 0.0), arrived_at=0.010)
        clock.tick(self._frame(1, 1.0), arrived_at=1.080)
        assert clock.late_frames == 1

    def test_repeated_seq_counts_as_duplicate_not_as_a_new_frame(self) -> None:
        clock = FrameClock(now=lambda: 0.0)
        frame = self._frame(3, 0.0)
        clock.tick(frame, arrived_at=0.0)
        clock.tick(frame, arrived_at=0.01)
        assert clock.frames == 1
        assert clock.duplicate_frames == 1

    def test_stats_percentiles_track_the_tail_not_the_mean(self) -> None:
        # 21 frames -> 20 intervals: 18 healthy, 2 stalls. The mean hides the
        # stalls; p95 is what tells you the agent occasionally acts on 200 ms
        # old pixels, which is the number that actually predicts misclicks.
        clock = FrameClock(now=lambda: 0.0)
        t = 0.0
        for i in range(21):
            clock.tick(self._frame(i, t), arrived_at=t)
            t += 0.2 if i in (9, 10) else 0.05
        stats = clock.stats()
        assert stats.max_interval_ms == pytest.approx(200.0)
        assert stats.p95_interval_ms == pytest.approx(200.0)
        assert stats.mean_interval_ms == pytest.approx(65.0)
        assert stats.mean_interval_ms < stats.p95_interval_ms

    def test_reset_clears_everything(self) -> None:
        clock = FrameClock(now=time.monotonic)
        clock.tick(self._frame(0, time.monotonic()))
        clock.reset()
        assert clock.frames == 0
        assert clock.achieved_fps == 0.0


# --------------------------------------------------------------------------
# Platform backends: importable everywhere, fail clearly at start()
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [PipeWirePortalCapture, ScreenCaptureKitCapture])
class TestPlatformBackends:
    def test_construction_never_touches_the_platform(self, cls: type) -> None:
        backend = cls(fps=15.0)
        assert isinstance(backend, CaptureBackend)
        assert backend.name in ("pipewire", "screencapturekit")

    def test_probe_or_start_raises_capture_unavailable_here(self, cls: type) -> None:
        # This box has no compositor and no Objective-C runtime, so at least one
        # of the two platform backends must refuse. Whichever one matches this
        # platform may in principle succeed on a developer machine, so accept a
        # clean failure or a clean start.
        backend = cls()
        try:
            backend.probe()
        except CaptureUnavailable as exc:
            assert exc.backend == backend.name
            assert exc.problem
            assert "try:" in str(exc) or exc.remedy == ""
        else:  # pragma: no cover - only on a real desktop
            backend.stop()

    def test_grab_before_start_raises(self, cls: type) -> None:
        with pytest.raises(CaptureNotStarted):
            cls().grab()


def test_platform_modules_import_without_their_dependencies() -> None:
    # The rule that makes CI possible: importing a backend must never probe the
    # platform. If this ever regresses, collection fails on the CI box first.
    import gielinor.capture.pipewire as pw
    import gielinor.capture.screencapturekit as sck

    assert pw.INSTALL_HINT
    assert sck.INSTALL_HINT


# --------------------------------------------------------------------------
# open_capture factory
# --------------------------------------------------------------------------


class TestOpenCapture:
    def test_explicit_replay(self, frames: list[np.ndarray]) -> None:
        cap = open_capture("replay", replay_source=frames, fps=10.0)
        assert isinstance(cap, ReplayCapture)
        assert cap.fps == 10.0

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown capture backend"):
            open_capture("magic")

    def test_explicit_platform_name_never_falls_back(self) -> None:
        # Asking for pipewire and silently getting a replay would turn "consent
        # refused" into "the agent clicks at nothing".
        cap = open_capture("pipewire")
        assert isinstance(cap, PipeWirePortalCapture)

    def test_auto_falls_back_to_replay_when_the_platform_backend_is_unavailable(
        self, frames: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("GIELINOR_ASSUME_HEADLESS", "1")
        cap = open_capture("auto", replay_source=frames)
        assert isinstance(cap, ReplayCapture)

    def test_auto_without_a_replay_source_explains_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("GIELINOR_ASSUME_HEADLESS", "1")
        with pytest.raises(CaptureUnavailable) as exc:
            open_capture("auto")
        assert "replay_source" in str(exc.value)

    def test_auto_drops_backend_specific_kwargs_on_the_fallback_path(
        self, frames: list[np.ndarray], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
        monkeypatch.delenv("DISPLAY", raising=False)
        monkeypatch.setenv("GIELINOR_ASSUME_HEADLESS", "1")
        cap = open_capture("auto", replay_source=frames, source_type="window")
        assert isinstance(cap, ReplayCapture)

    def test_platform_backend_name_is_one_of_the_known_ones(self) -> None:
        assert platform_backend() in KNOWN_BACKENDS
