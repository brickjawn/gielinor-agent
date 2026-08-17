#!/usr/bin/env python3
"""Record frames from a live capture backend into a replay directory.

Every test and every offline demo runs on recorded frames, so this is the tool
that turns a session with the real client into something reproducible. Record
once, then iterate on perception and planning forever without launching the
game.

    python scripts/record_replay.py --out captures/willows --seconds 30

Needs a display and a working capture backend. On Wayland the compositor will
ask for consent; that prompt is the whole security model, so it is not
bypassable and should not be.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gielinor.capture import CaptureUnavailable, FrameClock, open_capture

logger = logging.getLogger("record")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="directory to write PNGs into")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=2000)
    parser.add_argument(
        "--window", default=None, help="macOS: capture the window whose title contains this"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level="INFO", format="%(levelname)-7s %(message)s")
    kwargs = {"window_title_contains": args.window} if args.window else {}
    try:
        capture = open_capture(args.backend, fps=args.fps, **kwargs)
        capture.start()
    except CaptureUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    clock = FrameClock(budget_ms=1000.0 / args.fps)
    written = 0
    deadline = time.monotonic() + args.seconds
    try:
        while time.monotonic() < deadline and written < args.max_frames:
            frame = capture.grab()
            if frame is None:
                time.sleep(0.002)
                continue
            clock.tick(frame)
            # Zero-padded so lexicographic and natural order agree, which makes
            # the directory readable by anything, not just our loader.
            path = args.out / f"frame_{written:05d}.png"
            cv2.imwrite(str(path), frame.image)
            written += 1
    except KeyboardInterrupt:
        logger.info("interrupted")
    finally:
        capture.stop()

    stats = clock.stats()
    print(f"wrote {written} frames to {args.out}")
    print(
        f"achieved {stats.achieved_fps:.1f} fps | mean age {stats.mean_age_ms:.1f} ms | "
        f"p95 age {stats.p95_age_ms:.1f} ms | max age {stats.max_age_ms:.1f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
