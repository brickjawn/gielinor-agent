#!/usr/bin/env python3
"""Record the client canvas directly from X11, bypassing the portal entirely.

The PipeWire/portal backend in `gielinor.capture.pipewire` is the right design
for a Wayland-native client, and on this machine it does not work: KWin hands
back a session and a node id, then `pw_stream_connect` reports "target not
found" for window capture and delivers no buffers for monitor capture. That is
worth fixing, but it is not worth blocking Phase 0 on.

The client is a Java application, so it runs under XWayland, and XWayland
windows have real X11 drawables that can be read directly. Better still, the
Java canvas is its own child window at exactly the client's render size, so
reading it needs no cropping and no knowledge of window decorations or
compositor offsets -- which is precisely the property ADR 0002 depends on. A
cropped or scaled frame is not the glyph encoder's output and exact matching
silently stops meaning anything.

    python scripts/capture_x11.py --out captures/chopping --seconds 60 --fps 4

Every frame is size-checked as it is written, because a frame that is the wrong
size is not a frame you want to discover at calibration time.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path

IMPORT = "/usr/bin/import"
XDOTOOL = "/usr/bin/xdotool"
XWININFO = "/usr/bin/xwininfo"

#: RS2 fixed mode. The canvas child window is exactly this when the client is
#: in fixed mode; anything else means the client is in resizable or HD mode and
#: the calibration profile will not apply.
EXPECTED = (765, 503)

_CANVAS = re.compile(r"(0x[0-9a-f]+)\s+\"sun-awt-X11-XCanvasPeer\".*?(\d+)x(\d+)")


def find_canvas(title: str) -> tuple[int, int, int]:
    """Return ``(window_id, width, height)`` for the client's Java canvas.

    Resolved fresh on every call rather than cached: the AWT toolkit destroys
    and recreates these child windows when the client is resized or switches
    between the menu and the game, and a stale id silently captures the wrong
    window instead of failing.
    """
    found = subprocess.run(  # noqa: S603
        [XDOTOOL, "search", "--name", title], capture_output=True, text=True, check=False
    )
    ids = [line for line in found.stdout.split() if line.strip()]
    if not ids:
        raise SystemExit(f"no window matching {title!r}; is the client running?")

    for top in ids:
        tree = subprocess.run(  # noqa: S603
            [XWININFO, "-id", top, "-tree"], capture_output=True, text=True, check=False
        )
        m = _CANVAS.search(tree.stdout)
        if m:
            return int(m.group(1), 16), int(m.group(2)), int(m.group(3))
    raise SystemExit(f"found {title!r} but no sun-awt-X11-XCanvasPeer child window")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default="2009Scape")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--fps", type=float, default=4.0)
    args = ap.parse_args()

    wid, w, h = find_canvas(args.title)
    print(f"canvas 0x{wid:x} is {w}x{h}")
    if (w, h) != EXPECTED:
        print(
            f"WARNING: expected {EXPECTED[0]}x{EXPECTED[1]} (RS2 fixed mode).\n"
            "         Set the client to fixed mode / Standard detail, or the\n"
            "         calibration profile will not apply to these frames."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / args.fps
    deadline = time.monotonic() + args.seconds
    n = 0
    bad = 0

    while time.monotonic() < deadline:
        tick = time.monotonic()
        path = args.out / f"{n:06d}.png"
        proc = subprocess.run(  # noqa: S603
            # -silent suppresses the X bell. Without it every frame rings the
            # terminal bell, which at capture rates is a continuous alarm and
            # makes the operator want to stop the run early -- the opposite of
            # what a corpus needs.
            [IMPORT, "-silent", "-window", str(wid), str(path)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not path.exists():
            # The canvas was recreated mid-session (resize, menu toggle).
            # Re-resolve once rather than spending the rest of the run failing.
            print(f"  frame {n}: capture failed, re-resolving canvas")
            wid, w, h = find_canvas(args.title)
            bad += 1
        else:
            n += 1
            if n % 10 == 0:
                print(f"  {n} frames")
        slept = interval - (time.monotonic() - tick)
        if slept > 0:
            time.sleep(slept)

    print(f"\nwrote {n} frames to {args.out} ({bad} retries)")
    print(f"next: python scripts/verify_frames.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
