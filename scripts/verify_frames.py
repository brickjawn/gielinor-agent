#!/usr/bin/env python3
"""Verify captured frames are usable for calibration before you build on them.

ADR 0002 rests on one precondition: captured pixels are 1:1 with what the client
rendered. Exact glyph matching inverts a known encoder, and a resampled frame is
no longer that encoder's output -- it is a blurry photograph of it.

The failure is silent, which is why this script exists. A resampled corpus still
builds an atlas, still runs, still reports a number. The number is just quietly
wrong, and nothing points at the cause. Ten seconds here saves a day of
debugging perception code that was never the problem.

    python scripts/verify_frames.py assets/frames/raw

Exits non-zero if any frame fails, so it can gate a commit.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

# RS2 fixed mode. Override with --size if your client differs -- but note the
# calibration profile assumes this, so a different size means recalibrating.
EXPECTED = (765, 503)

# Chatbox region as a fraction of the frame, deliberately loose: this runs
# BEFORE calibration, so it cannot depend on a calibrated rect.
CHAT_BOX = (0.00, 0.68, 0.66, 1.00)  # x0, y0, x1, y1 as fractions

# Crisp RS chat text over parchment lands in the low tens of distinct colours.
# A x0.86 resample of the same content measured 1333; a real resampled
# screenshot measured 6927. 400 sits far above any plausible crisp frame and
# far below any resampled one, so the gap is wide and the threshold is not
# doing subtle work.
MAX_COLOURS = 400


def check(path: Path, expected: tuple[int, int]) -> tuple[str, list[str]]:
    """Returns ``("PASS" | "WARN" | "FAIL", notes)``.

    Size is the hard gate and colour count is only a warning, because the
    colour heuristic assumes the chat region holds text and nothing else. An
    NPC dialogue portrait, a level-up interface or an options menu all put
    anti-aliased artwork there and sail past the threshold on a frame that is
    pixel-perfect. Two of the ten frames the Phase 0 corpus needs are dialogue
    frames, so failing them would mean routinely overriding this script -- and
    a gate that is routinely overridden has stopped being a gate.
    """
    notes: list[str] = []
    im = Image.open(path)
    w, h = im.size

    if (w, h) != expected:
        sx, sy = w / expected[0], h / expected[1]
        notes.append(f"size {w}x{h}, expected {expected[0]}x{expected[1]}")
        # A near-uniform non-integer scale is the signature of a software
        # resize. Saying so turns "wrong size" into "here is what happened".
        if abs(sx - sy) < 0.01 and abs(sx - 1.0) > 0.01:
            notes.append(f"uniform x{sx:.4f} resize -- something rescaled this in transit")
        return "FAIL", notes

    a = np.array(im.convert("RGB"))
    x0, y0, x1, y1 = CHAT_BOX
    crop = a[int(h * y0) : int(h * y1), int(w * x0) : int(w * x1)]
    n = len(np.unique(crop.reshape(-1, 3), axis=0))
    notes.append(f"{n} colours in chat region")
    if n > MAX_COLOURS:
        notes.append(
            f"over {MAX_COLOURS}: either resampled, or artwork in the chat "
            "region (dialogue portrait, level-up, menu). Eyeball it."
        )
        return "WARN", notes
    return "PASS", notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("directory")
    ap.add_argument("--size", default="765x503")
    args = ap.parse_args()
    w_exp, h_exp = (int(v) for v in args.size.lower().split("x"))
    exp = (w_exp, h_exp)

    frames = sorted(Path(args.directory).glob("*.png"))
    if not frames:
        print(f"no PNGs in {args.directory}")
        return 1

    bad = 0
    warned = 0
    for f in frames:
        status, notes = check(f, exp)
        if status == "FAIL":
            bad += 1
        elif status == "WARN":
            warned += 1
        print(f"{status:4}  {f.name:36} {' | '.join(notes)}")

    print(f"\n{len(frames) - bad}/{len(frames)} the right size, {warned} to eyeball")
    if bad:
        print(
            "\nResampled frames cannot be fixed after the fact -- the information\n"
            "is gone. Recapture with a tool that writes the native framebuffer,\n"
            "and never route them through an app that might resize (chat, email,\n"
            "screenshot services)."
        )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
