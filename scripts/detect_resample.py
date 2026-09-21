#!/usr/bin/env python3
"""Detect a resampled frame from pixel structure instead of colour count.

## Why the colour heuristic was wrong, and why the size gate is not enough

`verify_frames.py` counted distinct colours in the chat region on the assumption
that the region holds only text. It does not -- NPC portraits, level-up
interfaces and menus put artwork there, and two of the ten required corpus
scenes are dialogue frames. That gate was structurally incapable of passing
frames the corpus requires: the frame list and the verifier were written in the
same sitting and contradicted each other on arrival.

Demoting colour to a warning and making size the hard gate is the right
immediate fix, but it reopens the case the gate existed for: a resize that lands
back at exactly 765x503 is invisible to a size check, and ADR 0002's exact glyph
matching dies silently on it.

## The signal that works

Any interpolating resize is a low-pass filter. Natively rendered UI carries
energy right at the pixel Nyquist limit -- 1px glyph strokes, 1px drop shadows,
colour-keyed sprite silhouettes -- and interpolation removes it. So measure
energy at that limit directly, as the ratio of second-difference to
first-difference magnitude, averaged over both axes:

    nyquist_ratio = mean(|d2I|) / mean(|dI|)

A one-pixel feature gives a second difference about twice its gradient; a smooth
ramp gives nearly zero. The ratio is normalised, so it measures *structure*
rather than contrast or brightness, and does not care what drew the edges.

## Measured on synthetics

RS2-style chat text (1px glyphs, hard +1/+1 shadow, no AA) against an
anti-aliased NPC portrait with a colour-keyed silhouette -- the exact case that
broke the colour heuristic:

| Transform | text | portrait |
| --- | --- | --- |
| **native** | **1.963** | **1.960** |
| bilinear x0.99 round-trip | 1.581 | 1.568 |
| bilinear x0.86 round-trip | 1.471 | 1.453 |
| lanczos x0.86 round-trip | 1.534 | 1.527 |
| bilinear x0.50 round-trip | 1.064 | 1.044 |
| JPEG q75, no resize | 1.883 | 1.876 |
| **NEAREST x0.86 round-trip** | **1.971** | **1.970** |

Text and portrait differ by at most 0.03 in every row, so the metric is
content-independent in a way the colour count never was. Even a x0.99 resize --
visually indistinguishable, dimensionally identical after round-trip -- lands at
1.57, far below any native frame.

## The blind spot, stated plainly

**Nearest-neighbour resizes are not caught.** NN duplicates and drops whole
columns rather than blending, so hard edges survive and the ratio stays high.
The glyph bitmaps are still wrecked, so matching would still fail -- this gate
just would not be the thing that tells you. `duplicate_line_fraction` rises
under NN and is reported as a diagnostic, but it is content-dependent (0.36 for
text against 0.17 for the portrait, both native) and cannot be a hard gate
without per-corpus calibration. Compositors, chat clients and screenshot
services interpolate, so this is a low-probability path -- but it is a real hole
and not a theoretical one.

## Calibrate before trusting the threshold

`MIN_NYQUIST_RATIO` is set from synthetic evidence. A real corpus does better:
`--calibrate` scores every native frame, generates a resampled copy of each as
the negatives you otherwise lack, and reports the achievable separation. Replace
the constant with what that measures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

#: Calibrated on the real corpus: 281 native frames across 8 scenes, against a
#: bilinear x0.86 round-trip of each. Native spans 1.711-1.832, resampled
#: 1.115-1.264, so the separation is +0.447 and this sits in the middle of it.
#:
#: Stable under corpus growth, which is the reason to believe it: recalculated
#: after the corpus grew 27% and gained inv_00, levelup and a second chopping
#: session, the separation and the suggested constant were unchanged to three
#: decimal places.
#:
#: The synthetic estimate was 1.75, and it was wrong in the direction that
#: matters. Real captures score lower than the synthetics (median 1.770 against
#: 1.96) because a real frame is mostly large flat regions -- grass, parchment,
#: the inventory panel -- which dilute the 1px detail the metric keys on. At
#: 1.75 the camera_alt, dialogue_npc and dialogue_options scenes fail entirely,
#: including the options frames that pin the option rects. That is the same
#: mistake the colour heuristic made: a constant chosen off synthetics that
#: rejects frames the corpus requires. Re-run --calibrate when the corpus grows.
MIN_NYQUIST_RATIO = 1.49

_LUMA = np.array([77, 150, 29])


def _luma(rgb: np.ndarray) -> np.ndarray:
    return ((rgb[..., :3].astype(np.int32) @ _LUMA) >> 8).astype(np.float64)


def nyquist_ratio(rgb: np.ndarray) -> float:
    """Energy at the pixel Nyquist limit, normalised by overall gradient energy.

    ~1.96 for a natively rendered frame, ~1.45-1.58 after any interpolating
    resize. Content-independent to within about 0.03.
    """
    g = _luma(rgb)
    out = []
    for axis in (1, 0):
        d1 = np.abs(np.diff(g, axis=axis))
        sl_a = (slice(None), slice(None, -2)) if axis else (slice(None, -2), slice(None))
        sl_b = (slice(None), slice(1, -1)) if axis else (slice(1, -1), slice(None))
        sl_c = (slice(None), slice(2, None)) if axis else (slice(2, None), slice(None))
        d2 = np.abs(g[sl_a] - 2 * g[sl_b] + g[sl_c])
        out.append(d2.mean() / max(d1.mean(), 1e-9))
    return float(np.mean(out))


def duplicate_line_fraction(rgb: np.ndarray) -> float:
    """Fraction of adjacent column/row pairs that are pixel-identical.

    Rises under nearest-neighbour resizing, which the ratio above cannot see.
    Content-dependent, so a diagnostic rather than a gate until calibrated.
    """
    g = _luma(rgb)
    cols = np.all(np.diff(g, axis=1) == 0, axis=0).mean()
    rows = np.all(np.diff(g, axis=0) == 0, axis=1).mean()
    return float(0.5 * (cols + rows))


def calibrate(directory: Path, scale: float) -> int:
    """Score every native frame and a resampled copy of each; report the gap.

    The copies are down-then-up, so they land at the original dimensions and are
    invisible to the size gate -- which is precisely the case this has to cover.
    """
    frames = sorted(directory.rglob("*.png"))
    if not frames:
        print(f"no PNGs under {directory}")
        return 1

    native, resampled = [], []
    for f in frames:
        im = Image.open(f).convert("RGB")
        native.append(nyquist_ratio(np.array(im)))
        small = im.resize(
            (max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR
        )
        resampled.append(nyquist_ratio(np.array(small.resize(im.size, Image.BILINEAR))))

    n, r = np.array(native), np.array(resampled)
    print(f"frames: {len(frames)}   negatives: bilinear x{scale} round-trip\n")
    print(f"{'':<12}{'min':>9}{'p05':>9}{'median':>9}{'p95':>9}{'max':>9}")
    for name, xs in (("native", n), ("resampled", r)):
        print(
            f"{name:<12}{xs.min():9.3f}{np.percentile(xs, 5):9.3f}"
            f"{np.median(xs):9.3f}{np.percentile(xs, 95):9.3f}{xs.max():9.3f}"
        )

    gap = n.min() - r.max()
    print(f"\nseparation (worst native - best resampled): {gap:+.3f}")
    if gap <= 0:
        print("OVERLAP -- this metric does not separate your corpus. Do not ship the gate.")
        return 1
    print(f"suggested MIN_NYQUIST_RATIO = {r.max() + gap / 2:.2f}")
    if gap < 0.15:
        print("NOTE: the margin is thin. Widen the corpus before relying on it.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="frame, or directory of frames")
    ap.add_argument("--threshold", type=float, default=MIN_NYQUIST_RATIO)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--scale", type=float, default=0.86)
    args = ap.parse_args()

    p = Path(args.path)
    if args.calibrate:
        return calibrate(p, args.scale)

    frames = sorted(p.rglob("*.png")) if p.is_dir() else [p]
    bad = 0
    for f in frames:
        arr = np.array(Image.open(f).convert("RGB"))
        ratio, dup = nyquist_ratio(arr), duplicate_line_fraction(arr)
        ok = ratio >= args.threshold
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {f.name:40} nyquist {ratio:.3f}  dup-lines {dup:.3f}")
    print(f"\n{len(frames) - bad}/{len(frames)} native")
    if bad:
        print("\nA resampled frame cannot be repaired -- the information is gone. Recapture.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
