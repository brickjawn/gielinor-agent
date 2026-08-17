#!/usr/bin/env python3
"""Check a calibration profile against a real screenshot, visually.

``assets/profiles/2009scape_fixed.toml`` ships with plausible RS2 fixed-mode
coordinates and ``calibrated = false``, because guessing is fine for bootstrap
and wrong for clicking. This draws every rect in the profile over a captured
frame so you can see, in one image, whether the inventory grid lines up with the
inventory and the chatbox rect actually contains the chatbox.

    python scripts/calibrate_profile.py --frame captures/willows/frame_00000.png
    python scripts/calibrate_profile.py --frame shot.png --out overlay.png

Correcting the numbers is a manual edit of the TOML. That is deliberate: a
script that guesses coordinates from one screenshot would be a third source of
truth, and the point of the profile is that there is exactly one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gielinor.config import Profile, load_profile

if TYPE_CHECKING:
    import numpy as np

    from gielinor.contracts import Rect

COLOURS = {  # BGR
    "viewport": (80, 220, 80),
    "chatbox": (220, 160, 60),
    "chat_input": (240, 200, 120),
    "inventory": (60, 160, 240),
    "minimap": (200, 80, 220),
    "slot": (40, 120, 200),
    "anchor": (60, 60, 230),
    "dialogue": (180, 180, 60),
}


def draw(image: np.ndarray, rect: Rect, colour: tuple[int, int, int], label: str = "") -> None:
    cv2.rectangle(image, (rect.x, rect.y), (rect.x + rect.w - 1, rect.y + rect.h - 1), colour, 1)
    if label:
        cv2.putText(
            image,
            label,
            (rect.x + 2, max(9, rect.y - 3)),
            cv2.FONT_HERSHEY_PLAIN,
            0.7,
            colour,
            1,
            cv2.LINE_AA,
        )


def overlay(image: np.ndarray, profile: Profile) -> np.ndarray:
    canvas = image.copy()
    for name in ("viewport", "chatbox", "chat_input", "inventory", "minimap"):
        draw(canvas, profile.region(name), COLOURS[name], name)  # type: ignore[arg-type]
    for i in range(profile.inventory_grid.capacity):
        draw(canvas, profile.inventory_grid.slot_rect(i), COLOURS["slot"])
    for name, spec in sorted(profile.ui_anchors.items()):
        draw(canvas, spec.to_rect(), COLOURS["anchor"], name)
    if profile.dialogue_options is not None:
        for i in range(profile.dialogue_options.max_lines):
            draw(canvas, profile.dialogue_options.line_rect(i), COLOURS["dialogue"])
    return canvas


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame", type=Path, required=True, help="a captured client screenshot")
    parser.add_argument("--profile", default="2009scape_fixed")
    parser.add_argument("--out", type=Path, default=Path("profile_overlay.png"))
    args = parser.parse_args(argv)

    image = cv2.imread(str(args.frame), cv2.IMREAD_COLOR)
    if image is None:
        print(f"could not read {args.frame}", file=sys.stderr)
        return 2
    profile = load_profile(args.profile)

    h, w = image.shape[:2]
    if (w, h) != profile.client_size:
        print(
            f"WARNING: frame is {w}x{h} but profile {profile.name} describes "
            f"{profile.client_w}x{profile.client_h}. Either the client is not in fixed "
            f"mode, or the capture includes window decorations. Every rect below will "
            f"be wrong until that matches.",
            file=sys.stderr,
        )

    cv2.imwrite(str(args.out), overlay(image, profile))
    print(f"wrote {args.out}")
    print(f"profile: {profile.name} (calibrated={profile.calibrated})")
    print(f"regions: viewport {profile.viewport.to_rect()}  minimap {profile.minimap.to_rect()}")
    print(
        f"{len(profile.anchor_names())} named UI anchors, {profile.inventory_grid.capacity} slots"
    )
    if not profile.calibrated:
        print(
            "\nWhen the rects line up, set `calibrated = true` in the TOML. Until then the "
            "runtime logs a warning on every start, which is the intended nag."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
