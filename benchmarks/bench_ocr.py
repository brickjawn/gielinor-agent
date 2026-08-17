#!/usr/bin/env python3
"""OCR benchmark: glyph matching vs a general-purpose engine, on identical input.

This is the headline number for the project, and it is worth being precise
about what it does and does not show.

WHAT IT SHOWS. Both engines get the same 765x503 synthetic frames and the same
chatbox region. The glyph matcher reads a bitmap font it has an exact atlas
for; Tesseract reads the same pixels with every advantage we can reasonably
give it — we try three preprocessings (raw crop, 4x upscale + Otsu, and the
binary mask our own shadow filter produced) and score its *best* result. The
gap that remains is the cost of using an anti-aliased-outline-font OCR on a
1-bit bitmap font at 12 px, which is the argument SPEC section 3 makes.

WHAT IT DOES NOT SHOW. The synthetic frames are drawn with the same atlas the
matcher decodes with, so the glyph arm's absolute accuracy on `clean` is
self-consistency, not a prediction for a real client. The transferable claims
are (a) the relative gap against a general engine on identical pixels, (b) the
degradation curve across difficulty levels, and (c) the per-line latency.

Tesseract is optional. If `pytesseract` or the `tesseract` binary is missing,
the comparison arm is skipped with a note and the glyph arm still runs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gielinor.perception.fonts import atlas_report, load_default_atlas  # noqa: E402
from gielinor.perception.layout import DEFAULT_LAYOUT  # noqa: E402
from gielinor.perception.metrics import summarise  # noqa: E402
from gielinor.perception.ocr import GlyphOCR, ShadowFilter  # noqa: E402
from gielinor.perception.synthetic import Difficulty, chat_frames  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

logger = logging.getLogger("bench_ocr")

DIFFICULTIES: tuple[tuple[str, Difficulty], ...] = (
    ("clean", Difficulty.clean()),
    ("busy", Difficulty.busy()),
    ("hostile", Difficulty.hostile()),
)


@dataclass
class Row:
    engine: str
    difficulty: str
    char_acc: float
    word_acc: float
    exact: float
    ms_per_line: float
    lines: int
    note: str = ""


# --------------------------------------------------------------------------
# Tesseract arm (optional)
# --------------------------------------------------------------------------


def _tesseract() -> Callable[[np.ndarray], str] | None:
    try:
        import pytesseract  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.info("pytesseract not importable (%s)", exc)
        return None
    try:
        version = pytesseract.get_tesseract_version()
    except Exception as exc:  # pragma: no cover - binary missing
        logger.info("tesseract binary unavailable (%s)", exc)
        return None
    logger.info("tesseract %s available; comparison arm enabled", version)

    def run(img: np.ndarray) -> str:
        # psm 6 == "a single uniform block of text", which is what a chat log is.
        return pytesseract.image_to_string(img, config="--psm 6")

    return run


def _tesseract_variants(crop: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Three preprocessings, so the comparison is not a straw man.

    General OCR is known to want big, black-on-white, denoised input, so we
    hand it exactly that in two of the three variants — including one built
    from our own shadow filter's output, i.e. with the segmentation problem
    already solved for it.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    from_mask = np.where(mask, np.uint8(0), np.uint8(255))
    from_mask = cv2.resize(from_mask, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_NEAREST)
    return [crop, otsu, from_mask]


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


def run_glyph_arm(
    name: str, difficulty: Difficulty, n_frames: int, seed: int, *, cache: bool = True
) -> Row:
    engine = GlyphOCR()
    if not cache:
        # A chat log repeats frame to frame, so the cache is doing real work
        # rather than flattering the benchmark. Reporting both is the honest
        # move: the cached row is the operational number, the uncached row is
        # the floor if every line on screen were new.
        engine.matcher._cache_size = 0
    frames = chat_frames(n_frames, difficulty=difficulty, seed=seed)
    for frame, _ in frames[:2]:  # warm the atlas index and numpy's first-touch paths
        engine.read_region(frame.image, DEFAULT_LAYOUT.chatbox, source="chatbox")

    truths: list[str] = []
    preds: list[str] = []
    elapsed = 0.0
    for frame, gt in frames:
        t0 = time.perf_counter()
        lines = engine.read_region(frame.image, DEFAULT_LAYOUT.chatbox, source="chatbox")
        elapsed += time.perf_counter() - t0
        truths.extend(gt.chat_lines)
        preds.extend(ln.text for ln in lines)

    stats = summarise(truths, preds)
    return Row(
        engine="glyph" if cache else "glyph (no cache)",
        difficulty=name,
        char_acc=stats["char_accuracy"],
        word_acc=stats["word_accuracy"],
        exact=stats["exact_lines"],
        ms_per_line=elapsed * 1000.0 / max(1, len(truths)),
        lines=len(truths),
        note=f"cache hits {engine.matcher.stats.cache_hits}/{engine.matcher.stats.decodes}",
    )


def run_tesseract_arm(
    name: str, difficulty: Difficulty, n_frames: int, seed: int, run: Callable[[np.ndarray], str]
) -> Row:
    frames = chat_frames(n_frames, difficulty=difficulty, seed=seed)
    rect = DEFAULT_LAYOUT.chatbox
    shadow = ShadowFilter()

    truths: list[str] = []
    preds: list[str] = []
    elapsed = 0.0
    for frame, gt in frames:
        crop = frame.image[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w]
        mask = shadow.apply(crop).mask
        best: list[str] = []
        best_score = -1.0
        for variant in _tesseract_variants(crop, mask):
            t0 = time.perf_counter()
            raw = run(variant)
            elapsed += time.perf_counter() - t0
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            score = summarise(gt.chat_lines, lines)["char_accuracy"]
            if score > best_score:
                best_score, best = score, lines
        truths.extend(gt.chat_lines)
        preds.extend(best)

    stats = summarise(truths, preds)
    return Row(
        engine="tesseract",
        difficulty=name,
        char_acc=stats["char_accuracy"],
        word_acc=stats["word_accuracy"],
        exact=stats["exact_lines"],
        ms_per_line=elapsed * 1000.0 / max(1, len(truths)),
        lines=len(truths),
        note="best of 3 preprocessings",
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def print_table(rows: Sequence[Row]) -> None:
    header = (
        f"{'engine':<17} {'difficulty':<9} {'char acc':>9} {'word acc':>9} "
        f"{'exact':>7} {'ms/line':>9} {'lines':>6}  note"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r.engine:<17} {r.difficulty:<9} {r.char_acc:>9.4f} {r.word_acc:>9.4f} "
            f"{r.exact:>7.3f} {r.ms_per_line:>9.2f} {r.lines:>6}  {r.note}"
        )


def print_comparison(rows: Sequence[Row]) -> None:
    by_key = {(r.engine, r.difficulty): r for r in rows}
    pairs = [(d, by_key.get(("glyph", d)), by_key.get(("tesseract", d))) for d, _ in DIFFICULTIES]
    if not any(t for _, _, t in pairs):
        print("\nTesseract arm did not run, so no comparison is available.")
        print("Install pytesseract and the tesseract binary and re-run to populate it.")
        return
    print("\nglyph vs tesseract, same pixels (tesseract -> glyph):")
    for name, g, t in pairs:
        if g is None or t is None:
            continue
        speed = t.ms_per_line / g.ms_per_line if g.ms_per_line else float("inf")
        print(
            f"  {name:<9} char {t.char_acc:.4f} -> {g.char_acc:.4f}   "
            f"word {t.word_acc:.4f} -> {g.word_acc:.4f}   "
            f"exact lines {t.exact:.3f} -> {g.exact:.3f}   {speed:.0f}x faster"
        )
    print(
        "\nCharacter accuracy flatters Tesseract here and the exact-line rate is the\n"
        "number to look at. A chat line is consumed by matching it against known\n"
        "server messages, so 'You get sone logs.' is not 92% useful, it is useless:\n"
        "one wrong glyph costs the whole line. Note also that Tesseract was handed\n"
        "our own shadow-filter mask as one of its three inputs — without that help\n"
        "its raw-crop numbers are considerably worse."
    )
    hostile = [(g, t) for n, g, t in pairs if n == "hostile" and g and t]
    if hostile and hostile[0][0].char_acc - hostile[0][1].char_acc < 0.05:
        g, t = hostile[0]
        print(
            "\nAnd the honest counterweight: under simulated lossy capture (`hostile`,\n"
            "per-pixel noise over the glyphs themselves) the *character* accuracies\n"
            f"converge — {t.char_acc:.4f} vs {g.char_acc:.4f}. Exact bitmap matching's\n"
            "advantage comes from the pixels being exact, and a capture path that\n"
            "perturbs them takes most of that away. The exact-line rate still separates\n"
            f"them ({t.exact:.3f} vs {g.exact:.3f}), because the glyph matcher's errors\n"
            "concentrate in a few unreadable lines while Tesseract's are spread thinly\n"
            "over almost every line — but anyone quoting the clean-input number should\n"
            "quote this one beside it."
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=25)
    # Tesseract runs ~1 s per call and we give it three preprocessings per
    # frame, so its arm gets its own (smaller) sample. Accuracy this far apart
    # does not need 25 frames to establish; the glyph arm keeps the full set.
    parser.add_argument("--tess-frames", type=int, default=6)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--no-tesseract", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    atlas = load_default_atlas()
    print("Gielinor OCR benchmark")
    print(f"  {atlas_report(atlas)}")
    print(f"  frames per difficulty: {args.frames}   seed: {args.seed}")
    print(f"  region: chatbox {DEFAULT_LAYOUT.chatbox}")
    print()

    rows: list[Row] = []
    for name, difficulty in DIFFICULTIES:
        rows.append(run_glyph_arm(name, difficulty, args.frames, args.seed))
    rows.append(run_glyph_arm("clean", DIFFICULTIES[0][1], args.frames, args.seed, cache=False))

    tess = None if args.no_tesseract else _tesseract()
    if tess is not None:
        print(
            f"running the tesseract arm on {args.tess_frames} frames per difficulty "
            f"(~1 s per call, 3 preprocessings per frame)...",
            flush=True,
        )
        for name, difficulty in DIFFICULTIES:
            rows.append(run_tesseract_arm(name, difficulty, args.tess_frames, args.seed, tess))
        print()
    else:
        print("NOTE: Tesseract comparison arm skipped (pytesseract or the binary is")
        print("      unavailable). The arm is wired up and runs automatically when both")
        print("      are installed: pip install pytesseract && apt install tesseract-ocr")
        print()

    print_table(rows)
    print_comparison(rows)

    print("\nCaveat, stated where the numbers are: the frames are rendered with the")
    print("same glyph atlas the matcher decodes with, so the glyph arm's absolute")
    print("accuracy on clean input measures pipeline self-consistency. The relative")
    print("gap on identical pixels and the degradation across difficulties are the")
    print("figures that carry over to a real client.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
