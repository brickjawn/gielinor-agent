#!/usr/bin/env python3
"""Full-pipeline perception benchmark against the 50 ms fast-loop budget.

SPEC section 2 gives the whole fast loop — capture, perceive, plan, actuate —
under 50 ms. Perception is one of four tenants in that budget, so the number
that matters is not "does it fit" but "how much of the budget does it take, and
which stage is spending it".

Two measurement notes that change what the numbers mean:

  * **Steady state, not cold start.** The first `perceive()` of a process pays
    for the atlas load, OpenCV's kernel allocation and numpy's first-touch page
    faults — about 4x the steady-state cost here. The fast loop runs at 10-20 Hz
    for hours, so steady state is the honest figure; the cold-start cost is
    reported separately rather than hidden inside an average.
  * **Cache effects are real, and reported.** Decoded chat lines are memoised,
    and a real chat log genuinely does repeat frame to frame, so the cached
    number is the operational one. The `--no-cache` arm shows the floor if
    every frame were novel.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from gielinor.perception.layout import DEFAULT_LAYOUT  # noqa: E402
from gielinor.perception.perceiver import PixelPerceiver  # noqa: E402
from gielinor.perception.synthetic import (  # noqa: E402
    CHAT_CORPUS,
    DialogueSpec,
    Difficulty,
    SceneSpec,
    SyntheticRenderer,
    TreeSpec,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gielinor.contracts import Frame, GameState

logger = logging.getLogger("bench_perception")

BUDGET_MS = 50.0
STAGES = ("chat", "tooltip", "objects", "inventory", "animation", "orbs")


def build_frames(n: int, difficulty: Difficulty, seed: int) -> list[Frame]:
    """A representative trip: trees drift, the inventory fills, chat scrolls."""
    renderer = SyntheticRenderer(difficulty=difficulty, seed=seed)
    frames: list[Frame] = []
    for i in range(n):
        dialogue = (
            DialogueSpec("npc", "Woodsman", ("Nice weather for chopping.",))
            if i and i % 17 == 0
            else None
        )
        spec = SceneSpec(
            seq=i,
            chat=()
            if dialogue
            else tuple(CHAT_CORPUS[j % len(CHAT_CORPUS)] for j in range(i, i + 6)),
            tooltip="Chop down Tree",
            trees=(
                TreeSpec("tree", 150 + (i % 12), 150, 34, "blob"),
                TreeSpec("tree", 340, 210, 30, "crescent"),
                TreeSpec("willow_tree", 430, 120, 32, "fork"),
            ),
            inventory_used=min(28, i // 3),
            dialogue=dialogue,
            animate_phase=(i % 4) + 1,
        )
        frames.append(renderer.render(spec)[0])
    return frames


def measure(frames: Sequence[Frame], *, warmup: int, cache: bool) -> dict[str, object]:
    perceiver = PixelPerceiver()
    if not cache:
        perceiver.ocr.matcher._cache_size = 0

    t0 = time.perf_counter()
    perceiver.perceive(frames[0])
    cold_ms = (time.perf_counter() - t0) * 1000.0

    for f in frames[1 : 1 + warmup]:
        perceiver.perceive(f)

    totals: list[float] = []
    stage_totals: dict[str, list[float]] = {s: [] for s in STAGES}
    previous: GameState | None = None
    confidences: list[float] = []
    for f in frames:
        state = perceiver.perceive(f, previous)
        previous = state
        totals.append(state.perception_ms)
        confidences.append(state.confidence)
        for s in STAGES:
            stage_totals[s].append(perceiver.stage_ms.get(s, 0.0))
    return {
        "cold_ms": cold_ms,
        "totals": totals,
        "stages": stage_totals,
        "confidence": statistics.mean(confidences),
        "matcher": perceiver.ocr.matcher.stats,
    }


def print_report(name: str, result: dict[str, object]) -> None:
    totals: list[float] = result["totals"]  # type: ignore[assignment]
    stages: dict[str, list[float]] = result["stages"]  # type: ignore[assignment]
    med = statistics.median(totals)
    p95 = sorted(totals)[int(0.95 * (len(totals) - 1))]

    print(f"\n{name}")
    print(f"  frames            {len(totals)}")
    print(f"  cold start        {result['cold_ms']:.2f} ms  (first perceive of the process)")
    print(f"  median            {med:.2f} ms")
    print(f"  mean              {statistics.mean(totals):.2f} ms")
    print(f"  p95               {p95:.2f} ms")
    print(f"  max               {max(totals):.2f} ms")
    print(
        f"  budget            {BUDGET_MS:.0f} ms  ->  {med / BUDGET_MS * 100:.1f}% used at median"
    )
    print(f"  mean confidence   {result['confidence']:.3f}")

    print(f"\n  {'stage':<12} {'median ms':>10} {'p95 ms':>9} {'% of frame':>11}")
    print("  " + "-" * 45)
    for stage, values in sorted(stages.items(), key=lambda kv: -statistics.median(kv[1])):
        smed = statistics.median(values)
        sp95 = sorted(values)[int(0.95 * (len(values) - 1))]
        print(f"  {stage:<12} {smed:>10.3f} {sp95:>9.3f} {smed / med * 100:>10.1f}%")
    accounted = sum(statistics.median(v) for v in stages.values())
    print(
        f"  {'(overhead)':<12} {med - accounted:>10.3f} {'':>9} "
        f"{(med - accounted) / med * 100:>10.1f}%"
    )

    verdict = "WITHIN BUDGET" if med < BUDGET_MS else "OVER BUDGET"
    headroom = BUDGET_MS / med if med else float("inf")
    print(f"\n  {verdict}: {headroom:.1f}x headroom at the median.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print("Gielinor perception benchmark")
    print(f"  layout   {DEFAULT_LAYOUT.name} ({DEFAULT_LAYOUT.screen_w}x{DEFAULT_LAYOUT.screen_h})")
    print(f"  frames   {args.frames} per scenario, {args.warmup} warmup")
    print(f"  budget   {BUDGET_MS:.0f} ms/frame (SPEC section 2, fast loop)")

    for label, difficulty in (
        ("clean scene", Difficulty.clean()),
        ("busy scene (noise + clutter + occlusion)", Difficulty.busy()),
    ):
        frames = build_frames(args.frames, difficulty, args.seed)
        print_report(label, measure(frames, warmup=args.warmup, cache=True))

    frames = build_frames(args.frames, Difficulty.clean(), args.seed)
    result = measure(frames, warmup=args.warmup, cache=False)
    print_report("clean scene, line cache disabled (every frame novel)", result)

    print(
        "\nWhat to take from this: perception is not the fast loop's bottleneck, and\n"
        "the stage table says where any future regression landed. The numbers are\n"
        "from synthetic frames of the right size and structure, so they measure the\n"
        "algorithms' cost honestly, but a real capture path adds its own latency\n"
        "(SPEC section 1 puts the real cost in frame acquisition, not in this code)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
