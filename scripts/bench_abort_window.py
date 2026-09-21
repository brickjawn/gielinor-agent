#!/usr/bin/env python3
"""Measure how long a withdrawal actually takes to take effect.

ADR 0005 claims the residual exposure after an abort is "one click call, not one
trajectory". That is a number, so it should be measured rather than asserted.
This measures the three segments that make it up:

1. **Executor latency** - abort.set() to the trajectory stopping, in-process.
2. **Socket latency** - the supervisor's write() to the actuator honouring it,
   over a real Unix socket with a real event loop.
3. **Commit gap** - the last abort check to the click driver being entered. This
   is the window the protocol cannot close, because the click is atomic.

What it does *not* measure is the cost of the OS-level synthetic input call,
which is machine- and backend-specific (uinput vs XTEST vs the Wayland portal).
That number has to come from the machine the agent runs on. Everything here is
the protocol's own overhead, which is the part this design controls.

    python scripts/bench_abort_window.py [--iters 300]
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gielinor.actuator.abortable import AbortableExecutor, Waypoint
from gielinor.actuator.client import ActuatorClient
from gielinor.actuator.server import ActuatorServer

# A plausible humanised trajectory: 24 waypoints over ~600 ms.
PATH = [Waypoint(10 + 8 * i, 10 + 5 * i, delay=0.025) for i in range(24)]


class TimedMouse:
    def __init__(self) -> None:
        self.entered_click: float | None = None
        self.last_move: float | None = None

    # Argument names are fixed by the MouseDriver protocol; this harness only
    # timestamps the calls and never reads the coordinates.
    async def move_to(self, x: int, y: int) -> None:  # noqa: ARG002
        self.last_move = time.perf_counter()
        await asyncio.sleep(0)

    async def click(self, x: int, y: int, button: str = "left") -> None:  # noqa: ARG002
        self.entered_click = time.perf_counter()


def summarise(name: str, samples: list[float], unit: str = "us") -> None:
    scale = 1e6 if unit == "us" else 1e3
    xs = sorted(s * scale for s in samples)
    n = len(xs)
    p = lambda q: xs[min(n - 1, int(q * n))]  # noqa: E731
    print(
        f"{name:<34} n={n:<5} median={statistics.median(xs):8.1f}{unit}  "
        f"p95={p(0.95):8.1f}{unit}  max={xs[-1]:8.1f}{unit}"
    )


async def bench_executor(iters: int) -> list[float]:
    out = []
    for _ in range(iters):
        mouse = TimedMouse()
        ex = AbortableExecutor(mouse=mouse)
        abort = asyncio.Event()
        task = asyncio.create_task(ex.run(1, PATH, abort))
        await asyncio.sleep(0.08)  # land mid-trajectory
        t0 = time.perf_counter()
        abort.set()
        res = await task
        out.append(time.perf_counter() - t0)
        if res.clicked:
            raise RuntimeError("abort did not prevent the click")
    return out


async def bench_socket(iters: int) -> list[float]:
    out = []
    with tempfile.TemporaryDirectory() as d:
        mouse = TimedMouse()
        server = ActuatorServer(
            executor=AbortableExecutor(mouse=mouse), build_path=lambda _x, _y: PATH
        )
        sock = str(Path(d) / "a.sock")
        srv = await server.serve(sock)
        client = await ActuatorClient.connect(sock)
        await client.recv()  # HELLO
        for _ in range(iters):
            seq = await client.exec_click(400, 300)
            await asyncio.sleep(0.08)
            t0 = time.perf_counter()
            await client.abort(seq, "bench")
            while True:
                f = await client.recv()
                if f is not None and f.seq == seq and f.type.value == "result":
                    break
            out.append(time.perf_counter() - t0)
            if f.payload["clicked"]:
                raise RuntimeError("abort did not prevent the click")
        await client.close()
        srv.close()
    return out


async def bench_commit_gap(iters: int) -> list[float]:
    """Time between the executor's final abort check and entering the click."""
    out = []
    for _ in range(iters):
        mouse = TimedMouse()
        ex = AbortableExecutor(mouse=mouse)
        res = await ex.run(1, PATH[:2], asyncio.Event())
        if not (res.clicked and mouse.entered_click and mouse.last_move):
            raise RuntimeError("uninterrupted run did not reach the click")
        out.append(mouse.entered_click - mouse.last_move)
    return out


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=200)
    args = ap.parse_args()

    print(
        f"trajectory: {len(PATH)} waypoints, {sum(w.delay for w in PATH) * 1000:.0f} ms nominal\n"
    )
    summarise("executor: set() -> stopped", await bench_executor(args.iters))
    summarise("socket: write() -> result", await bench_socket(max(20, args.iters // 4)))
    summarise("commit gap: check -> click", await bench_commit_gap(args.iters))
    print(
        "\nThe commit gap is the exposure the protocol cannot remove; the click is\n"
        "atomic by construction. Add your input backend's own latency to it, measured\n"
        "on the machine that runs the agent, before quoting a total anywhere."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
