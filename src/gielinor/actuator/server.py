"""The actuator process: one cursor, one command at a time, always listening.

The whole point of this file is one structural property, so it is worth stating
before the code: **the read loop never awaits the execution task.** An incoming
frame is parsed and dispatched; if it starts work, that work runs as a separate
task and the loop goes straight back to reading. An ABORT therefore reaches the
running command instead of waiting in line behind it, which is the difference
between a withdrawal mechanism and a decoration.

Two smaller decisions that are easy to get wrong:

*The current command is registered synchronously, before its task is created.*
Otherwise an ABORT that arrives in the same event-loop iteration as the EXEC
finds no current command and is discarded as stale — and the action it was meant
to stop proceeds. This race is rare, silent, and exactly the one that matters,
since the fastest aborts come from the fastest-changing state.

*Concurrent EXEC is refused, not queued.* There is one cursor. A queue would
mean actions executing against a world several seconds older than the one that
justified them, which is the problem this module exists to solve, reintroduced
with extra steps. The supervisor aborts and re-issues.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .abortable import AbortableExecutor, ExecStatus, Outcome, Waypoint
from .wire import FrameType, WireError, WireFrame, decode

log = logging.getLogger(__name__)

#: Builds the cursor trajectory for a target point. Supplied by `humanize.py`;
#: injected so the actuator, not the supervisor, owns the shape of the movement.
PathBuilder = Callable[[int, int], Sequence[Waypoint]]

#: Frames waiting to go out. ``None`` closes the pump.
OutQueue = asyncio.Queue["WireFrame | None"]


@dataclass
class _Current:
    seq: int
    abort: asyncio.Event


@dataclass
class ActuatorServer:
    executor: AbortableExecutor
    build_path: PathBuilder
    #: Mirrors `config.allow_real_input`. Reported in HELLO so the supervisor can
    #: refuse to run a live session against a dry-run actuator, and vice versa.
    live_input: bool = False

    _current: _Current | None = field(default=None, init=False)
    #: Aborts that named a command already finished. A rising count means the
    #: supervisor's validation loop is slower than its actions, which is a
    #: tuning signal rather than an error.
    stale_aborts: int = field(default=0, init=False)
    completed: list[Outcome] = field(default_factory=list, init=False)
    #: Strong references to in-flight execution tasks. The event loop only holds
    #: a weak reference, so a task nobody else references can be garbage
    #: collected mid-await -- which here would abandon a trajectory with the
    #: cursor already moving and never send its RESULT. `_current` is not enough:
    #: `_execute` clears it in its own `finally`, before the RESULT is queued.
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, init=False)

    async def serve(self, socket_path: str) -> asyncio.AbstractServer:
        return await asyncio.start_unix_server(self.handle, path=socket_path)

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        out: OutQueue = asyncio.Queue()
        pump = asyncio.create_task(self._pump(out, writer))
        await out.put(WireFrame(FrameType.HELLO, 0, {"live_input": self.live_input, "protocol": 1}))
        try:
            await self._read_loop(reader, out)
        finally:
            await out.put(None)
            await pump
            writer.close()

    # -- the loop that must not block -----------------------------------------

    async def _read_loop(self, reader: asyncio.StreamReader, out: OutQueue) -> None:
        while True:
            line = await reader.readline()
            if not line:
                return
            try:
                frame = decode(line)
            except WireError as exc:
                await out.put(WireFrame(FrameType.ERROR, 0, {"reason": str(exc)}))
                continue

            if frame.type is FrameType.EXEC:
                self._start(frame, out)
            elif frame.type is FrameType.ABORT:
                self._abort(frame)
            else:
                await out.put(
                    WireFrame(
                        FrameType.ERROR,
                        frame.seq,
                        {"reason": f"unexpected frame {frame.type.value}"},
                    )
                )

    def _start(self, frame: WireFrame, out: OutQueue) -> None:
        if self._current is not None:
            out.put_nowait(
                WireFrame(
                    FrameType.ERROR, frame.seq, {"reason": "busy", "running_seq": self._current.seq}
                )
            )
            return
        target = frame.payload.get("target")
        if not (isinstance(target, list) and len(target) == 2):
            out.put_nowait(WireFrame(FrameType.ERROR, frame.seq, {"reason": "missing target"}))
            return

        # Registered before the task exists, so an abort in the same loop
        # iteration is not mistaken for a stale one.
        abort = asyncio.Event()
        self._current = _Current(seq=frame.seq, abort=abort)
        task = asyncio.create_task(self._execute(frame, abort, out))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _abort(self, frame: WireFrame) -> None:
        cur = self._current
        if cur is not None and cur.seq == frame.seq:
            cur.abort.set()
            log.info("abort accepted for seq=%d: %s", frame.seq, frame.payload.get("reason", ""))
        else:
            self.stale_aborts += 1
            log.debug("stale abort seq=%d (current=%s)", frame.seq, cur.seq if cur else None)

    async def _execute(self, frame: WireFrame, abort: asyncio.Event, out: OutQueue) -> None:
        x, y = (int(v) for v in frame.payload["target"])
        try:
            outcome = await self.executor.run(
                seq=frame.seq,
                path=self.build_path(x, y),
                abort=abort,
                expect_hover=frame.payload.get("expect_hover"),
                button=frame.payload.get("button", "left"),
            )
        except Exception as exc:
            outcome = Outcome(ExecStatus.FAILED, False, frame.seq, 0, 0.0, repr(exc))
        finally:
            self._current = None
        self.completed.append(outcome)
        await out.put(
            WireFrame(
                FrameType.RESULT,
                frame.seq,
                {
                    "status": outcome.status.value,
                    "clicked": outcome.clicked,
                    "waypoints": outcome.waypoints_travelled,
                    "elapsed_s": round(outcome.elapsed_s, 6),
                    "detail": outcome.detail,
                    "hover_text": outcome.hover_text,
                },
            )
        )

    async def _pump(self, out: OutQueue, writer: asyncio.StreamWriter) -> None:
        """Single writer, so concurrent RESULT and ERROR frames cannot interleave
        mid-line on the wire."""
        while True:
            frame = await out.get()
            if frame is None:
                return
            writer.write(frame.encode())
            await writer.drain()
