"""Supervisor side of the actuator socket, plus the loop that watches an action.

:class:`ActionGuard` is the half of the protocol that lives outside the actuator.
The actuator can be interrupted; something has to decide to interrupt it. That
decision needs a GameState, so it belongs here, on the side of the boundary that
has one.

The shape is deliberately boring: while a command is in flight, re-run the same
validation that authorised it, once per fresh frame, against the *current*
state. The first time it fails, withdraw. This is not a new check — it is the
existing :func:`validate_action` moved from a one-shot gate to a continuous one,
which is the smallest change that makes validity a property maintained over the
action's lifetime rather than asserted at its start.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

from ..monotonic import MonotonicCounter
from .wire import FrameType, WireFrame, decode

log = logging.getLogger(__name__)

#: Returns a human-readable reason to withdraw, or None if the action is still
#: justified. Closes over whatever the supervisor's freshest GameState is.
Revalidator = Callable[[], str | None]


@dataclass
class ActuatorClient:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    seq: MonotonicCounter = field(default_factory=MonotonicCounter)

    @classmethod
    async def connect(cls, socket_path: str) -> ActuatorClient:
        reader, writer = await asyncio.open_unix_connection(socket_path)
        return cls(reader=reader, writer=writer)

    async def send(self, frame: WireFrame) -> None:
        self.writer.write(frame.encode())
        await self.writer.drain()

    async def recv(self) -> WireFrame | None:
        line = await self.reader.readline()
        return decode(line) if line else None

    async def exec_click(
        self, x: int, y: int, expect_hover: str | None = None, button: str = "left"
    ) -> int:
        s = self.seq.next()
        await self.send(
            WireFrame(
                FrameType.EXEC,
                s,
                {"target": [int(x), int(y)], "expect_hover": expect_hover, "button": button},
            )
        )
        return s

    async def abort(self, seq: int, reason: str) -> None:
        await self.send(WireFrame(FrameType.ABORT, seq, {"reason": reason}))

    async def close(self) -> None:
        self.writer.close()


@dataclass
class ActionGuard:
    client: ActuatorClient
    #: ~15 Hz. The guard cannot see changes faster than perception produces
    #: frames, so polling faster than the capture rate only burns CPU.
    poll_s: float = 1.0 / 15.0
    aborts_sent: int = field(default=0, init=False)

    async def run(
        self,
        x: int,
        y: int,
        revalidate: Revalidator,
        expect_hover: str | None = None,
    ) -> WireFrame:
        """Issue a click and keep it honest until it resolves.

        Returns the terminal RESULT (or ERROR) frame. Note that a withdrawal is
        a request, not a guarantee: the actuator may already be inside the click
        when it arrives, in which case the result says ``completed`` and the
        supervisor must treat the action as having happened. That is why
        :attr:`Outcome.clicked` exists separately from the status.
        """
        seq = await self.client.exec_click(x, y, expect_hover=expect_hover)
        pending = asyncio.create_task(self._await_terminal(seq))
        sent = False
        try:
            while not pending.done():
                await asyncio.wait({pending}, timeout=self.poll_s)
                if pending.done() or sent:
                    continue
                reason = revalidate()
                if reason is not None:
                    await self.client.abort(seq, reason)
                    self.aborts_sent += 1
                    sent = True
                    log.info("withdrew seq=%d: %s", seq, reason)
            return pending.result()
        finally:
            if not pending.done():
                pending.cancel()

    async def _await_terminal(self, seq: int) -> WireFrame:
        while True:
            frame = await self.client.recv()
            if frame is None:
                raise ConnectionError("actuator closed the socket mid-command")
            if frame.seq == seq and frame.type in (FrameType.RESULT, FrameType.ERROR):
                return frame
