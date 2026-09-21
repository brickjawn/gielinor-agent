"""Frame format for the actuator socket.

ADR 0004 put the actuator in its own process behind a Unix socket so that
synthetic input — a real OS privilege — sits behind a boundary the rest of the
agent cannot cross except by sending a message. ADR 0005 adds the requirement
that a message already being executed can be **withdrawn**, which the original
request/response shape cannot express.

Three properties the format has to have, each of which killed a simpler design:

**Abort must not queue behind exec.** If the socket is a strict
request/response pair, the supervisor's withdrawal sits in the send buffer until
the actuator finishes the very action it is trying to stop. The frames here are
independently addressed, and the actuator reads continuously while executing
(see :mod:`gielinor.actuator.server`), so an abort overtakes the work it cancels.

**Every frame names the command it concerns.** An abort that just says "stop"
races: the action it was meant for finishes, the next one starts, and the abort
kills the wrong command. ``seq`` comes from a
:class:`~gielinor.monotonic.MonotonicCounter`, so a late abort names a sequence
number that is no longer current and is discarded rather than misapplied. This
is the same never-recycle argument as ``oid``, one layer down.

**It is readable.** Newline-delimited JSON, not pickle and not a binary struct.
The traffic is a few dozen small frames per minute, so the encoding costs
nothing measurable, and being able to ``socat`` the socket during a failure is
worth more than the microseconds. Pickle would additionally mean the input
privilege is reachable by anything that can write to the socket — declining that
is free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

#: Frames are tiny; anything larger is a bug or a hostile writer, and reading it
#: is a hang. Refuse rather than grow.
MAX_FRAME_BYTES = 64 * 1024


class FrameType(StrEnum):
    HELLO = "hello"  # actuator -> supervisor on connect: capabilities, dry-run state
    EXEC = "exec"  # supervisor -> actuator: run this action
    ABORT = "abort"  # supervisor -> actuator: withdraw command `seq` if still running
    RESULT = "result"  # actuator -> supervisor: terminal outcome for `seq`
    ERROR = "error"  # actuator -> supervisor: frame rejected, command not started


class WireError(ValueError):
    """Malformed frame. Never raised for a well-formed frame the actuator
    declines to act on — that is an ERROR frame, which is an answer, not a
    parse failure."""


@dataclass(frozen=True, slots=True)
class WireFrame:
    type: FrameType
    seq: int
    payload: dict[str, Any] = field(default_factory=dict)

    def encode(self) -> bytes:
        body = json.dumps(
            {"t": self.type.value, "seq": self.seq, "p": self.payload},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        if len(body) + 1 > MAX_FRAME_BYTES:
            raise WireError(f"frame of {len(body)} bytes exceeds {MAX_FRAME_BYTES}")
        return body + b"\n"


def decode(line: bytes) -> WireFrame:
    if len(line) > MAX_FRAME_BYTES:
        raise WireError(f"frame of {len(line)} bytes exceeds {MAX_FRAME_BYTES}")
    try:
        obj = json.loads(line)
    except json.JSONDecodeError as exc:
        raise WireError(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise WireError("frame is not an object")
    try:
        ftype = FrameType(obj["t"])
    except (KeyError, ValueError) as exc:
        raise WireError(f"bad frame type: {obj.get('t')!r}") from exc
    seq = obj.get("seq")
    if not isinstance(seq, int) or isinstance(seq, bool) or seq < 0:
        raise WireError(f"bad seq: {seq!r}")
    payload = obj.get("p", {})
    if not isinstance(payload, dict):
        raise WireError("payload is not an object")
    return WireFrame(type=ftype, seq=seq, payload=payload)
