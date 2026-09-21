"""Interrupt invariants for the actuator.

One assertion recurs and it is the one that counts: **``clicked is False``**.
Status strings are narration; the click is the irreversible act. Every test that
claims an action was withdrawn checks that no click was issued, not that the
result said "aborted".

The tests run against a real Unix socket and a real asyncio server, because the
property under test is a concurrency property of the read loop. A mocked
transport would test a different program.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import pytest

from gielinor.actuator.abortable import (
    AbortableExecutor,
    ExecStatus,
    Waypoint,
    hover_matches,
    normalise_hover,
)
from gielinor.actuator.client import ActionGuard, ActuatorClient
from gielinor.actuator.server import ActuatorServer
from gielinor.actuator.wire import FrameType, WireError, WireFrame, decode

if TYPE_CHECKING:
    from pathlib import Path

# --- doubles -----------------------------------------------------------------


class RecordingMouse:
    def __init__(self) -> None:
        self.moves: list[tuple[int, int]] = []
        self.clicks: list[tuple[int, int, str]] = []
        #: When set to an Event, every move_to blocks on it. Lets a test hold the
        #: actuator inside execution while it sends an abort.
        self.gate: asyncio.Event | None = None
        self.first_move = asyncio.Event()

    async def move_to(self, x: int, y: int) -> None:
        self.moves.append((x, y))
        self.first_move.set()
        if self.gate is not None:
            await self.gate.wait()

    async def click(self, x: int, y: int, button: str = "left") -> None:
        self.clicks.append((x, y, button))


class FixedHover:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.reads = 0

    def read_hover_text(self) -> str | None:
        self.reads += 1
        return self.text


def straight_path(x: int, y: int, n: int = 5, delay: float = 0.0) -> list[Waypoint]:
    return [Waypoint(x=int(x * (i + 1) / n), y=int(y * (i + 1) / n), delay=delay) for i in range(n)]


@pytest.fixture
def mouse() -> RecordingMouse:
    return RecordingMouse()


async def start_server(
    tmp_path: Path, server: ActuatorServer
) -> tuple[ActuatorClient, asyncio.AbstractServer]:
    sock = str(tmp_path / "actuator.sock")
    srv = await server.serve(sock)
    client = await ActuatorClient.connect(sock)
    hello = await client.recv()
    assert hello is not None and hello.type is FrameType.HELLO
    return client, srv


async def terminal(client: ActuatorClient, seq: int) -> WireFrame:
    while True:
        f = await asyncio.wait_for(client.recv(), timeout=5.0)
        assert f is not None
        if f.seq == seq and f.type in (FrameType.RESULT, FrameType.ERROR):
            return f


# --- the executor in isolation ----------------------------------------------


@pytest.mark.asyncio
async def test_completes_and_clicks_when_nothing_interferes(mouse):
    ex = AbortableExecutor(mouse=mouse)
    out = await ex.run(1, straight_path(100, 100), asyncio.Event())
    assert out.status is ExecStatus.COMPLETED
    assert out.clicked is True
    assert mouse.clicks == [(100, 100, "left")]


@pytest.mark.asyncio
async def test_abort_before_start_issues_no_movement_and_no_click(mouse):
    abort = asyncio.Event()
    abort.set()
    out = await AbortableExecutor(mouse=mouse).run(1, straight_path(100, 100), abort)
    assert out.clicked is False
    assert out.status is ExecStatus.ABORTED
    assert mouse.moves == []


@pytest.mark.asyncio
async def test_abort_mid_travel_stops_and_never_clicks(mouse):
    abort = asyncio.Event()
    ex = AbortableExecutor(mouse=mouse)
    path = straight_path(200, 200, n=10, delay=0.02)
    task = asyncio.create_task(ex.run(1, path, abort))
    await asyncio.sleep(0.05)
    abort.set()
    out = await task
    assert out.clicked is False
    assert mouse.clicks == []
    assert 0 < out.waypoints_travelled < 10, "it should have started and then stopped"


@pytest.mark.asyncio
async def test_abort_during_a_long_dwell_is_not_made_to_wait_it_out(mouse):
    """A dwell is a race against the abort, not a sleep.

    If dwells were plain sleeps, this would take 5 s. The distinction matters
    because humanised trajectories are mostly dwell.
    """
    abort = asyncio.Event()
    ex = AbortableExecutor(mouse=mouse)
    path = [Waypoint(10, 10, delay=5.0), Waypoint(20, 20, delay=5.0)]
    t0 = time.perf_counter()
    task = asyncio.create_task(ex.run(1, path, abort))
    await asyncio.sleep(0.05)
    abort.set()
    out = await task
    elapsed = time.perf_counter() - t0
    assert out.clicked is False
    assert elapsed < 0.5, f"withdrawal waited out the dwell ({elapsed:.2f}s)"


@pytest.mark.asyncio
async def test_hover_mismatch_blocks_the_click(mouse):
    ex = AbortableExecutor(mouse=mouse, hover=FixedHover("Attack Giant rat"))
    out = await ex.run(1, straight_path(100, 100), asyncio.Event(), expect_hover="Chop down Tree")
    assert out.status is ExecStatus.REJECTED_HOVER
    assert out.clicked is False
    assert mouse.clicks == []
    assert "giant rat" in out.detail


@pytest.mark.asyncio
async def test_hover_match_allows_the_click(mouse):
    ex = AbortableExecutor(
        mouse=mouse, hover=FixedHover("Chop down Tree<col=ffffff> / 2 more options")
    )
    out = await ex.run(1, straight_path(100, 100), asyncio.Event(), expect_hover="Chop down Tree")
    assert out.status is ExecStatus.COMPLETED
    assert out.clicked is True


@pytest.mark.asyncio
async def test_empty_hover_is_a_rejection_not_a_pass(mouse):
    ex = AbortableExecutor(mouse=mouse, hover=FixedHover(None))
    out = await ex.run(1, straight_path(100, 100), asyncio.Event(), expect_hover="Chop down Tree")
    assert out.clicked is False


@pytest.mark.asyncio
async def test_requiring_hover_without_a_verifier_fails_closed(mouse):
    out = await AbortableExecutor(mouse=mouse).run(
        1, straight_path(100, 100), asyncio.Event(), expect_hover="Chop down Tree"
    )
    assert out.status is ExecStatus.FAILED
    assert out.clicked is False


@pytest.mark.asyncio
async def test_abort_arriving_during_the_hover_read_is_honoured(mouse):
    class AbortingHover:
        def __init__(self, ev: asyncio.Event) -> None:
            self.ev = ev

        def read_hover_text(self) -> str:
            self.ev.set()  # the world changed while we were reading
            return "Chop down Tree"

    abort = asyncio.Event()
    ex = AbortableExecutor(mouse=mouse, hover=AbortingHover(abort))
    out = await ex.run(1, straight_path(100, 100), abort, expect_hover="Chop down Tree")
    assert out.clicked is False
    assert out.status is ExecStatus.ABORTED


@pytest.mark.asyncio
async def test_click_failure_is_reported_as_possibly_landed():
    class BadClick(RecordingMouse):
        async def click(self, x, y, button="left"):
            raise OSError("uinput write failed")

    out = await AbortableExecutor(mouse=BadClick()).run(1, straight_path(10, 10), asyncio.Event())
    assert out.status is ExecStatus.FAILED
    assert out.clicked is True, "unknown must be treated as happened, or the supervisor double-acts"


def test_hover_normalisation():
    assert normalise_hover("Chop down  Tree<col=ffffff> / 2 more options") == "chop down tree"
    assert normalise_hover(None) == ""
    assert hover_matches("Chop down Tree", "chop down tree")
    assert not hover_matches("", "chop down tree")
    assert not hover_matches("Chop down Tree", "")


# --- the socket: the part that must not queue --------------------------------


@pytest.mark.asyncio
async def test_abort_reaches_a_command_that_is_already_executing(tmp_path, mouse):
    """The structural claim of ADR 0005, tested directly.

    The actuator is held inside `move_to`. If the read loop awaited execution,
    the abort would sit unread until the command finished and the click would
    land. It must be read and applied while the command is in flight.
    """
    mouse.gate = asyncio.Event()
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse), build_path=lambda x, y: straight_path(x, y)
    )
    client, srv = await start_server(tmp_path, server)
    try:
        seq = await client.exec_click(300, 200)
        await asyncio.wait_for(mouse.first_move.wait(), timeout=2.0)

        await client.abort(seq, "target despawned")
        await asyncio.sleep(0.05)  # the abort must be processed while we are still blocked
        mouse.gate.set()

        res = await terminal(client, seq)
        assert res.payload["clicked"] is False
        assert res.payload["status"] == ExecStatus.ABORTED.value
        assert mouse.clicks == []
    finally:
        await client.close()
        srv.close()


@pytest.mark.asyncio
async def test_abort_in_the_same_loop_iteration_as_exec_is_not_treated_as_stale(tmp_path, mouse):
    """The registration race.

    Both frames are written in one syscall, so the server sees them back to back
    before the execution task has run. If the current command were registered
    inside the task rather than at dispatch, this abort would find nothing
    current, be counted stale, and the click would land.
    """
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse),
        build_path=lambda x, y: straight_path(x, y, delay=0.01),
    )
    client, srv = await start_server(tmp_path, server)
    try:
        s = client.seq.next()
        blob = (
            WireFrame(FrameType.EXEC, s, {"target": [150, 150]}).encode()
            + WireFrame(FrameType.ABORT, s, {"reason": "premise expired"}).encode()
        )
        client.writer.write(blob)
        await client.writer.drain()

        res = await terminal(client, s)
        assert res.payload["clicked"] is False
        assert server.stale_aborts == 0
        assert mouse.clicks == []
    finally:
        await client.close()
        srv.close()


@pytest.mark.asyncio
async def test_stale_abort_does_not_kill_the_next_command(tmp_path, mouse):
    """An abort naming a finished command must be discarded, not reapplied.

    Without per-command sequence numbers this is the failure: a late withdrawal
    for action N arrives after N completed and silently cancels N+1, which was
    perfectly valid. Non-recycled seq is what makes the late frame identifiable.
    """
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse), build_path=lambda x, y: straight_path(x, y)
    )
    client, srv = await start_server(tmp_path, server)
    try:
        first = await client.exec_click(100, 100)
        r1 = await terminal(client, first)
        assert r1.payload["clicked"] is True

        await client.abort(first, "too late")
        await asyncio.sleep(0.02)

        second = await client.exec_click(120, 120)
        r2 = await terminal(client, second)
        assert r2.payload["clicked"] is True, (
            "a stale abort must not withdraw the following command"
        )
        assert server.stale_aborts == 1
        assert mouse.clicks == [(100, 100, "left"), (120, 120, "left")]
    finally:
        await client.close()
        srv.close()


@pytest.mark.asyncio
async def test_concurrent_exec_is_refused_and_does_not_disturb_the_running_one(tmp_path, mouse):
    mouse.gate = asyncio.Event()
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse), build_path=lambda x, y: straight_path(x, y)
    )
    client, srv = await start_server(tmp_path, server)
    try:
        first = await client.exec_click(100, 100)
        await asyncio.wait_for(mouse.first_move.wait(), timeout=2.0)
        second = await client.exec_click(200, 200)

        err = await terminal(client, second)
        assert err.type is FrameType.ERROR
        assert err.payload["reason"] == "busy"
        assert err.payload["running_seq"] == first

        mouse.gate.set()
        r1 = await terminal(client, first)
        assert r1.payload["status"] == ExecStatus.COMPLETED.value
        assert mouse.clicks == [(100, 100, "left")]
    finally:
        await client.close()
        srv.close()


@pytest.mark.asyncio
async def test_malformed_frame_is_answered_not_fatal(tmp_path, mouse):
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse), build_path=lambda x, y: straight_path(x, y)
    )
    client, srv = await start_server(tmp_path, server)
    try:
        client.writer.write(b"{not json}\n")
        await client.writer.drain()
        err = await asyncio.wait_for(client.recv(), timeout=2.0)
        assert err is not None and err.type is FrameType.ERROR

        seq = await client.exec_click(50, 50)
        res = await terminal(client, seq)
        assert res.payload["clicked"] is True, "the connection must survive a bad frame"
    finally:
        await client.close()
        srv.close()


def test_wire_rejects_oversized_and_ill_typed_frames():
    with pytest.raises(WireError):
        decode(b'{"t":"exec","seq":-1,"p":{}}')
    with pytest.raises(WireError):
        decode(b'{"t":"exec","seq":true,"p":{}}')
    with pytest.raises(WireError):
        decode(b'{"t":"nope","seq":1}')
    with pytest.raises(WireError):
        decode(b'{"t":"exec","seq":1,"p":[]}')
    with pytest.raises(WireError):
        WireFrame(FrameType.EXEC, 1, {"junk": "x" * 70_000}).encode()
    round_tripped = decode(WireFrame(FrameType.ABORT, 7, {"reason": "gone"}).encode())
    assert round_tripped == WireFrame(FrameType.ABORT, 7, {"reason": "gone"})


# --- the supervisor half -----------------------------------------------------


@pytest.mark.asyncio
async def test_guard_withdraws_when_revalidation_fails(tmp_path, mouse):
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse),
        build_path=lambda x, y: straight_path(x, y, n=20, delay=0.03),
    )
    client, srv = await start_server(tmp_path, server)
    try:
        calls = {"n": 0}

        def revalidate() -> str | None:
            calls["n"] += 1
            return "oid 7 retired" if calls["n"] >= 3 else None

        guard = ActionGuard(client=client, poll_s=0.02)
        res = await guard.run(300, 300, revalidate)
        assert res.payload["clicked"] is False
        assert guard.aborts_sent == 1
        assert mouse.clicks == []
    finally:
        await client.close()
        srv.close()


@pytest.mark.asyncio
async def test_guard_lets_a_still_valid_action_through(tmp_path, mouse):
    server = ActuatorServer(
        executor=AbortableExecutor(mouse=mouse),
        build_path=lambda x, y: straight_path(x, y, n=4, delay=0.01),
    )
    client, srv = await start_server(tmp_path, server)
    try:
        guard = ActionGuard(client=client, poll_s=0.02)
        res = await guard.run(80, 90, lambda: None)
        assert res.payload["clicked"] is True
        assert guard.aborts_sent == 0
    finally:
        await client.close()
        srv.close()
