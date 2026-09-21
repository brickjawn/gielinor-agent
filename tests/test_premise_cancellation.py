"""Escalation premise invariants.

The load-bearing assertion here is the accounting one: a cancelled request is
still a spent request. A test suite that let `requests_issued` drop on
cancellation would make the cost model optimistic in exactly the direction that
runs the agent out of quota mid-session.
"""

from __future__ import annotations

import asyncio

import pytest

from gielinor.planner.premise import Premise, PremiseScopedEscalation


async def slow(value: str, delay: float = 0.3) -> str:
    await asyncio.sleep(delay)
    return value


def test_premise_holds_reports_missing_oids():
    p = Premise(oids=frozenset({1, 2, 9}))
    assert p.holds({1, 2, 9}) is None
    assert "9" in (p.holds({1, 2}) or "")


def test_a_raising_predicate_invalidates_rather_than_propagates():
    def boom() -> bool:
        raise RuntimeError("registry is mid-swap")

    reason = Premise(still_true=boom).holds(set())
    assert reason is not None and "raised" in reason


@pytest.mark.asyncio
async def test_answer_returned_when_the_premise_survives():
    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(
        live_oids=lambda: {1, 2}, poll_s=0.01
    )
    got = await esc.ask(lambda: slow("chop oid 1", 0.05), Premise(oids=frozenset({1})))
    assert got == "chop oid 1"
    assert (esc.requests_issued, esc.completed, esc.cancelled) == (1, 1, 0)


@pytest.mark.asyncio
async def test_request_is_cancelled_when_its_oid_disappears():
    live = {1, 2}
    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(
        live_oids=lambda: set(live), poll_s=0.01
    )

    async def kill_it():
        await asyncio.sleep(0.05)
        live.discard(1)

    killer = asyncio.create_task(kill_it())
    got = await esc.ask(
        lambda: slow("too late", 2.0), Premise(oids=frozenset({1}), label="which tree")
    )
    assert got is None
    await killer
    assert esc.cancelled == 1
    assert esc.completed == 0


@pytest.mark.asyncio
async def test_cancellation_does_not_refund_the_request():
    """The accounting rule, asserted directly.

    OpenRouter counts a request at issue. If this ever passes with
    `requests_issued == 0`, the cost model is lying about the daily ceiling.
    """
    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(
        live_oids=lambda: set(), poll_s=0.01
    )
    await esc.ask(lambda: slow("x", 2.0), Premise(oids=frozenset({1})))  # premise dead at entry
    not_issued = esc.requests_issued

    esc2: PremiseScopedEscalation[str] = PremiseScopedEscalation(live_oids=lambda: {1}, poll_s=0.01)
    live = {1}
    esc2.live_oids = lambda: set(live)

    async def kill_it():
        await asyncio.sleep(0.03)
        live.clear()

    killer = asyncio.create_task(kill_it())
    await esc2.ask(lambda: slow("x", 2.0), Premise(oids=frozenset({1})))
    await killer

    assert not_issued == 0, "a premise already dead must not spend a request at all"
    assert esc2.requests_issued == 1, "a cancelled request is a spent request"
    assert esc2.cancelled == 1


@pytest.mark.asyncio
async def test_dead_premise_never_creates_the_coroutine():
    created = {"n": 0}

    def factory():
        created["n"] += 1
        return slow("x", 0.01)

    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(
        live_oids=lambda: set(), poll_s=0.01
    )
    assert await esc.ask(factory, Premise(oids=frozenset({5}))) is None
    assert created["n"] == 0


@pytest.mark.asyncio
async def test_in_flight_request_is_actually_torn_down():
    finished = {"yes": False}

    async def body():
        # No try/except around the sleep: CancelledError propagates on its own,
        # and the assertion below is that `finished` is never reached.
        await asyncio.sleep(2.0)
        finished["yes"] = True
        return "done"

    live = {1}
    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(
        live_oids=lambda: set(live), poll_s=0.01
    )

    async def kill_it():
        await asyncio.sleep(0.03)
        live.clear()

    killer = asyncio.create_task(kill_it())
    assert await esc.ask(body, Premise(oids=frozenset({1}))) is None
    await killer
    await asyncio.sleep(0.05)
    assert finished["yes"] is False, "the request kept running after cancellation"


@pytest.mark.asyncio
async def test_request_exceptions_still_surface():
    async def bad():
        raise ValueError("model returned nonsense")

    esc: PremiseScopedEscalation[str] = PremiseScopedEscalation(live_oids=lambda: {1}, poll_s=0.01)
    with pytest.raises(ValueError):
        await esc.ask(bad, Premise(oids=frozenset({1})))
    assert esc.requests_issued == 1
