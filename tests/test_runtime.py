"""Runtime tests: the FSM helper, the trace writer, and the two-loop handoff.

The interesting assertions are in ``TestStaleness`` and ``TestQueueHandoff``:
an action planned against frame N and arriving at frame N+60 must be revalidated
and dropped, and a queued action whose target has since vanished must not be
executed. That is the property SPEC section 2 asks for and the one that a naive
"send screenshot, do what it says" implementation gets wrong.
"""

from __future__ import annotations

import io
import json
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

import numpy as np
import pytest

from conftest import FakePerceiver, synth_frame
from gielinor.actuator.dryrun import DryRunActuator
from gielinor.actuator.validate import ValidationPolicy
from gielinor.capture.replay import ReplayCapture
from gielinor.contracts import (
    Action,
    ActionKind,
    DialogueState,
    Frame,
    GameState,
    Goal,
)
from gielinor.planner.escalation import Escalation, EscalationReason
from gielinor.planner.reflex import ReflexPlanner
from gielinor.runtime.loop import (
    ActionQueue,
    AgentLoop,
    InlineEscalationRunner,
    NullPerceiver,
    QueuedAction,
    ThreadEscalationRunner,
    default_perceiver,
)
from gielinor.runtime.state_machine import StateMachine, Transition, always
from gielinor.runtime.trace import (
    REDACTED,
    TraceWriter,
    read_trace,
    redact,
    summarise,
)

# ==========================================================================
# StateMachine
# ==========================================================================


class Ctx:
    def __init__(self, **kw: bool) -> None:
        self.__dict__.update(kw)

    def __getattr__(self, name: str) -> bool:  # unset flags read as False
        return False


class TestStateMachine:
    def _machine(self) -> StateMachine[str, Ctx]:
        return StateMachine(
            initial="idle",
            transitions=[
                Transition("panic", lambda c: c.alarm, name="alarm"),
                Transition("running", lambda c: c.go, sources=("idle",), name="start"),
                Transition("idle", always, sources=("running",), name="finish"),
            ],
        )

    def test_starts_in_the_initial_state(self) -> None:
        assert self._machine().state == "idle"

    def test_first_matching_transition_wins(self) -> None:
        m = self._machine()
        assert m.step(Ctx(alarm=True, go=True), now=0.0) == "panic"
        assert m.last_transition == "alarm"

    def test_source_filter_is_respected(self) -> None:
        m = self._machine()
        m.force("panic", now=0.0)
        # "start" only fires from idle, so a go=True context cannot escape panic.
        assert m.step(Ctx(go=True), now=1.0) == "panic"

    def test_no_matching_transition_stays_put(self) -> None:
        m = StateMachine(initial="a", transitions=[Transition("b", lambda c: c.never)])
        assert m.step(Ctx(), now=1.0) == "a"
        assert m.just_entered is False

    def test_just_entered_is_an_edge_not_a_level(self) -> None:
        m = self._machine()
        m.step(Ctx(go=True), now=1.0)
        assert m.just_entered is True
        m.step(Ctx(), now=2.0)  # running -> idle
        m.step(Ctx(), now=3.0)  # idle, no transition
        assert m.just_entered is False

    def test_self_transition_does_not_reset_the_state_clock(self) -> None:
        # Stuck detection depends on this: a machine that keeps deciding to wait
        # must still eventually look stuck.
        m = StateMachine(initial="a", transitions=[Transition("a", always, name="stay")])
        m.step(Ctx(), now=1.0)
        m.step(Ctx(), now=9.0)
        assert m.entered_at == 0.0
        assert m.time_in_state(9.0) == pytest.approx(9.0)

    def test_transition_records_history(self) -> None:
        m = self._machine()
        m.step(Ctx(go=True), now=1.0)
        m.step(Ctx(), now=2.0)
        assert [s for s, _ in m.timeline()] == ["idle", "running", "idle"]

    def test_reset_returns_to_the_initial_state(self) -> None:
        m = self._machine()
        m.step(Ctx(alarm=True), now=5.0)
        m.reset(now=0.0)
        assert m.state == "idle"
        assert len(m.timeline()) == 1

    def test_guard_exceptions_propagate(self) -> None:
        def boom(_: Ctx) -> bool:
            raise ZeroDivisionError("bad guard")

        m = StateMachine(initial="a", transitions=[Transition("b", boom)])
        with pytest.raises(ZeroDivisionError):
            m.step(Ctx(), now=0.0)


# ==========================================================================
# TraceWriter
# ==========================================================================


class TestTraceWriter:
    def test_writes_one_json_object_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        with TraceWriter(path, run_id="r1") as trace:
            trace.run_start(capture="replay")
            trace.note("hello", n=1)
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            record = json.loads(line)
            assert {"kind", "t", "wall", "run"} <= set(record)

    def test_records_are_readable_back(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        with TraceWriter(path) as trace:
            trace.note("a")
            trace.note("b")
        assert [r["message"] for r in read_trace(path)] == ["a", "b"]

    def test_a_truncated_final_line_is_skipped_not_fatal(self, tmp_path: Path) -> None:
        # Processes get killed mid-write; the trace has to survive that.
        path = tmp_path / "trace.jsonl"
        path.write_text('{"kind":"note","message":"ok"}\n{"kind":"note","mess')
        assert [r["message"] for r in read_trace(path)] == ["ok"]

    def test_writes_to_an_in_memory_stream(self) -> None:
        buffer = io.StringIO()
        trace = TraceWriter(buffer)
        trace.note("in memory")
        assert json.loads(buffer.getvalue())["message"] == "in memory"

    def test_disabled_writer_is_a_no_op_that_still_returns_records(self) -> None:
        trace = TraceWriter(None)
        record = trace.note("nowhere")
        assert record["message"] == "nowhere"

    def test_action_and_rejection_are_different_kinds(self, make_state, tree) -> None:
        buffer = io.StringIO()
        trace = TraceWriter(buffer)
        actuator = DryRunActuator()
        state = make_state(objects=[tree])
        trace.action(
            actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state), seq=1
        )
        trace.action(actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=99), state), seq=1)
        kinds = [json.loads(line)["kind"] for line in buffer.getvalue().strip().splitlines()]
        assert kinds == ["action", "rejection"]

    def test_action_records_carry_the_resolved_point(self, make_state, tree) -> None:
        buffer = io.StringIO()
        trace = TraceWriter(buffer)
        state = make_state(objects=[tree])
        result = DryRunActuator(seed=2).execute(
            Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state
        )
        record = trace.action(result, seq=3)
        assert record["point"]["x"] == (result.resolved_point.x if result.resolved_point else None)

    def test_non_serialisable_values_degrade_instead_of_raising(self) -> None:
        buffer = io.StringIO()
        TraceWriter(buffer).note("odd", value=np.zeros((2, 2)))
        assert "message" in json.loads(buffer.getvalue())

    def test_counts_by_kind(self) -> None:
        trace = TraceWriter(None)
        trace.note("a")
        trace.note("b")
        trace.escalation("stuck", seq=1)
        assert trace.counts["note"] == 2
        assert trace.counts["escalation"] == 1

    def test_is_thread_safe(self, tmp_path: Path) -> None:
        path = tmp_path / "trace.jsonl"
        trace = TraceWriter(path)

        def worker(n: int) -> None:
            for i in range(50):
                trace.note(f"w{n}-{i}")

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        trace.close()
        records = list(read_trace(path))
        assert len(records) == 200  # no interleaved/torn lines


class TestRedaction:
    @pytest.mark.parametrize(
        "key", ["api_key", "API_KEY", "authorization", "openrouter_api_key", "bearer_token"]
    )
    def test_credential_shaped_keys_are_blanked(self, key: str) -> None:
        assert redact({key: "sk-or-secret"})[key] == REDACTED

    def test_nested_structures_are_walked(self) -> None:
        out = redact({"a": [{"secret": "x"}, {"fine": "y"}]})
        assert out["a"][0]["secret"] == REDACTED
        assert out["a"][1]["fine"] == "y"

    def test_the_writer_redacts_on_the_way_out(self) -> None:
        buffer = io.StringIO()
        TraceWriter(buffer).note("oops", api_key="sk-or-v1-realkey")
        assert "sk-or-v1-realkey" not in buffer.getvalue()

    def test_ordinary_values_are_untouched(self) -> None:
        assert redact({"seq": 5, "label": "tree"}) == {"seq": 5, "label": "tree"}


class TestSummarise:
    def test_aggregates_the_numbers_that_justify_the_architecture(self) -> None:
        records = [
            {"kind": "frame", "total_ms": 10.0, "perception_ms": 4.0, "over_budget": False},
            {"kind": "frame", "total_ms": 80.0, "perception_ms": 60.0, "over_budget": True},
            {"kind": "action", "source": "reflex"},
            {"kind": "action", "source": "reflex"},
            {"kind": "action", "source": "llm"},
            {"kind": "rejection", "rejected_reason": "no_such_object: oid 9 gone"},
            {"kind": "escalation", "reason": "stuck"},
        ]
        summary = summarise(records)
        assert summary["frames"] == 2
        assert summary["over_budget_frames"] == 1
        assert summary["actions_by_source"] == {"reflex": 2, "llm": 1}
        assert summary["rejections_by_code"] == {"no_such_object": 1}
        assert summary["escalations_by_reason"] == {"stuck": 1}
        assert summary["mean_total_ms"] == 45.0

    def test_empty_trace_does_not_divide_by_zero(self) -> None:
        assert summarise([])["frames"] == 0


# ==========================================================================
# ActionQueue
# ==========================================================================


def queued(action: Action, *, seq: int = 0, at: float = 0.0) -> QueuedAction:
    return QueuedAction(action=action, planned_seq=seq, planned_at=at)


class TestActionQueue:
    def test_fifo(self) -> None:
        q = ActionQueue()
        q.put_many([queued(Action(ActionKind.NOOP)), queued(Action(ActionKind.WAIT))])
        assert [i.action.kind for i in q.drain()] == [ActionKind.NOOP, ActionKind.WAIT]

    def test_drain_empties(self) -> None:
        q = ActionQueue()
        q.put_many([queued(Action(ActionKind.NOOP))])
        q.drain()
        assert len(q) == 0

    def test_is_bounded(self) -> None:
        # Unbounded queue means unbounded lag, and every item is advice about a
        # world that no longer exists.
        q = ActionQueue(maxlen=2)
        q.put_many([queued(Action(ActionKind.NOOP)) for _ in range(5)])
        assert len(q) == 2
        assert q.dropped_overflow == 3


# ==========================================================================
# AgentLoop
# ==========================================================================


def build_loop(states: Sequence[GameState], **kw) -> tuple[AgentLoop, DryRunActuator]:
    frames = [synth_frame(i) for i in range(max(3, len(states)))]
    capture = ReplayCapture(frames, fps=15.0, loop=True)
    actuator = DryRunActuator(seed=0)
    loop = AgentLoop(
        capture=capture,
        actuator=actuator,
        goal=kw.pop("goal", Goal(name="chop_trees", params={"tree_label": "tree"})),
        perceiver=FakePerceiver(list(states)),
        reflex=kw.pop("reflex", ReflexPlanner()),
        target_hz=kw.pop("target_hz", 1000.0),
        sleep=kw.pop("sleep", lambda _: None),
        **kw,
    )
    return loop, actuator


class TestAgentLoopBasics:
    def test_a_single_step_produces_a_report(self, make_state, tree) -> None:
        loop, _ = build_loop([make_state(objects=[tree])])
        with loop:
            report = loop.step()
        assert report is not None
        assert report.seq == 0
        assert report.fsm_state == "click_tree"
        assert report.total_ms >= 0

    def test_clicks_the_tree_the_reflex_planner_chose(self, make_state, tree) -> None:
        loop, actuator = build_loop([make_state(objects=[tree])])
        with loop:
            loop.step()
        assert actuator.accepted
        assert actuator.accepted[0].action.target_oid == tree.oid
        assert actuator.accepted[0].resolved_point is not None

    def test_runs_for_a_bounded_number_of_frames(self, make_state, tree) -> None:
        loop, _ = build_loop([make_state(objects=[tree])])
        with loop:
            stats = loop.run(max_frames=10)
        assert stats.frames == 10
        assert stats.actions_dispatched > 0

    def test_returns_none_when_no_frame_is_ready(self, make_state) -> None:
        clock = {"t": 0.0}
        capture = ReplayCapture(
            [synth_frame(0)], fps=10.0, realtime=True, loop=True, now=lambda: clock["t"]
        )
        loop = AgentLoop(
            capture=capture,
            actuator=DryRunActuator(),
            goal=Goal(name="chop_trees"),
            perceiver=NullPerceiver(),
        )
        with loop:
            assert loop.step() is not None
            assert loop.step() is None
        assert loop.stats.idle_polls == 1

    def test_stops_when_the_source_dries_up(self, make_state, tree) -> None:
        capture = ReplayCapture([synth_frame(0)], loop=False)
        loop = AgentLoop(
            capture=capture,
            actuator=DryRunActuator(),
            goal=Goal(name="chop_trees"),
            perceiver=FakePerceiver([make_state(objects=[tree])]),
            target_hz=1000.0,
            sleep=lambda _: None,
        )
        with loop:
            stats = loop.run(max_frames=100, idle_limit=5)
        assert stats.frames == 1

    def test_stop_event_ends_the_run(self, make_state, tree) -> None:
        loop, _ = build_loop([make_state(objects=[tree])])
        event = threading.Event()
        original = loop.step

        def stepping():
            report = original()
            if loop.stats.frames >= 3:
                event.set()
            return report

        loop.step = stepping  # type: ignore[method-assign]
        with loop:
            stats = loop.run(stop_event=event, max_frames=999)
        assert stats.frames == 3

    def test_a_perception_failure_is_not_hidden(self, make_state) -> None:
        class Exploding:
            name = "boom"

            def perceive(self, frame: Frame, previous: GameState | None = None) -> GameState:
                raise RuntimeError("perception exploded")

        loop = AgentLoop(
            capture=ReplayCapture([synth_frame(0)]),
            actuator=DryRunActuator(),
            goal=Goal(name="chop_trees"),
            perceiver=Exploding(),
        )
        with pytest.raises(RuntimeError, match="perception exploded"), loop:
            loop.step()

    def test_budget_overruns_are_counted(self, make_state, tree) -> None:
        ticks = iter([i * 0.1 for i in range(500)])  # 100 ms per call
        loop, _ = build_loop(
            [make_state(objects=[tree])], frame_budget_ms=50.0, now=lambda: next(ticks)
        )
        with loop:
            loop.step()
        assert loop.stats.over_budget_frames == 1

    def test_null_perceiver_reports_no_confidence(self) -> None:
        frame = Frame(image=synth_frame(0), captured_at=1.0, seq=7)
        state = NullPerceiver().perceive(frame)
        assert state.seq == 7
        assert state.confidence == 0.0
        assert state.objects == ()

    def test_default_perceiver_never_explodes(self) -> None:
        # perception/ belongs to another workstream and may not exist yet.
        perceiver = default_perceiver()
        assert hasattr(perceiver, "perceive")


class TestTracing:
    def test_a_full_run_produces_a_readable_trace(self, tmp_path: Path, make_state, tree) -> None:
        path = tmp_path / "run.jsonl"
        loop, _ = build_loop([make_state(objects=[tree])], trace=TraceWriter(path))
        with loop:
            loop.run(max_frames=5)
        records = list(read_trace(path))
        kinds = {r["kind"] for r in records}
        assert {"run", "frame", "action"} <= kinds
        summary = summarise(records)
        assert summary["frames"] == 5
        assert summary["actions_by_source"]["reflex"] > 0

    def test_frame_records_carry_the_timing_breakdown(
        self, tmp_path: Path, make_state, tree
    ) -> None:
        path = tmp_path / "run.jsonl"
        loop, _ = build_loop([make_state(objects=[tree])], trace=TraceWriter(path))
        with loop:
            loop.step()
        frame_record = next(r for r in read_trace(path) if r["kind"] == "frame")
        timings = {"perception_ms", "plan_ms", "dispatch_ms", "total_ms", "age_ms"}
        assert timings <= set(frame_record)
        assert frame_record["fsm_state"] == "click_tree"

    def test_rejections_reach_the_trace(self, tmp_path: Path, make_state) -> None:
        path = tmp_path / "run.jsonl"
        dialogue = DialogueState(is_open=True, kind="options", options=("Yes.",))
        loop, _ = build_loop([make_state(dialogue=dialogue)], trace=TraceWriter(path))
        # SELECT_DIALOGUE resolves against a profile-less actuator, which fails
        # to locate the option and rejects. Whatever the cause, it is on record.
        with loop:
            loop.step()
        records = list(read_trace(path))
        assert any(r["kind"] in ("action", "rejection") for r in records)

    def test_run_end_carries_the_summary_stats(self, tmp_path: Path, make_state, tree) -> None:
        path = tmp_path / "run.jsonl"
        loop, _ = build_loop([make_state(objects=[tree])], trace=TraceWriter(path))
        with loop:
            loop.run(max_frames=3)
        end = [r for r in read_trace(path) if r["kind"] == "run" and r["event"] == "end"]
        assert end and end[0]["frames"] == 3


# ==========================================================================
# The two-loop handoff
# ==========================================================================


class TestQueueHandoff:
    def test_queued_action_is_executed_and_takes_precedence(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        runner = InlineEscalationRunner(
            plan_fn=lambda _s, _g, _e, _f: [
                Action(ActionKind.CLICK_UI, target_ui="inventory_tab", reason="from the model")
            ]
        )
        from gielinor.config import load_profile

        profile = load_profile()
        loop, actuator = build_loop([state], runner=runner)
        actuator.profile = profile
        with loop:
            loop.step()  # escalation may or may not fire; force one
            runner.submit(state, loop.goal, Escalation(reason=EscalationReason.STUCK), None)
            loop.step()
        assert any(r.action.target_ui == "inventory_tab" for r in actuator.accepted)

    def test_only_one_queued_action_runs_per_frame(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        loop, actuator = build_loop([state])
        with loop:
            loop.queue.put_many(
                [
                    QueuedAction(Action(ActionKind.NOOP, reason="a"), 0, 0.0),
                    QueuedAction(Action(ActionKind.NOOP, reason="b"), 0, 0.0),
                ]
            )
            loop.step()
            assert len(loop.queue) == 1
            loop.step()
            assert len(loop.queue) == 0
        reasons = [r.action.reason for r in actuator.accepted]
        assert reasons[:2] == ["a", "b"]

    def test_reflex_is_skipped_on_a_frame_that_spent_a_queued_action(
        self, make_state, tree
    ) -> None:
        loop, actuator = build_loop([make_state(objects=[tree])])
        with loop:
            loop.queue.put_many([QueuedAction(Action(ActionKind.NOOP, reason="model"), 0, 0.0)])
            report = loop.step()
        assert report is not None
        assert report.queued_used == 1
        assert [r.action.reason for r in actuator.accepted] == ["model"]

    def test_escalation_is_dispatched_once_then_held_by_the_cooldown(self, make_state) -> None:
        calls: list[Escalation] = []

        def plan_fn(s, g, e, f):
            calls.append(e)
            return []

        runner = InlineEscalationRunner(plan_fn=plan_fn)
        # confidence 0.0 => every frame is a novel state => escalates constantly
        loop, _ = build_loop(
            [make_state(confidence=0.0)], runner=runner, escalation_cooldown_s=100.0
        )
        with loop:
            loop.run(max_frames=6)
        assert len(calls) == 1
        assert loop.stats.escalations == 1
        assert loop.stats.escalations_dropped == 5

    def test_escalations_are_recorded_even_when_dropped(self, tmp_path: Path, make_state) -> None:
        path = tmp_path / "run.jsonl"
        loop, _ = build_loop([make_state(confidence=0.0)], trace=TraceWriter(path))
        with loop:
            loop.run(max_frames=2)
        escalations = [r for r in read_trace(path) if r["kind"] == "escalation"]
        assert escalations
        assert escalations[0]["dispatched"] is False
        assert escalations[0]["skipped_because"] == "no slow loop"

    def test_the_thread_runner_returns_actions_through_the_queue(self, make_state, tree) -> None:
        class FakeLLM:
            name = "fake-llm"

            async def aplan(self, state, goal, *, escalation=None, frame=None):
                return [Action(ActionKind.NOOP, reason="threaded")]

        runner = ThreadEscalationRunner(planner=FakeLLM())  # type: ignore[arg-type]
        state = make_state(objects=[tree])
        goal = Goal(name="chop_trees")
        assert runner.submit(state, goal, Escalation(reason=EscalationReason.STUCK), None) is True
        deadline = time.monotonic() + 5.0
        results: list[QueuedAction] = []
        while time.monotonic() < deadline and not results:
            results = runner.poll()
            time.sleep(0.005)
        runner.shutdown()
        assert [q.action.reason for q in results] == ["threaded"]
        assert results[0].planned_seq == state.seq
        assert results[0].source == "llm"

    def test_the_thread_runner_refuses_a_second_concurrent_question(self, make_state) -> None:
        started = threading.Event()
        release = threading.Event()

        class SlowLLM:
            name = "slow"

            async def aplan(self, state, goal, *, escalation=None, frame=None):
                started.set()
                release.wait(timeout=5.0)
                return []

        runner = ThreadEscalationRunner(planner=SlowLLM())  # type: ignore[arg-type]
        state = make_state()
        esc = Escalation(reason=EscalationReason.STUCK)
        assert runner.submit(state, Goal(name="g"), esc, None) is True
        started.wait(timeout=5.0)
        assert runner.busy is True
        assert runner.submit(state, Goal(name="g"), esc, None) is False
        release.set()
        runner.shutdown()

    def test_a_raising_slow_loop_does_not_kill_the_fast_loop(self, make_state) -> None:
        class BrokenLLM:
            name = "broken"

            async def aplan(self, state, goal, *, escalation=None, frame=None):
                raise RuntimeError("model exploded")

        runner = ThreadEscalationRunner(planner=BrokenLLM())  # type: ignore[arg-type]
        loop, _ = build_loop([make_state(confidence=0.0)], runner=runner)
        with loop:
            stats = loop.run(max_frames=4)
        assert stats.frames == 4


class TestStaleness:
    def test_an_action_planned_60_frames_ago_is_dropped(self, make_state, tree) -> None:
        # SPEC section 2's example: planned against frame N, arriving at N+60.
        states = [make_state(objects=[tree])]
        loop, actuator = build_loop(states, max_stale_frames=30, max_stale_s=1000.0)
        with loop:
            for _ in range(60):
                loop.step()
            loop.queue.put_many(
                [
                    QueuedAction(
                        Action(ActionKind.NOOP, reason="ancient"), planned_seq=0, planned_at=0.0
                    )
                ]
            )
            report = loop.step()
        assert report is not None
        assert report.queued_dropped == 1
        assert loop.stats.queued_stale == 1
        assert all(r.action.reason != "ancient" for r in actuator.accepted)

    def test_a_recent_action_survives(self, make_state, tree) -> None:
        loop, actuator = build_loop([make_state(objects=[tree])], max_stale_frames=30)
        with loop:
            loop.step()
            loop.queue.put_many(
                [
                    QueuedAction(
                        Action(ActionKind.NOOP, reason="fresh"), planned_seq=1, planned_at=0.06
                    )
                ]
            )
            report = loop.step()
        assert report is not None
        assert report.queued_used == 1
        assert any(r.action.reason == "fresh" for r in actuator.accepted)

    def test_wall_clock_staleness_catches_a_stalled_capture(self, make_state, tree) -> None:
        # Few frames elapsed but a lot of time did: the world still moved on.
        loop, _ = build_loop([make_state(objects=[tree])], max_stale_frames=1000, max_stale_s=0.05)
        with loop:
            for _ in range(3):
                loop.step()
            loop.queue.put_many(
                [
                    QueuedAction(
                        Action(ActionKind.NOOP, reason="old"), planned_seq=2, planned_at=-100.0
                    )
                ]
            )
            report = loop.step()
        assert report is not None
        assert report.queued_dropped == 1

    def test_a_queued_action_whose_target_vanished_is_rejected(self, make_state, tree) -> None:
        # Not stale by frame count, but the tree has been chopped down and its
        # oid went with it. This is why revalidation is against fresh state.
        with_tree = make_state(seq=0, objects=[tree])
        without = make_state(seq=1, objects=[])
        loop, actuator = build_loop([with_tree, without])
        with loop:
            loop.step()  # frame 0: tree present
            loop.queue.put_many(
                [
                    QueuedAction(
                        Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid, reason="stale target"),
                        planned_seq=0,
                        planned_at=0.0,
                    )
                ]
            )
            loop.step()  # frame 1: tree gone
        rejected = [r for r in actuator.rejected if r.action.reason == "stale target"]
        assert rejected
        assert "no_such_object" in (rejected[0].rejected_reason or "")

    def test_stale_drops_are_visible_in_the_trace(self, tmp_path: Path, make_state, tree) -> None:
        path = tmp_path / "run.jsonl"
        loop, _ = build_loop(
            [make_state(objects=[tree])], trace=TraceWriter(path), max_stale_frames=0
        )
        with loop:
            loop.step()
            loop.queue.put_many(
                [
                    QueuedAction(
                        Action(ActionKind.NOOP, reason="ancient"), planned_seq=-99, planned_at=0.0
                    )
                ]
            )
            loop.step()
        stale = [
            r
            for r in read_trace(path)
            if r["kind"] == "rejection" and "stale" in str(r.get("rejected_reason", ""))
        ]
        assert stale
        assert stale[0]["source"] == "llm"


class TestPolicyPlumbing:
    def test_chat_stays_disabled_end_to_end_by_default(self, make_state) -> None:
        loop, actuator = build_loop([make_state()])
        with loop:
            loop.queue.put_many(
                [QueuedAction(Action(ActionKind.TYPE_CHAT, text="hello world"), 0, 0.0)]
            )
            loop.step()
        assert actuator.rejected
        assert "chat_disabled" in (actuator.rejected[0].rejected_reason or "")

    def test_chat_can_be_enabled_deliberately(self, make_state) -> None:
        loop, actuator = build_loop([make_state()])
        actuator.policy = ValidationPolicy(allow_chat=True)
        with loop:
            loop.queue.put_many(
                [QueuedAction(Action(ActionKind.TYPE_CHAT, text="hello world"), 0, 0.0)]
            )
            loop.step()
        assert actuator.accepted
