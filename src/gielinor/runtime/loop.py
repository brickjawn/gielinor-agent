"""The two-loop orchestrator.

SPEC section 2 in code. A game tick is 600 ms; a free-tier OpenRouter round trip
is 700 ms to 6 s. So there are two loops at different clock rates and the whole
design question is how they hand work to each other without the slow one
stalling the fast one or the fast one acting on stale advice.

    fast loop (this thread, ~15 Hz, budget 50 ms)
      capture.grab() -> perceive() -> drain queue -> ReflexPlanner.plan()
                     -> validate -> actuator.execute()
                                          |
                                          | escalate() says: novel / dialogue /
                                          | stuck / goal transition
                                          v
    slow loop (worker, event-driven)   LLMPlanner.aplan() -> Action[] -> QUEUE

The three rules that make this work:

**The fast loop never blocks on the slow one.** Escalation is fire-and-forget:
``submit()`` returns immediately, and if a request is already in flight or the
cooldown has not elapsed, the escalation is *dropped*, not queued. A backlog of
questions about states that no longer exist is worthless.

**Queued actions are revalidated against fresh state.** An action planned
against frame 1400 and arriving at 1460 is checked twice: staleness (frames and
wall clock, since a stalled capture makes those disagree) and full
``validate_action`` against the current GameState. The tree may be gone; the oid
goes with it and the action is dropped with a reason in the trace.

**Queued actions take precedence for the frame they land on**, because they are
the answer to a question the reflex layer could not answer. Everything else is
reflex.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from gielinor.actuator.validate import (
    RejectionReason,
    ValidationPolicy,
    is_stale,
)
from gielinor.contracts import (
    Action,
    ActionResult,
    Actuator,
    CaptureBackend,
    Frame,
    GameState,
    Goal,
    Perceiver,
    PlayerState,
)
from gielinor.planner.escalation import Escalation, EscalationReason
from gielinor.planner.reflex import ReflexPlanner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Sequence

    from gielinor.planner.llm import LLMPlanner
    from gielinor.runtime.trace import TraceWriter

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Queue
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QueuedAction:
    """An action from the slow loop, tagged with the state it was planned against."""

    action: Action
    planned_seq: int
    planned_at: float
    source: str = "llm"
    reason: str = ""
    enqueued_at: float = 0.0


class ActionQueue:
    """Bounded, thread-safe FIFO. Bounded because unbounded means unbounded lag."""

    def __init__(self, maxlen: int = 16) -> None:
        self._items: deque[QueuedAction] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self.dropped_overflow = 0

    def put_many(self, items: Sequence[QueuedAction]) -> None:
        with self._lock:
            for item in items:
                if len(self._items) == self._items.maxlen:
                    self.dropped_overflow += 1
                self._items.append(item)

    def drain(self) -> list[QueuedAction]:
        with self._lock:
            items = list(self._items)
            self._items.clear()
        return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# --------------------------------------------------------------------------
# Slow loop runners
# --------------------------------------------------------------------------


class EscalationRunner(Protocol):
    """How the fast loop reaches the slow one without knowing how it is scheduled."""

    def submit(
        self, state: GameState, goal: Goal, escalation: Escalation, frame: Frame | None
    ) -> bool: ...
    def poll(self) -> list[QueuedAction]: ...
    @property
    def busy(self) -> bool: ...
    def shutdown(self) -> None: ...


@dataclass
class ThreadEscalationRunner:
    """Runs ``LLMPlanner.aplan`` on a single worker thread with its own event loop.

    A thread rather than a task on the main loop because the fast loop is
    synchronous by design — it is a tight sense/act cycle with a hard budget, and
    making it async to accommodate a subsystem that fires once a minute would be
    the tail wagging the dog.

    ``max_workers=1`` is deliberate: one outstanding question at a time. A second
    concurrent escalation would be about a state that the first one's answer is
    about to change.
    """

    planner: LLMPlanner
    on_result: Callable[[Any], None] | None = None

    _pool: ThreadPoolExecutor = field(init=False)
    _future: Future[list[QueuedAction]] | None = field(default=None, init=False)
    _done: deque[QueuedAction] = field(default_factory=deque, init=False)
    submitted: int = field(default=0, init=False)
    completed: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gielinor-slow")

    @property
    def busy(self) -> bool:
        return self._future is not None and not self._future.done()

    def submit(
        self, state: GameState, goal: Goal, escalation: Escalation, frame: Frame | None
    ) -> bool:
        if self.busy:
            return False
        self.submitted += 1
        self._future = self._pool.submit(self._run, state, goal, escalation, frame)
        return True

    def _run(
        self, state: GameState, goal: Goal, escalation: Escalation, frame: Frame | None
    ) -> list[QueuedAction]:
        import asyncio

        actions = asyncio.run(self.planner.aplan(state, goal, escalation=escalation, frame=frame))
        now = time.monotonic()
        return [
            QueuedAction(
                action=a,
                planned_seq=state.seq,
                planned_at=state.captured_at,
                source="llm",
                reason=escalation.reason.value,
                enqueued_at=now,
            )
            for a in actions
        ]

    def poll(self) -> list[QueuedAction]:
        if self._future is None or not self._future.done():
            return []
        future, self._future = self._future, None
        self.completed += 1
        try:
            results = future.result()
        except Exception:
            logger.exception("slow loop raised; dropping this escalation")
            return []
        if self.on_result is not None:
            self.on_result(self.planner)
        return results

    def shutdown(self) -> None:
        # No cancel_futures: an in-flight request is already paid for, and
        # letting it finish keeps the trace honest about what was asked.
        self._pool.shutdown(wait=False)


@dataclass
class InlineEscalationRunner:
    """Runs the planner synchronously at submit time.

    For tests and for the offline demo, where determinism is worth more than
    concurrency. Note this *does* block the fast loop — which is exactly why the
    default is the threaded runner.
    """

    plan_fn: Callable[[GameState, Goal, Escalation, Frame | None], Sequence[Action]]

    _pending: list[QueuedAction] = field(default_factory=list, init=False)
    submitted: int = field(default=0, init=False)
    completed: int = field(default=0, init=False)

    @property
    def busy(self) -> bool:
        return False

    def submit(
        self, state: GameState, goal: Goal, escalation: Escalation, frame: Frame | None
    ) -> bool:
        self.submitted += 1
        now = time.monotonic()
        try:
            actions = self.plan_fn(state, goal, escalation, frame)
        except Exception:
            logger.exception("inline escalation raised; dropping")
            return False
        self._pending.extend(
            QueuedAction(
                action=a,
                planned_seq=state.seq,
                planned_at=state.captured_at,
                source="llm",
                reason=escalation.reason.value,
                enqueued_at=now,
            )
            for a in actions
        )
        self.completed += 1
        return True

    def poll(self) -> list[QueuedAction]:
        items, self._pending = self._pending, []
        return items

    def shutdown(self) -> None:
        self._pending.clear()


# --------------------------------------------------------------------------
# Perception plumbing
# --------------------------------------------------------------------------


class NullPerceiver:
    """A `Perceiver` that reports an empty world.

    Stand-in so the runtime, the CLI and the tests are exercisable without the
    perception subsystem present. It is honest about what it is: confidence 0.0,
    which means the reflex layer immediately classes every frame as NOVEL_STATE
    rather than pretending to see an empty forest.
    """

    name = "null"

    def perceive(self, frame: Frame, previous: GameState | None = None) -> GameState:
        del previous
        return GameState(
            seq=frame.seq,
            captured_at=frame.captured_at,
            player=PlayerState(),
            confidence=0.0,
        )


def default_perceiver(profile_name: str | None = None) -> Perceiver:
    """Import the real perceiver lazily, falling back to the null one.

    Lazy on purpose: perception is a separate workstream with heavier imports
    (glyph atlases, template banks), and neither the tests nor `--dry-run` should
    fail to start because it is mid-write.

    The perceiver is built with the layout derived from the *same* calibration
    profile the actuator resolves click targets against, so there is one set of
    screen coordinates in the process rather than two. If the projection fails
    the perceiver still gets built, on its own defaults — degraded eyes beat no
    eyes — but that path logs loudly, because it means perception and the
    actuator now disagree about where the client is.
    """
    try:
        from gielinor import perception
    except ImportError:
        logger.warning("gielinor.perception is unavailable; using NullPerceiver")
        return NullPerceiver()

    kwargs: dict[str, object] = {}
    try:
        from gielinor.config import DEFAULT_PROFILE
        from gielinor.profiles import layout_for

        kwargs["layout"] = layout_for(profile_name or DEFAULT_PROFILE)
    except Exception:
        logger.warning(
            "could not derive a perception layout from the calibration profile; "
            "perception will use its built-in defaults and may disagree with the "
            "actuator about screen geometry",
            exc_info=True,
        )

    # Tried in order. `PixelPerceiver` is the real implementation; the others are
    # names a later refactor might plausibly use and cost nothing to check.
    for factory_name in ("PixelPerceiver", "build_perceiver", "default_perceiver"):
        factory = getattr(perception, factory_name, None)
        if not callable(factory):
            continue
        try:
            try:
                built = factory(**kwargs)
            except TypeError:
                # Factory does not take a layout; fall back rather than skip it.
                built = factory()
        except Exception:
            # Perception can legitimately fail to construct — no glyph atlas has
            # been built yet, for instance. That should degrade the run to
            # reflex-with-no-eyes, not end it before it starts.
            logger.warning("perception.%s could not be constructed", factory_name, exc_info=True)
            continue
        logger.info("using perception.%s", factory_name)
        return built  # type: ignore[no-any-return]
    logger.warning("gielinor.perception exposes no known factory; using NullPerceiver")
    return NullPerceiver()


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StepReport:
    """One fast-loop iteration, in numbers. Mirrors one `frame` trace record."""

    seq: int
    age_ms: float
    perception_ms: float
    plan_ms: float
    dispatch_ms: float
    total_ms: float
    fsm_state: str
    results: tuple[ActionResult, ...] = ()
    escalation: EscalationReason | None = None
    queued_used: int = 0
    queued_dropped: int = 0

    @property
    def over_budget(self) -> bool:
        return self.total_ms > 50.0


@dataclass
class LoopStats:
    frames: int = 0
    idle_polls: int = 0
    actions_dispatched: int = 0
    actions_rejected: int = 0
    escalations: int = 0
    escalations_dropped: int = 0
    queued_executed: int = 0
    queued_stale: int = 0
    over_budget_frames: int = 0
    total_ms: float = 0.0

    @property
    def mean_frame_ms(self) -> float:
        return self.total_ms / self.frames if self.frames else 0.0


@dataclass
class AgentLoop:
    """Owns the fast loop and the handoff to the slow one."""

    capture: CaptureBackend
    actuator: Actuator
    goal: Goal
    perceiver: Perceiver = field(default_factory=default_perceiver)
    reflex: ReflexPlanner = field(default_factory=ReflexPlanner)
    runner: EscalationRunner | None = None
    trace: TraceWriter | None = None
    policy: ValidationPolicy = field(default_factory=ValidationPolicy)

    target_hz: float = 15.0
    frame_budget_ms: float = 50.0
    max_stale_frames: int = 30
    max_stale_s: float = 3.0
    max_queued_actions: int = 16
    escalation_cooldown_s: float = 3.0
    now: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep

    queue: ActionQueue = field(init=False)
    stats: LoopStats = field(default_factory=LoopStats, init=False)
    previous_state: GameState | None = field(default=None, init=False)
    last_escalation_at: float = field(default=float("-inf"), init=False)
    last_report: StepReport | None = field(default=None, init=False)
    _started: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.queue = ActionQueue(maxlen=self.max_queued_actions)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self.capture.start()
        self._started = True
        if self.trace is not None:
            self.trace.run_start(
                capture=getattr(self.capture, "name", "?"),
                actuator=getattr(self.actuator, "name", "?"),
                perceiver=type(self.perceiver).__name__,
                reflex=self.reflex.name,
                goal=self.goal.name,
                goal_params=dict(self.goal.params),
                target_hz=self.target_hz,
                frame_budget_ms=self.frame_budget_ms,
                slow_loop=type(self.runner).__name__ if self.runner else "disabled",
            )

    def close(self) -> None:
        if self.runner is not None:
            self.runner.shutdown()
        self.capture.stop()
        if self.trace is not None:
            self.trace.run_end(**_stats_dict(self.stats))
            self.trace.close()
        self._started = False

    def __enter__(self) -> AgentLoop:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- one iteration -----------------------------------------------------

    def step(self) -> StepReport | None:
        """Run one fast-loop iteration. Returns None when no frame was ready."""
        if not self._started:
            self.start()
        t_start = self.now()
        frame = self.capture.grab()
        if frame is None:
            self.stats.idle_polls += 1
            return None

        age_ms = (self.now() - frame.captured_at) * 1000.0
        t0 = self.now()
        state = self.perceiver.perceive(frame, self.previous_state)
        perception_ms = (self.now() - t0) * 1000.0
        self.previous_state = state

        # 1. Drain the slow loop's answers into the queue, then spend at most
        #    one of them: they were planned against an older world and each one
        #    executed changes it again.
        if self.runner is not None:
            self.queue.put_many(self.runner.poll())
        results: list[ActionResult] = []
        queued_used, queued_dropped = self._drain_queue(state, results)

        # 2. Reflex planning. Skipped only if the slow loop's answer already
        #    acted this frame.
        t1 = self.now()
        plan_ms = 0.0
        if queued_used == 0:
            actions = self.reflex.plan(state, self.goal)
            plan_ms = (self.now() - t1) * 1000.0
            for action in actions:
                results.append(self._dispatch(action, state, source="reflex"))

        # 3. Escalation check. Never blocks; may be dropped on cooldown or
        #    because the slow loop is still thinking about the last one.
        escalation_reason = self._maybe_escalate(state, frame)

        dispatch_ms = (self.now() - t1) * 1000.0 - plan_ms
        total_ms = (self.now() - t_start) * 1000.0
        report = StepReport(
            seq=state.seq,
            age_ms=age_ms,
            perception_ms=perception_ms,
            plan_ms=plan_ms,
            dispatch_ms=dispatch_ms,
            total_ms=total_ms,
            fsm_state=self.reflex.fsm_state.value,
            results=tuple(results),
            escalation=escalation_reason,
            queued_used=queued_used,
            queued_dropped=queued_dropped,
        )
        self._record(report, state)
        self.last_report = report
        return report

    def _drain_queue(self, state: GameState, results: list[ActionResult]) -> tuple[int, int]:
        """Revalidate queued actions against *current* state; execute at most one."""
        used = 0
        dropped = 0
        for item in self.queue.drain():
            if used >= 1:
                # Anything after the first is re-queued rather than executed
                # back-to-back; the next frame will revalidate it too.
                self.queue.put_many([item])
                continue
            stale = is_stale(
                item.planned_seq,
                state,
                item.planned_at,
                max_frames=self.max_stale_frames,
                max_seconds=self.max_stale_s,
            )
            if stale is not None:
                dropped += 1
                self.stats.queued_stale += 1
                self._reject(item.action, stale, state, source=item.source)
                continue
            result = self._dispatch(item.action, state, source=item.source)
            results.append(result)
            if result.accepted:
                used += 1
                self.stats.queued_executed += 1
            else:
                dropped += 1
        return used, dropped

    def _dispatch(self, action: Action, state: GameState, *, source: str) -> ActionResult:
        result = self.actuator.execute(action, state)
        if result.accepted:
            self.stats.actions_dispatched += 1
        else:
            self.stats.actions_rejected += 1
        if self.trace is not None:
            self.trace.action(result, seq=state.seq, source=source)
        return result

    def _reject(
        self, action: Action, reason: RejectionReason, state: GameState, *, source: str
    ) -> None:
        self.stats.actions_rejected += 1
        logger.info("dropping %s action %s: %s", source, action.kind.value, reason)
        if self.trace is not None:
            self.trace.rejection(action, str(reason), seq=state.seq, source=source)

    def _maybe_escalate(self, state: GameState, frame: Frame) -> EscalationReason | None:
        reason = self.reflex.escalate(state)
        if reason is None:
            return None
        escalation = self.reflex.last_escalation or Escalation(reason=reason, seq=state.seq)
        if self.runner is None:
            self.stats.escalations_dropped += 1
            self._trace_escalation(escalation, state, dispatched=False, why="no slow loop")
            return reason
        now = self.now()
        if now - self.last_escalation_at < self.escalation_cooldown_s:
            self.stats.escalations_dropped += 1
            self._trace_escalation(escalation, state, dispatched=False, why="cooldown")
            return reason
        if self.runner.busy:
            self.stats.escalations_dropped += 1
            self._trace_escalation(escalation, state, dispatched=False, why="slow loop busy")
            return reason
        accepted = self.runner.submit(state, self.goal, escalation, frame)
        if accepted:
            self.last_escalation_at = now
            self.stats.escalations += 1
        else:
            self.stats.escalations_dropped += 1
        self._trace_escalation(
            escalation, state, dispatched=accepted, why="" if accepted else "runner refused"
        )
        return reason

    def _trace_escalation(
        self, escalation: Escalation, state: GameState, *, dispatched: bool, why: str = ""
    ) -> None:
        if self.trace is None:
            return
        self.trace.escalation(
            escalation.reason.value,
            seq=state.seq,
            detail=escalation.detail,
            accepted=dispatched,
            skipped_because=why,
            facts=dict(escalation.facts),
        )

    def _record(self, report: StepReport, state: GameState) -> None:
        self.stats.frames += 1
        self.stats.total_ms += report.total_ms
        over = report.total_ms > self.frame_budget_ms
        if over:
            self.stats.over_budget_frames += 1
            logger.debug(
                "frame %d took %.1fms (budget %.0fms)",
                report.seq,
                report.total_ms,
                self.frame_budget_ms,
            )
        if self.trace is not None:
            self.trace.frame(
                seq=report.seq,
                age_ms=report.age_ms,
                perception_ms=report.perception_ms,
                plan_ms=report.plan_ms,
                dispatch_ms=report.dispatch_ms,
                total_ms=report.total_ms,
                fsm_state=report.fsm_state,
                objects=len(state.objects),
                confidence=state.confidence,
                over_budget=over,
                queued_used=report.queued_used,
                queued_dropped=report.queued_dropped,
            )

    # -- driving -----------------------------------------------------------

    def run(
        self,
        *,
        max_frames: int | None = None,
        max_seconds: float | None = None,
        stop_event: threading.Event | None = None,
        idle_limit: int = 2000,
    ) -> LoopStats:
        """Drive the fast loop until a stop condition fires.

        ``idle_limit`` guards against a source that has quietly stopped
        producing: without it a non-looping replay that has run out would spin
        forever, which is a miserable way to discover your capture died.
        """
        self.start()
        period = 1.0 / self.target_hz if self.target_hz > 0 else 0.0
        deadline = None if max_seconds is None else self.now() + max_seconds
        consecutive_idle = 0
        try:
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if max_frames is not None and self.stats.frames >= max_frames:
                    break
                if deadline is not None and self.now() >= deadline:
                    break
                cycle_start = self.now()
                report = self.step()
                if report is None:
                    consecutive_idle += 1
                    if consecutive_idle >= idle_limit:
                        logger.warning("no frames for %d polls; stopping", consecutive_idle)
                        break
                else:
                    consecutive_idle = 0
                elapsed = self.now() - cycle_start
                if period > elapsed > 0:
                    self.sleep(period - elapsed)
        finally:
            logger.info(
                "loop finished: %d frames, %.1fms mean, %d dispatched, %d rejected, "
                "%d escalations (%d dropped), %d over budget",
                self.stats.frames,
                self.stats.mean_frame_ms,
                self.stats.actions_dispatched,
                self.stats.actions_rejected,
                self.stats.escalations,
                self.stats.escalations_dropped,
                self.stats.over_budget_frames,
            )
        return self.stats


def _stats_dict(stats: LoopStats) -> dict[str, Any]:
    return {
        "frames": stats.frames,
        "idle_polls": stats.idle_polls,
        "actions_dispatched": stats.actions_dispatched,
        "actions_rejected": stats.actions_rejected,
        "escalations": stats.escalations,
        "escalations_dropped": stats.escalations_dropped,
        "queued_executed": stats.queued_executed,
        "queued_stale": stats.queued_stale,
        "over_budget_frames": stats.over_budget_frames,
        "mean_frame_ms": round(stats.mean_frame_ms, 3),
    }


__all__ = [
    "ActionQueue",
    "AgentLoop",
    "EscalationRunner",
    "InlineEscalationRunner",
    "LoopStats",
    "NullPerceiver",
    "QueuedAction",
    "StepReport",
    "ThreadEscalationRunner",
    "default_perceiver",
]
