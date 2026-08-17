"""The fast loop's brain: a deterministic woodcutting state machine.

This is the planner that runs every frame, and by frame count it decides ~95% of
what the agent does. It is a plain FSM with no network, no model and no
allocation of consequence — the whole ``plan()`` call is microseconds, which is
what makes the 50 ms budget in SPEC section 2 achievable.

The behaviour it implements is real woodcutting, not a demo stub:

    BOOT -> FIND_TREE -> CLICK_TREE -> WAIT_ANIMATING -> FIND_TREE -> ...
                              |              |
                              |              +-- inventory full -> INVENTORY_FULL
                              +-- click didn't take (no animation within grace)
                                  -> back to FIND_TREE and pick again

with DIALOGUE pre-empting everything (a dialogue box blocks input, so nothing
else can make progress while one is open) and WALKING when the tree we want is
on the minimap but not in the viewport.

Two subtleties that took the shape of the code:

**Clicks are edge-triggered.** ``CLICK_TREE`` emits a click on *entry* only. A
level-triggered version re-clicks every frame at 15 Hz, which in the client
cancels the previous action and the character never chops anything. The retry
path is deliberately a bounce back through FIND_TREE so the target is re-chosen
against fresh perception rather than re-clicking a tree that may have gone.

**Target choice is total-ordered.** Given the same GameState the same tree is
always picked, because a planner that ties-breaks by dict order makes replay
diffs meaningless.

``escalate()`` implements the four triggers from SPEC section 2 and is the only
place the fast loop hands control up. Note it never *calls* anything: it returns
a reason and the runtime decides whether the slow loop has capacity.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from gielinor.contracts import (
    Action,
    ActionKind,
    DetectedObject,
    GameState,
    Goal,
    Point,
)
from gielinor.planner.escalation import Escalation, EscalationReason
from gielinor.runtime.state_machine import StateMachine, Transition, always

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from gielinor.config import Profile

logger = logging.getLogger(__name__)

TICK_MS = 600.0  # One game tick. Every wait below is expressed in these.


class WoodState(str, Enum):
    BOOT = "boot"
    FIND_TREE = "find_tree"
    CLICK_TREE = "click_tree"
    WAIT_ANIMATING = "wait_animating"
    WALKING = "walking"
    DIALOGUE = "dialogue"
    INVENTORY_FULL = "inventory_full"


# How long a state may last before "stuck" fires. WAIT_ANIMATING is generous
# because chopping a tree legitimately takes tens of seconds; FIND_TREE is not,
# because standing in a forest failing to see a tree means perception is wrong.
STATE_BUDGET_S: dict[WoodState, float] = {
    WoodState.BOOT: 5.0,
    WoodState.FIND_TREE: 12.0,
    WoodState.CLICK_TREE: 4.0,
    WoodState.WAIT_ANIMATING: 90.0,
    WoodState.WALKING: 30.0,
    WoodState.DIALOGUE: 20.0,
    WoodState.INVENTORY_FULL: 10.0,
}

# Grace period between issuing a click and expecting the animation to start:
# server tick + travel + the client's own animation ramp.
CLICK_GRACE_S = 2.4

# Dialogue options the reflex layer is willing to choose without asking a model.
# Anything not matching here is free-form and escalates — the allowlist is small
# on purpose, because "pick the option that looks right" is exactly the decision
# an injected line of dialogue would love to influence.
SAFE_OPTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*yes\b", re.I),
    re.compile(r"^\s*ok(ay)?\b", re.I),
    re.compile(r"^\s*continue\b", re.I),
    re.compile(r"click here to continue", re.I),
)


@dataclass(frozen=True, slots=True)
class ReflexContext:
    """Everything the guards look at, computed once per frame."""

    state: GameState
    goal: Goal
    tree_label: str
    target: DetectedObject | None
    walk_target: DetectedObject | None
    inventory_full: bool
    animating: bool
    moving: bool
    dialogue_open: bool
    dialogue_kind: str
    low_confidence: bool
    time_in_state: float

    @property
    def has_tree(self) -> bool:
        return self.target is not None

    @property
    def has_walk_target(self) -> bool:
        return self.walk_target is not None


@dataclass
class ReflexPlanner:
    """A local, deterministic `Planner`. Sub-millisecond, no network, no surprises."""

    profile: Profile | None = None
    name: str = "reflex"
    min_confidence: float = 0.35
    novelty_confidence: float = 0.55
    stuck_after_s: float | None = None
    click_grace_s: float = CLICK_GRACE_S

    machine: StateMachine[WoodState, ReflexContext] = field(init=False)
    last_escalation: Escalation | None = field(default=None, init=False)
    _last_seq: int | None = field(default=None, init=False)
    _ctx: ReflexContext | None = field(default=None, init=False)
    _full_since_seq: int | None = field(default=None, init=False)
    _escalated_full_seq: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.machine = StateMachine(initial=WoodState.BOOT, transitions=self._transitions())

    # -- FSM table ---------------------------------------------------------

    def _transitions(self) -> list[Transition[WoodState, ReflexContext]]:
        w = WoodState
        return [
            # A dialogue box swallows input; nothing else can make progress.
            Transition(w.DIALOGUE, lambda c: c.dialogue_open, name="dialogue-open"),
            Transition(w.INVENTORY_FULL, lambda c: c.inventory_full, name="inventory-full"),
            # The click landed: the character is chopping.
            Transition(w.WAIT_ANIMATING, lambda c: c.animating, name="animation-started"),
            # Still waiting to find out whether the click landed. Self-transition
            # so the state clock keeps running and `stuck` stays reachable.
            Transition(
                w.CLICK_TREE,
                lambda c: c.time_in_state <= self.click_grace_s,
                sources=(w.CLICK_TREE,),
                name="awaiting-click",
            ),
            Transition(
                w.WALKING,
                lambda c: c.moving and not c.has_tree,
                sources=(w.FIND_TREE, w.WALKING, w.BOOT),
                name="already-walking",
            ),
            # Entry edge for a click. Excludes CLICK_TREE as a source so a retry
            # has to bounce through FIND_TREE and re-choose a target.
            Transition(
                w.CLICK_TREE,
                lambda c: c.has_tree,
                sources=(w.FIND_TREE, w.WAIT_ANIMATING, w.WALKING, w.DIALOGUE, w.BOOT),
                name="tree-in-viewport",
            ),
            Transition(
                w.WALKING,
                lambda c: c.has_walk_target,
                sources=(w.FIND_TREE, w.CLICK_TREE, w.WAIT_ANIMATING, w.DIALOGUE, w.BOOT),
                name="tree-on-minimap",
            ),
            Transition(w.FIND_TREE, always, name="search"),
        ]

    # -- Planner protocol --------------------------------------------------

    def plan(self, state: GameState, goal: Goal) -> Sequence[Action]:
        ctx = self._observe(state, goal)
        now = state.captured_at
        fsm_state = self.machine.step(ctx, now=now)
        entered = self.machine.just_entered

        if fsm_state is WoodState.DIALOGUE:
            return self._plan_dialogue(ctx)
        if fsm_state is WoodState.INVENTORY_FULL:
            return self._plan_inventory_full(ctx)
        if fsm_state is WoodState.CLICK_TREE:
            if entered and ctx.target is not None:
                return (
                    Action(
                        kind=ActionKind.CLICK_OBJECT,
                        target_oid=ctx.target.oid,
                        reason=f"chop {ctx.target.label} (conf {ctx.target.confidence:.2f})",
                    ),
                )
            return (Action(kind=ActionKind.WAIT, ticks=1, reason="waiting for click to register"),)
        if fsm_state is WoodState.WAIT_ANIMATING:
            # Two ticks: chopping is a repeating animation and polling it every
            # frame buys nothing but log noise.
            return (Action(kind=ActionKind.WAIT, ticks=2, reason="chopping"),)
        if fsm_state is WoodState.WALKING:
            if entered and ctx.walk_target is not None:
                return (
                    Action(
                        kind=ActionKind.WALK_MINIMAP,
                        target_oid=ctx.walk_target.oid,
                        reason=f"walk to {ctx.walk_target.label}",
                    ),
                )
            return (Action(kind=ActionKind.WAIT, ticks=2, reason="walking"),)
        # FIND_TREE / BOOT: nothing actionable this frame. Waiting is a real
        # decision here, not a no-op: the next frame may show a tree, and
        # escalate() is what handles it never doing so.
        return (Action(kind=ActionKind.WAIT, ticks=1, reason="no target in view"),)

    def _plan_dialogue(self, ctx: ReflexContext) -> Sequence[Action]:
        dialogue = ctx.state.dialogue
        if dialogue is None:  # pragma: no cover - guard implies non-None
            return (Action(kind=ActionKind.NOOP, reason="dialogue vanished"),)
        if dialogue.kind in ("continue", "level_up", "npc") and not dialogue.options:
            return (
                Action(
                    kind=ActionKind.CONTINUE_DIALOGUE,
                    reason=f"advance {dialogue.kind} dialogue",
                ),
            )
        if dialogue.options:
            index = self._safe_option(dialogue.options)
            if index is not None:
                return (
                    Action(
                        kind=ActionKind.SELECT_DIALOGUE,
                        option_index=index,
                        reason=f"safe option {dialogue.options[index]!r}",
                    ),
                )
        # Free-form: hold still and let escalate() hand it to the slow loop.
        return (Action(kind=ActionKind.WAIT, ticks=1, reason="dialogue needs a decision"),)

    def _plan_inventory_full(self, ctx: ReflexContext) -> Sequence[Action]:
        on_full = ctx.goal.params.get("on_full", "escalate")
        if on_full == "stop":
            return (Action(kind=ActionKind.NOOP, reason="inventory full; goal says stop"),)
        return (Action(kind=ActionKind.WAIT, ticks=1, reason="inventory full; awaiting plan"),)

    @staticmethod
    def _safe_option(options: Sequence[str]) -> int | None:
        for i, option in enumerate(options):
            if any(p.search(option) for p in SAFE_OPTION_PATTERNS):
                return i
        return None

    # -- escalation --------------------------------------------------------

    def escalate(self, state: GameState) -> EscalationReason | None:
        """The four triggers from SPEC section 2, most specific first.

        Ordering is a decision, not an accident: a free-form dialogue while the
        inventory happens to be full is a dialogue problem, and answering "what
        now?" before dismissing the box would produce actions that cannot be
        dispatched because a dialogue is blocking input.

        Safe to call standalone — observation is idempotent per frame seq — but
        the runtime calls ``plan()`` first, so the FSM state is current.
        """
        ctx = self._ctx if (self._ctx and self._last_seq == state.seq) else self._observe(state)
        detail, facts = "", {}

        reason: EscalationReason | None = None
        if ctx.dialogue_open and self._is_freeform(ctx):
            reason = EscalationReason.FREEFORM_DIALOGUE
            dialogue = ctx.state.dialogue
            detail = f"{ctx.dialogue_kind} dialogue needs a reply"
            facts = {
                "speaker": (dialogue.speaker if dialogue and dialogue.speaker else "?"),
                "options": str(len(dialogue.options) if dialogue else 0),
            }
        elif ctx.inventory_full and self._full_is_new():
            reason = EscalationReason.GOAL_TRANSITION
            detail = "inventory filled; the goal needs a next step"
            facts = {"used": str(state.inventory.used)}
        elif self._is_stuck(ctx):
            reason = EscalationReason.STUCK
            elapsed = self.machine.time_in_state(state.captured_at)
            detail = (
                f"{self.machine.state.value} for {elapsed:.1f}s "
                f"(budget {self._budget(self.machine.state):.1f}s)"
            )
            facts = {"fsm_state": self.machine.state.value, "elapsed_s": f"{elapsed:.1f}"}
        elif self._is_novel(ctx):
            reason = EscalationReason.NOVEL_STATE
            detail = (
                f"perception confidence {state.confidence:.2f}"
                if ctx.low_confidence
                else "nothing recognisable in view and no rule applies"
            )
            facts = {"objects": str(len(state.objects)), "confidence": f"{state.confidence:.2f}"}

        if reason is None:
            return None
        self.last_escalation = Escalation(
            reason=reason, detail=detail, seq=state.seq, at=state.captured_at, facts=facts
        )
        logger.debug("escalating: %s", self.last_escalation)
        return reason

    def _is_freeform(self, ctx: ReflexContext) -> bool:
        dialogue = ctx.state.dialogue
        if dialogue is None or not dialogue.is_open:
            return False
        if dialogue.options:
            # Options the allowlist covers are mechanical; anything else is a
            # judgement call, and a judgement call driven by text a stranger
            # wrote is exactly what the slow loop exists to think about.
            return self._safe_option(dialogue.options) is None
        # An NPC's continue-style box is mechanical and handled locally. Another
        # *player* speaking is free-form by definition.
        return dialogue.kind == "player"

    def _full_is_new(self) -> bool:
        """Edge-triggered: a full inventory escalates once, not every frame."""
        already = self._escalated_full_seq
        if already is not None and self._full_since_seq == already:
            return False
        self._escalated_full_seq = self._full_since_seq
        return True

    def _is_stuck(self, ctx: ReflexContext) -> bool:
        # Measured against the machine, not against ctx: escalate() runs after
        # plan() has stepped the FSM, and the context's snapshot still describes
        # the state we were in a moment ago.
        elapsed = self.machine.time_in_state(ctx.state.captured_at)
        return elapsed > self._budget(self.machine.state)

    def _budget(self, state: WoodState) -> float:
        if self.stuck_after_s is not None:
            return self.stuck_after_s
        return STATE_BUDGET_S.get(state, 30.0)

    def _is_novel(self, ctx: ReflexContext) -> bool:
        if ctx.low_confidence:
            return True
        # Nothing detected at all, nothing happening, no dialogue: the rules have
        # nothing to say about this screen, which is the definition of novel.
        return (
            not ctx.state.objects
            and not ctx.animating
            and not ctx.moving
            and not ctx.dialogue_open
            and self.machine.state in (WoodState.FIND_TREE, WoodState.BOOT)
        )

    # -- observation -------------------------------------------------------

    def _observe(self, state: GameState, goal: Goal | None = None) -> ReflexContext:
        """Derive the guard context. Idempotent for a given ``state.seq``."""
        goal = goal or (self._ctx.goal if self._ctx else Goal(name="chop_trees"))
        tree_label = goal.params.get("tree_label", "tree")
        walk_label = goal.params.get("walk_label", tree_label)

        target = self._select_target(state, tree_label, space="viewport")
        walk_target = self._select_target(state, walk_label, space="minimap")
        full = state.inventory.is_full
        if full:
            if self._full_since_seq is None:
                self._full_since_seq = state.seq
        else:
            self._full_since_seq = None
            self._escalated_full_seq = None

        dialogue = state.dialogue
        ctx = ReflexContext(
            state=state,
            goal=goal,
            tree_label=tree_label,
            target=target,
            walk_target=walk_target,
            inventory_full=full,
            animating=state.player.is_animating,
            moving=state.player.is_moving,
            dialogue_open=bool(dialogue and dialogue.is_open),
            dialogue_kind=dialogue.kind if dialogue else "none",
            low_confidence=state.confidence < self.novelty_confidence,
            time_in_state=self.machine.time_in_state(state.captured_at),
        )
        self._ctx = ctx
        self._last_seq = state.seq
        return ctx

    def _select_target(self, state: GameState, label: str, *, space: str) -> DetectedObject | None:
        """Deterministic pick: most confident, then closest to centre, then lowest oid."""
        centre = self.profile.viewport_center() if self.profile is not None else None
        candidates = [
            o
            for o in state.objects
            if o.label == label and o.space == space and o.confidence >= self.min_confidence
        ]
        if not candidates:
            return None

        def sort_key(o: DetectedObject) -> tuple[float, float, int]:
            distance = _distance(o.click_point, centre) if centre is not None else 0.0
            # Confidence rounded before comparison so a 0.0001 difference does
            # not outrank "much closer to the middle of the screen".
            return (-round(o.confidence, 2), distance, o.oid)

        return min(candidates, key=sort_key)

    # -- introspection -----------------------------------------------------

    @property
    def fsm_state(self) -> WoodState:
        return self.machine.state

    def reset(self) -> None:
        self.machine.reset()
        self.last_escalation = None
        self._ctx = None
        self._last_seq = None
        self._full_since_seq = None
        self._escalated_full_seq = None


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


__all__ = [
    "CLICK_GRACE_S",
    "SAFE_OPTION_PATTERNS",
    "STATE_BUDGET_S",
    "ReflexContext",
    "ReflexPlanner",
    "WoodState",
]
