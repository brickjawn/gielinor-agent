"""Shared contracts for the Gielinor agent.

THIS FILE IS THE INTERFACE BOUNDARY. Every subsystem imports from here and
nothing here imports from a subsystem. If you are working on capture,
perception, planner, or actuator: you may ADD to this file, but changing an
existing field signature breaks the other three subsystems. Don't.

Data flows in exactly one direction:

    Frame -> Perception -> GameState -> Planner -> Action -> Actuator -> OS input

The security-relevant property of this design: `Action` targets are *symbolic*.
A planner (which may be a remote LLM reading untrusted in-game chat text) can
only refer to things perception has already found and labelled, by id. It can
never emit raw screen coordinates, shell strings, or file paths. Coordinate
resolution happens in the actuator, against the same GameState the planner saw.
See docs/adr/0003-untrusted-planner.md.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def center(self) -> Point:
        return Point(self.x + self.w // 2, self.y + self.h // 2)

    def contains(self, p: Point) -> bool:
        return self.x <= p.x < self.x + self.w and self.y <= p.y < self.y + self.h

    def clip_to(self, other: Rect) -> Rect:
        x0 = max(self.x, other.x)
        y0 = max(self.y, other.y)
        x1 = min(self.x + self.w, other.x + other.w)
        y1 = min(self.y + self.h, other.y + other.h)
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured video frame.

    `image` is BGR uint8, shape (H, W, 3) — OpenCV's native layout, so no
    conversion on the hot path. `captured_at` is time.monotonic() sampled as
    close to the compositor handoff as the backend allows; every latency
    number in the system is measured against it.
    """

    image: np.ndarray
    captured_at: float
    seq: int
    backend: str = "unknown"

    @property
    def size(self) -> tuple[int, int]:
        return self.image.shape[1], self.image.shape[0]

    def age_ms(self) -> float:
        return (time.monotonic() - self.captured_at) * 1000.0


@runtime_checkable
class CaptureBackend(Protocol):
    """Platform-specific frame source.

    Implementations: PipeWirePortalCapture (Wayland), ScreenCaptureKitCapture
    (macOS), ReplayCapture (disk, used by every test and by CI).
    """

    name: str

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(self) -> Frame | None:
        """Return the newest frame, or None if none is ready. Non-blocking."""
        ...


# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------

TextSource = Literal[
    "chatbox",  # the scrolling message log
    "dialogue",  # NPC/player dialogue box text
    "dialogue_option",  # a selectable option line
    "overhead",  # text rendered above a player/NPC head
    "tooltip",  # upper-left "Chop down Tree" hover text
    "xp_drop",
    "level_up",
    "unknown",
]


@dataclass(frozen=True, slots=True)
class TextLine:
    """One line of OCR'd game text.

    `text` is UNTRUSTED when source is chatbox/overhead — it may have been
    typed by another player. It is never interpolated into a prompt without
    passing through planner.sanitize.wrap_untrusted(). See ADR 0003.
    """

    text: str
    bbox: Rect
    source: TextSource
    color: tuple[int, int, int] = (255, 255, 255)
    confidence: float = 1.0
    speaker: str | None = None

    @property
    def is_untrusted(self) -> bool:
        return self.source in ("chatbox", "overhead")


@dataclass(frozen=True, slots=True)
class DetectedObject:
    """A clickable thing found in the 3D viewport or on the minimap.

    `oid` is stable across frames for as long as the tracker can associate the
    detection; it is the ONLY handle a planner gets. `click_point` is chosen by
    perception (not the planner) and is already inside the object's clickable
    mask.
    """

    oid: int
    label: str  # "tree", "oak_tree", "npc", "bank_booth", "ground_item"
    bbox: Rect
    click_point: Point
    confidence: float
    space: Literal["viewport", "minimap", "ui"] = "viewport"
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InventorySlot:
    index: int  # 0..27
    occupied: bool
    bbox: Rect
    item_label: str | None = None
    quantity: int | None = None


@dataclass(frozen=True, slots=True)
class InventoryState:
    slots: tuple[InventorySlot, ...] = ()

    @property
    def used(self) -> int:
        return sum(1 for s in self.slots if s.occupied)

    @property
    def is_full(self) -> bool:
        return len(self.slots) == 28 and self.used == 28

    @property
    def free(self) -> int:
        return 28 - self.used


@dataclass(frozen=True, slots=True)
class PlayerState:
    hitpoints: int | None = None
    max_hitpoints: int | None = None
    prayer: int | None = None
    run_energy: int | None = None
    is_animating: bool = False  # derived from viewport pixel churn, not a stat
    is_moving: bool = False


@dataclass(frozen=True, slots=True)
class DialogueState:
    """Set when a dialogue/chat interface is occupying the chatbox."""

    is_open: bool
    kind: Literal["npc", "player", "options", "level_up", "continue", "none"] = "none"
    speaker: str | None = None
    body: tuple[str, ...] = ()
    options: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MinimapState:
    player_world_pos: Point | None = None  # if a position solver is available
    compass_angle: float | None = None
    bbox: Rect | None = None


@dataclass(frozen=True, slots=True)
class GameState:
    """Everything the agent believes about the current frame.

    Produced by perception, consumed by planner and actuator. Immutable so a
    planner call can hold a snapshot while new frames keep arriving.
    """

    seq: int
    captured_at: float
    player: PlayerState = field(default_factory=PlayerState)
    inventory: InventoryState = field(default_factory=InventoryState)
    objects: tuple[DetectedObject, ...] = ()
    text: tuple[TextLine, ...] = ()
    dialogue: DialogueState | None = None
    minimap: MinimapState = field(default_factory=MinimapState)
    confidence: float = 1.0
    perception_ms: float = 0.0

    def object_by_id(self, oid: int) -> DetectedObject | None:
        for o in self.objects:
            if o.oid == oid:
                return o
        return None

    def objects_labelled(self, label: str) -> tuple[DetectedObject, ...]:
        return tuple(o for o in self.objects if o.label == label)


@runtime_checkable
class Perceiver(Protocol):
    """Turns a Frame into a GameState. Must be pure and side-effect free."""

    def perceive(self, frame: Frame, previous: GameState | None = None) -> GameState: ...


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


class ActionKind(str, Enum):
    """The complete set of things the agent is physically able to do.

    This enum is the capability boundary. Adding a member grants the planner —
    including a remote model — a new power over the user's machine. Adding one
    requires a corresponding validator in actuator.validate and a test that
    proves the validator rejects out-of-bounds input.
    """

    CLICK_OBJECT = "click_object"  # target_oid: a DetectedObject in this state
    CLICK_UI = "click_ui"  # target_ui: a name from the UI anchor registry
    SELECT_DIALOGUE = "select_dialogue"  # option_index into dialogue.options
    CONTINUE_DIALOGUE = "continue_dialogue"
    TYPE_CHAT = "type_chat"  # text: length- and charset-capped
    WALK_MINIMAP = "walk_minimap"  # target_oid of a minimap-space object
    WAIT = "wait"  # ticks: 1..50
    NOOP = "noop"


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind
    target_oid: int | None = None
    target_ui: str | None = None
    option_index: int | None = None
    text: str | None = None
    ticks: int = 1
    reason: str = ""  # planner's stated justification, for the trace log

    def __post_init__(self) -> None:
        if self.ticks < 0:
            raise ValueError("ticks must be >= 0")


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: Action
    accepted: bool
    rejected_reason: str | None = None
    dispatched_at: float | None = None
    resolved_point: Point | None = None


# --------------------------------------------------------------------------
# Planning
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Goal:
    """A standing objective, e.g. 'chop willows until inventory full'."""

    name: str
    params: dict[str, str] = field(default_factory=dict)
    description: str = ""


@runtime_checkable
class Planner(Protocol):
    """Chooses the next action.

    Two implementations exist and they are NOT interchangeable in timing:

      ReflexPlanner  — local, deterministic, sub-millisecond, runs every frame.
      LLMPlanner     — remote OpenRouter call, 700ms-6s, runs only when the
                       reflex layer escalates (novel state, dialogue, stuck).

    The runtime always consults the reflex planner first. See ADR 0001.
    """

    name: str

    def plan(self, state: GameState, goal: Goal) -> Sequence[Action]: ...


@runtime_checkable
class Actuator(Protocol):
    """Resolves symbolic actions to OS input events.

    Implementations: UinputActuator (Linux), QuartzActuator (macOS),
    DryRunActuator (logs only — the default, and what tests use).
    """

    name: str

    def execute(self, action: Action, state: GameState) -> ActionResult: ...


__all__ = [
    "Action",
    "ActionKind",
    "ActionResult",
    "Actuator",
    "CaptureBackend",
    "DetectedObject",
    "DialogueState",
    "Frame",
    "GameState",
    "Goal",
    "InventorySlot",
    "InventoryState",
    "MinimapState",
    "Perceiver",
    "Planner",
    "PlayerState",
    "Point",
    "Rect",
    "TextLine",
    "TextSource",
]
