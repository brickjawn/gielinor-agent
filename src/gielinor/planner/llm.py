"""The slow loop's planner: compact context in, validated Actions out.

Two decisions here matter more than the code.

**It sends text, not pixels, by default.** The naive version of this project
ships a screenshot to a VLM every time. That is slower (vision models on the
free tier are the slow ones), larger (hundreds of KB of base64), and *worse*,
because a model reading a 765x503 JPEG has to re-derive what perception already
knows exactly: which objects exist, what they are, where the inventory stands.
So the default context is a compact text rendering of the GameState — oids,
labels, confidences, counts — and an image is attached only when the escalation
reason is NOVEL_STATE, i.e. precisely when the extracted state is the thing that
failed. See ``EscalationReason.wants_image``.

**The parser assumes the response is hostile.** Not because the model is an
adversary, but because it may have just read a line of chat written by one, and
because free models produce malformed JSON at a rate that would otherwise take
down the run. So parsing is total: every failure mode yields "drop this action,
record why" rather than an exception. In particular it rejects

* unknown ``kind`` values (the closed enum is the capability boundary);
* ``target_oid`` values not present in the state the model was shown;
* any attempt to smuggle coordinates (``x``, ``y``, ``point``, ``screen_x``...),
  which is the shape an injected instruction takes when it wants to click
  something perception never found;
* out-of-range ticks (clamped) and over-long chat text (coerced, then still
  validated downstream).

Whatever survives parsing still goes through ``actuator.validate.validate_action``
against *fresh* state at dispatch. Parsing is a filter, not a guarantee.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gielinor.actuator.validate import MAX_CHAT_LEN, MAX_TICKS, sanitize_chat_text
from gielinor.contracts import Action, ActionKind, Frame, GameState, Goal
from gielinor.planner.escalation import Escalation, EscalationReason
from gielinor.planner.openrouter import (
    DEFAULT_REASONING_MODELS,
    DEFAULT_TEXT_MODELS,
    DEFAULT_VISION_MODELS,
    AllModelsFailed,
    AuthError,
    ChatResponse,
    OpenRouterClient,
    build_image_data_url,
    user_message,
)
from gielinor.planner.prompts import (
    RESPONSE_REMINDER,
    STATE_HEADER,
    SYSTEM_PROMPT,
    build_few_shot_messages,
)
from gielinor.planner.sanitize import make_nonce, wrap_untrusted

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

MAX_ACTIONS = 3
MAX_OBJECTS_IN_PROMPT = 24
MAX_TEXT_LINES_IN_PROMPT = 12

# Keys that indicate the model is trying to express a screen position. There is
# no coordinate field in the schema, so their presence means the response was
# either confused or steered; either way the action is dropped whole rather than
# silently stripped of the offending key.
COORDINATE_KEYS: frozenset[str] = frozenset(
    {
        "x",
        "y",
        "px",
        "py",
        "point",
        "pos",
        "position",
        "coords",
        "coordinates",
        "screen_x",
        "screen_y",
    }
)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


@dataclass(frozen=True, slots=True)
class ParseResult:
    actions: tuple[Action, ...]
    errors: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.actions)


# --------------------------------------------------------------------------
# Context building
# --------------------------------------------------------------------------


def render_state(
    state: GameState,
    goal: Goal,
    escalation: Escalation | None = None,
    *,
    nonce: str | None = None,
    max_objects: int = MAX_OBJECTS_IN_PROMPT,
    max_text: int = MAX_TEXT_LINES_IN_PROMPT,
) -> str:
    """Render a GameState as compact text for the prompt.

    Deliberately not JSON: this is read by a language model, and a flat labelled
    listing costs fewer tokens and confuses small models less than nested JSON.
    Untrusted text is fenced with a single nonce shared by every block in the
    prompt, so the boundary is consistent.
    """
    nonce = nonce or make_nonce()
    params = " ".join(f"{k}={v}" for k, v in sorted(goal.params.items()))
    lines: list[str] = [
        f"{STATE_HEADER}",
        f"GOAL: {goal.name}" + (f" ({params})" if params else ""),
    ]
    if escalation is not None:
        lines.append(f"ESCALATION: {escalation.reason.value} - {escalation.detail}")

    player = state.player
    hp = f"{player.hitpoints}/{player.max_hitpoints}" if player.hitpoints is not None else "unknown"
    lines.append(
        f"PLAYER: animating={str(player.is_animating).lower()} "
        f"moving={str(player.is_moving).lower()} hp={hp}"
    )
    inv = state.inventory
    full = " (FULL)" if inv.is_full else ""
    lines.append(f"INVENTORY: {inv.used}/{len(inv.slots) or 28} used{full}")

    if state.objects:
        lines.append("OBJECTS:")
        # Most confident first and capped: a model given 90 objects picks worse
        # than one given the 24 that matter, and the tail is mostly noise.
        ranked = sorted(state.objects, key=lambda o: (-o.confidence, o.oid))[:max_objects]
        for o in ranked:
            attrs = (
                " " + " ".join(f"{k}={v}" for k, v in sorted(o.attrs.items())) if o.attrs else ""
            )
            lines.append(
                f"  oid={o.oid} label={o.label} space={o.space} conf={o.confidence:.2f}{attrs}"
            )
        if len(state.objects) > max_objects:
            lines.append(f"  ... and {len(state.objects) - max_objects} more, less confident")
    else:
        lines.append("OBJECTS: (none)")

    dialogue = state.dialogue
    if dialogue is not None and dialogue.is_open:
        speaker = dialogue.speaker or "?"
        lines.append(f"DIALOGUE: kind={dialogue.kind} speaker={speaker}")
        if dialogue.body:
            lines.append(
                wrap_untrusted(
                    "\n".join(dialogue.body),
                    nonce=nonce,
                    source="a dialogue box",
                    keep_newlines=True,
                )
            )
        for i, option in enumerate(dialogue.options):
            lines.append(f"  OPTION {i}: {option}")

    untrusted = [t for t in state.text if t.is_untrusted]
    trusted = [t for t in state.text if not t.is_untrusted]
    if trusted:
        lines.append("TEXT:")
        lines.extend(f"  [{t.source}] {t.text}" for t in trusted[:max_text])
    if untrusted:
        body = "\n".join(
            f"{t.speaker + ': ' if t.speaker else ''}{t.text}" for t in untrusted[:max_text]
        )
        lines.append(wrap_untrusted(body, nonce=nonce, source="in-game chat", keep_newlines=True))
    if not state.text:
        lines.append("TEXT: (none)")

    lines.append("")
    lines.append(RESPONSE_REMINDER)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------


def extract_json(text: str) -> Any | None:
    """Find the JSON payload in a model response. Returns None if there is none.

    Handles, in order: a clean document, a ```json fence, and JSON with prose
    wrapped around it (scan for the first balanced object or array). Free models
    do all three despite being asked for the first.
    """
    stripped = text.strip()
    for candidate in _json_candidates(stripped):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _json_candidates(text: str) -> list[str]:
    candidates = [text]
    fence = _FENCE.search(text)
    if fence:
        candidates.append(fence.group(1).strip())
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : i + 1])
                    break
    return candidates


def parse_actions(
    text: str,
    state: GameState,
    *,
    max_actions: int = MAX_ACTIONS,
    allow_chat: bool = False,
) -> ParseResult:
    """Turn a model response into Actions. Never raises; records every rejection."""
    errors: list[str] = []
    payload = extract_json(text)
    if payload is None:
        return ParseResult((), (f"no JSON found in {len(text)}-char response",))

    rationale = ""
    if isinstance(payload, dict):
        rationale = str(payload.get("rationale", ""))[:200]
        raw_actions = payload.get("actions")
        if raw_actions is None:
            # A bare single action object is a common shape; accept it.
            raw_actions = [payload] if "kind" in payload else []
    elif isinstance(payload, list):
        raw_actions = payload
    else:
        return ParseResult((), (f"top-level JSON is {type(payload).__name__}, not object/list",))

    if not isinstance(raw_actions, list):
        return ParseResult((), ("'actions' is not a list",), rationale)

    actions: list[Action] = []
    for i, raw in enumerate(raw_actions):
        if len(actions) >= max_actions:
            errors.append(f"truncated at {max_actions} actions ({len(raw_actions)} offered)")
            break
        action, error = _parse_one(raw, state, index=i, allow_chat=allow_chat)
        if error is not None:
            errors.append(error)
        if action is not None:
            actions.append(action)
    return ParseResult(tuple(actions), tuple(errors), rationale)


def _parse_one(
    raw: Any, state: GameState, *, index: int, allow_chat: bool
) -> tuple[Action | None, str | None]:
    if not isinstance(raw, dict):
        return None, f"action[{index}] is {type(raw).__name__}, not an object"

    smuggled = COORDINATE_KEYS & {str(k).lower() for k in raw}
    if smuggled:
        # The interesting rejection. A planner is not allowed to name a place on
        # the screen, only a thing perception found there.
        return None, (
            f"action[{index}] carries coordinate keys {sorted(smuggled)}; the schema has "
            "no coordinate field and symbolic targets are the whole point"
        )

    kind_raw = raw.get("kind")
    if not isinstance(kind_raw, str):
        return None, f"action[{index}] has no string 'kind'"
    try:
        kind = ActionKind(kind_raw.strip().lower())
    except ValueError:
        return None, f"action[{index}] kind={kind_raw!r} is not in the closed action set"

    reason = str(raw.get("reason", ""))[:160]
    ticks = _clamp_ticks(raw.get("ticks"))

    if kind in (ActionKind.CLICK_OBJECT, ActionKind.WALK_MINIMAP):
        oid = _as_int(raw.get("target_oid"))
        if oid is None:
            return None, f"action[{index}] {kind.value} needs an integer target_oid"
        obj = state.object_by_id(oid)
        if obj is None:
            return None, (
                f"action[{index}] targets oid={oid}, which is not in the state the model "
                f"was shown (seq={state.seq})"
            )
        wanted_space = "minimap" if kind is ActionKind.WALK_MINIMAP else "viewport"
        if obj.space != wanted_space and not (
            kind is ActionKind.CLICK_OBJECT and obj.space == "ui"
        ):
            return None, (f"action[{index}] {kind.value} targets a {obj.space}-space object")
        return Action(kind=kind, target_oid=oid, ticks=ticks, reason=reason), None

    if kind is ActionKind.CLICK_UI:
        target = raw.get("target_ui")
        if not isinstance(target, str) or not target:
            return None, f"action[{index}] click_ui needs a string target_ui"
        return Action(kind=kind, target_ui=target.strip(), ticks=ticks, reason=reason), None

    if kind is ActionKind.SELECT_DIALOGUE:
        idx = _as_int(raw.get("option_index"))
        if idx is None or idx < 0:
            return None, f"action[{index}] select_dialogue needs a non-negative option_index"
        return Action(kind=kind, option_index=idx, ticks=ticks, reason=reason), None

    if kind is ActionKind.TYPE_CHAT:
        if not allow_chat:
            return None, f"action[{index}] type_chat rejected: chat is disabled"
        text = raw.get("text")
        if not isinstance(text, str):
            return None, f"action[{index}] type_chat needs string text"
        cleaned = sanitize_chat_text(text, max_len=MAX_CHAT_LEN)
        if not cleaned:
            return None, f"action[{index}] type_chat text was empty after sanitising"
        return Action(kind=kind, text=cleaned, ticks=ticks, reason=reason), None

    # CONTINUE_DIALOGUE / WAIT / NOOP carry no target.
    return Action(kind=kind, ticks=ticks, reason=reason), None


def _clamp_ticks(value: Any) -> int:
    ticks = _as_int(value)
    if ticks is None:
        return 1
    return max(0, min(MAX_TICKS, ticks))


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


# --------------------------------------------------------------------------
# The planner
# --------------------------------------------------------------------------


@dataclass
class LLMPlanner:
    """A `Planner` backed by OpenRouter. Async-first; runs only on escalation."""

    client: OpenRouterClient
    name: str = "llm"
    max_actions: int = MAX_ACTIONS
    allow_chat: bool = False
    attach_image: bool = True
    image_max_width: int = 765
    text_models: Sequence[str] = DEFAULT_TEXT_MODELS
    vision_models: Sequence[str] = DEFAULT_VISION_MODELS
    reasoning_models: Sequence[str] = DEFAULT_REASONING_MODELS
    include_few_shot: bool = True

    last_response: ChatResponse | None = field(default=None, init=False)
    last_result: ParseResult | None = field(default=None, init=False)
    calls: int = field(default=0, init=False)
    failures: int = field(default=0, init=False)

    async def aplan(
        self,
        state: GameState,
        goal: Goal,
        *,
        escalation: Escalation | None = None,
        frame: Frame | None = None,
    ) -> Sequence[Action]:
        """Ask the model what to do. Returns () on any failure — never raises.

        A failed escalation must not take down the fast loop; the reflex planner
        keeps playing and the trace records that the slow loop came back empty.
        """
        self.calls += 1
        messages = self.build_messages(state, goal, escalation, frame)
        models = self._model_chain(escalation, has_image=_has_image(messages))
        try:
            response = await self.client.chat(messages, models=models)
        except AuthError:
            self.failures += 1
            logger.error("openrouter auth failed; escalations are disabled until the key is fixed")
            return ()
        except AllModelsFailed as exc:
            self.failures += 1
            logger.warning("escalation produced nothing: %s", exc)
            return ()
        except Exception:
            self.failures += 1
            logger.exception("unexpected error during escalation")
            return ()

        self.last_response = response
        result = parse_actions(
            response.text, state, max_actions=self.max_actions, allow_chat=self.allow_chat
        )
        self.last_result = result
        for error in result.errors:
            logger.warning("llm response rejected: %s", error)
        logger.info(
            "llm plan: %d action(s) from %s in %.0fms (%s)",
            len(result.actions),
            response.model,
            response.latency_ms,
            result.rationale[:80] or "no rationale",
        )
        return result.actions

    def plan(self, state: GameState, goal: Goal) -> Sequence[Action]:
        """Synchronous `Planner` conformance. Prefer ``aplan`` in the runtime.

        Exists so ``LLMPlanner`` genuinely satisfies the Protocol rather than
        almost satisfying it. Refuses to run inside a live event loop, where
        blocking on a coroutine would deadlock the loop it was called from.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.aplan(state, goal))
        raise RuntimeError(
            "LLMPlanner.plan() called from a running event loop; await aplan() instead"
        )

    def build_messages(
        self,
        state: GameState,
        goal: Goal,
        escalation: Escalation | None = None,
        frame: Frame | None = None,
    ) -> list[dict[str, Any]]:
        nonce = make_nonce()
        context = render_state(state, goal, escalation, nonce=nonce)
        image_url: str | None = None
        wants_image = escalation is not None and escalation.reason.wants_image
        if self.attach_image and wants_image and frame is not None:
            try:
                image_url = build_image_data_url(frame.image, max_width=self.image_max_width)
            except (ValueError, ImportError):
                logger.warning("could not encode the frame; sending text only", exc_info=True)

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.include_few_shot:
            messages.extend(build_few_shot_messages())
        messages.append(user_message(context, image_url))
        return messages

    def _model_chain(self, escalation: Escalation | None, *, has_image: bool) -> Sequence[str]:
        if has_image:
            return self.vision_models
        if escalation is not None and escalation.reason.wants_reasoning_model:
            return self.reasoning_models
        return self.text_models


def _has_image(messages: Sequence[dict[str, Any]]) -> bool:
    content = messages[-1].get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content
    )


__all__ = [
    "COORDINATE_KEYS",
    "MAX_ACTIONS",
    "EscalationReason",
    "LLMPlanner",
    "ParseResult",
    "extract_json",
    "parse_actions",
    "render_state",
]
