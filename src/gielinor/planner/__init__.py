"""Planners: one protocol, two implementations, very different clocks.

``ReflexPlanner`` runs every frame and answers in microseconds. ``LLMPlanner``
runs only when the reflex layer escalates and answers in seconds. They are not
interchangeable and the runtime never treats them as such — see SPEC section 2
and ``runtime.loop``.
"""

from __future__ import annotations

from gielinor.contracts import Planner
from gielinor.planner.escalation import Escalation, EscalationReason
from gielinor.planner.llm import LLMPlanner, ParseResult, parse_actions, render_state
from gielinor.planner.openrouter import (
    DEFAULT_TEXT_MODELS,
    DEFAULT_VISION_MODELS,
    AllModelsFailed,
    AuthError,
    ChatResponse,
    OpenRouterClient,
    OpenRouterError,
)
from gielinor.planner.reflex import ReflexPlanner, WoodState
from gielinor.planner.sanitize import strip_control_chars, wrap_untrusted

__all__ = [
    "DEFAULT_TEXT_MODELS",
    "DEFAULT_VISION_MODELS",
    "AllModelsFailed",
    "AuthError",
    "ChatResponse",
    "Escalation",
    "EscalationReason",
    "LLMPlanner",
    "OpenRouterClient",
    "OpenRouterError",
    "ParseResult",
    "Planner",
    "ReflexPlanner",
    "WoodState",
    "parse_actions",
    "render_state",
    "strip_control_chars",
    "wrap_untrusted",
]
