"""Escalation vocabulary shared by the reflex planner and the runtime.

``EscalationReason`` is the four triggers from SPEC section 2, as an enum. It
lives here rather than in ``contracts.py`` because it is planner-internal: the
actuator never sees one, and nothing outside ``planner`` and ``runtime`` needs
to name one. (``contracts.py`` is owned by the lead and does not define it.)

The reason matters downstream, not just as a label:

* it decides whether the slow loop sends a **screenshot** — only NOVEL_STATE, by
  definition the case where extracted text and labels were not enough;
* it selects the prompt template;
* it picks the model chain (reasoning-heavy models cost seconds, so they are
  reserved for the cases that plausibly need reasoning);
* it is the primary grouping key when reading a trace afterwards — "how often
  did we escalate, and why" is the single most informative number about whether
  the reflex layer is carrying its share.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EscalationReason(str, Enum):
    NOVEL_STATE = "novel_state"
    FREEFORM_DIALOGUE = "freeform_dialogue"
    STUCK = "stuck"
    GOAL_TRANSITION = "goal_transition"

    @property
    def wants_image(self) -> bool:
        """Only send pixels when the extracted state is what failed.

        A screenshot costs a vision-capable model (slower, fewer free slugs) and
        tens of kilobytes of base64. For dialogue, stuck and goal transitions,
        perception has already produced everything the model can use.
        """
        return self is EscalationReason.NOVEL_STATE

    @property
    def wants_reasoning_model(self) -> bool:
        return self in (EscalationReason.STUCK, EscalationReason.GOAL_TRANSITION)


@dataclass(frozen=True, slots=True)
class Escalation:
    """A trigger plus enough context to explain it in the trace."""

    reason: EscalationReason
    detail: str = ""
    seq: int = -1
    at: float = 0.0
    facts: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.reason.value}({self.detail})" if self.detail else self.reason.value


__all__ = ["Escalation", "EscalationReason"]
