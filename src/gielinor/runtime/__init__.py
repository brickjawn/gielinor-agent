"""Runtime: the orchestrator, the trace writer, and the FSM helper they share.

Exports are lazy, and that is structural rather than an optimisation. The reflex
planner needs ``runtime.state_machine``; the runtime loop needs
``planner.reflex``. An eager ``__init__`` here turns that legitimate two-way
dependency between *packages* into an import cycle between *modules*, because
importing any submodule runs its parent's ``__init__`` first. Resolving names on
demand keeps ``from gielinor.runtime.state_machine import StateMachine`` free of
everything else in the package.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - re-exported lazily via __getattr__
    from gielinor.runtime.loop import (  # noqa: F401
        ActionQueue,
        AgentLoop,
        EscalationRunner,
        InlineEscalationRunner,
        LoopStats,
        NullPerceiver,
        QueuedAction,
        StepReport,
        ThreadEscalationRunner,
        default_perceiver,
    )
    from gielinor.runtime.state_machine import (  # noqa: F401
        StateMachine,
        Transition,
        always,
    )
    from gielinor.runtime.trace import (  # noqa: F401
        TraceWriter,
        read_trace,
        summarise,
    )

_EXPORTS: dict[str, str] = {
    "ActionQueue": "loop",
    "AgentLoop": "loop",
    "EscalationRunner": "loop",
    "InlineEscalationRunner": "loop",
    "LoopStats": "loop",
    "NullPerceiver": "loop",
    "QueuedAction": "loop",
    "StepReport": "loop",
    "ThreadEscalationRunner": "loop",
    "default_perceiver": "loop",
    "StateMachine": "state_machine",
    "Transition": "state_machine",
    "always": "state_machine",
    "TraceWriter": "trace",
    "read_trace": "trace",
    "summarise": "trace",
}


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})


__all__ = sorted(_EXPORTS)
