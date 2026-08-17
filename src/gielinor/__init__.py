"""gielinor-agent: a pixels-only agent for a locally-run 2009scape server.

Everything it knows arrives through a screenshot; everything it does leaves as a
synthetic mouse or key event. No memory reads, no injected plugin, no packet
interception. See SPEC.md for the architecture and README.md for what it is.

Subpackages are exposed lazily. Importing ``gielinor`` must stay cheap and must
not drag in OpenCV, httpx, or a platform SDK — the CLI's ``--help`` should not
cost two seconds, and a machine missing a capture dependency should still be
able to import the package and be told so clearly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_LAZY_SUBMODULES = frozenset({"actuator", "capture", "config", "contracts", "planner", "runtime"})

if TYPE_CHECKING:  # pragma: no cover - import-time cost is the point
    from gielinor import actuator, capture, config, contracts, planner, runtime


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_SUBMODULES})


__all__ = [
    "__version__",
    "actuator",
    "capture",
    "config",
    "contracts",
    "planner",
    "runtime",
]
