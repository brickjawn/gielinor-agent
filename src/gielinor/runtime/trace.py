"""Structured JSONL trace: what actually happened, frame by frame.

A recorded trace is the most valuable artefact this project produces. A video of
the agent chopping trees proves it worked once; a trace says *why* it worked —
which frames blew the 50 ms budget, how often the reflex layer handled things
alone, what the model was asked and what it answered, and every action that was
refused and on what grounds. It replays, diffs between runs, and survives being
read six months later.

Format is one JSON object per line, because that is greppable, streamable, and
survives a process being killed mid-run — which a single top-level JSON document
would not.

Every record carries ``kind``, ``t`` (monotonic, matching Frame.captured_at) and
``wall`` (unix time, for correlating with anything else on the machine). Values
that cannot be serialised degrade to their repr rather than raising: a tracer
that can crash the run it is observing is worse than no tracer.

Redaction is applied to every record on the way out. The API key should never
reach here in the first place, but "should never" is not an argument, and the
cost of a keyword check per record is nothing next to a leaked credential in a
file people are encouraged to share.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterator

    from gielinor.contracts import Action, ActionResult


logger = logging.getLogger(__name__)

SENSITIVE_SUBSTRINGS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "authorization",
    "auth_token",
    "secret",
    "password",
    "bearer",
    "token",
)
REDACTED = "***redacted***"

TraceKind = str  # "run" | "frame" | "action" | "rejection" | "escalation" | "llm" | "note"


def redact(value: Any, *, depth: int = 0) -> Any:
    """Recursively blank anything whose *key* looks like a credential."""
    if depth > 6:
        return "<max depth>"
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            key = str(k)
            if any(s in key.lower() for s in SENSITIVE_SUBSTRINGS):
                out[key] = REDACTED
            else:
                out[key] = redact(v, depth=depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [redact(v, depth=depth + 1) for v in value]
    return value


@dataclass
class TraceWriter:
    """Append-only JSONL sink. Thread-safe; the slow loop writes from a worker."""

    sink: Path | str | IO[str] | None = None
    flush_every: int = 1
    now: Callable[[], float] = time.monotonic
    run_id: str = ""
    enabled: bool = True

    _fh: IO[str] | None = field(default=None, init=False)
    _owns_fh: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _written: int = field(default=0, init=False)
    counts: Counter[str] = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        if not self.enabled or self.sink is None:
            return
        if isinstance(self.sink, (str, Path)):
            path = Path(self.sink)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("a", encoding="utf-8")
            self._owns_fh = True
        else:
            self._fh = self.sink

    # -- core --------------------------------------------------------------

    def emit(self, kind: TraceKind, **fields: Any) -> dict[str, Any]:
        """Write one record. Returns it, so callers can log or assert on it."""
        record: dict[str, Any] = {
            "kind": kind,
            "t": round(self.now(), 6),
            "wall": round(time.time(), 3),
        }
        if self.run_id:
            record["run"] = self.run_id
        record.update(redact(fields))
        self.counts[kind] += 1
        if self._fh is not None:
            with self._lock:
                # default=str: a stray Point or ndarray degrades to its repr
                # instead of taking down the run it was supposed to document.
                self._fh.write(json.dumps(record, default=str, separators=(",", ":")) + "\n")
                self._written += 1
                if self.flush_every and self._written % self.flush_every == 0:
                    self._fh.flush()
        return record

    def close(self) -> None:
        if self._fh is not None:
            with suppress(ValueError, OSError):
                self._fh.flush()
            if self._owns_fh:
                with suppress(OSError):
                    self._fh.close()
            self._fh = None

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- typed helpers -----------------------------------------------------

    def run_start(self, **meta: Any) -> dict[str, Any]:
        return self.emit("run", event="start", **meta)

    def run_end(self, **meta: Any) -> dict[str, Any]:
        return self.emit("run", event="end", **meta)

    def frame(
        self,
        *,
        seq: int,
        age_ms: float,
        perception_ms: float,
        plan_ms: float,
        dispatch_ms: float,
        total_ms: float,
        fsm_state: str,
        objects: int,
        confidence: float,
        over_budget: bool = False,
        **extra: Any,
    ) -> dict[str, Any]:
        return self.emit(
            "frame",
            seq=seq,
            age_ms=round(age_ms, 3),
            perception_ms=round(perception_ms, 3),
            plan_ms=round(plan_ms, 3),
            dispatch_ms=round(dispatch_ms, 3),
            total_ms=round(total_ms, 3),
            fsm_state=fsm_state,
            objects=objects,
            confidence=round(confidence, 3),
            over_budget=over_budget,
            **extra,
        )

    def action(
        self, result: ActionResult, *, seq: int, source: str = "reflex", **extra: Any
    ) -> dict[str, Any]:
        action = result.action
        kind = "action" if result.accepted else "rejection"
        return self.emit(
            kind,
            seq=seq,
            source=source,
            action=_action_dict(action),
            accepted=result.accepted,
            rejected_reason=result.rejected_reason,
            point=(
                {"x": result.resolved_point.x, "y": result.resolved_point.y}
                if result.resolved_point
                else None
            ),
            **extra,
        )

    def rejection(
        self, action: Action, reason: str, *, seq: int, source: str = "queue", **extra: Any
    ) -> dict[str, Any]:
        return self.emit(
            "rejection",
            seq=seq,
            source=source,
            action=_action_dict(action),
            accepted=False,
            rejected_reason=reason,
            **extra,
        )

    def escalation(
        self, reason: str, *, seq: int, detail: str = "", accepted: bool = True, **extra: Any
    ) -> dict[str, Any]:
        return self.emit(
            "escalation",
            seq=seq,
            reason=reason,
            detail=detail,
            dispatched=accepted,
            **extra,
        )

    def llm(
        self,
        *,
        seq: int,
        model: str = "",
        latency_ms: float = 0.0,
        attempts: int = 0,
        actions: int = 0,
        errors: tuple[str, ...] | list[str] = (),
        rationale: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        return self.emit(
            "llm",
            seq=seq,
            model=model,
            latency_ms=round(latency_ms, 1),
            attempts=attempts,
            actions=actions,
            errors=list(errors),
            rationale=rationale,
            **extra,
        )

    def note(self, message: str, **extra: Any) -> dict[str, Any]:
        return self.emit("note", message=message, **extra)


def _action_dict(action: Action) -> dict[str, Any]:
    kind = action.kind.value if hasattr(action.kind, "value") else str(action.kind)
    out: dict[str, Any] = {"kind": kind, "ticks": action.ticks}
    if action.target_oid is not None:
        out["target_oid"] = action.target_oid
    if action.target_ui is not None:
        out["target_ui"] = action.target_ui
    if action.option_index is not None:
        out["option_index"] = action.option_index
    if action.text is not None:
        out["text"] = action.text
    if action.reason:
        out["reason"] = action.reason
    return out


# --------------------------------------------------------------------------
# Reading traces back
# --------------------------------------------------------------------------


def read_trace(path: Path | str) -> Iterator[dict[str, Any]]:
    """Stream records from a trace file, skipping any truncated final line."""
    with Path(path).open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                logger.warning("trace %s line %d is not valid JSON; skipping", path, lineno)


def summarise(records: Iterator[dict[str, Any]] | list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate a trace into the numbers worth putting in a README.

    Deliberately includes the reflex/LLM split and the budget overrun count:
    those two together are the claim the architecture makes, so they should be
    the easiest numbers in the project to check.
    """
    kinds: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    escalations: Counter[str] = Counter()
    rejections: Counter[str] = Counter()
    totals: list[float] = []
    perception: list[float] = []
    over_budget = 0
    frames = 0

    for record in records:
        kind = str(record.get("kind", "?"))
        kinds[kind] += 1
        if kind == "frame":
            frames += 1
            totals.append(float(record.get("total_ms", 0.0)))
            perception.append(float(record.get("perception_ms", 0.0)))
            if record.get("over_budget"):
                over_budget += 1
        elif kind == "action":
            sources[str(record.get("source", "?"))] += 1
        elif kind == "rejection":
            reason = str(record.get("rejected_reason") or "?").split(":", 1)[0]
            rejections[reason] += 1
        elif kind == "escalation":
            escalations[str(record.get("reason", "?"))] += 1

    ordered = sorted(totals)
    return {
        "frames": frames,
        "records": sum(kinds.values()),
        "kinds": dict(kinds),
        "actions_by_source": dict(sources),
        "escalations_by_reason": dict(escalations),
        "rejections_by_code": dict(rejections),
        "mean_total_ms": round(sum(totals) / len(totals), 2) if totals else 0.0,
        "p95_total_ms": round(ordered[int(0.95 * (len(ordered) - 1))], 2) if ordered else 0.0,
        "max_total_ms": round(max(totals), 2) if totals else 0.0,
        "mean_perception_ms": round(sum(perception) / len(perception), 2) if perception else 0.0,
        "over_budget_frames": over_budget,
    }


__all__ = [
    "REDACTED",
    "SENSITIVE_SUBSTRINGS",
    "TraceWriter",
    "read_trace",
    "redact",
    "summarise",
]
