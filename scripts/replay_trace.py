#!/usr/bin/env python3
"""Read a JSONL trace back as a human-readable timeline.

A trace file you can replay is worth more than a video: it says what the agent
saw, what it decided, how long each stage took, and — most usefully — what it
*refused* to do and why. This prints it as a narrative, one line per event, with
the frame timings alongside.

    python scripts/replay_trace.py traces/demo.jsonl
    python scripts/replay_trace.py traces/demo.jsonl --only rejection escalation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gielinor.runtime.trace import read_trace, summarise

GLYPH = {
    "run": "==",
    "frame": "  ",
    "action": "->",
    "rejection": "XX",
    "escalation": "^^",
    "llm": "@@",
    "note": "..",
}


def describe(record: dict[str, object]) -> str:
    kind = record.get("kind", "?")
    if kind == "run":
        event = record.get("event")
        if event == "start":
            return (
                f"run start: capture={record.get('capture')} "
                f"actuator={record.get('actuator')} goal={record.get('goal')}"
            )
        return (
            f"run end: {record.get('frames')} frames, "
            f"{record.get('actions_dispatched')} dispatched, "
            f"{record.get('actions_rejected')} rejected, "
            f"{record.get('escalations')} escalations"
        )
    if kind == "frame":
        return (
            f"frame {record.get('seq'):>5} {record.get('fsm_state', '?'):<16} "
            f"objects={record.get('objects', 0):<3} "
            f"total={record.get('total_ms', 0):.2f}ms "
            f"(perceive {record.get('perception_ms', 0):.2f} / plan {record.get('plan_ms', 0):.2f} "
            f"/ dispatch {record.get('dispatch_ms', 0):.2f})"
            + ("  OVER BUDGET" if record.get("over_budget") else "")
        )
    action = record.get("action") or {}
    target = (
        action.get("target_oid")
        or action.get("target_ui")
        or action.get("option_index")
        or action.get("ticks")
    )
    if kind == "action":
        point = record.get("point")
        where = f" at ({point['x']},{point['y']})" if point else ""
        return (
            f"[{record.get('source')}] {action.get('kind')} {target}{where}"
            f"  -- {action.get('reason', '')}"
        )
    if kind == "rejection":
        return (
            f"[{record.get('source')}] REJECTED {action.get('kind')} {target}: "
            f"{record.get('rejected_reason')}"
        )
    if kind == "escalation":
        state = (
            "dispatched"
            if record.get("dispatched")
            else f"skipped ({record.get('skipped_because')})"
        )
        return f"escalate {record.get('reason')} [{state}] -- {record.get('detail', '')}"
    if kind == "llm":
        return (
            f"model {record.get('model')} in {record.get('latency_ms')}ms -> "
            f"{record.get('actions')} action(s); errors={record.get('errors')}"
        )
    return str(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="event kinds to show")
    parser.add_argument("--no-frames", action="store_true", help="hide per-frame timing lines")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    if not args.path.exists():
        print(f"no such trace: {args.path}", file=sys.stderr)
        return 2

    if not args.summary_only:
        for record in read_trace(args.path):
            kind = str(record.get("kind", "?"))
            if args.only and kind not in args.only:
                continue
            if args.no_frames and kind == "frame":
                continue
            print(f"{record.get('t', 0.0):>10.3f} {GLYPH.get(kind, '??')} {describe(record)}")
        print()

    print("summary")
    for key, value in summarise(read_trace(args.path)).items():
        print(f"  {key:24s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
