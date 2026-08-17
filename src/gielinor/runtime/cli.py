"""``gielinor`` command line entry point.

Wires settings -> capture -> perception -> planners -> actuator -> loop, and
nothing else. Every default here is the safe one: the dry-run actuator, no
network, and a replay source rather than a live screen. Turning any of those on
takes an explicit flag, and the flag names say what they do.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gielinor.actuator import open_actuator
from gielinor.capture import CaptureUnavailable, open_capture
from gielinor.config import Settings, load_profile
from gielinor.contracts import Goal
from gielinor.planner.reflex import ReflexPlanner
from gielinor.runtime.loop import AgentLoop, ThreadEscalationRunner, default_perceiver
from gielinor.runtime.trace import TraceWriter, read_trace, summarise

logger = logging.getLogger("gielinor")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gielinor",
        description="Pixels-only agent for a locally-run 2009scape server.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the agent loop")
    run.add_argument("--replay", type=Path, help="directory of PNG frames to replay")
    run.add_argument("--capture", default=None, help="auto|replay|pipewire|screencapturekit")
    run.add_argument("--actuator", default=None, help="dryrun|uinput|quartz (default dryrun)")
    run.add_argument(
        "--allow-real-input",
        action="store_true",
        help="REQUIRED to use a real actuator. Moves your actual mouse and keyboard.",
    )
    run.add_argument("--frames", type=int, default=None, help="stop after N frames")
    run.add_argument("--seconds", type=float, default=None, help="stop after N seconds")
    run.add_argument("--hz", type=float, default=None, help="target fast-loop rate")
    run.add_argument("--goal", default=None, help="goal name (default from settings)")
    run.add_argument("--trace", type=Path, default=None, help="write a JSONL trace here")
    run.add_argument("--profile", default=None, help="calibration profile name")
    run.add_argument("--llm", action="store_true", help="enable the slow loop (needs an API key)")
    run.add_argument("--log-level", default=None)

    show = sub.add_parser("trace", help="summarise a JSONL trace")
    show.add_argument("path", type=Path)

    prof = sub.add_parser("profile", help="print a calibration profile")
    prof.add_argument("name", nargs="?", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    logging.basicConfig(
        level=getattr(args, "log_level", None) or settings.log_level,
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
    )

    if args.command == "trace":
        summary = summarise(read_trace(args.path))
        for key, value in summary.items():
            print(f"{key:24s} {value}")
        return 0

    if args.command == "profile":
        profile = load_profile(args.name or settings.actuator.profile)
        print(profile.model_dump_json(indent=2))
        return 0

    return _run(args, settings)


def _run(args: argparse.Namespace, settings: Settings) -> int:
    profile = load_profile(args.profile or settings.actuator.profile)
    replay_dir = args.replay or settings.capture.replay_dir
    try:
        capture = open_capture(
            args.capture or settings.capture.backend,
            replay_source=replay_dir,
            fps=settings.capture.fps,
        )
    except CaptureUnavailable as exc:
        logger.error("%s", exc)
        return 2

    actuator_name = args.actuator or settings.actuator.backend
    actuator = open_actuator(
        actuator_name,
        profile=profile,
        allow_real_input=args.allow_real_input or settings.actuator.allow_real_input,
    )

    reflex = ReflexPlanner(
        profile=profile,
        min_confidence=settings.actuator.min_object_confidence,
        novelty_confidence=settings.planner.novelty_confidence,
    )

    runner = None
    if args.llm:
        if not settings.has_api_key():
            logger.error("--llm needs OPENROUTER_API_KEY; running reflex-only would be quieter")
            return 2
        from gielinor.planner.llm import LLMPlanner
        from gielinor.planner.openrouter import OpenRouterClient

        secret = settings.openrouter_api_key
        key = secret.get_secret_value() if secret else None
        client = OpenRouterClient(
            api_key=key,
            models=settings.planner.text_models,
            timeout_s=settings.planner.timeout_s,
            max_retries=settings.planner.max_retries,
        )
        runner = ThreadEscalationRunner(
            planner=LLMPlanner(
                client=client,
                text_models=settings.planner.text_models,
                vision_models=settings.planner.vision_models,
                allow_chat=settings.actuator.allow_chat,
                attach_image=settings.planner.attach_image,
            )
        )

    trace_path = args.trace or settings.runtime.trace_path
    trace = TraceWriter(trace_path) if trace_path else None
    goal = Goal(name=args.goal or settings.runtime.goal, params=dict(settings.runtime.goal_params))

    loop = AgentLoop(
        capture=capture,
        actuator=actuator,
        goal=goal,
        perceiver=default_perceiver(),
        reflex=reflex,
        runner=runner,
        trace=trace,
        target_hz=args.hz or settings.runtime.target_hz,
        frame_budget_ms=settings.runtime.frame_budget_ms,
        max_stale_frames=settings.runtime.max_stale_frames,
        max_stale_s=settings.runtime.max_stale_s,
        escalation_cooldown_s=settings.planner.escalation_cooldown_s,
    )
    try:
        with loop:
            stats = loop.run(max_frames=args.frames, max_seconds=args.seconds)
    except KeyboardInterrupt:
        logger.info("interrupted")
        return 130
    except CaptureUnavailable as exc:
        logger.error("%s", exc)
        return 2

    print(
        f"{stats.frames} frames | {stats.mean_frame_ms:.1f} ms mean | "
        f"{stats.actions_dispatched} dispatched | {stats.actions_rejected} rejected | "
        f"{stats.escalations} escalations | {stats.over_budget_frames} over budget"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
