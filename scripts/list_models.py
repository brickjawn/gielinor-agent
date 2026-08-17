#!/usr/bin/env python3
"""Re-check which OpenRouter models are free right now.

SPEC section 1 lists the free slugs verified in Aug 2026 and warns that the
roster churns. This is how you find out what it churned into: the defaults in
``planner.openrouter`` are a starting point, not a promise, and a slug that
retired last week shows up here as simply absent.

    OPENROUTER_API_KEY=... python scripts/list_models.py
    OPENROUTER_API_KEY=... python scripts/list_models.py --vision

Touches the network. Nothing else in the project does by default.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gielinor.planner.openrouter import (
    DEFAULT_TEXT_MODELS,
    DEFAULT_VISION_MODELS,
    AuthError,
    OpenRouterClient,
)


def accepts_images(model: dict) -> bool:
    modalities = (model.get("architecture") or {}).get("input_modalities") or []
    return "image" in modalities


async def run(args: argparse.Namespace) -> int:
    client = OpenRouterClient()
    try:
        async with client:
            models = await client.list_models(free_only=not args.all)
    except AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    configured = set(DEFAULT_TEXT_MODELS) | set(DEFAULT_VISION_MODELS)
    shown = [m for m in models if not args.vision or accepts_images(m)]

    print(f"{'slug':58s} {'ctx':>9s}  img  configured")
    print("-" * 90)
    for model in shown:
        slug = str(model.get("id", "?"))
        ctx = model.get("context_length") or 0
        print(
            f"{slug:58s} {ctx:>9,}  "
            f"{'yes' if accepts_images(model) else ' - ':>3s}  "
            f"{'<-- in use' if slug in configured else ''}"
        )
    print(f"\n{len(shown)} model(s)")

    missing = sorted(configured - {str(m.get("id")) for m in models})
    if missing:
        print("\nconfigured slugs that are NO LONGER in the free roster:")
        for slug in missing:
            print(f"  {slug}")
        print("update planner.openrouter DEFAULT_*_MODELS, or set them via settings.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="include paid models")
    parser.add_argument("--vision", action="store_true", help="only models that accept images")
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
