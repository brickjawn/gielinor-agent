"""Pixels-to-GameState perception for the Gielinor agent (workstream C).

Everything in this package is deliberately *local and deterministic*: it runs
inside the 50 ms fast-loop budget described in SPEC.md section 2, so nothing
here may touch the network, spawn a thread pool, or block on I/O.

The public entry point is `PixelPerceiver`, which implements the
`contracts.Perceiver` protocol. The submodules are usable standalone and are
benchmarked individually, because "perception was slow" is a useless bug report
compared to "glyph matching was slow".
"""

from __future__ import annotations

from .animation import AnimationDetector
from .chatbox import ChatboxReader
from .detect import ObjectSignature, ObjectTracker, ViewportDetector
from .fonts import GlyphAtlas, build_atlas_from_frames, load_default_atlas
from .inventory import InventoryReader
from .layout import LayoutProfile, RegionSpec
from .ocr import ColorFilter, GlyphMatcher, GlyphOCR, ShadowFilter, read_line, read_region
from .perceiver import PixelPerceiver

__all__ = [
    "AnimationDetector",
    "ChatboxReader",
    "ColorFilter",
    "GlyphAtlas",
    "GlyphMatcher",
    "GlyphOCR",
    "InventoryReader",
    "LayoutProfile",
    "ObjectSignature",
    "ObjectTracker",
    "PixelPerceiver",
    "RegionSpec",
    "ShadowFilter",
    "ViewportDetector",
    "build_atlas_from_frames",
    "load_default_atlas",
    "read_line",
    "read_region",
]
