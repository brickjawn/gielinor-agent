"""Glyph-exact OCR for RS2 bitmap text (SPEC section 3).

The pipeline is three cheap stages, in this order, and the order is the whole
trick:

  1. `ShadowFilter` — the client draws every glyph twice: once in black at
     (+1,+1), then the glyph itself on top. Shadow pixels are therefore a
     near-perfect *locator*: shift the dark mask up-left by one and you land on
     text, and almost nothing else in a 3-D scene produces that signature.
  2. `ColorFilter` — RS text colour is semantic. Isolating a colour classifies
     the message before a single character is read, and (used as stage 2 of the
     shadow filter) recovers the glyph pixels the shadow shift necessarily
     loses, see `ShadowFilter` for why.
  3. `GlyphMatcher` — exact per-pixel comparison against the atlas. Not
     correlation, not a distance threshold: the renderer does no anti-aliasing,
     so any mismatch is a real mismatch and fuzziness only buys false positives.

Performance notes, since this runs every frame inside a 50 ms budget: columns
of the binary strip are packed into 9-bit integers once per line with numpy,
the first column of each glyph is a dict key into candidate lists, candidates
are tried widest-first (provably safe, see `Glyph`), and decoded lines are
memoised on the packed bits — a chat log is mostly identical frame to frame.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import cv2
import numpy as np

from ..contracts import Rect, TextLine, TextSource
from .fonts import Glyph, GlyphAtlas, load_default_atlas

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Colour registry
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextColor:
    name: str
    rgb: tuple[int, int, int]
    semantic: str
    note: str = ""


# CALIBRATION WARNING. These are *plausible defaults* for a 2009-era client in
# transparent-chat mode, reconstructed from the RS2 palette conventions, not
# measured from the user's install. Interface skin, chat transparency, and the
# 2009scape build all move them. `scripts`-equivalent calibration is:
# screenshot a known message, sample the glyph interior, compare to this table.
# Anything relying on exact equality here should use `ColorFilter(tol=...)`.
RS_TEXT_COLORS: dict[str, TextColor] = {
    "white": TextColor("white", (255, 255, 255), "game", "server/game messages, transparent chat"),
    "black": TextColor("black", (0, 0, 0), "game", "same messages with opaque chat background"),
    "yellow": TextColor("yellow", (255, 255, 0), "trade", "trade/duel requests, ::yellow effect"),
    "red": TextColor("red", (255, 0, 0), "warning", "warnings, ::red effect"),
    "green": TextColor("green", (0, 255, 0), "chat_effect", "::green effect"),
    "cyan": TextColor("cyan", (0, 255, 255), "chat_effect", "::cyan effect"),
    "purple": TextColor("purple", (255, 0, 255), "chat_effect", "::purple effect"),
    "blue": TextColor("blue", (0, 0, 255), "private", "private message body"),
    "clan": TextColor("clan", (127, 0, 127), "clan", "clan chat"),
    "level_up": TextColor("level_up", (0, 0, 128), "level_up", "skill advance notification"),
    "overhead": TextColor("overhead", (255, 255, 0), "overhead", "text above a player's head"),
    "option": TextColor("option", (255, 255, 0), "option", "highlighted dialogue option"),
}


def classify_color(rgb: tuple[int, int, int], tol: int = 24) -> TextColor | None:
    """Nearest registered colour within `tol` (L-inf), or None.

    Returning None matters: an unrecognised colour is a signal that the profile
    needs recalibration, and the perceiver folds it into its confidence.
    """
    best: TextColor | None = None
    best_d = tol + 1
    for tc in RS_TEXT_COLORS.values():
        d = max(abs(int(a) - int(b)) for a, b in zip(tc.rgb, rgb, strict=True))
        if d < best_d:
            best, best_d = tc, d
    return best if best_d <= tol else None


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Binary text mask plus whatever the filter learned about the pixels."""

    mask: np.ndarray  # (h, w) bool
    rgb: tuple[int, int, int] | None = None
    color_name: str | None = None
    semantic: str | None = None
    row_rgb: np.ndarray | None = None  # (h, 3) int16, -1 where no text was found

    @property
    def ink(self) -> int:
        return int(self.mask.sum())

    def rgb_for(self, y0: int, y1: int) -> tuple[int, int, int] | None:
        """Colour of the text on rows [y0, y1].

        A chat log holds several colours at once and colour *is* the message
        type, so a per-region answer would throw away the classification for
        every line but one.
        """
        if self.row_rgb is None:
            return self.rgb
        band = self.row_rgb[y0 : y1 + 1]
        band = band[band[:, 0] >= 0]
        if band.size == 0:
            return self.rgb
        vals, counts = np.unique(band, axis=0, return_counts=True)
        top = vals[int(np.argmax(counts))]
        return int(top[0]), int(top[1]), int(top[2])


@runtime_checkable
class TextFilter(Protocol):
    """Turns a BGR crop into a binary text mask. Stateless and reusable."""

    name: str

    def apply(self, bgr: np.ndarray) -> FilterResult: ...


# Two hot-path shortcuts, both measured rather than assumed on a 500x120 crop:
#   numpy's `arr.max(axis=2)` over a length-3 axis costs 1.66 ms; chaining
#   `np.maximum` over the three channel views costs 0.06 ms.
#   `np.abs(arr.astype(int16) - c).max(axis=2) <= tol` costs 2.17 ms;
#   `cv2.inRange` expresses the identical L-inf ball in 0.05 ms.
# Perception runs this per colour per region per frame, so a 40x factor here is
# the difference between fitting the 50 ms budget and not.


def channel_max(bgr: np.ndarray) -> np.ndarray:
    out: np.ndarray = np.maximum(np.maximum(bgr[:, :, 0], bgr[:, :, 1]), bgr[:, :, 2])
    return out


def within_tol(bgr: np.ndarray, rgb: tuple[int, int, int], tol: int) -> np.ndarray:
    """Boolean mask of pixels within `tol` (L-inf) of `rgb`."""
    b, g, r = int(rgb[2]), int(rgb[1]), int(rgb[0])
    lo = np.array([max(0, b - tol), max(0, g - tol), max(0, r - tol)], dtype=np.uint8)
    hi = np.array([min(255, b + tol), min(255, g + tol), min(255, r + tol)], dtype=np.uint8)
    hit: np.ndarray = cv2.inRange(bgr, lo, hi) != 0
    return hit


def _modal_colors(
    bgr: np.ndarray, mask: np.ndarray, *, top_k: int = 1, min_count: int = 1
) -> list[tuple[tuple[int, int, int], int]]:
    """The `top_k` most common BGR values among masked pixels, as (RGB, count).

    Mode rather than mean: averaging a mixed pixel set gives a colour that
    exists nowhere in the image, which then matches nothing exactly. Top-k
    rather than top-1 because a chat log holds several message colours at once
    and each one is a separate, individually-classifiable message.
    """
    px = bgr[mask]
    if px.size == 0:
        return []
    packed = (px[:, 0].astype(np.uint32) << 16) | (px[:, 1].astype(np.uint32) << 8) | px[:, 2]
    vals, counts = np.unique(packed, return_counts=True)
    order = np.argsort(counts)[::-1][:top_k]
    out: list[tuple[tuple[int, int, int], int]] = []
    for i in order:
        c = int(counts[i])
        if c < min_count:
            break
        v = int(vals[i])
        out.append((((v >> 0) & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF), c))
    return out


def _reconstruct_diagonal(seed: np.ndarray, colour: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """Grow the shadow-derived seed back into the full glyph. Exactly.

    The renderer's model gives an algebraic identity: with G the glyph mask,
    the surviving shadow is `shift(G,+1,+1) \\ G`, so shifting it back yields
    `seed = {p in G : p+(1,1) not in G}`. Equivalently, `p in G \\ seed` iff
    `p+(1,1) in G`. So G is the closure of `seed` under "add p when p+(1,1) is
    already in G and p has the right colour" — a diagonal flood fill along the
    shadow direction, bounded by the glyph height.

    That reconstruction is provably complete (walk down-right from any glyph
    pixel until you leave the glyph; the last pixel inside is a seed, and the
    whole chain back is in G), and far tighter than the obvious alternative of
    keeping whole connected components: a scene pixel that merely *touches* the
    text no longer comes along, it has to sit on an unbroken same-colour
    diagonal chain leading into the glyph.

    `max_iter` is a bound, not a tolerance: no diagonal chain inside a glyph
    can outrun the glyph box height, so 10 settles any 12 px font. The loop
    still exits early when it converges.
    """
    g = seed & colour
    for _ in range(max_iter):
        up = np.zeros_like(g)
        up[:-1, :-1] = g[1:, 1:]
        nxt = g | (up & colour)
        if not (nxt ^ g).any():
            break
        g = nxt
    grown: np.ndarray = g
    return grown


@dataclass(slots=True)
class ShadowFilter:
    """Locate text by its hard 1 px drop shadow, then recover the full glyph.

    Naively, `shift(shadow, -1, -1)` *is* the glyph — except it isn't. The
    client paints the shadow first and the glyph over it, so a shadow pixel
    only survives where the glyph does not cover it. Shifting back therefore
    yields `glyph \\ shift(glyph, -1, -1)`: every glyph pixel whose down-right
    diagonal neighbour is also ink is missing. On thin strokes that is a few
    percent; on a bold font it guts the glyph.

    So the shadow is used for what it is uniquely good at — saying *where* text
    is and *what colour* it is — and the recovered modal colour then does the
    actual segmentation. `require_shadow_support` additionally drops connected
    components with no shadow evidence at all, which is what kills scene pixels
    that happen to match the text colour (green text over grass).
    """

    shadow_max: int = 56  # a "shadow" pixel is this dark in every channel
    text_min: int = 70  # ...and the glyph above it is at least this bright
    color_tol: int = 24  # L-inf tolerance when re-segmenting by recovered colour
    max_colors: int = 6  # distinct text colours to recover per region
    min_color_pixels: int = 6  # absolute floor on a colour's seed count
    min_color_share: float = 0.10  # ...as a share of the *most common* colour
    name: str = "shadow"

    def seed_mask(self, bgr: np.ndarray) -> np.ndarray:
        """Pixels that look like glyph-over-shadow. Cheap, and the whole trick."""
        bright = channel_max(bgr)
        shadow = bright <= self.shadow_max
        seed = np.zeros_like(shadow)
        seed[:-1, :-1] = shadow[1:, 1:]
        seed &= bright >= self.text_min
        return seed

    def apply_multi(self, bgr: np.ndarray) -> list[FilterResult]:
        """One `FilterResult` per recovered text colour, most ink first.

        Splitting by colour before segmenting is what makes a busy background
        survivable: each pass asks "where is the *cyan* text", which a random
        scene answers with a scattering of unconnected pixels, instead of
        "where is text", which it answers with everything.
        """
        if bgr.ndim != 3:
            raise ValueError("ShadowFilter expects an (h, w, 3) BGR image")
        h, w = bgr.shape[:2]
        if h < 2 or w < 2:
            return []

        seed = self.seed_mask(bgr)
        if not seed.any():
            return []

        # Isolated seeds are used for neither colour voting nor reconstruction:
        # scene texture produces plenty of them. Glyph strokes are connected, so
        # a real glyph loses at most its 'i' dot from the *vote*, and the
        # reconstruction puts that back via the full seed set.
        strong = seed & (_neighbour_count(seed) >= 1)
        vote_mask = strong if strong.any() else seed
        votes = _modal_colors(
            bgr, vote_mask, top_k=self.max_colors * 4, min_count=self.min_color_pixels
        )
        if not votes:
            votes = _modal_colors(bgr, seed, top_k=1)
        # Thresholding against the *leading* colour rather than the total is
        # what keeps a one-line cyan message alive in a chatbox dominated by
        # white, while still discarding scene colours: a real message is within
        # an order of magnitude of the busiest message, scene speckle is not.
        if votes:
            floor = max(self.min_color_pixels, int(self.min_color_share * votes[0][1]))
            votes = [v for v in votes if v[1] >= floor]

        out: list[FilterResult] = []
        accepted: list[tuple[int, int, int]] = []
        for rgb, _count in votes:
            if len(accepted) >= self.max_colors:
                break
            # Near-duplicates are one message rendered over a noisy capture, not
            # two messages; a second pass on them costs time and yields a worse
            # copy of a line we already have.
            if any(
                max(abs(a - b) for a, b in zip(rgb, prev, strict=True)) <= self.color_tol
                for prev in accepted
            ):
                continue
            colour = within_tol(bgr, rgb, self.color_tol)
            sub_seed = seed & colour
            box = _bbox(sub_seed)
            if box is None:
                continue
            # Reconstruction only ever grows up-left of a seed, so cropping to
            # the seed's bounding box (plus the glyph-box margin) is lossless
            # and turns a full-region flood fill into a per-message one.
            y0, y1, x0, x1 = box
            y0, x0 = max(0, y0 - _RECON_MARGIN), max(0, x0 - _RECON_MARGIN)
            y1, x1 = min(h - 1, y1), min(w - 1, x1)
            mask = np.zeros((h, w), dtype=bool)
            mask[y0 : y1 + 1, x0 : x1 + 1] = _reconstruct_diagonal(
                seed[y0 : y1 + 1, x0 : x1 + 1], colour[y0 : y1 + 1, x0 : x1 + 1]
            )
            if not mask.any():
                continue
            accepted.append(rgb)
            row_rgb = np.full((h, 3), -1, dtype=np.int16)
            row_rgb[mask.any(axis=1)] = rgb
            tc = classify_color(rgb)
            out.append(
                FilterResult(
                    mask=mask,
                    rgb=rgb,
                    color_name=tc.name if tc else None,
                    semantic=tc.semantic if tc else None,
                    row_rgb=row_rgb,
                )
            )
        out.sort(key=lambda r: r.ink, reverse=True)
        return out

    def apply(self, bgr: np.ndarray) -> FilterResult:
        """Union of every colour pass — for callers that just want "the text"."""
        parts = self.apply_multi(bgr)
        if not parts:
            h, w = bgr.shape[:2]
            return FilterResult(np.zeros((h, w), dtype=bool))
        mask = parts[0].mask.copy()
        row_rgb = parts[0].row_rgb.copy() if parts[0].row_rgb is not None else None
        for p in parts[1:]:
            mask |= p.mask
            if row_rgb is not None and p.row_rgb is not None:
                fill = (row_rgb[:, 0] < 0) & (p.row_rgb[:, 0] >= 0)
                row_rgb[fill] = p.row_rgb[fill]
        return FilterResult(
            mask=mask,
            rgb=parts[0].rgb,
            color_name=parts[0].color_name,
            semantic=parts[0].semantic,
            row_rgb=row_rgb,
        )


_RECON_MARGIN = 12  # glyph-box slack around a seed bbox before reconstruction


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(mask.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])


def _neighbour_count(mask: np.ndarray) -> np.ndarray:
    """4-neighbour ink count per pixel, without a convolution."""
    m = mask.astype(np.uint8)
    n = np.zeros_like(m)
    n[1:, :] += m[:-1, :]
    n[:-1, :] += m[1:, :]
    n[:, 1:] += m[:, :-1]
    n[:, :-1] += m[:, 1:]
    return n


@dataclass(slots=True)
class ColorFilter:
    """Exact or tolerance-based colour isolation, optionally inverted.

    The inverted mode is not symmetry for its own sake: dialogue text is dark
    on a light parchment panel, so "everything that is not the background" is
    the correct segmentation there, while chat text is light on a dark scene.
    """

    rgb: tuple[int, int, int]
    tol: int = 0  # 0 == exact match
    invert: bool = False
    semantic: str | None = None
    name: str = "color"

    @classmethod
    def named(cls, key: str, *, tol: int = 0, invert: bool = False) -> ColorFilter:
        tc = RS_TEXT_COLORS[key]
        return cls(rgb=tc.rgb, tol=tol, invert=invert, semantic=tc.semantic, name=f"color:{key}")

    def apply(self, bgr: np.ndarray) -> FilterResult:
        if bgr.ndim != 3:
            raise ValueError("ColorFilter expects an (h, w, 3) BGR image")
        hit = within_tol(bgr, self.rgb, self.tol)
        mask = ~hit if self.invert else hit
        tc = classify_color(self.rgb)
        return FilterResult(
            mask=mask,
            rgb=self.rgb,
            color_name=tc.name if tc else None,
            semantic=self.semantic or (tc.semantic if tc else None),
        )


@dataclass(slots=True)
class ChainFilter:
    """Intersection of several filters, e.g. shadow-located AND yellow."""

    filters: tuple[TextFilter, ...]
    name: str = "chain"

    def apply(self, bgr: np.ndarray) -> FilterResult:
        out: FilterResult | None = None
        for f in self.filters:
            r = f.apply(bgr)
            if out is None:
                out = r
            else:
                out = FilterResult(
                    mask=out.mask & r.mask,
                    rgb=out.rgb or r.rgb,
                    color_name=out.color_name or r.color_name,
                    semantic=out.semantic or r.semantic,
                )
        if out is None:
            raise ValueError("ChainFilter needs at least one filter")
        return out


# --------------------------------------------------------------------------
# Glyph matching
# --------------------------------------------------------------------------


@dataclass(slots=True)
class MatcherStats:
    decodes: int = 0
    cache_hits: int = 0
    glyph_probes: int = 0
    total_ms: float = 0.0

    @property
    def ms_per_decode(self) -> float:
        return self.total_ms / self.decodes if self.decodes else 0.0


@dataclass(frozen=True, slots=True)
class DecodeResult:
    text: str
    confidence: float
    unknown: int = 0
    x0: int = 0
    x1: int = 0
    y0: int = 0
    y1: int = 0

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class GlyphMatcher:
    """Exact bitmap matcher with a first-column index and a line cache."""

    def __init__(
        self,
        atlas: GlyphAtlas | None = None,
        *,
        space_gap: int = 2,
        cache_size: int = 512,
        max_align_offsets: int = 7,
    ) -> None:
        self.atlas = atlas or load_default_atlas()
        self.space_gap = space_gap
        self.max_align_offsets = max_align_offsets
        self.stats = MatcherStats()
        self._cache: OrderedDict[bytes, DecodeResult] = OrderedDict()
        self._cache_size = cache_size
        self._weights = (1 << np.arange(self.atlas.height, dtype=np.uint32))[:, None]
        # Each glyph is stored as its list of packed column integers alongside
        # the bitmap. Comparing `list[int] == list[int]` is exactly equivalent
        # to comparing the bitmaps (a 9-bit key is a lossless encoding of a
        # column) and roughly 5x faster: numpy's per-call overhead dominates
        # utterly on 9xN arrays, which is what every glyph probe is.
        self._index: dict[int, list[tuple[Glyph, list[int]]]] = {}
        for g in self.atlas.glyphs():
            if g.width == 0:
                continue
            cols = [int(c) for c in (g.mask.astype(np.uint32) * self._weights).sum(axis=0)]
            self._index.setdefault(cols[0], []).append((g, cols))
        # Widest first. Safe because every glyph's last column carries ink, so a
        # wide glyph can never equal a narrow one padded with blanks; and needed
        # because 'r' really is a prefix of 'n' in most bitmap fonts.
        for cands in self._index.values():
            cands.sort(key=lambda gc: gc[0].width, reverse=True)

    # -- public API ---------------------------------------------------------

    def decode_strip(self, mask: np.ndarray) -> DecodeResult:
        """Decode a single line of text from a binary mask.

        `mask` may be taller than the glyph box; the vertical alignment is
        searched, baseline-first, because a line of "hello" has no ascender to
        anchor on and guessing wrong shifts every glyph into nonsense.
        """
        t0 = time.perf_counter()
        mask = np.ascontiguousarray(np.asarray(mask, dtype=bool))
        key = hashlib.blake2b(
            np.packbits(mask).tobytes() + str(mask.shape).encode(), digest_size=16
        ).digest()
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.stats.cache_hits += 1
            self.stats.decodes += 1
            self.stats.total_ms += (time.perf_counter() - t0) * 1000.0
            return hit

        result = self._decode_uncached(mask)

        self._cache[key] = result
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        self.stats.decodes += 1
        self.stats.total_ms += (time.perf_counter() - t0) * 1000.0
        return result

    # -- internals ----------------------------------------------------------

    def _decode_uncached(self, mask: np.ndarray) -> DecodeResult:
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            return DecodeResult("", 0.0)
        r0, r1 = int(rows[0]), int(rows[-1])
        box = self.atlas.height
        total_ink = int(mask.sum())

        # Pack every column of the whole strip into one integer per column, once.
        # A window at offset `t` is then just `col >> t` masked to the box
        # height, so trying several vertical alignments costs a shift instead of
        # a fresh 2-D reduction each time.
        column_bits = self._pack_columns(mask)
        box_mask = (1 << box) - 1

        best: DecodeResult | None = None
        for t in self._alignment_candidates(r0, r1, box):
            if column_bits is not None and t >= 0:
                keys = [(c >> t) & box_mask for c in column_bits]
            else:
                keys = self._pack_columns(self._window(mask, t, box), force=True) or []
            cand = self._walk(keys, total_ink)
            if best is None or (cand.confidence, -cand.unknown) > (best.confidence, -best.unknown):
                best = cand
            if cand.confidence >= 1.0 and cand.unknown == 0:
                break  # a perfect decode cannot be improved on
        if best is None:  # no alignment candidate produced a usable walk
            return DecodeResult("", 0.0, 0, 0, 0, r0, r1)
        return DecodeResult(best.text, best.confidence, best.unknown, best.x0, best.x1, r0, r1)

    @staticmethod
    def _pack_columns(mask: np.ndarray, *, force: bool = False) -> list[int] | None:
        """Each column as an integer, bit i = row i. None if too tall to pack."""
        h = int(mask.shape[0])
        if h > 63 and not force:
            return None
        rows = min(h, 63)
        weights = (1 << np.arange(rows, dtype=np.uint64))[:, None]
        packed = (mask[:rows].astype(np.uint64) * weights).sum(axis=0)
        return [int(v) for v in packed]

    def _alignment_candidates(self, r0: int, r1: int, box: int) -> list[int]:
        """Glyph-box tops worth trying, most likely first.

        Ordering is a real prior, not a guess: ink almost always bottoms out on
        the last cap row (`baseline - 1`) or on the descender floor, so those
        two offsets decode the overwhelming majority of lines on the first try.
        """
        prefer = [r1 - (self.atlas.baseline - 1), r1 - (box - 1)]
        lo = min(r0, r1 - box + 1)
        hi = max(r0, r1 - box + 1)
        rest = [t for t in range(lo, hi + 1) if t not in prefer]
        ordered = [*prefer, *rest]
        seen: set[int] = set()
        out: list[int] = []
        for t in ordered:
            if t in seen:
                continue
            seen.add(t)
            out.append(t)
            if len(out) >= self.max_align_offsets:
                break
        return out

    @staticmethod
    def _window(mask: np.ndarray, top: int, box: int) -> np.ndarray:
        """The `box`-row slice at `top`, zero-padded when it runs off the edge."""
        h, w = mask.shape
        if top >= 0 and top + box <= h:
            return mask[top : top + box]
        out = np.zeros((box, w), dtype=bool)
        src0, src1 = max(0, top), min(h, top + box)
        if src1 > src0:
            out[src0 - top : src1 - top] = mask[src0:src1]
        return out

    def _walk(self, key_list: list[int], total_ink: int) -> DecodeResult:
        """Left-to-right baseline walk over one glyph-box-high strip."""
        w = len(key_list)
        ink = [i for i, k in enumerate(key_list) if k]
        if not ink:
            return DecodeResult("", 0.0)

        space_adv = max(1, self.atlas.space_advance)
        chars: list[str] = []
        matched_ink = 0
        unknown = 0
        x = ink[0]
        x_start = x
        x_end = x
        # Stop at the last inked column. A chat line occupies half its region,
        # and stepping through the blank remainder one column at a time cost as
        # much as the decode itself.
        x_limit = ink[-1] + 1
        pen: int | None = None
        index = self._index

        while x < x_limit:
            if key_list[x] == 0:
                x += 1
                continue
            if pen is not None:
                gap = x - pen
                if gap >= self.space_gap:
                    chars.append(" " * max(1, round(gap / space_adv)))
            matched = False
            for g, gcols in index.get(key_list[x], ()):
                gw = g.width
                if x + gw > w:
                    continue
                self.stats.glyph_probes += 1
                if key_list[x : x + gw] == gcols:
                    chars.append(g.char)
                    matched_ink += g.ink
                    x_end = x + gw
                    x += g.advance
                    pen = x
                    matched = True
                    break
            if not matched:
                unknown += 1
                chars.append("�")
                nxt = next((i for i in range(x + 1, x_limit) if key_list[i]), None)
                if nxt is None:
                    break
                x = nxt
                pen = None

        text = "".join(chars).rstrip()
        coverage = matched_ink / total_ink if total_ink else 0.0
        n = max(1, len(text.replace(" ", "")))
        confidence = max(0.0, min(1.0, coverage * (1.0 - unknown / n)))
        return DecodeResult(text, confidence, unknown, x_start, x_end)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _Band:
    y0: int
    y1: int


class GlyphOCR:
    """Region-level OCR: crop, filter, split into lines, decode each."""

    def __init__(
        self,
        atlas: GlyphAtlas | None = None,
        *,
        matcher: GlyphMatcher | None = None,
        default_filter: TextFilter | None = None,
        min_band_height: int = 3,
        min_line_ink: int = 6,
    ) -> None:
        self.matcher = matcher or GlyphMatcher(atlas)
        self.default_filter: TextFilter = default_filter or ShadowFilter()
        self.min_band_height = min_band_height
        self.min_line_ink = min_line_ink

    # -- public API ---------------------------------------------------------

    def read_line(
        self,
        img: np.ndarray,
        rect: Rect,
        filt: TextFilter | None = None,
        *,
        source: TextSource = "unknown",
    ) -> TextLine:
        """Decode `rect` as a single line of text.

        `source` is not cosmetic. `TextLine.is_untrusted` is derived from it and
        the planner's prompt-injection defence keys off that, so passing the
        default "unknown" for chatbox text would silently downgrade a security
        control. Callers that know the region must say so; `PixelPerceiver`
        drives this from the layout profile precisely so it cannot be forgotten.
        """
        crop, abs_rect = _crop(img, rect)
        if crop.size == 0:
            return TextLine("", abs_rect, source, confidence=0.0)
        best: TextLine | None = None
        for res in _apply_multi(filt or self.default_filter, crop):
            if res.ink < self.min_line_ink:
                continue
            dec = self.matcher.decode_strip(res.mask)
            line = _to_textline(dec, abs_rect, source, res, band=(dec.y0, dec.y1))
            if best is None or _score(line) > _score(best):
                best = line
        return best if best is not None else TextLine("", abs_rect, source, confidence=0.0)

    def read_region(
        self,
        img: np.ndarray,
        rect: Rect,
        filt: TextFilter | None = None,
        *,
        source: TextSource = "unknown",
        max_lines: int = 32,
    ) -> list[TextLine]:
        """Decode every text line inside `rect`, top to bottom.

        Lines are found per colour and then merged, because two colour passes
        can claim the same rows — one of them correctly, the other because a
        scene colour happened to survive reconstruction. The merge keeps the
        better decode rather than emitting both, so a caller never sees the
        same screen line twice.
        """
        crop, abs_rect = _crop(img, rect)
        if crop.size == 0:
            return []
        found: list[tuple[_Band, TextLine]] = []
        for res in _apply_multi(filt or self.default_filter, crop):
            for band in _row_bands(res.mask, self.min_band_height):
                strip = res.mask[band.y0 : band.y1 + 1]
                if int(strip.sum()) < self.min_line_ink:
                    continue
                dec = self.matcher.decode_strip(strip)
                if dec.is_empty:
                    continue
                line_rect = Rect(
                    abs_rect.x, abs_rect.y + band.y0, abs_rect.w, band.y1 - band.y0 + 1
                )
                found.append(
                    (
                        band,
                        _to_textline(
                            dec, line_rect, source, res, y_offset=0, band=(band.y0, band.y1)
                        ),
                    )
                )

        out = _merge_overlapping(found)
        out.sort(key=lambda ln: (ln.bbox.y, ln.bbox.x))
        if len(out) > max_lines:
            logger.debug("read_region hit max_lines=%d in %s", max_lines, rect)
        return out[:max_lines]


def _to_textline(
    dec: DecodeResult,
    rect: Rect,
    source: TextSource,
    res: FilterResult,
    *,
    y_offset: int | None = None,
    band: tuple[int, int] | None = None,
) -> TextLine:
    x0 = rect.x + dec.x0
    y0 = rect.y + (dec.y0 if y_offset is None else y_offset)
    w = max(1, dec.x1 - dec.x0)
    h = max(1, (dec.y1 - dec.y0 + 1) if y_offset is None else rect.h)
    rgb = res.rgb_for(*band) if band is not None else res.rgb
    return TextLine(
        text=dec.text,
        bbox=Rect(x0, y0, w, h),
        source=source,
        color=rgb or (255, 255, 255),
        confidence=dec.confidence,
    )


def _apply_multi(filt: TextFilter, crop: np.ndarray) -> list[FilterResult]:
    """Per-colour results if the filter offers them, else its single result."""
    multi = getattr(filt, "apply_multi", None)
    if callable(multi):
        return list(multi(crop))
    res = filt.apply(crop)
    return [res] if res.ink else []


def _score(line: TextLine) -> tuple[float, int]:
    return (line.confidence, len(line.text.strip()))


def _merge_overlapping(
    found: Sequence[tuple[_Band, TextLine]], overlap: float = 0.5
) -> list[TextLine]:
    """Keep the best decode per screen line when colour passes collide."""
    kept: list[tuple[_Band, TextLine]] = []
    for band, line in sorted(found, key=lambda bl: _score(bl[1]), reverse=True):
        clash = False
        for kb, _ in kept:
            lo, hi = max(band.y0, kb.y0), min(band.y1, kb.y1)
            if hi < lo:
                continue
            shorter = min(band.y1 - band.y0, kb.y1 - kb.y0) + 1
            if (hi - lo + 1) / shorter >= overlap:
                clash = True
                break
        if not clash:
            kept.append((band, line))
    return [ln for _, ln in kept]


def _crop(img: np.ndarray, rect: Rect) -> tuple[np.ndarray, Rect]:
    full = Rect(0, 0, int(img.shape[1]), int(img.shape[0]))
    r = rect.clip_to(full)
    if r.w <= 0 or r.h <= 0:
        return img[0:0, 0:0], r
    return img[r.y : r.y + r.h, r.x : r.x + r.w], r


def _row_bands(mask: np.ndarray, min_height: int) -> list[_Band]:
    """Split a region mask into horizontal text bands separated by blank rows."""
    rows: np.ndarray = np.asarray(mask.any(axis=1))
    bands: list[_Band] = []
    start: int | None = None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if y - start >= min_height:
                bands.append(_Band(start, y - 1))
            start = None
    if start is not None and len(rows) - start >= min_height:
        bands.append(_Band(start, len(rows) - 1))
    return bands


# --------------------------------------------------------------------------
# Module-level convenience (a shared default engine, so the atlas loads once)
# --------------------------------------------------------------------------

_DEFAULT_ENGINE: GlyphOCR | None = None


def default_engine() -> GlyphOCR:
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None:
        _DEFAULT_ENGINE = GlyphOCR()
    return _DEFAULT_ENGINE


def read_line(
    img: np.ndarray,
    rect: Rect,
    filt: TextFilter | None = None,
    *,
    source: TextSource = "unknown",
) -> TextLine:
    return default_engine().read_line(img, rect, filt, source=source)


def read_region(
    img: np.ndarray,
    rect: Rect,
    filt: TextFilter | None = None,
    *,
    source: TextSource = "unknown",
) -> list[TextLine]:
    return default_engine().read_region(img, rect, filt, source=source)


__all__ = [
    "RS_TEXT_COLORS",
    "ChainFilter",
    "ColorFilter",
    "DecodeResult",
    "FilterResult",
    "GlyphMatcher",
    "GlyphOCR",
    "MatcherStats",
    "ShadowFilter",
    "TextColor",
    "TextFilter",
    "classify_color",
    "default_engine",
    "read_line",
    "read_region",
]
