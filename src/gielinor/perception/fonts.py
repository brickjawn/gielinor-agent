"""Bitmap glyph atlas: character -> 1-bit mask + advance width.

Why a bespoke atlas instead of a font file: the OCR in `ocr.py` does *exact*
per-pixel matching (SPEC section 3), which only works if the atlas holds the
literal pixels the client rasterises. A TTF plus a rasteriser reintroduces
hinting, subpixel positioning and anti-aliasing — the three things that make
general OCR fail on this input in the first place.

Two atlas sources exist and they must never be confused:

  * `build_synthetic_atlas()` — procedurally generated, ships in the repo as
    `assets/fonts/synthetic_p12.npz`. It is *RS-like*, not RS: 1-bit, no
    anti-aliasing, hard 1 px edges, variable advance widths, ~12 px line box.
    It exists so the whole perception stack has deterministic, labelled input
    with perfect ground truth and zero third-party assets.
  * `build_atlas_from_frames()` — derives the real atlas from frames the *user*
    captures from their own client. We do not ship Jagex font data; see that
    function's docstring.

Storage format is a single compressed npz: all glyph masks packed side by side
into one 2-D array plus offset/width/advance vectors and a JSON metadata blob.
One file, no pickle, loads in ~1 ms.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Synthetic font design grid
# --------------------------------------------------------------------------
#
# Glyphs are authored on a 5x7 grid (the classic pixel-font cell), then trimmed
# horizontally so advance widths vary the way a real proportional bitmap font's
# do. Descender glyphs are shifted down two rows, which is what makes the box
# 9 rows tall while cap height stays 7. Line height is 12 to match RS p12.

DESIGN_W = 5
DESIGN_H = 7
DESCENDER_DROP = 2
GLYPH_H = DESIGN_H + DESCENDER_DROP  # 9: the vertical box every glyph shares
BASELINE = DESIGN_H  # first row below the cap band
LINE_HEIGHT = 12
SPACE_ADVANCE = 4
SIDE_BEARING = 1  # blank columns between adjacent glyphs
DESCENDERS = frozenset("gjpqy")

_ON = "#"

# One entry per printable ASCII codepoint, 7 rows of 5 columns, '/'-separated.
# Authored so that no two glyphs share a bitmap once placed in the 9-row box —
# `GlyphAtlas.duplicate_pairs()` proves it and a test asserts it, because two
# identical glyphs would make exact matching ambiguous rather than merely wrong.
_DESIGN: dict[str, str] = {
    " ": "...../...../...../...../...../...../.....",
    "!": "..#../..#../..#../..#../..#../...../..#..",
    '"': ".#.#./.#.#./...../...../...../...../.....",
    "#": ".#.#./.#.#./#####/.#.#./#####/.#.#./.#.#.",
    "$": "..#../.####/#.#../.###./..#.#/####./..#..",
    "%": "##.../##..#/...#./..#../.#.../#..##/...##",
    "&": ".##../#..#./#.#../.#.../#.#.#/#..#./.##.#",
    "'": "..#../..#../...../...../...../...../.....",
    "(": "...#./..#../.#.../.#.../.#.../..#../...#.",
    ")": ".#.../..#../...#./...#./...#./..#../.#...",
    "*": "...../#.#.#/.###./#####/.###./#.#.#/.....",
    "+": "...../..#../..#../#####/..#../..#../.....",
    ",": "...../...../...../...../.##../.##../.#...",
    "-": "...../...../...../#####/...../...../.....",
    ".": "...../...../...../...../...../.##../.##..",
    "/": "....#/...#./..#../..#../.#.../#..../#....",
    "0": ".###./#...#/#..##/#.#.#/##..#/#...#/.###.",
    "1": "..#../.##../..#../..#../..#../..#../.###.",
    "2": ".###./#...#/....#/...#./..#../.#.../#####",
    "3": "#####/...#./..##./....#/....#/#...#/.###.",
    "4": "...#./..##./.#.#./#..#./#####/...#./...#.",
    "5": "#####/#..../####./....#/....#/#...#/.###.",
    "6": "..##./.#.../#..../####./#...#/#...#/.###.",
    "7": "#####/#...#/....#/...#./..#../..#../..#..",
    "8": ".###./#...#/#...#/.###./#...#/#...#/.###.",
    "9": ".###./#...#/#...#/.####/....#/...#./.##..",
    ":": "...../.##../.##../...../.##../.##../.....",
    ";": "...../.##../.##../...../.##../.##../.#...",
    "<": "...#./..#../.#.../#..../.#.../..#../...#.",
    "=": "...../...../#####/...../#####/...../.....",
    ">": ".#.../..#../...#./....#/...#./..#../.#...",
    "?": ".###./#...#/....#/...#./..#../...../..#..",
    "@": ".###./#...#/....#/#.##./#.#.#/#.#.#/.##..",
    "A": ".###./#...#/#...#/#####/#...#/#...#/#...#",
    "B": "####./#...#/#...#/####./#...#/#...#/####.",
    "C": ".###./#...#/#..../#..../#..../#...#/.###.",
    "D": "###../#..#./#...#/#...#/#...#/#..#./###..",
    "E": "#####/#..../#..../####./#..../#..../#####",
    "F": "#####/#..../#..../####./#..../#..../#....",
    "G": ".###./#...#/#..../#.###/#...#/#...#/.####",
    "H": "#...#/#...#/#...#/#####/#...#/#...#/#...#",
    "I": ".###./..#../..#../..#../..#../..#../.###.",
    "J": "..###/...#./...#./...#./...#./#..#./.##..",
    "K": "#...#/#..#./#.#../##.../#.#../#..#./#...#",
    "L": "#..../#..../#..../#..../#..../#..../#####",
    "M": "#...#/##.##/#.#.#/#.#.#/#...#/#...#/#...#",
    "N": "#...#/##..#/#.#.#/#..##/#...#/#...#/#...#",
    "O": ".###./#...#/#...#/#...#/#...#/#...#/.###.",
    "P": "####./#...#/#...#/####./#..../#..../#....",
    "Q": ".###./#...#/#...#/#...#/#.#.#/#..#./.##.#",
    "R": "####./#...#/#...#/####./#.#../#..#./#...#",
    "S": ".####/#..../#..../.###./....#/....#/####.",
    "T": "#####/..#../..#../..#../..#../..#../..#..",
    "U": "#...#/#...#/#...#/#...#/#...#/#...#/.###.",
    "V": "#...#/#...#/#...#/#...#/#...#/.#.#./..#..",
    "W": "#...#/#...#/#...#/#.#.#/#.#.#/##.##/#...#",
    "X": "#...#/#...#/.#.#./..#../.#.#./#...#/#...#",
    "Y": "#...#/#...#/.#.#./..#../..#../..#../..#..",
    "Z": "#####/....#/...#./..#../.#.../#..../#####",
    "[": "..###/..#../..#../..#../..#../..#../..###",
    "\\": "#..../#..../.#.../..#../..#../...#./....#",
    "]": "###../..#../..#../..#../..#../..#../###..",
    "^": "..#../.#.#./#...#/...../...../...../.....",
    "_": "...../...../...../...../...../...../#####",
    "`": ".#.../..#../...../...../...../...../.....",
    "a": "...../...../.###./....#/.####/#...#/.####",
    "b": "#..../#..../####./#...#/#...#/#...#/####.",
    "c": "...../...../.####/#..../#..../#..../.####",
    "d": "....#/....#/.####/#...#/#...#/#...#/.####",
    "e": "...../...../.###./#...#/#####/#..../.###.",
    "f": "..##./.#..#/.#.../####./.#.../.#.../.#...",
    "g": "...../...../.####/#...#/.####/....#/.###.",
    "h": "#..../#..../#.##./##..#/#...#/#...#/#...#",
    "i": "..#../...../.##../..#../..#../..#../.###.",
    "j": "...#./...../...#./...#./...#./#..#./.##..",
    "k": "#..../#..../#..#./#.#../##.../#.#../#..#.",
    "l": ".##../..#../..#../..#../..#../..#../.###.",
    "m": "...../...../##.#./#.#.#/#.#.#/#...#/#...#",
    "n": "...../...../#.##./##..#/#...#/#...#/#...#",
    "o": "...../...../.###./#...#/#...#/#...#/.###.",
    "p": "...../...../####./#...#/####./#..../#....",
    "q": "...../...../.####/#...#/.####/....#/....#",
    "r": "...../...../#.##./##.../#..../#..../#....",
    "s": "...../...../.####/#..../.###./....#/####.",
    "t": ".#.../.#.../####./.#.../.#.../.#..#/..##.",
    "u": "...../...../#...#/#...#/#...#/#..##/.##.#",
    "v": "...../...../#...#/#...#/#...#/.#.#./..#..",
    "w": "...../...../#...#/#...#/#.#.#/#.#.#/.#.#.",
    "x": "...../...../#...#/.#.#./..#../.#.#./#...#",
    "y": "...../...../#...#/#...#/.####/....#/.###.",
    "z": "...../...../#####/...#./..#../.#.../#####",
    "{": "...##/..#../..#../.#.../..#../..#../...##",
    "|": "..#../..#../..#../..#../..#../..#../..#..",
    "}": "##.../..#../..#../...#./..#../..#../##...",
    "~": "...../...../.#..#/#.#.#/#..#./...../.....",
}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Glyph:
    """One character's bitmap, trimmed horizontally, in the shared 9-row box.

    Invariant relied on by the matcher: the first and last columns of `mask`
    both contain ink (except for zero-width glyphs like space). That is what
    makes "try the widest candidate first" provably safe — no wide glyph can
    ever equal a narrow glyph padded with blank columns.
    """

    char: str
    mask: np.ndarray  # (GLYPH_H, w) bool
    advance: int

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def ink(self) -> int:
        return int(self.mask.sum())


class GlyphAtlas:
    """A character set's bitmaps plus the metrics needed to walk a line."""

    __slots__ = ("_glyphs", "baseline", "height", "line_height", "name", "space_advance")

    def __init__(
        self,
        glyphs: Mapping[str, Glyph],
        *,
        height: int = GLYPH_H,
        baseline: int = BASELINE,
        line_height: int = LINE_HEIGHT,
        space_advance: int = SPACE_ADVANCE,
        name: str = "unnamed",
    ) -> None:
        for ch, g in glyphs.items():
            if g.mask.shape[0] != height:
                raise ValueError(f"glyph {ch!r} has height {g.mask.shape[0]}, expected {height}")
        self._glyphs: dict[str, Glyph] = dict(glyphs)
        self.height = int(height)
        self.baseline = int(baseline)
        self.line_height = int(line_height)
        self.space_advance = int(space_advance)
        self.name = name

    # -- mapping-ish access -------------------------------------------------

    def __len__(self) -> int:
        return len(self._glyphs)

    def __contains__(self, ch: object) -> bool:
        return ch in self._glyphs

    def __getitem__(self, ch: str) -> Glyph:
        return self._glyphs[ch]

    def get(self, ch: str) -> Glyph | None:
        return self._glyphs.get(ch)

    @property
    def chars(self) -> tuple[str, ...]:
        return tuple(sorted(self._glyphs))

    def glyphs(self) -> tuple[Glyph, ...]:
        return tuple(self._glyphs[c] for c in sorted(self._glyphs))

    # -- introspection ------------------------------------------------------

    def duplicate_pairs(self) -> list[tuple[str, str]]:
        """Character pairs whose bitmaps are identical — i.e. undecodable.

        Exact matching cannot distinguish these, so any non-empty result is a
        font bug, not an OCR bug. Kept as a method (not a test helper) so a
        user-derived atlas can be validated the same way.
        """
        seen: dict[bytes, str] = {}
        dupes: list[tuple[str, str]] = []
        for ch, g in sorted(self._glyphs.items()):
            if g.width == 0:
                continue
            key = g.mask.tobytes() + b"|" + bytes([g.width])
            prior = seen.get(key)
            if prior is not None:
                dupes.append((prior, ch))
            else:
                seen[key] = ch
        return dupes

    def text_width(self, text: str) -> int:
        return sum(self._glyphs[c].advance for c in text if c in self._glyphs)

    # -- rendering ----------------------------------------------------------

    def render(self, text: str) -> np.ndarray:
        """Rasterise `text` to a bool mask of shape (height, width).

        Used by the synthetic frame generator, and by anyone wanting to eyeball
        a user-derived atlas. Deliberately trivial: pen walks left to right,
        advance per glyph, no kerning — exactly what the matcher assumes.
        """
        total = self.text_width(text)
        out = np.zeros((self.height, max(total, 1)), dtype=bool)
        pen = 0
        for ch in text:
            g = self._glyphs.get(ch)
            if g is None:
                logger.debug("atlas %s has no glyph for %r; skipping", self.name, ch)
                continue
            if g.width:
                out[:, pen : pen + g.width] |= g.mask
            pen += g.advance
        return out[:, :total] if total else out[:, :0]

    # -- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        chars = self.chars
        widths = np.array([self._glyphs[c].width for c in chars], dtype=np.int32)
        advances = np.array([self._glyphs[c].advance for c in chars], dtype=np.int32)
        offsets = np.concatenate([[0], np.cumsum(widths)]).astype(np.int32)
        sheet = np.zeros((self.height, int(widths.sum())), dtype=np.uint8)
        for i, c in enumerate(chars):
            g = self._glyphs[c]
            if g.width:
                sheet[:, offsets[i] : offsets[i + 1]] = g.mask.astype(np.uint8)
        meta = {
            "name": self.name,
            "height": self.height,
            "baseline": self.baseline,
            "line_height": self.line_height,
            "space_advance": self.space_advance,
            "chars": "".join(chars),
            "format": 1,
        }
        np.savez_compressed(
            p,
            sheet=np.packbits(sheet, axis=None),
            sheet_shape=np.array(sheet.shape, dtype=np.int32),
            widths=widths,
            advances=advances,
            offsets=offsets,
            meta=np.array(json.dumps(meta)),
        )
        logger.info("wrote atlas %s (%d glyphs) to %s", self.name, len(chars), p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> GlyphAtlas:
        with np.load(Path(path), allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            shape = tuple(int(v) for v in z["sheet_shape"])
            sheet = np.unpackbits(z["sheet"], count=shape[0] * shape[1]).reshape(shape).astype(bool)
            widths = z["widths"]
            advances = z["advances"]
            offsets = z["offsets"]
        chars = meta["chars"]
        glyphs: dict[str, Glyph] = {}
        for i, ch in enumerate(chars):
            w = int(widths[i])
            mask = sheet[:, int(offsets[i]) : int(offsets[i]) + w].copy()
            glyphs[ch] = Glyph(char=ch, mask=mask, advance=int(advances[i]))
        return cls(
            glyphs,
            height=int(meta["height"]),
            baseline=int(meta["baseline"]),
            line_height=int(meta["line_height"]),
            space_advance=int(meta["space_advance"]),
            name=str(meta["name"]),
        )


# --------------------------------------------------------------------------
# Synthetic atlas construction
# --------------------------------------------------------------------------


def _parse_design(rows: str) -> np.ndarray:
    grid = rows.split("/")
    if len(grid) != DESIGN_H or any(len(r) != DESIGN_W for r in grid):
        raise ValueError(f"design must be {DESIGN_H}x{DESIGN_W}, got {[len(r) for r in grid]}")
    return np.array([[c == _ON for c in row] for row in grid], dtype=bool)


def _place(design: np.ndarray, descend: bool) -> np.ndarray:
    box = np.zeros((GLYPH_H, DESIGN_W), dtype=bool)
    top = DESCENDER_DROP if descend else 0
    box[top : top + DESIGN_H] = design
    return box


def build_synthetic_atlas(name: str = "synthetic_p12") -> GlyphAtlas:
    """Procedurally build the stand-in RS-like 12 px bitmap font.

    Deterministic and dependency-free on purpose: regenerating the npz on any
    machine must produce byte-identical output, otherwise "the OCR scores 0.99"
    stops being a reproducible claim.
    """
    glyphs: dict[str, Glyph] = {}
    for ch, rows in _DESIGN.items():
        box = _place(_parse_design(rows), ch in DESCENDERS)
        cols = np.flatnonzero(box.any(axis=0))
        if cols.size == 0:  # space and friends carry no ink
            glyphs[ch] = Glyph(ch, np.zeros((GLYPH_H, 0), dtype=bool), SPACE_ADVANCE)
            continue
        trimmed = box[:, cols[0] : cols[-1] + 1].copy()
        glyphs[ch] = Glyph(ch, trimmed, int(trimmed.shape[1]) + SIDE_BEARING)
    atlas = GlyphAtlas(glyphs, name=name)
    dupes = atlas.duplicate_pairs()
    if dupes:  # pragma: no cover - guarded by a test
        raise ValueError(f"synthetic font has ambiguous glyphs: {dupes}")
    return atlas


def default_atlas_path() -> Path:
    """Location of the checked-in synthetic atlas (repo_root/assets/fonts)."""
    return Path(__file__).resolve().parents[3] / "assets" / "fonts" / "synthetic_p12.npz"


_DEFAULT_ATLAS: GlyphAtlas | None = None


def load_default_atlas(path: str | Path | None = None) -> GlyphAtlas:
    """Load the synthetic atlas, regenerating it if the asset is absent.

    Self-healing rather than fatal, because a missing asset must never be the
    reason a test suite that is supposed to need nothing but numpy fails.
    """
    global _DEFAULT_ATLAS
    if path is None and _DEFAULT_ATLAS is not None:
        return _DEFAULT_ATLAS
    p = Path(path) if path is not None else default_atlas_path()
    if p.exists():
        atlas = GlyphAtlas.load(p)
    else:
        logger.warning("atlas %s missing; regenerating synthetic font", p)
        atlas = build_synthetic_atlas()
        try:
            atlas.save(p)
        except OSError:  # read-only checkout, container, whatever
            logger.warning("could not persist regenerated atlas to %s", p)
    if path is None:
        _DEFAULT_ATLAS = atlas
    return atlas


# --------------------------------------------------------------------------
# Deriving a real atlas from the user's own frames
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtlasSample:
    """One labelled training line: a binary text mask and the string it spells."""

    mask: np.ndarray  # (h, w) bool, output of ShadowFilter/ColorFilter
    text: str


def build_atlas_from_frames(
    samples: Iterable[AtlasSample],
    *,
    height: int = GLYPH_H,
    baseline: int = BASELINE,
    line_height: int = LINE_HEIGHT,
    min_votes: int = 1,
    name: str = "derived",
) -> GlyphAtlas:
    """Derive a glyph atlas from frames the *user* captured from their client.

    This is the supported path to a production atlas, and the reason there is
    no font file in this repo. RuneScape's `p12_full`/`b12_full` bitmaps are
    Jagex's copyrighted assets: redistributing them (or a transcription of
    them, which is the same bytes with extra steps) is not ours to do. Deriving
    them locally from pixels the user already has on screen is, and it has the
    side benefit of capturing whatever the user's client actually renders —
    version, scaling and interface skin included — instead of what we guessed.

    The intended workflow:

      1. Put the client in a state with known text on screen (the login screen,
         a `::` command echo, or any fixed server message).
      2. Capture frames, crop the text region, run `ShadowFilter` over it to get
         a binary mask, and pair each mask with the string you know it spells.
      3. Feed those pairs here and `.save()` the result.

    Algorithm: split each mask into connected column runs (a bitmap font at
    this size has a blank column between glyphs, so runs == glyphs), align the
    runs with the non-space characters of the label, and vote per character
    across all samples. Samples whose run count disagrees with the label are
    dropped rather than force-aligned — a misaligned sample poisons every glyph
    after the misalignment, so silence beats a plausible-looking wrong atlas.
    """
    votes: dict[str, dict[bytes, int]] = {}
    shapes: dict[bytes, tuple[int, int]] = {}
    advance_obs: dict[str, list[int]] = {}
    used = 0

    for sample in samples:
        mask = np.asarray(sample.mask, dtype=bool)
        if mask.ndim != 2 or mask.shape[0] < height:
            logger.debug("sample rejected: bad shape %s", mask.shape)
            continue
        rows = np.flatnonzero(mask.any(axis=1))
        if rows.size == 0:
            continue
        top = max(0, min(int(rows[0]), int(rows[-1]) - height + 1))
        band = mask[top : top + height]
        if band.shape[0] != height:
            continue

        runs = _column_runs(band)
        expected = [c for c in sample.text if not c.isspace()]
        if len(runs) != len(expected):
            logger.debug(
                "sample %r rejected: %d column runs vs %d glyphs",
                sample.text,
                len(runs),
                len(expected),
            )
            continue

        used += 1
        for i, (ch, (x0, x1)) in enumerate(zip(expected, runs, strict=True)):
            bits = band[:, x0:x1].copy()
            key = bits.tobytes()
            shapes[key] = bits.shape
            votes.setdefault(ch, {})[key] = votes.setdefault(ch, {}).get(key, 0) + 1
            if i + 1 < len(runs):
                advance_obs.setdefault(ch, []).append(runs[i + 1][0] - x0)

    if not used:
        raise ValueError("no usable samples: every line's segmentation disagreed with its label")

    glyphs: dict[str, Glyph] = {}
    for ch, options in votes.items():
        key, count = max(options.items(), key=lambda kv: kv[1])
        if count < min_votes:
            continue
        shape = shapes[key]
        bits = np.frombuffer(key, dtype=bool).reshape(shape).copy()
        obs = advance_obs.get(ch)
        advance = int(np.median(obs)) if obs else shape[1] + SIDE_BEARING
        glyphs[ch] = Glyph(ch, bits, max(advance, shape[1]))

    if " " not in glyphs:
        glyphs[" "] = Glyph(" ", np.zeros((height, 0), dtype=bool), SPACE_ADVANCE)

    logger.info("derived atlas %s: %d glyphs from %d usable samples", name, len(glyphs), used)
    return GlyphAtlas(glyphs, height=height, baseline=baseline, line_height=line_height, name=name)


def _column_runs(band: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of ink-bearing columns, as [start, stop) pairs."""
    col: np.ndarray = np.asarray(band.any(axis=0))
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for x, on in enumerate(col):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, int(col.size)))
    return runs


def atlas_report(atlas: GlyphAtlas) -> str:
    """Human-readable metrics summary, used by the benchmarks."""
    widths = [g.width for g in atlas.glyphs() if g.width]
    advances = [g.advance for g in atlas.glyphs() if g.width]
    return (
        f"atlas={atlas.name} glyphs={len(atlas)} box={atlas.height}px "
        f"baseline={atlas.baseline} line_height={atlas.line_height} "
        f"width={min(widths)}-{max(widths)} advance={min(advances)}-{max(advances)}"
    )


def main() -> int:  # pragma: no cover - CLI glue
    """`python -m gielinor.perception.fonts` regenerates the shipped atlas."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    atlas = build_synthetic_atlas()
    path = atlas.save(default_atlas_path())
    logger.info("%s -> %s", atlas_report(atlas), path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
