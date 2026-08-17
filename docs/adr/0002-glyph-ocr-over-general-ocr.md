# ADR 0002 — Exact glyph-bitmap matching instead of general-purpose OCR

- Status: accepted
- Date: 2026-08-15
- Deciders: project lead
- Related: [`src/gielinor/contracts.py`](../../src/gielinor/contracts.py) (`TextLine`,
  `TextSource`), [latency budget](../latency-budget.md), `scripts/build_font_atlas.py`

## Context

The agent reads text from screenshots: chatbox lines, NPC dialogue, the
upper-left hover tooltip ("Chop down Tree"), overhead player text, XP drops,
level-up messages. `TextLine` in `contracts.py` is one of the two things
perception produces, and several reflex rules key off it directly.

The input has properties that a general OCR engine is not built for:

- The font is a **bitmap font at 12-16 px**, not an outline font rasterized at
  runtime.
- Each glyph is drawn with a **hard 1 px drop shadow at offset (+1, +1)** in
  near-black, so the text stays legible over the 3D scene.
- **There is no anti-aliasing.** A pixel is either the text colour or it is
  untouched background. The renderer blits a 1-bit coverage mask.
- Text is laid out on **integer pixel advances**. There is no subpixel
  positioning.
- The background is an **arbitrary 3D scene** — a forest, a bank interior, water,
  another player's cape — so "the background" has no stable statistics.
- Text **colour is semantic**. RS uses a small fixed palette where the colour of
  a chatbox line tells you the message type before you have read a character.

The reflex loop's whole budget is 50 ms per frame (see
[latency-budget.md](../latency-budget.md)). Tesseract alone is typically 30-200 ms
per invocation, which is between 60% and 400% of the entire budget for one
stage.

## Decision

Recover the glyph bitmaps directly and look them up in an exact-match table.
Four steps, all numpy and OpenCV:

1. **Shadow filter.** Find candidate shadow pixels by darkness, shift the mask
   by (-1, -1) to land on the glyph body, and keep only pixels that have a
   shadow correctly positioned behind them. Two thresholded array operations
   over the region of interest.
2. **Colour filter.** Intersect with the known text palette for the region.
   Because the palette is semantic, this simultaneously classifies the line
   (public chat, game message, level-up) and rejects scene pixels that happened
   to survive step 1.
3. **Glyph match.** Compare the recovered binary mask against the atlas. Every
   pixel must match — not correlated, not thresholded similarity, **exact**.
4. **Segment on the baseline**, walk left to right, match each column band, and
   emit a `TextLine` with its bbox, colour, and inferred `TextSource`.

The palette values and the region rectangles live in the calibration profile
(`assets/profiles/*.toml`), never inline in code — same rule as coordinates in
`SPEC.md` §1.

Expected results: **>99% character accuracy, sub-millisecond per line**, against
roughly 60-80% for Tesseract on the same input. Those figures are the target,
not a measurement; `benchmarks/` exists to turn them into one, and the
comparison is worth publishing because it is concrete and falsifiable.

## Why exact matching is correct here rather than brittle

"Exact pixel match" normally signals a fragile heuristic. Here it is the
opposite, and the reason is worth stating precisely, because it is the whole
technical argument.

**With anti-aliasing, a glyph's pixels are not a function of the glyph.** They
depend on subpixel phase, on the background being composited against, on gamma,
on hinting, and on the rasterizer version. The letter `a` at x=100.0 and at
x=100.5 produces different pixel values; the same `a` over grass and over sky
produces different pixel values at every edge pixel. An exact-match table would
need an entry per glyph per subpixel phase per background, which is
combinatorially hopeless. That is *why* general OCR is statistical: the signal
genuinely is ambiguous, so approximate inference is the right tool.

**Without anti-aliasing, the rendered output is a function of the glyph id and
the text colour, and nothing else.** The renderer walks a 1-bit mask and writes
the text colour where the mask is set. Position is integer, so there is no
phase. Compositing is a hard write, so there is no background dependence *in
the set pixels*. There are exactly N distinct bitmaps for N glyphs, and each is
byte-identical every single time it appears on screen.

That changes the nature of the problem. The image is not a noisy observation of
the string; it is a **lossless, deterministic serialization** of it. Exact
matching is not a heuristic that happens to work — it is the inverse of a known
encoder. We are not recognizing text, we are decoding it.

The only place the background survives is where the mask is clear, and that is
exactly what steps 1 and 2 remove. Which is why the shadow filter is not a
preprocessing nicety: it is the step that turns an ambiguous image back into
the unambiguous mask the renderer started from.

**Why the shadow filter is so effective.** Text colour alone is ambiguous over
a 3D scene — white pixels are also clouds, snow, and item icons. But the
conjunction is not. A pixel of colour C at (x, y) *with* a near-black pixel at
(x+1, y+1) that is itself part of a 1-pixel-offset copy of the same mask is
essentially unique in a natural image. Natural scenes do not contain
self-similar structures displaced by exactly one pixel diagonally. The filter
is two thresholds and a shifted AND over the ROI, and it separates text from
scene almost perfectly for a handful of microseconds of work.

### The precondition, stated so it can be checked

Correctness rests on one assumption: **the captured pixels are the rendered
pixels, 1:1**. Anything that resamples breaks it, and the realistic culprits
are specific:

- fractional Wayland output scaling, or capturing a scaled-down copy of the
  screen;
- macOS Retina capture configured in points rather than backing-store pixels
  (see [platform-notes.md](../platform-notes.md));
- a lossy or chroma-subsampled capture path;
- the client's own resizable/HD mode applying GPU filtering to the UI.

So this is not "assume 1:1 and hope". A startup self-test matches a known UI
sprite byte-for-byte and refuses to run if it fails. That converts a class of
silent, weird, intermittent misreads into one loud error at launch, which is
the trade worth making. The failure mode of a broken precondition should be
obvious, not subtle.

## Alternatives considered

### Tesseract

The default choice, and wrong on three independent grounds.

Its LSTM recognizer was trained on document text: outline fonts at effectively
150-300 DPI, dark on light, anti-aliased or cleanly binarized. Its own guidance
asks for roughly 30 px of x-height; we have 12-16 px of total glyph height,
somewhere between a third and a half of what it wants, and upscaling invents
detail rather than recovering it.

Its layout analysis assumes a page. It looks for text blocks and lines against
a background with consistent statistics. Over a forest canopy there are no such
blocks, so the segmentation stage fails before recognition is reached.

Its language model actively hurts. Tesseract will happily bend `1 x Willow
logs` toward dictionary words. For text we intend to pattern-match against
game messages, a plausible wrong answer is worse than no answer, because it
passes downstream checks.

And it costs 30-200 ms, against a 50 ms whole-frame budget.

### EasyOCR and PaddleOCR

Both are scene-text pipelines: a detector (CRAFT, DB) that proposes text
regions, then a recognizer. The detectors are trained on signage and
photographed text — large, high-contrast, generally anti-aliased. 12 px UI
glyphs sit below their effective scale, and a missed detection means the
recognizer never sees the line at all.

They also drag in a PyTorch or Paddle runtime: hundreds of megabytes of
dependency, a model download, and — if run on GPU — contention with the game
client for the same device, which is the same self-interference problem
described in [ADR 0001](0001-two-loop-architecture.md) option (d).

Rejected for the same underlying reason as Tesseract, which is worth naming
once: **all three are approximate inference over an ambiguous signal, and our
signal is not ambiguous.** Applying a statistical model to a lossless
deterministic encoding trades correctness away and gets nothing back. When you
know the encoder, invert it.

### A small CNN classifier trained on the glyph set

Would work, would reach high accuracy, and would still be strictly worse: it
needs training data, a runtime, and a nondeterministic failure mode, in
exchange for approximating a lookup table that we can build exactly. It also
makes the >99% figure a model property rather than a structural guarantee.

### Reading the client's memory or hooking its renderer

Out of scope by the project's founding constraint — this is a pixels-only
agent. Noted for completeness because it is the obvious "just cheat" answer,
and rejecting it is what makes the perception work interesting.

## Prior art

The RuneScape tooling community reached the same conclusion independently.
`ollydev/SimpleOCR`, in the Simba scripting ecosystem, is a bitmap-font OCR
that matches glyph bitmaps directly against font atlases and handles the
shadow/colour structure of RS text, rather than wrapping a trained recognizer.
It emerged after that community's practical experience with general OCR on this
exact input.

Convergent evolution is evidence. Two independent efforts, the same constraint,
the same answer, is a stronger signal than either one alone. It is also a
useful reference implementation to check behaviour against — though its API and
licence should be verified before anything is vendored, rather than assumed
from this document.

## Legal note

**We do not ship Jagex font files, and no font binary enters this repository.**

The glyph atlas is *derived at setup time* from frames the user captures on
their own machine, by `scripts/build_font_atlas.py`. The output is a per-user
artifact, gitignored, and never included in a release. Tests use a
synthetically generated atlas, so CI never needs game assets — which also keeps
`SPEC.md` §6's "every test passes with no display, no network, and no game
running" true.

Two reasons this distinction matters. First, it is the same posture the
2009scape project itself takes toward game assets: the software is AGPL, the
cache is the user's problem. Second, a measured bitmap and a font program are
different artifacts. Font *programs* are protected as software (and in the UK,
typefaces have their own protection); a bitmap of glyph shapes measured from a
running program is a substantially weaker thing to be holding. Deriving locally
rather than redistributing keeps us on the better side of that line and, more
practically, means there is nothing to take down.

This is engineering caution, not legal advice.

## Consequences

### Good

- Sub-millisecond lines, which is what makes text usable in a 50 ms loop at all.
  Chatbox reading stops being a budget question.
- Accuracy that is structural rather than statistical: given the precondition,
  a correct match is *certain*, not probable.
- Message type falls out of the colour filter for free, before any character is
  decoded — a cheap and reliable routing signal for `TextSource`.
- No heavyweight dependency. numpy and OpenCV are already present for template
  matching; nothing new is added to satisfy `SPEC.md` §6's dependency rule.
- Unknown glyphs are *detectable*. A mask that matches nothing in the atlas is
  an explicit unknown, which can lower `TextLine.confidence` and, if it matters,
  escalate to the slow loop. A statistical recognizer would silently return its
  best wrong guess instead.

### Bad

- **A setup step exists.** The user must capture frames and build an atlas
  before text works. That is friction on first run, and it needs to fail
  clearly when skipped.
- **The precondition is real and will be violated by someone.** Fractional
  scaling and Retina point/pixel confusion are common, and without the startup
  self-test the symptom would be baffling.
- **Retargeting means rebuilding.** A different client, font, or era needs a new
  atlas. This is by design — it is a data change, not a code change — but it is
  still work.
- **Coverage is only what was captured.** A glyph that never appeared in the
  capture set is missing from the atlas. Atlas construction needs a
  completeness check against the expected character set, and a clear report of
  what is absent.
- **Effective palettes drift.** Client settings and interface variants can
  change text colours. Keeping them in the profile TOML contains this, but
  someone still has to notice and update the profile.
