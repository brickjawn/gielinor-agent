# Latency budget

> **Every number in the fast-loop and slow-loop tables below is an estimate.**
> Nothing here has been measured on this codebase yet. The figures come from
> published characteristics of the libraries and APIs involved, from arithmetic
> on data volumes and memory bandwidth, and from the target's own constraints.
> They exist so that the design has a falsifiable claim attached to it and so
> that `benchmarks/` has something to disagree with. Where a number is an
> assumption dressed as precision, it is marked. The measurement methodology at
> the end is how these become real.

## Why 50 ms

The fast loop targets **under 50 ms end to end**, which is 20 Hz.

That number is *not* derived from the 600 ms game tick, and it would be
dishonest to imply it is. Sampling a 600 ms process only needs about 300 ms to
see every tick boundary. The 50 ms target comes from four other requirements:

- **Transients.** The click marker, XP drop text, and hitsplats persist for a
  few hundred milliseconds. A 300 ms sampler misses them at unlucky phase; a
  50 ms sampler sees each one several times.
- **Animation detection.** `PlayerState.is_animating` is derived from viewport
  pixel churn, not from a stat readout. Distinguishing "chopping" from "idle"
  needs several samples inside one animation cycle.
- **Tick phase.** To time an action into a specific tick, you have to know where
  the tick boundary is. Sampling at 50 ms locates it to about 8% of a tick.
  Sampling at 300 ms does not locate it at all.
- **Jitter headroom.** A loop designed to exactly its deadline misses it roughly
  half the time. Designing to 50 ms and degrading to 100 ms (10 Hz) under load
  is the intended behaviour, not a failure.

## Fast loop, per frame

Assumptions for the table: a 1920x1080 capture source, the 765x503 2009scape
viewport cropped from it, four active templates, and a chatbox with up to eight
visible lines. Timings are single-core CPU on a mid-range modern desktop.
"Best" and "worst" are intended as roughly p10 and p95 of the eventual
distribution, not hard bounds.

| # | Stage | Best (ms) | Worst (ms) | Status |
| --- | --- | --- | --- | --- |
| 1 | Frame acquisition — Wayland (PipeWire SHM buffer) | 3.0 | 8.0 | assumed |
| 1' | Frame acquisition — macOS (ScreenCaptureKit `CMSampleBuffer`) | 4.0 | 10.0 | assumed |
| 2 | Viewport crop (slice + `ascontiguousarray`) | 0.2 | 0.4 | assumed |
| 3 | BGRA to BGR conversion, post-crop | 0.2 | 0.5 | assumed |
| 4 | Colour threshold, 3 ranges over the viewport | 0.6 | 1.5 | assumed |
| 5 | Template match, 4 templates | 4.0 | 12.0 | assumed |
| 6 | Glyph OCR, chatbox + tooltip | 0.5 | 4.0 | assumed |
| 7 | Inventory and UI panel read, 28 slots | 0.3 | 0.8 | assumed |
| 8 | State assembly and object tracking | 0.3 | 0.8 | assumed |
| 9 | `ReflexPlanner.plan()` | 0.1 | 0.2 | assumed |
| 10 | Action validate + dispatch over Unix socket | 0.2 | 0.6 | assumed |
| 11 | Interpreter overhead, allocation, GC, scheduling | 1.0 | 3.0 | assumed |
| | **Total (Wayland)** | **10.4** | **31.8** | |
| | **Total (macOS)** | **11.4** | **33.8** | |

Against a 50 ms budget that is **21% to 68% utilisation** across both platforms
(Wayland best case to macOS worst case). The budget holds, with real but not
generous headroom.

### Notes on individual lines

**1 — Frame acquisition** is the largest single line on both platforms and the
one we control least. On Wayland the cost is a PipeWire event-loop wakeup plus
a buffer dequeue; the compositor has already done the copy if the stream
negotiated shared-memory (`MemPtr`) buffers. If it negotiates DMA-BUF instead,
the number changes character entirely — naive CPU reads from mapped GPU memory
over PCIe are uncached and can cost 5-15 ms, so the capture backend should
request SHM and treat a DMA-BUF negotiation as a configuration error worth
logging. On macOS the cost is a dispatch-queue hop plus
`CVPixelBufferLockBaseAddress` and a copy out of the IOSurface. See
[platform-notes.md](platform-notes.md) for both.

**2 before 3 is deliberate.** Converting the full 1920x1080 BGRA frame costs
8.3 MB of reads and 6.2 MB of writes, roughly 1-2 ms. Cropping to the 765x503
viewport first drops that to 1.5 MB and 1.15 MB — under 0.5 ms. Same result,
about a millisecond saved, purely from ordering. The requested stage order in
this document lists conversion before crop; the implementation does the
opposite, and this is why.

**5 — Template matching is the dominant compute term** and the first thing to
optimise if the budget is tight. `SPEC.md` cites 1-3 ms per template over the
full viewport, and four templates puts us at 4-12 ms. The obvious win is to
stop searching the whole viewport: run the colour threshold first (line 4, and
cheap), take the bounding boxes of its connected components, and match only
inside those. For an object class that is colour-distinctive — which is most of
them — that shrinks the search area by 3-5x. This is the reason line 4 sits
before line 5 rather than being folded into it.

**6 — Glyph OCR** hits its worst case only when text has changed. Most frames
show an unchanged chatbox, so a cheap dirty check on the chatbox region skips
the stage entirely. The 0.5 ms best case is that check plus one tooltip line.
The reasoning behind the approach is in
[ADR 0002](adr/0002-glyph-ocr-over-general-ocr.md).

**8 and 11 are Python, not mathematics.** Building ~20 frozen dataclasses and
running IoU association over them is trivially little arithmetic; the cost is
interpreter overhead and allocation. This is also where a generation-2 GC pause
will show up as a p99 spike — see the methodology section.

**10 excludes everything after the write.** The compositor's own processing, the
Java client's input handling, and the server's tick alignment are all outside
this budget and outside our instrumentation. Only an external measurement sees
them.

## Slow loop, per escalation

| Stage | Best (ms) | Worst (ms) | Status |
| --- | --- | --- | --- |
| Escalation decision | 0.1 | 0.5 | assumed |
| Build compact context (GameState to ~1-2 KB JSON) | 0.5 | 2.0 | assumed |
| Viewport JPEG encode, q80 (image requests only) | 3.0 | 8.0 | assumed |
| Base64 encode of 40-80 KB payload | 0.2 | 0.5 | assumed |
| TCP + TLS handshake (0 if connection is warm) | 0 | 150 | assumed |
| Request upload, ~110 KB with image | 10 | 200 | assumed |
| Provider queueing (free tier) | 100 | 3000+ | uncontrolled |
| Server-side prefill + decode | 400 | 2500 | assumed |
| Response parse and schema validation | 0.3 | 1.0 | assumed |
| **Total (sum of line items)** | **~510** | **~5900** | |

The worst case lands on `SPEC.md`'s stated 6 s ceiling. The best case does not:
the line items sum to about 510 ms against an observed floor of 700 ms. That gap
is not rounding, it is a signal that one of the estimates is optimistic, and the
most likely candidate is provider queueing — the one line we neither control nor
can predict. Rather than inflate a line to make the total match, the discrepancy
is left visible; it is the first thing the slow-loop benchmark should resolve.

Local compute is roughly 4-12 ms of that total. **Better than 99% of slow-loop
latency is network and provider time**, which is the entire justification for
the architecture in [ADR 0001](adr/0001-two-loop-architecture.md): there is
nothing to optimise locally, so the only available move is to stop waiting.

### The one lever we actually control

Server-side latency is approximately `constant + output_tokens / rate`. At the
30-60 tokens/s a large free-tier model realistically sustains, **every 10 output
tokens costs 170-330 ms**. A chatty 300-token response is 5-10 seconds of pure
decode.

So: cap `max_tokens` hard (96 is a reasonable starting point for an action
list), demand terse JSON with no prose, and forbid explanations in the response
schema. `Action.reason` is a short string for the trace log, not an invitation
to think out loud. Prompt engineering here is latency engineering.

The second lever is **avoiding images**. An image request costs the JPEG encode,
roughly 8x the upload bytes, and several hundred milliseconds of extra
server-side prefill. `SPEC.md` §1 already says to prefer text-only models when
perception has extracted what the model needs; the budget is why.

### Why this cannot be synchronous

At the 6 s worst case, blocking the fast loop would mean:

- **10 game ticks** with no agent, during which the character stands still;
- **120 dropped frames** at 20 Hz, and every transient inside them lost;
- **no capture at all** for six seconds, so the state we resume with is not a
  continuation of the state we planned against, it is a fresh observation of a
  world that moved.

And the failure is driven by the tail, not the median. A design tuned to a
700 ms p50 is broken by the 6 s p99, and free-tier latency has a fat right tail
by construction — it is scheduled behind paying traffic.

Hence: the slow loop runs off-thread, returns into a queue, and the fast loop
drains and revalidates. At most **one call in flight**. Escalations raised while
a call is outstanding are *coalesced to the newest trigger*, not queued — a
trigger that is four seconds old is as worthless as an action that is four
seconds old, and queueing them would build a backlog that guarantees every
response is stale.

Staleness budgets are per action kind, because perishability varies by an order
of magnitude:

| Kind | Budget | Reason |
| --- | --- | --- |
| `CLICK_OBJECT`, `WALK_MINIMAP` | ~2 ticks (1.2 s) | The object may be gone, dead, or occluded |
| `SELECT_DIALOGUE`, `CONTINUE_DIALOGUE` | ~5 ticks (3 s) | Dialogue blocks the client and persists |
| `TYPE_CHAT` | ~5 ticks (3 s) | Conversational relevance decays |
| Goal transitions | ~10 s | Goals are not frame-scoped |

## Measurement methodology

### What to instrument

Per-stage timers around each of the eleven fast-loop lines, plus:

- **End-to-end frame latency**: `Frame.captured_at` to `ActionResult.dispatched_at`.
  This is the number that matters and the only one worth quoting.
- **`Frame.age_ms()` at the moment perception starts** — separates "we are slow"
  from "the frame was already old when we got it". These have different fixes.
- **Slow-loop queue depth and time-in-queue**, plus the count of actions dropped
  at revalidation and the reason for each.
- **Reflex/LLM decision split per session.** Not a latency metric, but the one
  that detects the failure mode [ADR 0001](adr/0001-two-loop-architecture.md)
  warns about, where reflex rules quietly absorb the planner's job.

### Percentiles, not means

Report **p50, p95, p99, and max**. A mean of 20 ms with a p99 of 300 ms is a
broken loop that looks healthy in a mean, and the p99 is exactly where the
interesting bugs live — GC pauses, buffer starvation, a compositor hiccup.

Keep a fixed-size ring buffer (2048 samples is a few frames of memory and about
two minutes of history at 20 Hz) per stage and dump a histogram on exit. **Do
not log per-frame at INFO.** At 20 Hz a formatted log line per stage per frame
is 220 records/second, and the logging call becomes a stage in the budget it is
supposed to be measuring.

Note that this histogram buffer is a different thing from the *live* rolling
window in [`capture/clock.py`](../src/gielinor/capture/clock.py), which is
deliberately short (a few seconds) because it answers "is the loop healthy right
now" rather than "what did this session look like". `FrameClock` also draws the
distinction that matters most here: **interval** is how fast the source is,
**age** is how stale the pixels are, and a source can be fast and stale
simultaneously if something in the path is buffering. Only age predicts
misclicks.

Discard the first ~100 frames. OpenCV initialises lazily, numpy allocations
fault in pages on first touch, and the CPU has not settled into a frequency
state. Warmup samples will otherwise dominate the max and make every percentile
above p95 meaningless.

### The clock trap

**`time.time()` is the wrong clock and using it here is a real bug, not a style
preference.**

`time.time()` reads `CLOCK_REALTIME`. NTP can step it backward, which produces
*negative durations*; NTP slew changes its rate, which silently stretches or
compresses measured intervals; a manual clock change, a timezone daemon, or a
VM resume can move it by hours. Every one of those corrupts a latency histogram,
and the corruption is indistinguishable from a genuine outlier — you get a
mysterious 3-hour p99 or, worse, a plausible-looking 60 ms that was actually 50.

**`time.monotonic()` is the right clock.** It cannot go backward and its rate is
not adjusted. On Linux it is `CLOCK_MONOTONIC`; on macOS it is
`mach_absolute_time()`. Neither advances while the system is suspended, which is
the behaviour we want: a closed laptop lid should not be recorded as a
four-hour frame.

**`time.perf_counter()` versus `time.monotonic()` — the distinction that matters
here is the epoch, not the resolution.** On CPython/Linux both are backed by
`CLOCK_MONOTONIC` and are effectively identical in precision. But
`CLOCK_MONOTONIC`'s reference point is machine-global (boot), so two processes
on the same machine can compare timestamps directly, whereas `perf_counter`'s
reference point is documented as undefined.

Because the actuator runs in a **separate process** ([ADR 0003](adr/0003-untrusted-planner.md)),
any timestamp that crosses the Unix socket must be `time.monotonic()`. That is
why `Frame.captured_at` is monotonic in `contracts.py` and why `Frame.age_ms()`
is safe to call from either process. Use `perf_counter()` only for deltas that
begin and end inside one process, such as the individual stage timers.

Neither clock should ever be serialised as an absolute time in a log a human
reads. Log a monotonic timestamp for arithmetic and a separate wall-clock time
for orientation, and never subtract one from the other.

### The part we cannot measure from inside

Stamping `captured_at` when Python receives the buffer measures *when we noticed
the frame*, not when the compositor produced it. The gap between them is real
and is invisible to that method. The honest fix is to use the buffer's own
presentation timestamp:

- **PipeWire** attaches an `SPA_META_Header` carrying a pts, and `pw_stream_get_time`
  gives the mapping into the graph clock. PipeWire's clock is `CLOCK_MONOTONIC`
  based, so the conversion into `time.monotonic()`'s domain is direct.
- **ScreenCaptureKit** gives `CMSampleBufferGetPresentationTimeStamp`, which is on
  the host time base (`CMClockGetHostTimeClock`) — that is `mach_absolute_time`,
  the same base as `time.monotonic()` on macOS. The conversion is exact.

Doing this makes `Frame.age_ms()` mean what its docstring claims. Falling back
to an arrival stamp is acceptable, but the backend should then report which it
used, because the two numbers are not comparable and mixing them across
platforms would make a benchmark meaningless.

### Ground truth for the headline number

Instrumentation cannot see the display pipeline, the Java client's input
handling, or the server's tick alignment. The only way to get a true
click-to-reaction figure is external: point a high-frame-rate camera at the
screen, issue a synthetic click, and count frames until the game reacts. A
240 fps phone camera gives 4.2 ms resolution, which is plenty at this scale.

This is the number to quote in a write-up, precisely because it includes the
parts we cannot instrument and cannot fake.

### Two benchmarks, not one

- **Compute-only**, driven by `ReplayCapture` over recorded frames. Deterministic,
  runs in CI with no display and no network per `SPEC.md` §6, and can therefore
  carry a regression threshold that fails a build.
- **End-to-end**, on real hardware with a live capture backend. Platform- and
  machine-dependent, so it cannot gate CI, but it is the only one that includes
  line 1 of the fast-loop table — the largest line and the one most likely to
  surprise.

Reporting only the first would be flattering and misleading. Reporting only the
second gives no regression signal. Both, labelled.
