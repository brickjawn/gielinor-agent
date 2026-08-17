# Phased roadmap

The repo currently passes 424 tests, benchmarks its OCR at 1.00 exact-line
accuracy, and runs a full woodcutting loop end to end. None of that has touched
a real game client. Every coordinate is a guess, every colour signature is a
guess, and the headline OCR number is measured against frames drawn with the
same font the matcher decodes. The purpose of the phases below is to convert
that scaffold into something with real numbers behind it, in an order where the
cheap-to-fix mistakes surface before the expensive ones.

## The constraint that shapes everything

Work happens in two places and they have different capabilities.

**This sandbox** has the repo, Python, and PyPI. It has no display, no game
client, and — verified, not assumed — **no route to OpenRouter**; the egress
proxy returns 403. So it can run algorithms, tests, and offline benchmarks, and
nothing else.

**Your machine** is the only place a real frame or a real API response exists.

Those two facts mean the project has exactly one bridge: **recorded frames go
in, trace files come out.** You capture a session locally and drop the PNGs into
the repo; `ReplayCapture` then serves them to every downstream stage, so
perception, the reflex loop, and the planner can all be developed against real
pixels without a game running. This is why `ReplayCapture` was built first and
why every test runs headless — not as a testing convenience, but because it is
the only development loop available.

The practical consequence: **phases 1–3 are cheap and fast, phases 0, 4, and 5
require your attention on a real machine.** Batch the local work.

---

## Phase 0 — First contact

*Where:* your machine. *Effort:* an evening. *Blocks:* everything.

Get a client running and get real pixels into the repo. Nothing else.

The client choice matters more than it looks. The [Saradomin
launcher](https://2009scape.org/site/game_guide/play.html) is the maintained
path but connects to the live community server, which means real people are
around — fine for capture, wrong for an agent that clicks. There are archived
`Singleplayer-Edition-Windows` / `-Linux` builds on the [2009scape
GitHub](https://github.com/2009scape), and the server source itself is on
GitLab. Prefer fully local: it makes runs repeatable, removes other players from
the picture entirely, and means a runaway loop inconveniences nobody.

Capture 60–100 frames covering the states the agent actually has to distinguish:
idle, walking, mid-chop animation, inventory at 0 / partial / 28, an NPC
dialogue, an options dialogue, a level-up, and a busy scene with other entities
on screen. Cover a couple of camera angles — the detector will be judged on
them later.

**Exit criteria**
- Frames committed under `assets/frames/raw/`, named by state.
- `python -m gielinor run --capture replay --replay-dir assets/frames/raw` walks
  through them without raising.
- One frame confirmed to be a lossless, unscaled capture. If the client was
  captured through any kind of scaling, exact glyph matching degrades to roughly
  Tesseract's accuracy and the entire OCR premise evaporates — so check this
  once, deliberately, before building anything on top of it.

---

## Phase 1 — Calibration

*Where:* sandbox, from Phase 0 frames. *Effort:* short. *Risk if skipped:* high.

`assets/profiles/2009scape_fixed.toml` has `calibrated = false` and nags on
every start, because every rect in it is a plausible RS2-era guess. Replace them
with measurements, and derive a real glyph atlas from your own frames rather
than the synthetic stand-in.

This phase is early on purpose. A wrong rect does not throw — it silently
returns the wrong crop, and the symptom appears three subsystems downstream as
"the OCR is bad" or "it keeps missing trees". Fixing coordinates costs an hour;
debugging the consequences of wrong coordinates costs a day.

**Exit criteria**
- `calibrated = true`, with every rect verified against a real frame.
- Glyph atlas built by `build_atlas_from_frames()` from real client pixels, not
  `synthetic_p12.npz`.
- `tests/test_profiles.py` still green — it asserts perception and the actuator
  cannot drift apart, which matters most right after someone edits coordinates.
- HSV signatures in `detect.py` re-tuned against real foliage. Grass and tree
  canopy share a hue, so this discriminates on saturation and value; the current
  values are guesses.

---

## Phase 2 — Perception truth

*Where:* sandbox. *Effort:* medium — hand-labelling is the cost.

Replace self-consistency numbers with real ones. Hand-label 20–30 of the Phase 0
frames: exact chat strings, inventory counts, tree bounding boxes, dialogue
state. Then run the perceiver over them and measure.

Expect the numbers to drop, and let them. The current 1.0000 is honest about
being self-consistent — the renderer draws with the same atlas the matcher
reads — but a real number, even a worse one, is the only kind worth putting in a
README. The figure to watch is **exact-line rate**, not character accuracy,
because a chat line is consumed by matching it against known server messages and
one wrong glyph costs the whole line.

**Exit criteria**
- Real-frame accuracy for OCR, inventory occupancy, tree detection, and dialogue
  classification, committed as a benchmark that reruns.
- README numbers sourced from real frames, with the synthetic ones kept as a
  clearly-labelled ceiling.
- Every detector that scores badly either fixed or documented as a known gap.

---

## Phase 3 — Reflex loop on real frames

*Where:* sandbox. *Effort:* short.

Replay a real chopping session and confirm the reflex FSM produces a coherent
action sequence with the LLM entirely disconnected. Still nothing executes.

The number this phase produces is the **escalation rate** — what fraction of
frames the reflex layer cannot handle alone. That figure decides whether the
two-loop architecture is actually carrying its weight. If the reflex layer
handles 95%+ of frames, the design is validated. If it escalates constantly, the
reflex layer needs more rules, not a better model, and it is much cheaper to
learn that here than after wiring up the API.

**Exit criteria**
- A trace over real frames showing the full FIND_TREE → CLICK → WAIT → full-inventory cycle.
- Escalation rate measured and, if high, understood.
- Frame time on real frames within the 50 ms budget (synthetic frames currently
  run 9–12 ms; real ones will be slower and this is where you find out by how much).

---

## Phase 4 — Planner, live

*Where:* your machine — this is the phase the sandbox cannot do at all.

First real OpenRouter calls. Put your key in `.env` (it is gitignored;
`.env.example` is not) and run the escalation path against real captured states.

Three things get measured here, and all three are currently assumptions:

1. **Latency.** `docs/latency-budget.md` assumes 700 ms–6 s and flags that its
   own line items only sum to ~510 ms, which is probably optimistic. Real p50
   and p95 settle that.
2. **Which model.** `google/gemma-4-26b-a4b-it:free` is the current primary on
   paper. Only three of OpenRouter's free models are usable VLMs, so the
   fallback chain is short and worth testing under real rate limits.
3. **The injection defence.** Feed real player chat through it. The design
   argument is that capability restriction beats content filtering — there is no
   drop or trade verb to reach for — and that argument deserves an adversarial
   test rather than a confident paragraph.

**Exit criteria**
- Measured latency distribution replacing the assumed one in the budget doc.
- Model chain verified against live rate limits.
- Dialogue handled correctly end to end on real frames.
- An adversarial chat-injection attempt tried and its outcome documented.

---

## Phase 5 — First real click

*Where:* your machine. *Gate, not a phase.*

The first time `allow_real_input=True` is set. Everything before this ran with
`DryRunActuator`, which does the entire dispatch path and stops one call short of
the syscall.

Before flipping it, three things must exist and be tested: a kill switch that
does not depend on the agent's own loop being healthy, a watchdog that halts on
repeated validation rejections, and a hard session cap. The reason to build them
first is that synthetic input is a genuine OS privilege — the same mechanism that
clicks a tree can type into any focused window — and a stuck loop with hands on
the mouse is not something you want to be debugging while it runs.

This is also where the actuator becomes a separate process, per
[ADR 0004](adr/0004-process-split-not-container-split.md). Worth noting that this
is not extra work bolted onto the phase: the kill switch and the privilege
boundary are the *same mechanism*. An in-process handler is stuck exactly when
you need it — which is the case the kill switch exists for. A separate process
you can signal, or whose socket you can close, stops input regardless of what the
planner is doing. Build it once and both requirements are satisfied.

Everything else stays in one virtual environment on the same machine. There is no
container in this path.

The target for this phase is deliberately tiny: **one click, on one tree,
correctly, once.** Then stop and read the trace.

**Exit criteria**
- Actuator running as its own process behind the socket, speaking only validated
  `Action` messages.
- Kill switch verified working *while the loop is running*, and verified again
  with the agent loop deliberately wedged — the second test is the one that
  matters, and the easy one to skip.
- One correct click, with the trace confirming it resolved from the intended `oid`.
- Click coordinates land inside the object mask, not merely inside the bbox.

---

## Phase 6 — Closed loop and hardening

*Where:* your machine, increasingly unattended.

Full woodcutting: chop until the inventory fills, handle dialogue, recover from
the states that actually happen — a tree despawning mid-swing, the camera getting
dragged, a random event, a misclick that opens the wrong interface.

Recovery is the real work here. The happy path already works in replay; what
separates a demo from a system is what it does when perception is wrong, and the
answer should be "notices, and re-observes" rather than "carries on confidently".

**Exit criteria**
- 30 minutes unattended without intervention.
- Trace-derived stats: actions per minute, rejection rate, escalations, recoveries.
- Every failure it hit either handled or logged as a known limitation.

---

## Phase 7 — The write-up

The portfolio artifact is not the repo, it is the explanation.

The material is largely already there: three ADRs, a latency budget with its own
discrepancy flagged rather than smoothed over, an OCR benchmark that argues
against its own flattering number, and a threat model whose central claim is that
capability restriction beats content filtering. What it needs is the real numbers
from phases 2–4 and a short piece framing the actual insight — that the naive
version of this project is a screenshot-to-VLM loop that fails visibly, and that
the two-loop split with escalation policy and action staleness is where the
engineering lives.

A replayable trace file is worth more here than a video. It shows judgment.

---

## Sequencing notes

- **Do not reorder 1 before 0 or 5 before 4.** Calibration needs real frames;
  real input needs a planner you trust.
- **Phases 1–3 are one sandbox sitting** once Phase 0 frames exist. Batch them.
- **Phase 4 can start during 2–3** if you want live latency numbers early; it
  only needs the API key and a captured `GameState`, not a finished perceiver.
- **Keep `calibrated = false` until it is genuinely true.** The nag on startup is
  load-bearing.
