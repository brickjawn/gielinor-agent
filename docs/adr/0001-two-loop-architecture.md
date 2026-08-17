# ADR 0001 — Two loops: a local reflex loop and a remote planner loop

- Status: accepted
- Date: 2026-08-15
- Deciders: project lead
- Related: [`src/gielinor/contracts.py`](../../src/gielinor/contracts.py) (`Planner`
  protocol), [ADR 0003](0003-untrusted-planner.md), [latency budget](../latency-budget.md)

## Context

Two numbers set this design, and they do not overlap.

A RuneScape 2 game tick is **600 ms**. Every server-side state change — a
woodcutting roll, a step of movement, a combat hit — lands on a tick boundary.
An agent that wants to act on the tick it observed has 600 ms to see, decide,
and click.

A round trip to a free-tier OpenRouter model is **700 ms to 6 s**. The floor is
a small text model returning a short response over a warm connection. The
ceiling is a 26B vision model that has to ingest a screenshot, queued behind
other free-tier traffic. Neither end of that range fits in 600 ms, and the
distribution has a long right tail we do not control.

The consequence is not "the LLM is a bit slow". It is that **an LLM cannot be
in the tick loop at all**. Any design where a frame waits on a model response
before an action is issued is already broken; the only question is whether it
fails by being late or by falling over when the API has a bad minute.

There is a second, quieter constraint. Free-tier quota is finite — on the order
of tens of requests per minute and a few hundred to ~1000 per day, and the
exact numbers change (which is why `scripts/list_models.py` re-checks the
roster at runtime rather than trusting a table in a doc). Any design that calls
the model on a fixed schedule has to answer the quota question, and most
schedules answer it badly.

## Decision

Split the agent into two concurrent loops running at different clock rates,
communicating through a queue.

**Fast loop — local, 10-20 Hz, no network.**

```
capture.grab() -> perceive() -> ReflexPlanner.plan() -> actuator.execute()
```

Entirely OpenCV, numpy, and glyph OCR. Budget under 50 ms end to end, which at
20 Hz gives 12 observations per game tick. It handles every question with a
deterministic answer: is the inventory full, is the character still animating,
is there a tree in the viewport, did a dialogue box appear, did HP drop, is the
click I issued still pending. The reflex planner is a plain state machine. It
never calls out, and it carries roughly 95% of runtime decisions by frame
count.

**Slow loop — remote, event-driven, best-effort.**

```
escalation trigger -> build compact context -> OpenRouter -> Action[] -> queue
```

It fires on an event, never on a timer. The triggers are enumerated in
`SPEC.md` §2: novel state, free-form dialogue, stuck state machine, goal
transition. It returns actions into a queue that the fast loop drains and
**revalidates against the current `GameState`**. An action planned against
frame 1400 that arrives at frame 1460 is checked again; if the tree it named is
gone, the action is dropped rather than executed. See ADR 0003 for why
revalidation is a security property and not just a correctness one.

The two loops meet at exactly one place: the `Planner` protocol in
`contracts.py`. `ReflexPlanner` and `LLMPlanner` satisfy the same interface and
are explicitly *not* interchangeable in timing — the docstring says so, because
the type system cannot.

## Alternatives considered

### (a) One LLM call per tick

The obvious first idea, and structurally impossible rather than merely
expensive. The best free-tier latency we measured expectations for (700 ms) is
already over the 600 ms budget before adding capture, encode, and parse. Even
granting a hypothetical 200 ms model, three problems remain:

- **Quota.** 600 ms per tick is 1.67 requests/second, 6000 requests/hour. A
  free-tier daily allowance in the hundreds is gone in minutes.
- **Tail latency.** A loop whose period is set by a p50 breaks on the p99. One
  4-second stall means seven ticks with no agent, during which the character
  stands still in a game where standing still is how you die.
- **Liveness coupling.** The agent's ability to act at all becomes a function
  of a third party's uptime. A 30-second API blip is a 30-second freeze.

Rejected. This is the design the two-loop split exists to avoid.

### (b) One LLM call every N ticks, on a timer

The natural retreat from (a), and the version most people build second. It
survives the latency argument by decoupling call rate from tick rate, but it
fails on quota and on fit.

The quota arithmetic is unforgiving. At 1 Hz, a ~1000-request daily allowance
is exhausted in about 17 minutes of play. To survive a three-hour session you
need roughly one call every 11 seconds — and 11 seconds is 18 game ticks, far
too slow to answer a dialogue box that is blocking the character right now.
There is no N that is both affordable and responsive, because the two
requirements pull in opposite directions.

The deeper objection is that the need for a model is **bursty and
event-shaped, not periodic**. Twenty minutes of uneventful woodcutting need
zero model calls; a single unexpected interface needs one immediately. A timer
samples too often during the former and too rarely during the latter. This is
the same argument as interrupt-driven versus polled I/O, and it has the same
answer.

Rejected. Escalation triggers are strictly better than any fixed N.

### (c) A pure scripted bot with no LLM

Honesty first: for the narrow task — chop willows until the inventory is full,
bank, repeat — a scripted bot is *better*. It is faster, free, deterministic,
and the RuneScape tooling ecosystem (Simba, SRL, and the RuneLite plugin
world) has two decades of evidence that it works. Claiming otherwise to justify
an LLM would be dishonest.

The case against is narrower than it looks. Scripted bots are brittle exactly
where the world is open-ended: an NPC says something the author did not
anticipate, an interface appears that has no rule, a goal completes and the
question "what now" has no scripted answer. Those cases are rare by frame count
and dominant by failure count.

So the decision is not to reject (c) but to **contain** it. The `ReflexPlanner`
*is* a scripted bot. It runs every frame and does almost all the work. The
model is not a replacement for scripting; it is what handles the residue that
scripting handles badly. Framing it as a two-loop split rather than "an LLM
plays RuneScape" is the accurate description of what was built.

### (d) A small VLM running locally instead of a remote one

The most interesting alternative, because it attacks the constraint directly:
if the network round trip is the problem, remove the network.

Take the realistic candidate class, 7B-12B open vision-language models
(Qwen2.5-VL-7B, InternVL-8B, MiniCPM-V 8B, Llama-3.2-11B-Vision, Gemma-3-12B).
Work the numbers on a well-equipped consumer desktop — call it an RTX 4070-class
card, 12 GB VRAM, roughly 500 GB/s of memory bandwidth.

**Memory.** At Q4_K_M a 7B model is about 4.2-4.7 GB of weights; a 12B is
7.0-7.5 GB. The vision tower (typically a 300-400M ViT) is usually kept at
FP16, adding 0.6-0.8 GB. KV cache is negligible at these context lengths — a
7B with grouped-query attention costs on the order of 50-60 KB per token, so a
2000-token context is ~110 MB. A 7B fits comfortably in 8 GB; a 12B fits in
12 GB with little room to spare.

The room to spare matters, because **2009scape is running on the same GPU**.
It is a Java 3D client that wants VRAM and, more importantly, wants GPU time.
Inference and rendering contend, and the thing being starved is the renderer
producing the frames we are trying to read. The agent would be degrading its
own input.

**Throughput.** Two phases, and prefill is the one that hurts.

- *Prefill.* A 765x503 viewport becomes somewhere between ~400 and ~1500 image
  tokens depending on the model's patch and tiling scheme (dynamic-resolution
  models like Qwen2-VL land near the bottom of that range; models that tile
  into multiple 336px crops land near the top). Realistic llama.cpp prefill for
  a 7B Q4 on this class of card is 1500-3000 tokens/s, so 400-1500 image tokens
  is roughly 0.15-1.0 s. Add the ViT encoder forward pass, 0.1-0.3 s. Call it
  **0.3-1.3 s before the first output token exists**.
- *Decode.* Memory-bandwidth bound: 500 GB/s against 4.5 GB of weights gives a
  theoretical 110 tokens/s, realistically 50-70. A minimal JSON action is
  30-60 tokens, so **0.5-1.2 s**. A 12B at ~7.5 GB roughly halves that rate.

Total on dedicated hardware: **1-2.5 s**, against a 600 ms tick. Squeezing to
~400 ms is conceivable with a tiny image, a 20-token output, and a card doing
nothing else — but the card *is* doing something else, and there is no margin
left for the game.

The laptop case is worse. An M3 Pro has roughly 150 GB/s of unified bandwidth,
giving 20-30 tokens/s decode for a 7B Q4, and Metal prefill for a thousand
image tokens runs into seconds. CPU-only is not worth costing.

There is also an accuracy argument. At 7-12B, reading a 12px bitmap font over a
3D scene is something these models do poorly and unpredictably — worse than the
forty lines of numpy described in [ADR 0002](0002-glyph-ocr-over-general-ocr.md),
which does it exactly and in under a millisecond. So the local VLM cannot
replace the fast loop on either latency or accuracy grounds.

And if it cannot replace the fast loop, its only remaining job is the slow
loop — where a free remote 26B model is larger, more capable, and costs nothing
but latency the queue already absorbs.

Rejected, **but not wrong**. Option (d) solves a different problem: offline
operation, privacy, or independence from a third-party API. Because `LLMPlanner`
sits behind the `Planner` protocol, swapping it for a local llama.cpp or vLLM
backend is a config change, not a redesign. That door is deliberately left open.

## Consequences

### Good

- **Bounded blast radius.** The API being slow, rate-limited, or down degrades
  the agent to "competent scripted bot". It does not stop it. There is a
  behavioural floor that does not depend on anything remote.
- **The interesting 95% is deterministic.** The fast loop is pure functions
  over frames, so it is testable with `ReplayCapture` against recorded frames,
  in CI, with no display and no network — which is the non-negotiable in
  `SPEC.md` §6.
- **The slow loop is testable too**, for the same reason: it is a function from
  a serialized `GameState` to an `Action[]`, so recorded request/response pairs
  make it deterministic in tests. Only `@pytest.mark.live` tests touch the network.
- **Cost is zero and stays zero.** Event-driven escalation keeps call volume in
  the tens per session, comfortably inside free-tier quota, without a budget
  guard bolted on afterwards.
- **Model choice becomes a config value.** Nothing in the fast loop knows which
  model answered, or whether one did.

### Bad

- **Two clocks means races, permanently.** Action staleness is not a bug to be
  fixed; it is an intrinsic property of planning against a snapshot that ages.
  Revalidation at dispatch mitigates it but cannot eliminate it — the window
  between validating and dispatching is small, not zero.
- **The escalation policy is now hand-written logic with its own bugs.** A
  missed escalation presents as an agent that is inexplicably stuck. A spurious
  one burns quota and can loop. This policy is the least testable part of the
  system because its correctness is defined against situations we have not
  enumerated.
- **Immutability has a cost.** `GameState` is frozen so the slow loop can hold a
  snapshot while frames keep arriving. That means allocating a fresh state
  object every frame at 20 Hz, which is fine at this scale and would not be at
  200 Hz.
- **Debugging spans time.** A bad action's cause may be four seconds and seven
  frames upstream. This is why `Action.reason` exists and why the trace log
  must carry both the frame `seq` an action was planned against and the one it
  was dispatched on.
- **The reflex layer will try to eat the planner's job.** Every time the model
  is too slow or too wrong, the cheapest fix is one more reflex rule. Repeat
  that enough times and the project is a scripted bot with a vestigial LLM
  attached — which would make the write-up dishonest even if the code worked.
  The mitigation is to treat the reflex/LLM decision split as a first-class
  metric and log it per session, so the erosion is visible rather than
  discovered later.
- **No single place answers "why did it do that".** The causal chain crosses a
  queue and a process boundary. The trace log is the only reconstruction, so it
  is load-bearing rather than a nicety.
