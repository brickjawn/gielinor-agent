# ADR 0005: Interruptible actuation and stable identity

- **Status:** accepted
- **Date:** 2026-09-20
- **Amends:** [ADR 0004](0004-process-split-not-container-split.md) (actuator contract),
  [ADR 0001](0001-two-loop-architecture.md) (escalation policy)
- **Supersedes nothing.** ADR 0003's capability restriction is unchanged and is
  relied on here.

## Context

ADR 0001 split the agent into a reflex loop and a remote planner because a 600 ms
tick cannot accommodate a 0.7–6 s round trip. The escalation path was then guarded
by three checks, all of which run **before** the action starts:

1. **Dual-clock staleness** — the GameState the planner saw must be no more than
   `MAX_STALE_FRAMES` (30) frames and `MAX_STALE_S` (3.0) seconds old.
2. **Referential validity** — `validate_action` re-resolves every `oid` in the
   action against the *current* GameState.
3. **Semantic preconditions** — inventory space for a gather, dialogue open for
   a dialogue reply, and so on.

Three problems with that, in increasing order of severity.

### The budget is smaller than the latency it guards, and by accident

At the 15 Hz capture rate, `MAX_STALE_FRAMES = 30` is a **2.0 s** ceiling.
`MAX_STALE_S = 3.0` never binds. So the effective budget is a frame count that
nobody chose as a time, sitting inside a latency distribution of 0.7–6 s.

The majority of escalations therefore return an answer that is discarded on
arrival. Every one of those cost a request against a 1,000/day account-wide
ceiling (see [cost-model.md](../cost-model.md)), and requests are the scarce
resource. The system is designed to spend its scarcest input on answers it throws
away.

### Validity is asserted once, but has to hold for a second

`humanize.py` spends 200–1000 ms moving the cursor before it clicks. All three
checks above have finished by the time the first pixel of that movement happens.
The window between "this action is valid" and "this action takes effect" is
therefore the whole trajectory, and nothing looks again inside it. The tree gets
chopped by another player, an NPC walks onto the tile, the camera rotates — and
the click lands anyway, with every gate green.

ADR 0004 makes this worse rather than better: the actuator is a separate process
with no GameState, so even in principle it cannot re-check. The boundary that
protects the input privilege also blinds the thing holding it.

### `oid` reuse turns a dropped action into a wrong action

`validate_action` is only as good as the binding from `oid` to object. If oids
come from a per-frame detection index — or from any pool that recycles — then
oid 7 being "a tree" when the planner was asked and "a different tree" when the
click lands is a case where **every check passes**. A missing oid fails loudly.
A recycled oid fails silently, and silence is what makes it the dangerous one.

## Decision

Three changes. The first two are mechanism; the third is policy and is the one
that actually fixes the economics.

### 1. Validation becomes continuous, and actuation becomes interruptible

The actuator socket gains sequence-numbered frames (`EXEC`, `ABORT`, `RESULT`,
`ERROR`) and the actuator reads them **continuously while executing**, so an
`ABORT` overtakes the command it withdraws instead of queueing behind it. On the
supervisor side, `ActionGuard` re-runs the same `validate_action` once per fresh
frame for as long as a command is in flight, and withdraws on the first failure.

Inside the actuator, the trajectory is interruptible and the click is not:

- Before each waypoint, the abort flag is checked.
- Dwells are a race against the abort event, not a sleep, so a withdrawal during
  a 400 ms pause is honoured immediately rather than after the pause.
- The final check happens immediately before the click. The click itself
  completes — there is no useful state between press and release.

Sequence numbers come from a never-recycling counter, so an `ABORT` that arrives
after its command finished names a sequence that is no longer current and is
discarded. Without that, a late withdrawal for action *N* silently cancels the
perfectly valid action *N+1*.

One subtlety that is easy to get wrong and is pinned by a test: the current
command is registered **at dispatch, synchronously**, not inside its task.
Otherwise an `ABORT` arriving in the same event-loop iteration as its `EXEC`
finds nothing current, is counted stale, and the action proceeds — and the
fastest withdrawals come from the fastest-changing state, so this race selects
for exactly the cases that matter.

Measured on this protocol (200 iterations, 24-waypoint / 600 ms trajectory):

| Segment | Median | p95 |
| --- | --- | --- |
| `abort.set()` → trajectory stopped, in-process | 38 µs | 47 µs |
| Supervisor `write()` → `RESULT` received, over the Unix socket | 332 µs | 430 µs |
| Last abort check → click driver entered | 8 µs | 11 µs |

Reproduce with `scripts/bench_abort_window.py`. These are the protocol's own
overhead only; the OS synthetic-input call has to be measured on the machine
that runs the agent and added to the third row.

So the exposure drops from *the whole trajectory* (200–1000 ms) to *the commit
gap* (~10 µs plus the input backend). Not zero. Bounded, small, and measured,
which is the most an asynchronous world offers.

### 2. Identity is monotonic, appearance-checked, and confirmed by the client

- `oid` comes from a `MonotonicCounter`. There is no `free()`. A retired oid is
  retired forever, and `IdentityRegistry.was_retired()` lets the validator tell
  "the world moved" (expected, count it) from "the planner invented a reference"
  (a model-behaviour signal, log it).
- Association combines IoU, a 4×4×4 quantised RGB histogram, and bounding-box
  shape. Every candidate pair must clear a **gate** before it can be matched at
  all, so a merely-best-available pair still loses. On gate failure the detection
  gets a **new oid** rather than inheriting an old one — a spurious new oid costs
  one dropped escalation, a wrong match costs a wrong click, and the asymmetry
  points one way.
- Immediately before clicking, the actuator reads the client's own hover text
  ("Chop down Tree") and requires an exact match after normalisation. This is
  the only check in the system that interrogates the **actual** click target
  rather than a model of it, and it catches the residual case where GameState is
  still valid but the projection from object to pixel is not.

The hover check does not give the actuator judgement. It gets one rect, one
glyph-matched string, and an equality comparison — no `GameState`, no
`Perceiver`. ADR 0004's boundary survives intact.

### 3. Escalate only on questions whose answers do not expire

Mechanism 1 stops a stale action from executing. It does **not** recover the
request that produced it. Cancelling an in-flight HTTP request frees a
concurrency slot and saves wall-clock, but the quota was consumed the moment the
request was issued; there is no refund, and claiming otherwise would make the
cost model wrong in the optimistic direction. So early cancellation is worth
implementing — and it is, via task cancellation on the `httpx` call, scoped to
the premise — but it is a latency optimisation, not a budget one.

The budget fix is upstream, and it is a rule about what escalation is *for*:

> An escalation must ask a question whose answer is still correct when it
> arrives. If the premise has a shorter expected lifetime than the p95 planner
> latency, it is not an escalation — it is a reflex the FSM has to answer, or a
> decision that must be made before the premise forms.

"Which tree should I work on next" survives six seconds. "Is there a click target
at these coordinates right now" does not, and must never leave the machine. In
practice this means escalations are **goal-level** and are issued at goal
boundaries, where the agent has time to spare, rather than mid-activity where it
does not.

`MAX_STALE_FRAMES` and `MAX_STALE_S` are consequently both wrong today, because
one is a frame count that became a time by accident and the other never binds.
Both get derived from a single measured latency distribution in Phase 4
(`bench_planner_latency.py`), set at the p95 of *measured* round trip on the
agent's own machine, and expressed in one unit with the other derived from the
capture rate.

This also sharpens the case for the middle tier discussed in the Jev notes: a
70–500 ms typed decision model sits comfortably inside any plausible budget,
which turns the discard rate from a majority into a rounding error. That is a
better argument for a three-tier planner than "it is faster" was.

## Consequences

**Good.**

- Validity becomes a property maintained over an action's lifetime rather than
  asserted at its start.
- The dangerous failure (right check, wrong object) is structurally excluded
  rather than mitigated: no recycled identifier exists to be confused.
- `ActionGuard.aborts_sent`, `ActuatorServer.stale_aborts`, and
  `Track.occlusions` are all cheap trace fields that make "the agent behaved
  oddly" diagnosable instead of mysterious.
- The actuator now has a defensible reason to refuse: `REJECTED_HOVER` is the
  client disagreeing with us, which is worth surfacing.

**Costs and open edges.**

- More moving parts in the process that holds the input privilege. Mitigated by
  the surface staying tiny — the actuator gained one rect and one string compare,
  not a perception system.
- `FAILED` with `clicked=True` is deliberate. If a click driver raises, the press
  may have landed, and an action of unknown effect must be treated as having
  happened or the supervisor double-acts. This makes some failures
  non-idempotent by design.
- A withdrawal is a request, not a guarantee. The supervisor must read
  `clicked`, not the status string, and `Outcome.clicked` exists separately from
  `Outcome.status` for exactly this reason.
- Concurrent `EXEC` is refused rather than queued. One cursor, one command. A
  queue would reintroduce the staleness problem with extra steps.
- The association thresholds (`MIN_IOU`, `MAX_APPEARANCE_DIST`, `MAX_COST`,
  `DEFAULT_MAX_MISSES`) are chosen, not measured. They are deliberately tight,
  because a tight gate fails toward a new oid and a loose one fails toward a
  wrong oid. Phase 1 replaces them with values fitted to labelled frames.
- Greedy matching, not Hungarian. At RS2 viewport object counts the two agree
  almost always, and where they differ the gate rejects the marginal pair anyway.
  Revisit if labelled data says otherwise; do not add scipy speculatively.

## Verification

39 tests across `tests/test_oid_monotonicity.py`, `tests/test_actuator_abort.py`
and `tests/test_premise_cancellation.py`. They assert `clicked is False` rather
than status strings, and run against a real Unix socket and a real event loop,
because the central claim is a concurrency property of the read loop and a
mocked transport would test a different program.

A green suite on first run is not evidence, so eight mutations were introduced
to confirm the tests detect the defects they describe rather than passing by
construction:

| Mutation | Tests failed |
| --- | --- |
| Recycle oids from a free pool | 3 |
| Register the current command inside the task | 1 |
| Read loop awaits execution (ADR 0004's original shape) | 4 |
| Drop the final abort check before the click | 1 |
| Dwell becomes a plain `sleep` | 1 |
| Remove the association gate | 1 |
| Remove the hover comparison | 2 |
| Decrement `requests_issued` on cancellation | 1 |

Each mutation is the mistake the corresponding design point exists to prevent.
`ruff check`, `ruff format --check` and `mypy --strict src/gielinor` are clean.
