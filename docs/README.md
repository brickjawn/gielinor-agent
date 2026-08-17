# Documentation index

Design records and engineering notes for the Gielinor agent — a pixels-only
agent that plays a locally-run 2009scape server by screen-reading and clicking.
Start with [`SPEC.md`](../SPEC.md) for the architecture and
[`src/gielinor/contracts.py`](../src/gielinor/contracts.py) for the interface;
these documents explain why both look the way they do.

## Architecture decision records

| ADR | Title | Summary |
| --- | --- | --- |
| [0001](adr/0001-two-loop-architecture.md) | Two loops: a local reflex loop and a remote planner loop | A 600 ms game tick cannot accommodate a 700 ms-6 s model round trip, so the agent splits into a 20 Hz local loop and an event-driven remote one — with the rejected alternatives, including a local 7-12B VLM, costed out. |
| [0002](adr/0002-glyph-ocr-over-general-ocr.md) | Exact glyph-bitmap matching instead of general-purpose OCR | The client renders a 12-16 px bitmap font with no anti-aliasing, which makes the screenshot a lossless encoding of the string rather than a noisy observation of it — so exact matching inverts a known encoder rather than guessing. |
| [0003](adr/0003-untrusted-planner.md) | The LLM planner is an untrusted component | In-game chat is attacker-controlled text and synthetic input is a real OS privilege, so the planner gets symbolic-only targets, a closed eight-member capability enum, and no input handle: capability restriction beats content filtering. |

## Engineering notes

| Document | Summary |
| --- | --- |
| [latency-budget.md](latency-budget.md) | Line-item breakdown of the 50 ms fast-loop target and the 0.7-6 s slow loop, with every figure flagged as an estimate, plus measurement methodology and why `time.time()` is the wrong clock. |
| [platform-notes.md](platform-notes.md) | Practical setup for Wayland (portal ScreenCast + PipeWire, `/dev/uinput` udev rule, compositor differences) and macOS (ScreenCaptureKit, the two separate TCC grants, and the gotchas that produce plausible-looking wrong images). |
| [deployment.md](deployment.md) | The host/container split and why it falls out of ADR 0003 rather than being chosen — the component holding the API key cannot move your mouse, and the one that can holds no credential. Covers the four ways the key can arrive and why a file beats an environment variable. |
| [roadmap.md](roadmap.md) | Phases 0-7 with explicit exit criteria. Ordered so calibration mistakes surface before real input exists, and structured around the one bridge between the sandbox and a real machine: recorded frames in, traces out. |

## Reading order

For a reviewer with fifteen minutes: ADR 0001, then the "Why 50 ms" and
"Measurement methodology" sections of the latency budget, then ADR 0003.

For picking the work back up: [roadmap.md](roadmap.md), which says what is
measured versus assumed and what the next phase actually is.
