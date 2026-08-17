# ADR 0004 — Isolate the actuator by process, not by container

- Status: accepted
- Date: 2026-08-17
- Deciders: project lead
- Related: [ADR 0003](0003-untrusted-planner.md) (the privilege this isolates),
  [deployment.md](../deployment.md), [roadmap.md](../roadmap.md) (Phase 5)
- Supersedes: the container-as-deployment-shape framing in the first draft of
  `deployment.md`

## Context

ADR 0003 establishes that synthetic input is a real OS privilege and that the
planner is untrusted. It concludes that the actuator should not live in the same
process as the thing reading attacker-controlled chat text. That conclusion is
sound and is not revisited here.

What is revisited is how far to take it. An earlier draft went further and put
perception, the planner and the runtime in a container, leaving capture and the
actuator on the host — a split justified on the grounds that capture and input
need host devices anyway, so the boundary is free.

It is not free, and the review that prompted this ADR asked the right question:
is the system being fragmented in the name of efficiency? Worth answering
directly, because the answer is no on two counts. Nothing here is about
efficiency — the two-loop split is forced by a 600 ms tick meeting a
multi-second round trip, and this boundary is about privilege. And the
fragmentation was not all one decision. Four boundaries had been conflated:

| Boundary | Nature |
| --- | --- |
| Reflex loop vs LLM loop | Forced by latency. Not a decision. |
| Development sandbox vs target machine | Forced by hardware. Not a decision. |
| Actuator as its own process | A choice. Cheap. |
| Compute half in a container | A choice. Not cheap. |

Only the last two were ever decisions, and they had been bundled together as
though accepting one meant accepting both.

## Decision

**Keep the process boundary. Drop the container from the development and
deployment path.**

The actuator runs as a separate process on the host, behind a Unix socket
speaking the validated `Action` schema. Perception, the planner and the runtime
run in one ordinary virtual environment on the same machine.

The `Dockerfile` and `docker-compose.yml` stay in the repository as the CI
environment definition and as a reference deployment. They are not part of the
development loop.

## Rationale

**The security property comes from the process split, not the container.** The
guarantee worth having is that the component holding the API key cannot reach an
input device, and that the component that can reach one holds no credential.
Two processes achieve that. A container around one of them adds namespace
isolation between two pieces of software written by the same person, running as
the same user, on a machine they already fully control. That is not where the
risk was.

**The process split pays for itself twice.** Phase 5 needs a kill switch that
works when the agent loop is wedged — precisely the case where an in-process
handler would be stuck too. A separate actuator process *is* that kill switch:
signal it, or close the socket, and input stops regardless of what the planner
is doing. The isolation and the safety mechanism are the same piece of work,
which is what makes this boundary cheap where the container's is not.

**The container's costs land in the worst phase.** Containerizing means IPC for
the whole state-and-action path rather than just the input hop, serialization
inside a 50 ms frame budget, version skew between two artifacts, and no single
stack trace. Phase 5 and 6 are when "the click landed 4 px off the tree" gets
debugged, and that is the bug that suffers most from crossing a process boundary
for reasons unrelated to it.

**Nothing had been built, so the choice was still free.** The Dockerfile and the
deployment doc existed; the IPC did not. Deciding now costs nothing, whereas
discovering the same conclusion in Phase 6 would mean unpicking a protocol.

## What was rejected, and what would reverse it

**Full container split.** Rejected for the reasons above, not because it is
wrong in general — for a multi-tenant or multi-operator deployment it would be
right, and the secrets mechanism it motivated is worth keeping either way. It
becomes correct again if the agent ever runs somewhere the operator does not
control, or if a second person runs it.

**Single process throughout.** Rejected because it forfeits the Phase 5 kill
switch, and because the coupling is easy to introduce and hard to remove:
in-process code accretes shared state within days, and the boundary gets more
expensive to add every week it is deferred.

**A devcontainer as a third environment.** Rejected by default. It earns its
place only for work away from the Linux box, and the environment it would
reproduce is not the one that matters — Codespaces is headless, so
`PipeWirePortalCapture` and `UinputActuator`, the two most platform-fragile
components in the system, can never be exercised there.

## Consequences

Good:

- One development environment instead of three. One venv, one machine, one
  `pytest` run.
- The kill switch and the privilege boundary are the same mechanism, so neither
  is a bolt-on.
- Input bugs are debuggable in a single process for everything except the final
  hop, and that hop is small and well-defined.
- The container remains available and documented, so the option is preserved
  rather than closed.

Bad, and accepted:

- The privilege boundary is weaker than it looks. Two processes under the same
  uid on the same machine is an architectural boundary, not a security one — the
  planner process could read the actuator's socket, or its memory, if it were
  actively hostile. It defends against a confused planner, not a compromised
  one. A separate uid would make it real; v1 does not do that, and the honest
  framing is that this raises the cost of a mistake rather than preventing an
  attack.
- The Dockerfile is now less exercised than the code it builds. CI builds it, but
  nobody runs the agent that way day to day, so it will rot slowly. The
  mitigation is that CI catches a broken build; nothing catches a subtly wrong
  runtime configuration.
- `secrets_dir` and the `_FILE` convention were motivated by Compose secrets and
  are now used without Compose. They still work — `secrets_dir` is only a
  directory path — but the plumbing is more general than the current deployment
  needs.
