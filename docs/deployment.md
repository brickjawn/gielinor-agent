# Deployment and secret handling

## The split, and why it falls out of the threat model

The agent does not containerize as one unit, and the line it splits along is not
arbitrary — it is the same line ADR 0003 already drew for security reasons.

| | Runs where | Needs | Holds the key? |
| --- | --- | --- | --- |
| capture | host | a display to read | no |
| perception | container | nothing but pixels | no |
| planner | container | the network | **yes** |
| runtime | container | nothing | no |
| actuator | host | an input device to write | no |

Capture and the actuator need host devices, so they cannot meaningfully be
containerized. That is a constraint, not a choice. But the useful part is what
the constraint produces for free: **the component that holds the API key has no
ability to move your mouse, and the component that can move your mouse holds no
credential.**

That is worth stating explicitly because it is the kind of property that is
cheap when the architecture is set up for it and nearly impossible to retrofit.
ADR 0003's argument is that capability restriction beats content filtering; this
is that argument applied one layer out, at the process and container boundary
rather than at the action enum.

The host-side actuator bridge — a Unix socket speaking the validated `Action`
schema — is specified but **not yet built**. It lands in Phase 5, which is the
first time real input exists at all. Until then the container runs
`DryRunActuator` and there is nothing on the other side of the socket to talk to.

## Getting the key in

Four mechanisms, in the order they are consulted:

1. **Passed in code.** Tests, and callers that already hold it.
2. **`OPENROUTER_API_KEY`** in the environment. Dev convenience and the operator
   override — `OPENROUTER_API_KEY=... docker compose up` beats the deployment's
   configuration, which is what you want at 2am.
3. **`OPENROUTER_API_KEY_FILE`** pointing at a path. The `_FILE` convention that
   Postgres, MySQL and most secret-aware images already use.
4. **`secrets_dir`** — a file named `openrouter_api_key` under `/run/secrets`,
   which is exactly where Compose mounts a declared secret, on tmpfs.

Both filename casings are accepted. Environment variables are conventionally
SHOUTY while Docker secrets are conventionally lowercase (`docker secret create
openrouter_api_key`), and supporting only one means a correctly-declared secret
is silently not found — which presents to the user as "my key doesn't work"
rather than as a naming mismatch.

### Why a file rather than an environment variable

An operational knob is fine in the environment. A credential is not, and the
reasons are concrete rather than stylistic:

- Environment variables appear in `docker inspect`.
- They are readable from `/proc/<pid>/environ` by anything running as the same
  user.
- They are captured by crash handlers and error reporters.
- They are inherited by **every** child process the agent spawns.

A file read once at startup into a `SecretStr` has none of those properties, and
`/run/secrets` is tmpfs, so the value never touches disk and is gone when the
container stops.

### Setup

```bash
mkdir -p secrets
printf '%s' "$OPENROUTER_API_KEY" > secrets/openrouter_api_key   # printf, not echo — no trailing newline
chmod 600 secrets/openrouter_api_key
docker compose up
```

`printf` rather than `echo` matters more than it looks. `echo` appends a
newline, and a key with `\n` on the end fails auth with a 401 that is
indistinguishable from a wrong key. The loader strips trailing whitespace
defensively for exactly this reason, and there is a test pinning that behaviour,
but it is still the single most common way this goes wrong.

`secrets/` is in both `.gitignore` and `.dockerignore`.

### Confirming it worked

`Settings.api_key_source()` reports the *mechanism*, never the value:

```
unset | env | env:OPENROUTER_API_KEY_FILE | secrets_dir:/run/secrets/... | dotenv
```

Seeing `secrets_dir` when you expected `env` is most of the debugging on a
container that will not pick up its credential. When it genuinely cannot tell —
a caller overrode `_secrets_dir` without saying so — it returns
`supplied-directly-or-unknown` rather than guessing, because a debugging aid
that confidently reports the wrong answer costs more time than one that shrugs.

## Container hardening

The compose file sets `read_only: true`, `cap_drop: ALL`,
`no-new-privileges:true`, a tmpfs `/tmp`, and memory and PID limits, and the
image runs as uid 10001. None of this is required to make the agent work — it is
the same capability-restriction argument applied to the runtime. The agent needs
no privileged operation inside the container, so granting one would be pure
downside.

Volumes are deliberately asymmetric: frames and profiles mount **read-only**
(the container has no business editing the corpus it is being evaluated
against), and `traces/` is the only writable mount, because a trace file is the
actual output of a run.

## What is not here, and why

**No secrets-manager SDK.** Vault, 1Password Connect and the cloud secret
managers all solve a problem this project does not have: rotating credentials
across many services owned by many people. There is one key, one consumer, one
operator. Compose secrets plus the `_FILE` convention covers it, and — this is
the part that matters — an injector that writes to either location composes with
this without a code change. If SOPS or Vault Agent gets added later, it writes
the file and the application is none the wiser.

**No secret in the image.** The Dockerfile copies named paths rather than
`COPY . .`, and `.dockerignore` excludes `.env`, `secrets/`, `*.key` and `*.pem`
as a second independent guard. Both matter: a credential baked into a layer
survives `docker rmi` of the tag, gets pushed to any registry the image reaches,
and stays readable in the layer history long after the file was "deleted".
`tests/test_secrets.py` asserts the `.dockerignore` entries still exist, because
the point of a second guard is that it holds when the first is edited by someone
who did not know it was load-bearing.
