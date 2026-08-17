## What and why

<!-- What changes, and what problem it solves. Link the roadmap phase if relevant. -->

## Verification

- [ ] `python -m pytest` green
- [ ] `ruff check` and `ruff format --check` clean
- [ ] `python -m mypy --strict src/gielinor` clean

## Safety review

Tick only what applies, and say why in the box below if any are ticked. These
are the changes that alter what the agent is *able* to do, as opposed to what it
does — see SECURITY.md.

- [ ] Adds or changes a member of `ActionKind`
- [ ] Changes a rule in `actuator/validate.py`
- [ ] Changes a default for `allow_real_input` or `allow_chat`
- [ ] Changes how the API key is read, stored, or passed
- [ ] Sends new data to a third-party API
- [ ] Touches `tests/test_safety_invariants.py`

<!-- If any box above is ticked, explain the reasoning here. A safety invariant
     changing is fine when it is deliberate; it is not fine when it is silent. -->

## Calibration

- [ ] Does not touch `assets/profiles/*.toml`
- [ ] Touches a profile, and `tests/test_profiles.py` still passes
