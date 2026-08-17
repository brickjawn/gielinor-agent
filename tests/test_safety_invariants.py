"""Invariants that CI exists to protect, as opposed to behaviour it verifies.

Ordinary tests fail when code is wrong. These fail when code is *dangerous*, and
they are deliberately annoying to satisfy: each one pins a property that is
cheap to erode by accident and expensive to notice afterwards.

The distinction matters for how you respond to a failure here. A normal test
going red means "you broke something, fix it". One of these going red means
"you changed the security posture of the project — was that deliberate?" The
answer is sometimes yes, and then you update the pin in the same commit, with
the reasoning in the message. What must not happen is the posture drifting
without anyone deciding.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from gielinor.actuator import open_actuator
from gielinor.config import ActuatorSettings, Settings
from gielinor.contracts import ActionKind

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src" / "gielinor"


# --------------------------------------------------------------------------
# The capability boundary
# --------------------------------------------------------------------------

# Every verb the agent is physically able to perform. This set IS the blast
# radius: an LLM reading attacker-controlled game chat can only ever ask for
# something in here, so what is absent matters more than what is present.
#
# Conspicuously absent, and intended to stay that way for v1: dropping items,
# trading, banking, entering a bank PIN, logging out, using an item on another
# player. ADR 0003's argument is that capability restriction beats content
# filtering — a prompt injection saying "drop your items" fails because there is
# no drop verb to reach for, not because a regex caught the string.
EXPECTED_ACTION_KINDS = frozenset(
    {
        "click_object",
        "click_ui",
        "select_dialogue",
        "continue_dialogue",
        "type_chat",
        "walk_minimap",
        "wait",
        "noop",
    }
)

# Verbs that would meaningfully widen the blast radius. Named explicitly so that
# adding one trips a test whose failure message says why it is a big deal,
# rather than only tripping the generic count check.
FORBIDDEN_VERB_HINTS = (
    "drop",
    "trade",
    "bank",
    "pin",
    "logout",
    "log_out",
    "sell",
    "buy",
    "destroy",
    "alch",
    "duel",
    "follow",
    "run_shell",
    "exec",
)


def test_action_kind_set_is_pinned() -> None:
    """Adding a verb must be a deliberate act, not a side effect.

    If this fails because you added a capability on purpose: update
    EXPECTED_ACTION_KINDS in the same commit, add the matching rule to
    actuator/validate.py, and say in the commit message what the new verb can do
    that the old ones could not.
    """
    actual = {k.value for k in ActionKind}
    added = actual - EXPECTED_ACTION_KINDS
    removed = EXPECTED_ACTION_KINDS - actual
    assert not added, (
        f"new capability granted to the planner: {sorted(added)}. "
        "This widens what a remote model reading untrusted chat text can ask for. "
        "See ADR 0003 before pinning it."
    )
    assert not removed, f"capability removed: {sorted(removed)} — update the pin if intended"


@pytest.mark.parametrize("hint", FORBIDDEN_VERB_HINTS)
def test_no_high_risk_verb_exists(hint: str) -> None:
    """No action kind may hint at irreversible or item-losing operations.

    These are the verbs a prompt injection would actually want. The defence is
    that they do not exist.
    """
    offenders = [k.name for k in ActionKind if hint in k.value.lower()]
    assert not offenders, (
        f"ActionKind contains {offenders}, matching high-risk verb {hint!r}. "
        "An irreversible verb reachable from the planner needs a threat-model "
        "update in ADR 0003, not just a validator rule."
    )


def test_every_action_kind_has_a_validation_rule() -> None:
    """A verb with no validator is a verb that accepts anything.

    validate.py builds a dispatch table and refuses to import if a kind is
    missing, so this mostly guards against that guard being removed.
    """
    from gielinor.actuator import validate as validate_mod

    source = (Path(validate_mod.__file__)).read_text()
    for kind in ActionKind:
        assert kind.name in source or kind.value in source, (
            f"{kind.name} has no reference in actuator/validate.py"
        )


# --------------------------------------------------------------------------
# Real input stays off unless someone means it
# --------------------------------------------------------------------------


def test_real_input_defaults_off_in_settings() -> None:
    assert ActuatorSettings().allow_real_input is False
    assert ActuatorSettings().backend == "dryrun"
    assert Settings(_env_file=None).actuator.allow_real_input is False  # type: ignore[call-arg]


def test_chat_defaults_off() -> None:
    """TYPE_CHAT exists but is disabled by default.

    Chat is the one verb that emits attacker-visible output, so it is the one
    where an injection could get leverage even without an item-losing verb.
    """
    assert ActuatorSettings().allow_chat is False


def test_default_actuator_cannot_reach_a_device() -> None:
    """`open_actuator()` with no arguments must not produce something that clicks."""
    actuator = open_actuator()
    assert type(actuator).__name__ == "DryRunActuator"


def test_real_backend_refuses_without_explicit_flag() -> None:
    """Naming a real backend is not sufficient; the flag is a second, separate lock.

    Two independent refusals means a single misconfigured env var cannot arm
    the thing that moves your mouse.
    """
    for backend in ("uinput", "quartz"):
        with pytest.raises((RuntimeError, ValueError, PermissionError, NotImplementedError)):
            open_actuator(backend, allow_real_input=False)


# --------------------------------------------------------------------------
# Coordinates live in profiles, not in code
# --------------------------------------------------------------------------

# Files legitimately allowed to construct geometry from literals: the synthetic
# frame renderer has to draw a client-sized image, layout holds the fallback
# defaults, and the bridge maps between the two representations.
_COORD_ALLOWLIST = {
    "perception/synthetic.py",
    "perception/layout.py",
    "profiles.py",
    "config.py",
}


def _module_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts)


def test_no_inline_screen_coordinates() -> None:
    """Geometry may only enter the system through a profile.

    The first version of this check matched bare integers in screen-sized
    ranges and was useless: it flagged HTTP 500/502/503/504 in the retry logic
    and 512 as a buffer size. A check that cries wolf gets suppressed, and a
    suppressed check protects nothing.

    So it matches *structure* instead — the two shapes an actual violation
    takes. Someone hardcoding geometry writes either `Rect(548, 205, 190, 261)`
    or a literal crop like `img[345:475, 7:513]`. Both are unambiguous; a bare
    number is not.
    """
    offenders: list[str] = []
    for path in _module_files():
        rel = path.relative_to(SRC).as_posix()
        if rel in _COORD_ALLOWLIST:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            # Rect(x, y, w, h) built entirely from literals.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in {"Rect", "Point"}
                and len(node.args) >= 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, int) for a in node.args)
            ):
                literals = [a.value for a in node.args if isinstance(a, ast.Constant)]
                if any(v > 32 for v in literals):
                    offenders.append(f"{rel}:{node.lineno} -> {node.func.id}{tuple(literals)}")
            # A literal array crop: img[345:475, 7:513].
            #
            # BOTH bounds must be literal ints. Requiring both is what separates
            # a crop from a string truncation -- `reason[:64]` and
            # `rationale[:200]` are all over the planner and are not geometry.
            # An earlier version flagged them and would have been deleted within
            # a week for crying wolf.
            if isinstance(node, ast.Slice):
                lower, upper = node.lower, node.upper
                if (
                    isinstance(lower, ast.Constant)
                    and isinstance(lower.value, int)
                    and isinstance(upper, ast.Constant)
                    and isinstance(upper.value, int)
                    and max(lower.value, upper.value) > 32
                ):
                    offenders.append(
                        f"{rel}:{node.lineno} -> literal crop [{lower.value}:{upper.value}]"
                    )
    assert not offenders, (
        "geometry built from literals outside a calibration profile:\n  "
        + "\n  ".join(offenders)
        + "\nAsk a Profile or LayoutProfile for the rect instead, "
        "or allowlist the file with a reason."
    )


# --------------------------------------------------------------------------
# Nothing that holds a credential may print one
# --------------------------------------------------------------------------


# Identifiers that plausibly *hold* a credential value, as opposed to naming one.
# Bare `key` is deliberately absent. In this codebase it means a keyboard key
# (uinput, quartz) or a dict key (layout, cli) far more often than a credential,
# and including it produced four false positives and zero true ones.
_SECRET_BEARING = {"api_key", "_api_key", "apikey", "api_token", "token", "secret", "credential"}


def test_no_module_logs_a_credential_value() -> None:
    """Catch credential logging before review has to.

    The naive version of this check matched the words "secret" or "API_KEY"
    anywhere in a logging call, which flagged every *correct* line in the
    codebase — `logger.warning("secret file %s is empty", path)` logs a path,
    and `logger.error("--llm needs OPENROUTER_API_KEY")` logs a variable name.
    Both are exactly what you want.

    What actually leaks a key is *interpolating a variable that holds one*. So
    this inspects the call's arguments rather than its message text: a Name or
    Attribute argument whose identifier looks credential-bearing, or an f-string
    interpolating one. That distinction is the whole difference between a check
    people keep and a check people delete.
    """
    offenders: list[str] = []
    for path in _module_files():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_log = (isinstance(func, ast.Attribute) and func.attr in _LOG_METHODS) or (
                isinstance(func, ast.Name) and func.id == "print"
            )
            if not is_log:
                continue
            for name in _interpolated_names(node):
                if name.lower() in _SECRET_BEARING:
                    offenders.append(
                        f"{path.relative_to(SRC).as_posix()}:{node.lineno}: "
                        f"logs credential-bearing name {name!r}"
                    )
    assert not offenders, (
        "credential value passed to a log call:\n  "
        + "\n  ".join(offenders)
        + "\nLog the *source* of a key (see Settings.api_key_source), never the value."
    )


_LOG_METHODS = {"debug", "info", "warning", "warn", "error", "exception", "critical", "log"}


def _interpolated_names(call: ast.Call) -> list[str]:
    """Identifiers whose *value* reaches the formatted output of a log call."""
    names: list[str] = []

    def collect(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)

    # Positional args after the format string, plus anything inside an f-string.
    for arg in call.args[1:]:
        collect(arg)
    for arg in call.args:
        if isinstance(arg, ast.JoinedStr):
            for part in arg.values:
                if isinstance(part, ast.FormattedValue):
                    collect(part.value)
    return names


def test_no_print_in_library_code() -> None:
    """print() bypasses log level and redaction, and in a container it is the
    difference between a filtered log stream and unfiltered stdout."""
    offenders: list[str] = []
    for path in _module_files():
        if path.relative_to(SRC).as_posix().startswith(("runtime/cli.py",)):
            continue  # a CLI is allowed to talk to a terminal
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                offenders.append(f"{path.relative_to(SRC).as_posix()}:{node.lineno}")
    assert not offenders, f"print() in library code: {offenders}"


# --------------------------------------------------------------------------
# The repo itself
# --------------------------------------------------------------------------


def test_gitignore_covers_secrets_and_captures() -> None:
    patterns = {
        line.strip()
        for line in (REPO / ".gitignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for required in (".env", "secrets/"):
        assert required in patterns, f".gitignore must cover {required}"


def test_no_credential_shaped_string_is_committed() -> None:
    """A cheap in-repo backstop so the check does not depend on CI being wired.

    gitleaks in CI is the real scanner; this catches the specific token shapes
    this project could plausibly leak, and runs on every local `pytest`.
    """
    shapes = (
        re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
        re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    )
    skip_dirs = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
    offenders: list[str] = []
    for path in REPO.rglob("*"):
        if not path.is_file() or set(path.parts) & skip_dirs:
            continue
        if path.suffix in {".npz", ".png", ".zip", ".jpg"}:
            continue
        if path.name == "test_safety_invariants.py":
            continue  # the patterns above would match themselves
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for shape in shapes:
            if shape.search(text):
                offenders.append(f"{path.relative_to(REPO)} matches {shape.pattern}")
    assert not offenders, "credential-shaped string committed:\n  " + "\n  ".join(offenders)
