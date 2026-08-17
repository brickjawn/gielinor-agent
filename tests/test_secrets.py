"""How the OpenRouter key gets in, and the ways it must not get out.

The interesting assertions here are the negative ones. A credential that loads
correctly is table stakes; a credential that never appears in a repr, a
model_dump, a trace record or a log line is the actual requirement, and those
are exactly the paths that regress silently because nothing fails when they do.

Every test isolates the environment. `Settings` reads process env, `.env` and
`secrets_dir`, so a leaked variable from one test changes the result of another
in a way that looks like flakiness.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from gielinor.config import Settings

KEY_ENV = "OPENROUTER_API_KEY"
FILE_ENV = "OPENROUTER_API_KEY_FILE"
# Deliberately NOT shaped like a real OpenRouter key. A realistic-looking fake
# in a committed test is a permanent false positive for every secret scanner
# that ever runs on this repo, and the code path under test does not validate
# key format / so realism buys nothing here.
SENTINEL = "test-key-not-a-real-credential"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No ambient key, and no repo .env, for any test in this module."""
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.delenv(FILE_ENV, raising=False)
    monkeypatch.chdir(tmp_path)


def _settings(**kw: object) -> Settings:
    # _env_file=None so a stray .env can never influence a result here.
    return Settings(_env_file=None, **kw)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_absent_key_is_not_an_error() -> None:
    """No key must degrade to "no slow loop", never to a crash.

    The reflex planner needs no network at all, so a missing credential costs a
    capability and nothing else. Failing hard here would turn a degraded run
    into an outage.
    """
    s = _settings()
    assert s.has_api_key() is False
    assert s.openrouter_api_key is None
    assert s.api_key_source() == "unset"


def test_missing_secrets_dir_emits_no_warning() -> None:
    """/run/secrets does not exist off-container; that is normal, not notable.

    A UserWarning on every single Settings() construction trains people to
    ignore warnings, which is how the one that matters gets missed.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _settings()
    assert not [w for w in caught if "run/secrets" in str(w.message)]


@pytest.mark.parametrize("filename", ["openrouter_api_key", "OPENROUTER_API_KEY"])
def test_secrets_dir_both_casings(tmp_path: Path, filename: str) -> None:
    """Docker secrets are conventionally lowercase; env vars are SHOUTY.

    Accepting only one casing means a correctly-declared Compose secret is
    silently not found, which presents to the user as "my key doesn't work".
    """
    d = tmp_path / "secrets"
    d.mkdir()
    (d / filename).write_text(SENTINEL)
    s = _settings(_secrets_dir=str(d))
    assert s.has_api_key()
    assert s.openrouter_api_key is not None
    assert s.openrouter_api_key.get_secret_value() == SENTINEL


def test_trailing_newline_is_stripped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`echo key > file` appends \\n, and a key with \\n fails auth as a 401.

    That 401 is indistinguishable from a wrong key, so this is worth an
    explicit test rather than trusting it stays right.
    """
    f = tmp_path / "key.txt"
    f.write_text(SENTINEL + "\n")
    monkeypatch.setenv(FILE_ENV, str(f))
    s = _settings()
    assert s.openrouter_api_key is not None
    assert s.openrouter_api_key.get_secret_value() == SENTINEL


def test_file_indirection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "key.txt"
    f.write_text(SENTINEL)
    monkeypatch.setenv(FILE_ENV, str(f))
    s = _settings()
    assert s.has_api_key()
    assert s.api_key_source() == "env:OPENROUTER_API_KEY_FILE"


def test_env_beats_file_indirection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An operator debugging with `OPENROUTER_API_KEY=... docker compose up`
    expects that to win over whatever the deployment configured."""
    f = tmp_path / "key.txt"
    f.write_text("sk-or-v1-from-the-file")
    monkeypatch.setenv(FILE_ENV, str(f))
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    s = _settings()
    assert s.openrouter_api_key is not None
    assert s.openrouter_api_key.get_secret_value() == SENTINEL
    assert s.api_key_source() == "env"


@pytest.mark.parametrize(
    ("what", "make"),
    [
        ("missing", lambda p: p / "nope.txt"),
        ("empty", lambda p: _write(p / "empty.txt", "")),
        ("whitespace only", lambda p: _write(p / "ws.txt", "   \n\t ")),
        ("a directory", lambda p: _mkdir(p / "adir")),
    ],
)
def test_unusable_secret_file_degrades_quietly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str, make: object
) -> None:
    """Every malformed-secret case lands on "no key", never on a traceback."""
    target = make(tmp_path)  # type: ignore[operator]
    monkeypatch.setenv(FILE_ENV, str(target))
    s = _settings()
    assert s.has_api_key() is False, f"{what} should not produce a key"


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


def _mkdir(p: Path) -> Path:
    p.mkdir()
    return p


# --------------------------------------------------------------------------
# Not leaking
# --------------------------------------------------------------------------


def test_key_absent_from_repr_and_dump(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    s = _settings()
    assert s.has_api_key()
    for rendered in (repr(s), str(s), str(s.model_dump()), s.model_dump_json()):
        assert SENTINEL not in rendered


def test_key_absent_from_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Construction and source-reporting must not log the value.

    api_key_source() exists precisely so that a startup line can say *where* the
    key came from without saying what it is — seeing "secrets_dir" when you
    expected "env" is most of the debugging on a container that isn't picking
    up its credential.
    """
    monkeypatch.setenv(KEY_ENV, SENTINEL)
    with caplog.at_level("DEBUG"):
        s = _settings()
        source = s.api_key_source()
    assert source == "env"
    assert SENTINEL not in caplog.text


def test_source_reports_mechanism_not_value(tmp_path: Path) -> None:
    d = tmp_path / "secrets"
    d.mkdir()
    (d / "openrouter_api_key").write_text(SENTINEL)
    s = _settings(_secrets_dir=str(d))
    source = s.api_key_source(secrets_dir=d)
    assert source.startswith("secrets_dir:")
    assert SENTINEL not in source


def test_source_admits_ignorance_rather_than_guessing(tmp_path: Path) -> None:
    """When the secrets dir isn't disclosed, say unknown — do not claim "init".

    A debugging aid that confidently reports the wrong source costs more time
    than one that shrugs.
    """
    d = tmp_path / "secrets"
    d.mkdir()
    (d / "openrouter_api_key").write_text(SENTINEL)
    s = _settings(_secrets_dir=str(d))
    assert s.api_key_source() == "supplied-directly-or-unknown"


# --------------------------------------------------------------------------
# The build context
# --------------------------------------------------------------------------


def test_dockerignore_excludes_secrets() -> None:
    """A `COPY . .` bakes credentials into a layer that survives image deletion.

    The Dockerfile copies named paths rather than `.`, so this is the second
    guard — and the point of a second guard is that it holds when the first one
    is edited by someone who didn't know it was load-bearing.
    """
    root = Path(__file__).resolve().parent.parent
    ignored = (root / ".dockerignore").read_text().splitlines()
    patterns = {line.strip() for line in ignored if line.strip() and not line.startswith("#")}
    for required in (".env", "secrets/", "*.key", "*.pem"):
        assert required in patterns, f".dockerignore must exclude {required}"


def test_env_example_carries_no_real_key() -> None:
    """.env.example is committed; .env is not. Keep it that way."""
    root = Path(__file__).resolve().parent.parent
    text = (root / ".env.example").read_text()
    assert "sk-or-v1-" not in text.replace("sk-or-v1-...", "")
