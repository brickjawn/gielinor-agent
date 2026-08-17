"""Configuration: environment-driven settings plus on-disk calibration profiles.

Two different kinds of configuration live here and they are deliberately not the
same mechanism.

*Settings* are operational knobs (which backend, which models, timeouts). They
change per run, so they come from the environment with a ``GIELINOR_`` prefix
and nested ``__`` delimiters, e.g. ``GIELINOR_CAPTURE__FPS=30``. The one
exception is ``OPENROUTER_API_KEY``, which keeps its conventional unprefixed
name because that is what every other tool on the machine already calls it.

*Profiles* are screen geometry (viewport/chatbox/inventory/minimap rects). They
change per client build and per resolution, not per run, so they are TOML data
files under ``assets/profiles/``. SPEC section 1 makes retargeting a data change
rather than a code change: no coordinate may appear inline in a module, and the
only way to get one is to ask a Profile for it by name.

*Secrets* are a third thing again, and the difference matters once this runs in a
container. An operational knob is fine in the environment; a credential is not.
Environment variables are visible in ``docker inspect``, readable from
``/proc/<pid>/environ``, captured by crash handlers, and inherited by every
child process the agent spawns. A file read once at startup has none of those
properties. So the key can arrive four ways and they are ordered deliberately:

1. passed explicitly in code (tests, and callers that already hold it)
2. ``OPENROUTER_API_KEY`` in the environment (dev convenience, operator override)
3. ``OPENROUTER_API_KEY_FILE`` pointing at a path (the ``_FILE`` convention that
   Postgres, MySQL and most secret-aware images already use)
4. ``secrets_dir`` — a file named ``openrouter_api_key`` under ``/run/secrets``,
   which is exactly where Docker Compose mounts a declared secret, on tmpfs

Nothing here reaches for a secrets manager SDK. Compose secrets plus the
``_FILE`` convention covers the deployment shape this project actually has, and
an injector that writes to either location composes with it without a code
change. See ``docs/deployment.md``.
"""

from __future__ import annotations

import logging
import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from gielinor.contracts import Point, Rect

logger = logging.getLogger(__name__)

# Repository root as seen from an editable/source checkout. Packaged installs
# ship assets/ alongside the package, so both locations are searched.
_PKG_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_DIR.parent.parent
PROFILE_SEARCH_PATH: tuple[Path, ...] = (
    _REPO_ROOT / "assets" / "profiles",
    _PKG_DIR / "assets" / "profiles",
)

DEFAULT_PROFILE = "2009scape_fixed"

RegionName = Literal["viewport", "chatbox", "chat_input", "inventory", "minimap"]


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


class RectSpec(BaseModel):
    """A rectangle as it appears in a profile TOML: ``[x, y, w, h]`` semantics."""

    model_config = {"frozen": True, "extra": "forbid"}

    x: int = Field(ge=0)
    y: int = Field(ge=0)
    w: int = Field(gt=0)
    h: int = Field(gt=0)

    def to_rect(self) -> Rect:
        return Rect(self.x, self.y, self.w, self.h)


class InventoryGrid(BaseModel):
    """Where the 28 inventory slots sit, expressed as a grid rather than 28 rects.

    A grid is one row of TOML to recalibrate instead of 28, and it makes an
    off-by-one in the pitch obvious the moment slot 27 lands in the wrong place.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    origin_x: int = Field(ge=0)
    origin_y: int = Field(ge=0)
    cell_w: int = Field(gt=0)
    cell_h: int = Field(gt=0)
    pitch_x: int = Field(gt=0)
    pitch_y: int = Field(gt=0)
    cols: int = Field(default=4, gt=0)
    rows: int = Field(default=7, gt=0)

    @property
    def capacity(self) -> int:
        return self.cols * self.rows

    def slot_rect(self, index: int) -> Rect:
        if not 0 <= index < self.capacity:
            raise IndexError(f"inventory slot {index} outside 0..{self.capacity - 1}")
        col, row = index % self.cols, index // self.cols
        return Rect(
            self.origin_x + col * self.pitch_x,
            self.origin_y + row * self.pitch_y,
            self.cell_w,
            self.cell_h,
        )


class LineGrid(BaseModel):
    """A stack of evenly spaced text lines, e.g. the dialogue option list.

    Perception normally reports option lines with real bboxes; this is the
    fallback for when it hasn't (options detected by interface sprite alone).
    """

    model_config = {"frozen": True, "extra": "forbid"}

    origin_x: int = Field(ge=0)
    origin_y: int = Field(ge=0)
    width: int = Field(gt=0)
    line_h: int = Field(gt=0)
    pitch_y: int = Field(gt=0)
    max_lines: int = Field(default=5, gt=0)

    def line_rect(self, index: int) -> Rect:
        if not 0 <= index < self.max_lines:
            raise IndexError(f"line {index} outside 0..{self.max_lines - 1}")
        return Rect(self.origin_x, self.origin_y + index * self.pitch_y, self.width, self.line_h)


class PerceptionExtras(BaseModel):
    """Regions only the perception layer reads, never a click target.

    These live here rather than in ``perception.LayoutProfile`` defaults so
    that this file stays the single place a calibration number is written. The
    rects shared with the actuator (viewport, chatbox, inventory, minimap) are
    NOT repeated here; ``gielinor.profiles.layout_from_profile`` derives those
    from the top-level fields. See that module for why.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    side_panel: RectSpec | None = None
    tooltip: RectSpec | None = None
    hp_orb: RectSpec | None = None
    prayer_orb: RectSpec | None = None
    run_orb: RectSpec | None = None
    dialogue_speaker_h: int = Field(default=16, gt=0)
    player_box_w: int = Field(default=70, gt=0)
    player_box_h: int = Field(default=96, gt=0)
    player_box_dy: int = 6


class Profile(BaseModel):
    """Screen geometry for one client build at one window size.

    Every rect is in *client* coordinates (origin at the client window's
    top-left), never desktop coordinates, so the profile survives the window
    being moved.

    This model is the single source of truth for screen geometry. The
    perception layer does not carry a second copy: it consumes a
    ``LayoutProfile`` derived from this one by ``gielinor.profiles``. Two
    independently-editable coordinate tables is how you get an agent that
    detects a tree correctly and then clicks 4 px off it after someone
    recalibrates one file and not the other.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    name: str
    description: str = ""
    calibrated: bool = False
    client_w: int = Field(gt=0)
    client_h: int = Field(gt=0)
    viewport: RectSpec
    chatbox: RectSpec
    chat_input: RectSpec
    inventory: RectSpec
    minimap: RectSpec
    inventory_grid: InventoryGrid
    dialogue_options: LineGrid | None = None
    ui_anchors: dict[str, RectSpec] = Field(default_factory=dict)
    perception: PerceptionExtras = Field(default_factory=PerceptionExtras)

    @field_validator("ui_anchors")
    @classmethod
    def _anchor_names_are_identifiers(cls, v: dict[str, RectSpec]) -> dict[str, RectSpec]:
        # UI anchor names cross the planner boundary as `Action.target_ui`, so
        # they are kept to a boring charset: no spaces, no path separators, no
        # anything that could be mistaken for a filename or a shell token.
        for name in v:
            if not name.replace("_", "").isalnum():
                raise ValueError(f"ui anchor name {name!r} must be alphanumeric/underscore")
        return v

    @property
    def client_size(self) -> tuple[int, int]:
        return self.client_w, self.client_h

    def region(self, name: RegionName) -> Rect:
        specs: dict[RegionName, RectSpec] = {
            "viewport": self.viewport,
            "chatbox": self.chatbox,
            "chat_input": self.chat_input,
            "inventory": self.inventory,
            "minimap": self.minimap,
        }
        return specs[name].to_rect()

    def anchor(self, name: str) -> Rect | None:
        """Resolve a UI anchor name to a rect, or None if this profile lacks it.

        Returns None rather than raising: a missing anchor is a *validation*
        outcome (the action gets rejected with a reason that reaches the trace),
        not an exception that should unwind the frame loop.
        """
        if name.startswith("inventory_slot_"):
            suffix = name.removeprefix("inventory_slot_")
            if suffix.isdigit() and int(suffix) < self.inventory_grid.capacity:
                return self.inventory_grid.slot_rect(int(suffix))
            return None
        spec = self.ui_anchors.get(name)
        return spec.to_rect() if spec else None

    def anchor_names(self) -> frozenset[str]:
        dynamic = {f"inventory_slot_{i}" for i in range(self.inventory_grid.capacity)}
        return frozenset(self.ui_anchors) | frozenset(dynamic)

    def dialogue_option_rect(self, index: int) -> Rect | None:
        if self.dialogue_options is None or index >= self.dialogue_options.max_lines:
            return None
        return self.dialogue_options.line_rect(index)

    def viewport_center(self) -> Point:
        return self.viewport.to_rect().center

    def contains_client(self, r: Rect) -> bool:
        clipped = r.clip_to(Rect(0, 0, self.client_w, self.client_h))
        return clipped == r


def profile_path(name_or_path: str | Path) -> Path:
    """Resolve a profile name (``2009scape_fixed``) or an explicit path to a file."""
    candidate = Path(name_or_path)
    if candidate.suffix == ".toml" and candidate.exists():
        return candidate
    for root in PROFILE_SEARCH_PATH:
        p = root / f"{candidate.name}.toml"
        if p.exists():
            return p
    searched = ", ".join(str(r) for r in PROFILE_SEARCH_PATH)
    raise FileNotFoundError(f"no profile {name_or_path!r} found; searched {searched}")


def load_profile(name_or_path: str | Path = DEFAULT_PROFILE) -> Profile:
    """Read and validate a calibration profile. Not cached: profiles are edited live."""
    path = profile_path(name_or_path)
    with path.open("rb") as fh:
        raw: dict[str, Any] = tomllib.load(fh)
    raw.setdefault("name", path.stem)
    profile = Profile.model_validate(raw)
    if not profile.calibrated:
        logger.warning(
            "profile %s is marked uncalibrated; coordinates are plausible RS2 defaults "
            "and must be checked against a real capture before real input is enabled",
            profile.name,
        )
    return profile


@lru_cache(maxsize=4)
def cached_profile(name: str = DEFAULT_PROFILE) -> Profile:
    """Hot-path accessor. The loop resolves anchors per action; TOML per frame is silly."""
    return load_profile(name)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


class CaptureSettings(BaseModel):
    model_config = {"extra": "forbid"}

    backend: Literal["auto", "replay", "pipewire", "screencapturekit"] = "auto"
    fps: Annotated[float, Field(gt=0, le=240)] = 20.0
    replay_dir: Path | None = None
    replay_loop: bool = True
    replay_realtime: bool = False


class PlannerSettings(BaseModel):
    model_config = {"extra": "forbid"}

    # SPEC section 1: verified free OpenRouter slugs, text-first because
    # perception has usually already extracted what the model needs.
    text_models: tuple[str, ...] = (
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "openai/gpt-oss-20b:free",
        "google/gemma-4-26b-a4b-it:free",
    )
    vision_models: tuple[str, ...] = (
        "google/gemma-4-26b-a4b-it:free",
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    )
    timeout_s: Annotated[float, Field(gt=0)] = 20.0
    connect_timeout_s: Annotated[float, Field(gt=0)] = 5.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    backoff_base_s: Annotated[float, Field(gt=0)] = 0.5
    max_actions_per_response: Annotated[int, Field(ge=1, le=16)] = 4
    escalation_cooldown_s: Annotated[float, Field(ge=0)] = 3.0
    attach_image: bool = True
    image_max_width: Annotated[int, Field(gt=0)] = 765
    stuck_after_s: Annotated[float, Field(gt=0)] = 30.0
    novelty_confidence: Annotated[float, Field(ge=0, le=1)] = 0.55


class ActuatorSettings(BaseModel):
    model_config = {"extra": "forbid"}

    backend: Literal["dryrun", "uinput", "quartz"] = "dryrun"
    # Two independent locks. `allow_real_input` must be set here AND passed
    # explicitly to the backend constructor; nothing reads this and silently
    # grants itself input.
    allow_real_input: bool = False
    allow_chat: bool = False
    profile: str = DEFAULT_PROFILE
    min_object_confidence: Annotated[float, Field(ge=0, le=1)] = 0.35
    move_speed: Annotated[float, Field(gt=0)] = 1.0
    rng_seed: int | None = None


class RuntimeSettings(BaseModel):
    model_config = {"extra": "forbid"}

    target_hz: Annotated[float, Field(gt=0, le=120)] = 15.0
    frame_budget_ms: Annotated[float, Field(gt=0)] = 50.0
    max_stale_frames: Annotated[int, Field(ge=0)] = 30
    max_stale_s: Annotated[float, Field(ge=0)] = 3.0
    max_queued_actions: Annotated[int, Field(ge=1)] = 16
    trace_path: Path | None = None
    goal: str = "chop_trees"
    goal_params: dict[str, str] = Field(default_factory=lambda: {"tree_label": "tree"})


#: Where Docker Compose mounts declared secrets. Overridable so a test can point
#: at a tmp_path and so a non-Docker injector can write somewhere else.
DEFAULT_SECRETS_DIR = os.environ.get("GIELINOR_SECRETS_DIR", "/run/secrets")

#: Passed to pydantic-settings only when the directory actually exists. On any
#: machine that is not a container ``/run/secrets`` is absent, and handing
#: pydantic-settings a missing path makes it emit a UserWarning on every single
#: construction of Settings. Warning noise on the normal path trains people to
#: ignore warnings, which is how the one that matters gets missed.
_SECRETS_DIR_IF_PRESENT: str | None = (
    DEFAULT_SECRETS_DIR if Path(DEFAULT_SECRETS_DIR).is_dir() else None
)

#: The ``_FILE`` indirection suffix. `FOO_FILE=/path` means "read FOO from /path".
SECRET_FILE_SUFFIX = "_FILE"  # noqa: S105 - a variable-name suffix, not a credential


def _read_secret_file(path: str | Path, *, label: str) -> str | None:
    """Read a single-line secret, or return None with a specific reason logged.

    Never raises. A missing or unreadable secret file has to degrade to "no key"
    rather than crash the process, because the reflex loop is fully functional
    without one — losing the slow loop is a capability regression, not an
    outage, and taking the whole agent down for it would be the wrong trade.

    Trailing newlines are stripped because every editor and most `echo`
    invocations add one, and a key with `\\n` on the end fails auth with a
    401 that looks exactly like a wrong key.
    """
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.debug("%s: no secret file at %s", label, p)
        return None
    except OSError as exc:
        logger.warning("%s: cannot read secret file %s (%s)", label, p, exc)
        return None
    value = raw.strip()
    if not value:
        logger.warning("%s: secret file %s is empty", label, p)
        return None
    return value


class Settings(BaseSettings):
    """Top-level settings object. Construct once, pass it down; never a global."""

    model_config = SettingsConfigDict(
        env_prefix="GIELINOR_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        secrets_dir=_SECRETS_DIR_IF_PRESENT,
        extra="ignore",
        frozen=True,
    )

    # SecretStr so an accidental repr()/model_dump() of Settings in a log line
    # or a trace record prints '**********' instead of the key.
    #
    # Both alias casings are accepted because the two delivery mechanisms
    # disagree by convention: environment variables are SHOUTY, while a Docker
    # secret is conventionally named in lowercase (`docker secret create
    # openrouter_api_key`) and lands as a lowercase filename. Accepting one
    # casing only would mean a correctly-declared Compose secret silently not
    # being found, which presents as "my key isn't working".
    openrouter_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENROUTER_API_KEY", "openrouter_api_key"),
    )
    log_level: str = "INFO"
    capture: CaptureSettings = Field(default_factory=CaptureSettings)
    planner: PlannerSettings = Field(default_factory=PlannerSettings)
    actuator: ActuatorSettings = Field(default_factory=ActuatorSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    @model_validator(mode="before")
    @classmethod
    def _resolve_secret_files(cls, data: Any) -> Any:
        """Apply the ``_FILE`` indirection before normal validation.

        This runs after pydantic-settings has already consulted init args, the
        environment, ``.env`` and ``secrets_dir``, so it only fills a gap — an
        explicitly supplied key always wins over a path pointing at one. That
        ordering is what makes ``OPENROUTER_API_KEY=... docker compose up``
        still work as an override when someone is debugging.
        """
        if not isinstance(data, dict):
            return data
        for field_alias in ("OPENROUTER_API_KEY", "openrouter_api_key"):
            if data.get(field_alias):
                return data
        pointer = os.environ.get("OPENROUTER_API_KEY" + SECRET_FILE_SUFFIX)
        if pointer:
            value = _read_secret_file(pointer, label="OPENROUTER_API_KEY_FILE")
            if value:
                data["OPENROUTER_API_KEY"] = value
        return data

    def profile(self) -> Profile:
        return cached_profile(self.actuator.profile)

    def has_api_key(self) -> bool:
        return self.openrouter_api_key is not None and bool(
            self.openrouter_api_key.get_secret_value()
        )

    def api_key_source(self, secrets_dir: str | Path | None = None) -> str:
        """Which mechanism supplied the key, for a startup log line.

        Reports the *mechanism*, never the value. Seeing "secrets_dir" in the
        logs when you expected "env" is most of the debugging on a container
        that isn't picking up its credential.

        This reconstructs the answer by inspecting the same places
        pydantic-settings consulted, rather than observing which source actually
        won — pydantic-settings does not expose that, and wrapping its source
        stack to find out is more machinery than a log line justifies. The
        consequence is that a caller who overrode ``_secrets_dir`` has to pass
        the same value here for the answer to be exact. When it cannot tell, it
        says so rather than guessing: a debugging aid that confidently reports
        the wrong source is worse than one that admits ignorance.
        """
        if not self.has_api_key():
            return "unset"
        if os.environ.get("OPENROUTER_API_KEY"):
            return "env"
        if os.environ.get("OPENROUTER_API_KEY" + SECRET_FILE_SUFFIX):
            return "env:OPENROUTER_API_KEY_FILE"
        for directory in (secrets_dir, DEFAULT_SECRETS_DIR):
            if directory is None:
                continue
            for candidate in ("openrouter_api_key", "OPENROUTER_API_KEY"):
                path = Path(directory) / candidate
                if path.is_file():
                    return f"secrets_dir:{path}"
        if Path(".env").is_file():
            return "dotenv"
        return "supplied-directly-or-unknown"


__all__ = [
    "DEFAULT_PROFILE",
    "DEFAULT_SECRETS_DIR",
    "PROFILE_SEARCH_PATH",
    "SECRET_FILE_SUFFIX",
    "ActuatorSettings",
    "CaptureSettings",
    "InventoryGrid",
    "LineGrid",
    "PlannerSettings",
    "Profile",
    "RectSpec",
    "RuntimeSettings",
    "Settings",
    "cached_profile",
    "load_profile",
    "profile_path",
]
