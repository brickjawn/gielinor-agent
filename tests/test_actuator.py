"""Actuator tests, with most of the weight on validation.

``validate_action`` is the last thing standing between a remote model that has
just read attacker-controlled chat and a synthetic click, so it gets tested like
a security boundary rather than like a helper: every ActionKind, every rejection
code, and the specific attacks SPEC section 4 calls out.
"""

from __future__ import annotations

import random

import pytest

from conftest import make_object
from gielinor.actuator import open_actuator
from gielinor.actuator.base import RealInputRefused
from gielinor.actuator.dryrun import DryRunActuator
from gielinor.actuator.humanize import (
    bezier_path,
    click_delay_ms,
    fitts_movement_ms,
    jitter_point_in,
    mouse_path,
    sample_reaction_time_ms,
    typing_delays_ms,
)
from gielinor.actuator.quartz import QuartzActuator
from gielinor.actuator.resolve import ResolutionError, resolve_action
from gielinor.actuator.uinput import UinputActuator
from gielinor.actuator.validate import (
    MAX_CHAT_LEN,
    MAX_TICKS,
    RejectionCode,
    ValidationPolicy,
    is_stale,
    sanitize_chat_text,
    validate_action,
)
from gielinor.config import Profile  # noqa: TC001  (used in fixture signatures)
from gielinor.contracts import (
    Action,
    ActionKind,
    Actuator,
    DialogueState,
    GameState,
    Point,
    Rect,
    TextLine,
)

CHATTY = ValidationPolicy(allow_chat=True)


def code(action: Action, state: GameState, **kw: object) -> RejectionCode | None:
    reason = validate_action(action, state, **kw)  # type: ignore[arg-type]
    return reason.code if reason else None


# --------------------------------------------------------------------------
# Totality: the property that makes this a boundary rather than a filter
# --------------------------------------------------------------------------


class TestValidationIsTotal:
    def test_every_action_kind_has_a_rule(self, make_state) -> None:
        from gielinor.actuator.validate import _RULES

        assert set(_RULES) == set(ActionKind), (
            "an ActionKind with no validator is a hole in the capability boundary"
        )

    def test_every_kind_can_be_validated_without_raising(self, make_state) -> None:
        state = make_state()
        for kind in ActionKind:
            result = validate_action(Action(kind=kind), state)
            assert result is None or result.code in set(RejectionCode)

    def test_a_kind_outside_the_enum_is_rejected_not_executed(self, make_state) -> None:
        # Action is a plain dataclass and does not type-check its fields at
        # runtime, so a JSON parser that forgot to coerce, or a caller who
        # simply invented a verb, lands here. It must not fall through.
        state = make_state()
        smuggled = Action(kind="drop_all", target_oid=1)  # type: ignore[arg-type]
        assert code(smuggled, state) is RejectionCode.UNKNOWN_ACTION_KIND

    def test_noop_and_wait_are_always_fine(self, make_state) -> None:
        state = make_state()
        assert validate_action(Action(kind=ActionKind.NOOP), state) is None
        assert validate_action(Action(kind=ActionKind.WAIT, ticks=3), state) is None


# --------------------------------------------------------------------------
# CLICK_OBJECT: symbolic targets only
# --------------------------------------------------------------------------


class TestClickObject:
    def test_accepts_an_oid_present_in_the_state(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        assert validate_action(Action(ActionKind.CLICK_OBJECT, target_oid=41), state) is None

    def test_rejects_an_oid_absent_from_the_state(self, make_state, tree) -> None:
        # The central rule: the planner may only refer to things perception
        # found in the state it was judged against.
        state = make_state(objects=[tree])
        assert (
            code(Action(ActionKind.CLICK_OBJECT, target_oid=999), state)
            is RejectionCode.NO_SUCH_OBJECT
        )

    def test_rejects_an_oid_that_existed_in_an_earlier_frame(self, make_state, tree) -> None:
        # Exactly the staleness case: valid when planned, invalid at dispatch.
        planned_against = make_state(seq=100, objects=[tree])
        now = make_state(seq=160, objects=[])
        action = Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid)
        assert validate_action(action, planned_against) is None
        assert code(action, now) is RejectionCode.NO_SUCH_OBJECT

    def test_requires_a_target(self, make_state) -> None:
        assert code(Action(ActionKind.CLICK_OBJECT), make_state()) is RejectionCode.MISSING_TARGET

    def test_rejects_a_minimap_object(self, make_state) -> None:
        obj = make_object(7, "tree", space="minimap")
        state = make_state(objects=[obj])
        assert (
            code(Action(ActionKind.CLICK_OBJECT, target_oid=7), state) is RejectionCode.WRONG_SPACE
        )

    def test_rejects_a_low_confidence_detection(self, make_state) -> None:
        obj = make_object(7, "tree", confidence=0.10)
        state = make_state(objects=[obj])
        assert (
            code(Action(ActionKind.CLICK_OBJECT, target_oid=7), state)
            is RejectionCode.LOW_CONFIDENCE
        )

    def test_confidence_floor_is_policy(self, make_state) -> None:
        obj = make_object(7, "tree", confidence=0.10)
        state = make_state(objects=[obj])
        lenient = ValidationPolicy(min_object_confidence=0.05)
        action = Action(ActionKind.CLICK_OBJECT, target_oid=7)
        assert validate_action(action, state, policy=lenient) is None


class TestWalkMinimap:
    def test_accepts_a_minimap_object(self, make_state) -> None:
        obj = make_object(9, "tree", space="minimap")
        state = make_state(objects=[obj])
        assert validate_action(Action(ActionKind.WALK_MINIMAP, target_oid=9), state) is None

    def test_rejects_a_viewport_object(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        assert (
            code(Action(ActionKind.WALK_MINIMAP, target_oid=tree.oid), state)
            is RejectionCode.WRONG_SPACE
        )

    def test_rejects_an_unknown_oid(self, make_state) -> None:
        assert (
            code(Action(ActionKind.WALK_MINIMAP, target_oid=3), make_state())
            is RejectionCode.NO_SUCH_OBJECT
        )


# --------------------------------------------------------------------------
# TYPE_CHAT: the only verb that emits attacker-influenced text
# --------------------------------------------------------------------------


class TestTypeChat:
    def test_is_disabled_by_default(self, make_state) -> None:
        action = Action(ActionKind.TYPE_CHAT, text="hello")
        assert code(action, make_state()) is RejectionCode.CHAT_DISABLED

    def test_accepts_ordinary_text_when_enabled(self, make_state) -> None:
        action = Action(ActionKind.TYPE_CHAT, text="hi, nice woodcutting spot!")
        assert validate_action(action, make_state(), policy=CHATTY) is None

    def test_rejects_text_over_the_cap(self, make_state) -> None:
        action = Action(ActionKind.TYPE_CHAT, text="a" * (MAX_CHAT_LEN + 1))
        assert code(action, make_state(), policy=CHATTY) is RejectionCode.CHAT_TOO_LONG

    def test_accepts_text_at_exactly_the_cap(self, make_state) -> None:
        action = Action(ActionKind.TYPE_CHAT, text="a" * MAX_CHAT_LEN)
        assert validate_action(action, make_state(), policy=CHATTY) is None

    @pytest.mark.parametrize(
        "text",
        [
            "hello\nworld",  # newline would submit early
            "hello\tworld",  # tab moves focus
            "hello\x00world",  # NUL
            "héllo wörld",  # non-ascii
            "‮reversed",  # bidi override
            "emoji 🌳",
        ],
    )
    def test_rejects_characters_the_client_does_not_accept(self, make_state, text: str) -> None:
        action = Action(ActionKind.TYPE_CHAT, text=text)
        assert code(action, make_state(), policy=CHATTY) is RejectionCode.CHAT_CHARSET

    def test_rejects_empty_and_whitespace_only(self, make_state) -> None:
        assert (
            code(Action(ActionKind.TYPE_CHAT, text="   "), make_state(), policy=CHATTY)
            is RejectionCode.CHAT_EMPTY
        )

    def test_rejects_a_client_command(self, make_state) -> None:
        # "::something" is a command to the client, not speech. Different verb,
        # different privilege.
        action = Action(ActionKind.TYPE_CHAT, text="::teleport home")
        assert code(action, make_state(), policy=CHATTY) is RejectionCode.CHAT_COMMAND

    def test_sanitize_helper_coerces_but_does_not_authorise(self) -> None:
        cleaned = sanitize_chat_text("hey\nthere \x07friend " + "x" * 200)
        assert "\n" not in cleaned
        assert len(cleaned) <= MAX_CHAT_LEN


# --------------------------------------------------------------------------
# Remaining kinds
# --------------------------------------------------------------------------


class TestDialogueActions:
    def test_select_requires_an_open_dialogue(self, make_state) -> None:
        assert (
            code(Action(ActionKind.SELECT_DIALOGUE, option_index=0), make_state())
            is RejectionCode.NO_DIALOGUE
        )

    def test_select_bounds_are_checked(self, make_state) -> None:
        dialogue = DialogueState(is_open=True, kind="options", options=("Yes.", "No."))
        state = make_state(dialogue=dialogue)
        assert validate_action(Action(ActionKind.SELECT_DIALOGUE, option_index=1), state) is None
        assert (
            code(Action(ActionKind.SELECT_DIALOGUE, option_index=2), state)
            is RejectionCode.BAD_OPTION_INDEX
        )
        assert (
            code(Action(ActionKind.SELECT_DIALOGUE, option_index=-1), state)
            is RejectionCode.BAD_OPTION_INDEX
        )

    def test_continue_requires_an_open_dialogue(self, make_state) -> None:
        assert code(Action(ActionKind.CONTINUE_DIALOGUE), make_state()) is RejectionCode.NO_DIALOGUE

    def test_continue_refuses_while_options_are_pending(self, make_state) -> None:
        dialogue = DialogueState(is_open=True, kind="options", options=("Yes.", "No."))
        assert (
            code(Action(ActionKind.CONTINUE_DIALOGUE), make_state(dialogue=dialogue))
            is RejectionCode.NO_DIALOGUE
        )

    def test_continue_accepts_a_continue_box(self, make_state) -> None:
        dialogue = DialogueState(is_open=True, kind="continue")
        state = make_state(dialogue=dialogue)
        assert validate_action(Action(ActionKind.CONTINUE_DIALOGUE), state) is None


class TestClickUi:
    def test_known_anchor_accepted(self, make_state) -> None:
        action = Action(ActionKind.CLICK_UI, target_ui="inventory_tab")
        assert validate_action(action, make_state()) is None

    def test_unknown_anchor_rejected(self, make_state) -> None:
        assert (
            code(Action(ActionKind.CLICK_UI, target_ui="../../etc/passwd"), make_state())
            is RejectionCode.UNKNOWN_UI_ANCHOR
        )

    def test_missing_anchor_rejected(self, make_state) -> None:
        assert code(Action(ActionKind.CLICK_UI), make_state()) is RejectionCode.MISSING_TARGET

    def test_profile_narrows_the_allowlist(self, make_state, profile: Profile) -> None:
        assert (
            validate_action(
                Action(ActionKind.CLICK_UI, target_ui="inventory_slot_27"),
                make_state(),
                profile=profile,
            )
            is None
        )
        assert (
            code(
                Action(ActionKind.CLICK_UI, target_ui="inventory_slot_28"),
                make_state(),
                profile=profile,
            )
            is RejectionCode.UNKNOWN_UI_ANCHOR
        )


class TestTicks:
    def test_within_range_is_fine(self, make_state) -> None:
        assert validate_action(Action(ActionKind.WAIT, ticks=MAX_TICKS), make_state()) is None

    def test_above_the_cap_is_rejected(self, make_state) -> None:
        assert (
            code(Action(ActionKind.WAIT, ticks=MAX_TICKS + 1), make_state())
            is RejectionCode.TICKS_OUT_OF_RANGE
        )

    def test_negative_is_refused_by_the_contract_itself(self) -> None:
        with pytest.raises(ValueError, match="ticks must be >= 0"):
            Action(kind=ActionKind.WAIT, ticks=-1)

    def test_bool_is_not_an_int(self, make_state) -> None:
        action = Action(ActionKind.WAIT, ticks=True)  # type: ignore[arg-type]
        assert code(action, make_state()) is RejectionCode.TICKS_OUT_OF_RANGE


class TestExtraneousFields:
    def test_ignored_by_default(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        action = Action(ActionKind.WAIT, target_oid=tree.oid, ticks=1)
        assert validate_action(action, state) is None

    def test_rejected_under_the_strict_policy(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        strict = ValidationPolicy(reject_extraneous_fields=True)
        action = Action(ActionKind.WAIT, target_oid=tree.oid, ticks=1)
        assert code(action, state, policy=strict) is RejectionCode.UNEXPECTED_FIELD


# --------------------------------------------------------------------------
# Staleness
# --------------------------------------------------------------------------


class TestStaleness:
    def test_fresh_action_survives(self, make_state) -> None:
        current = make_state(seq=105, captured_at=7.0)
        assert is_stale(100, current, 6.9, max_frames=30, max_seconds=3.0) is None

    def test_too_many_frames_elapsed(self, make_state) -> None:
        # SPEC's example verbatim: planned at 1400, arriving at 1460.
        current = make_state(seq=1460, captured_at=100.0)
        reason = is_stale(1400, current, 99.0, max_frames=30, max_seconds=3.0)
        assert reason is not None
        assert reason.code is RejectionCode.STALE
        assert "60 frames" in reason.detail

    def test_too_much_wall_clock_elapsed_even_with_few_frames(self, make_state) -> None:
        # A stalled capture produces few frames but plenty of wall clock; the
        # world moved on regardless of how many frames we saw it happen in.
        current = make_state(seq=102, captured_at=110.0)
        reason = is_stale(100, current, 100.0, max_frames=30, max_seconds=3.0)
        assert reason is not None
        assert "10.00s ago" in reason.detail


# --------------------------------------------------------------------------
# humanize: pure, seeded, reproducible
# --------------------------------------------------------------------------


class TestHumanize:
    def test_same_seed_gives_identical_paths(self) -> None:
        a = bezier_path(Point(0, 0), Point(300, 200), random.Random(7))
        b = bezier_path(Point(0, 0), Point(300, 200), random.Random(7))
        assert a == b

    def test_different_seeds_give_different_paths(self) -> None:
        a = bezier_path(Point(0, 0), Point(300, 200), random.Random(1))
        b = bezier_path(Point(0, 0), Point(300, 200), random.Random(2))
        assert a != b

    def test_endpoints_are_exact(self) -> None:
        start, end = Point(11, 13), Point(417, 233)
        path = bezier_path(start, end, random.Random(3))
        assert path[0] == start
        assert path[-1] == end

    def test_path_is_not_a_straight_line(self) -> None:
        start, end = Point(0, 0), Point(400, 0)
        path = bezier_path(start, end, random.Random(5), curvature=0.3)
        assert any(p.y != 0 for p in path), "a perfectly straight path is the tell"

    def test_degenerate_move_is_handled(self) -> None:
        path = bezier_path(Point(5, 5), Point(5, 5), random.Random(0))
        assert path[0] == path[-1] == Point(5, 5)

    def test_jitter_stays_inside_the_box_even_with_absurd_sigma(self) -> None:
        box = Rect(100, 100, 10, 10)
        rng = random.Random(0)
        for _ in range(500):
            p = jitter_point_in(box, rng, sigma_frac=5.0)
            assert box.contains(p)

    def test_jitter_clusters_near_the_bias_point(self) -> None:
        box = Rect(0, 0, 200, 200)
        bias = Point(150, 40)
        rng = random.Random(11)
        samples = [jitter_point_in(box, rng, bias=bias) for _ in range(400)]
        mean_x = sum(p.x for p in samples) / len(samples)
        mean_y = sum(p.y for p in samples) / len(samples)
        assert abs(mean_x - bias.x) < 8
        assert abs(mean_y - bias.y) < 8

    def test_jitter_rejects_a_degenerate_rect(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            jitter_point_in(Rect(0, 0, 0, 0), random.Random(0))

    def test_fitts_is_monotonic_in_distance_and_target_size(self) -> None:
        near = fitts_movement_ms(50, 40)
        far = fitts_movement_ms(800, 40)
        assert far > near
        big = fitts_movement_ms(400, 200)
        small = fitts_movement_ms(400, 8)
        assert small > big

    def test_fitts_speed_multiplier_shortens_the_move(self) -> None:
        assert fitts_movement_ms(400, 40, speed=2.0) < fitts_movement_ms(400, 40, speed=1.0)

    def test_reaction_times_are_skewed_and_bounded(self) -> None:
        rng = random.Random(42)
        samples = [sample_reaction_time_ms(rng) for _ in range(2000)]
        assert all(110.0 <= s <= 1400.0 for s in samples)
        mean = sum(samples) / len(samples)
        median = sorted(samples)[len(samples) // 2]
        # Lognormal: mean sits above the median. A constant or a Gaussian would
        # fail this, and a constant is the thing we are avoiding.
        assert mean > median
        assert len({round(s, 3) for s in samples}) > 1000

    def test_mouse_path_timings_are_monotonic_and_start_at_zero(self) -> None:
        steps, end = mouse_path(Point(0, 0), Rect(300, 200, 40, 40), random.Random(9))
        times = [s.t_ms for s in steps]
        assert times[0] == 0.0
        assert times == sorted(times)
        assert times[-1] > 0
        assert Rect(300, 200, 40, 40).contains(end)
        assert steps[-1].point == end

    def test_mouse_path_honours_an_explicit_click_point(self) -> None:
        target = Rect(300, 200, 40, 40)
        want = Point(305, 214)
        steps, end = mouse_path(Point(0, 0), target, random.Random(9), click_point=want)
        assert end == want
        assert steps[-1].point == want

    def test_click_delay_and_typing_delays_are_bounded(self) -> None:
        rng = random.Random(4)
        assert all(18.0 <= click_delay_ms(rng) <= 400.0 for _ in range(200))
        delays = typing_delays_ms("hello world", rng)
        assert len(delays) == len("hello world")
        assert all(25.0 <= d <= 600.0 for d in delays)


# --------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------


class TestResolve:
    def test_object_resolves_inside_its_bbox(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        res = resolve_action(
            Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state, rng=random.Random(0)
        )
        assert res.point is not None
        assert tree.bbox.contains(res.point)

    def test_object_resolution_is_seeded(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        action = Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid)
        a = resolve_action(action, state, rng=random.Random(5)).point
        b = resolve_action(action, state, rng=random.Random(5)).point
        assert a == b

    def test_ui_anchor_needs_a_profile(self, make_state, profile: Profile) -> None:
        action = Action(ActionKind.CLICK_UI, target_ui="inventory_tab")
        with pytest.raises(ResolutionError, match="no geometry"):
            resolve_action(action, make_state(), rng=random.Random(0))
        res = resolve_action(action, make_state(), profile=profile, rng=random.Random(0))
        assert res.point is not None
        assert profile.anchor("inventory_tab").contains(res.point)  # type: ignore[union-attr]

    def test_dialogue_option_prefers_perceptions_bbox(self, make_state) -> None:
        lines = [
            TextLine(text="Yes.", bbox=Rect(20, 380, 200, 14), source="dialogue_option"),
            TextLine(text="No.", bbox=Rect(20, 402, 200, 14), source="dialogue_option"),
        ]
        state = make_state(
            text=lines,
            dialogue=DialogueState(is_open=True, kind="options", options=("Yes.", "No.")),
        )
        res = resolve_action(
            Action(ActionKind.SELECT_DIALOGUE, option_index=1), state, rng=random.Random(0)
        )
        assert res.point is not None
        assert lines[1].bbox.contains(res.point)

    def test_dialogue_option_falls_back_to_the_profile_grid(
        self, make_state, profile: Profile
    ) -> None:
        state = make_state(dialogue=DialogueState(is_open=True, kind="options", options=("Yes.",)))
        res = resolve_action(
            Action(ActionKind.SELECT_DIALOGUE, option_index=0),
            state,
            profile=profile,
            rng=random.Random(0),
        )
        assert res.point is not None

    def test_wait_resolves_to_no_input_at_all(self, make_state) -> None:
        res = resolve_action(Action(ActionKind.WAIT, ticks=2), make_state())
        assert res.is_input is False
        assert res.point is None

    def test_vanished_object_raises_rather_than_guessing(self, make_state) -> None:
        with pytest.raises(ResolutionError, match="vanished"):
            resolve_action(Action(ActionKind.CLICK_OBJECT, target_oid=77), make_state())


# --------------------------------------------------------------------------
# DryRunActuator
# --------------------------------------------------------------------------


class TestDryRun:
    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(DryRunActuator(), Actuator)

    def test_accepts_and_resolves_a_valid_click(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        actuator = DryRunActuator(seed=1)
        result = actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        assert result.accepted
        assert result.resolved_point is not None
        assert tree.bbox.contains(result.resolved_point)
        assert result.dispatched_at is not None

    def test_rejects_without_resolving(self, make_state) -> None:
        actuator = DryRunActuator()
        result = actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=1), make_state())
        assert result.accepted is False
        assert result.rejected_reason is not None
        assert "no_such_object" in result.rejected_reason
        assert result.resolved_point is None
        assert actuator.gestures == []

    def test_plans_a_full_gesture_without_performing_it(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        actuator = DryRunActuator(seed=3)
        actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        gesture = actuator.gestures[0]
        assert gesture.waypoints > 2
        assert gesture.reaction_ms > 0
        assert gesture.travel_ms > 0
        assert gesture.total_ms == pytest.approx(
            gesture.reaction_ms + gesture.travel_ms + gesture.hold_ms
        )

    def test_is_reproducible_with_a_seed(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        action = Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid)
        first = DryRunActuator(seed=99).execute(action, state).resolved_point
        second = DryRunActuator(seed=99).execute(action, state).resolved_point
        assert first == second

    def test_cursor_follows_the_last_click(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        actuator = DryRunActuator(seed=1)
        result = actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        assert actuator.cursor == result.resolved_point

    def test_history_separates_accepted_from_rejected(self, make_state, tree) -> None:
        state = make_state(objects=[tree])
        actuator = DryRunActuator()
        actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=404), state)
        assert len(actuator.accepted) == 1
        assert len(actuator.rejected) == 1
        assert len(actuator.history) == 2

    def test_on_result_hook_sees_everything(self, make_state, tree) -> None:
        seen = []
        actuator = DryRunActuator(on_result=seen.append)
        state = make_state(objects=[tree])
        actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        actuator.execute(Action(ActionKind.CLICK_OBJECT, target_oid=1), state)
        assert len(seen) == 2

    def test_logs_at_info(self, make_state, tree, caplog: pytest.LogCaptureFixture) -> None:
        state = make_state(objects=[tree])
        with caplog.at_level("INFO", logger="gielinor.actuator.dryrun"):
            DryRunActuator().execute(Action(ActionKind.CLICK_OBJECT, target_oid=tree.oid), state)
        assert any("DRY-RUN" in r.message for r in caplog.records)

    def test_ui_click_without_a_profile_is_a_rejection_not_a_crash(self, make_state) -> None:
        result = DryRunActuator().execute(
            Action(ActionKind.CLICK_UI, target_ui="inventory_tab"), make_state()
        )
        assert result.accepted is False


# --------------------------------------------------------------------------
# Real actuators: constructing one must be deliberate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [UinputActuator, QuartzActuator])
class TestRealActuatorsRefuseByDefault:
    def test_construction_refuses_without_explicit_consent(self, cls: type) -> None:
        with pytest.raises(RealInputRefused) as exc:
            cls()
        assert "allow_real_input=True" in str(exc.value)

    def test_the_refusal_happens_before_any_platform_work(self, cls: type) -> None:
        # No import of evdev/Quartz, no /dev/uinput probe: refusing is the first
        # thing that happens, so this test passes on a machine that has neither.
        with pytest.raises(RealInputRefused):
            cls(profile=None)


class TestOpenActuator:
    def test_defaults_to_dry_run(self) -> None:
        assert isinstance(open_actuator(), DryRunActuator)

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown actuator"):
            open_actuator("keyboard-cannon")

    def test_real_backend_without_consent_raises_rather_than_downgrading(self) -> None:
        # Silently downgrading would make "why is nothing happening" the failure
        # mode, which is worse than an exception.
        with pytest.raises(RealInputRefused):
            open_actuator("uinput")

    def test_allow_real_input_is_inert_for_dry_run(self) -> None:
        assert isinstance(open_actuator("dryrun", allow_real_input=True), DryRunActuator)

    def test_passes_the_profile_through(self, profile: Profile, make_state) -> None:
        actuator = open_actuator("dryrun", profile=profile)
        result = actuator.execute(
            Action(ActionKind.CLICK_UI, target_ui="inventory_tab"), make_state()
        )
        assert result.accepted
