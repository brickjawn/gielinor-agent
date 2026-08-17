"""Perception tests: detection, oid stability, inventory, dialogue, animation.

Everything here runs against `synthetic.py` frames — no display, no network, no
game, no captured screenshots. The properties asserted are the ones other
workstreams depend on and cannot check themselves:

  * oids are stable frame to frame (the planner's only handle on the world),
  * click points are inside the object (the actuator clicks them blind),
  * `is_full` is right (it ends the woodcutting trip),
  * confidence moves with evidence (it drives LLM escalation).
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:  # workstream B owns packaging; tests stay runnable meanwhile
    sys.path.insert(0, str(_SRC))

from gielinor.contracts import Frame, GameState, Perceiver, Rect  # noqa: E402
from gielinor.perception.animation import AnimationDetector  # noqa: E402
from gielinor.perception.chatbox import ChatboxReader  # noqa: E402
from gielinor.perception.detect import (  # noqa: E402
    DEFAULT_SIGNATURES,
    Detection,
    ObjectSignature,
    ObjectTracker,
    TemplateMatcher,
    ViewportDetector,
    click_point_for,
)
from gielinor.perception.inventory import InventoryReader  # noqa: E402
from gielinor.perception.layout import DEFAULT_LAYOUT, LayoutProfile  # noqa: E402
from gielinor.perception.perceiver import PixelPerceiver  # noqa: E402
from gielinor.perception.synthetic import (  # noqa: E402
    CHAT_CORPUS,
    DialogueSpec,
    Difficulty,
    SceneSpec,
    SyntheticRenderer,
    TreeSpec,
    tracking_sequence,
)

PERCEPTION_BUDGET_MS = 50.0  # SPEC section 2


@pytest.fixture()
def renderer() -> SyntheticRenderer:
    return SyntheticRenderer()


def _scene(**kw: object) -> SceneSpec:
    base = {"seq": 1, "trees": (TreeSpec("tree", 150, 150, 32, "blob"),), "inventory_used": 6}
    base.update(kw)  # type: ignore[arg-type]
    return SceneSpec(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The generator itself — if it is not deterministic, nothing below means anything
# --------------------------------------------------------------------------


def test_renderer_is_deterministic(renderer: SyntheticRenderer) -> None:
    a, _ = renderer.render(_scene())
    b, _ = SyntheticRenderer().render(_scene())
    assert np.array_equal(a.image, b.image)


def test_renderer_produces_a_fixed_mode_frame(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene())
    assert frame.image.shape == (DEFAULT_LAYOUT.screen_h, DEFAULT_LAYOUT.screen_w, 3)
    assert frame.image.dtype == np.uint8
    assert frame.backend == "synthetic"


def test_static_scene_is_pixel_identical_between_frames(renderer: SyntheticRenderer) -> None:
    # Churn detection is meaningless if the renderer never holds still.
    a, _ = renderer.render(_scene(seq=1))
    b, _ = renderer.render(_scene(seq=2))
    assert np.array_equal(a.image, b.image)


# --------------------------------------------------------------------------
# Object detection
# --------------------------------------------------------------------------


def _iou(a: Rect, b: Rect) -> float:
    i = a.clip_to(b)
    inter = i.w * i.h
    return inter / (a.w * a.h + b.w * b.h - inter) if inter else 0.0


def test_detector_finds_every_tree(renderer: SyntheticRenderer) -> None:
    spec = _scene(
        trees=(
            TreeSpec("tree", 120, 140, 32, "blob"),
            TreeSpec("tree", 300, 120, 30, "blob"),
            TreeSpec("willow_tree", 420, 240, 34, "blob"),
        )
    )
    frame, gt = renderer.render(spec)
    dets = ViewportDetector().detect(frame.image)
    assert len(dets) == len(gt.objects)
    for label, rect in gt.objects:
        best = max(dets, key=lambda d: _iou(d.bbox, rect))
        assert _iou(best.bbox, rect) > 0.7, f"{label} bbox {rect} vs {best.bbox}"
        assert best.label == label


def test_detector_ignores_the_grass_it_shares_a_hue_with(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene(trees=()))
    assert ViewportDetector().detect(frame.image) == []


def test_miscalibrated_signature_is_rejected_not_reported(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene())
    everything = ObjectSignature(label="bogus", hsv_lo=(0, 0, 0), hsv_hi=(179, 255, 255))
    assert ViewportDetector([everything]).detect(frame.image) == []


def test_detection_survives_partial_occlusion() -> None:
    r = SyntheticRenderer(difficulty=Difficulty(occlusion=0.3))
    frame, gt = r.render(_scene(trees=(TreeSpec("tree", 200, 170, 38, "blob"),)))
    dets = ViewportDetector().detect(frame.image)
    assert len(dets) >= 1
    assert _iou(dets[0].bbox, gt.objects[0][1]) > 0.5


# --------------------------------------------------------------------------
# Click points — the property the actuator depends on
# --------------------------------------------------------------------------


def _eroded(mask: np.ndarray, iters: int = 2) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask.astype(np.uint8), k, iterations=iters).astype(bool)


@pytest.mark.parametrize("shape", ["blob", "fork", "crescent"])
def test_click_point_is_inside_the_eroded_mask(renderer: SyntheticRenderer, shape: str) -> None:
    frame, _ = renderer.render(_scene(trees=(TreeSpec("tree", 220, 170, 38, shape),)))
    dets = ViewportDetector().detect(frame.image)
    assert dets, f"no detection for {shape}"
    for det in dets:
        lx = det.click_point.x - det.bbox.x
        ly = det.click_point.y - det.bbox.y
        assert det.mask[ly, lx], f"{shape}: click point is not even on the object"
        assert _eroded(det.mask)[ly, lx], f"{shape}: click point is on the boundary"


@pytest.mark.parametrize("shape", ["fork", "crescent"])
def test_bbox_centre_would_have_missed_on_concave_shapes(
    renderer: SyntheticRenderer, shape: str
) -> None:
    # Proves the test above is not vacuous: on these silhouettes the obvious
    # implementation (bbox centre) lands on background.
    frame, _ = renderer.render(_scene(trees=(TreeSpec("tree", 220, 170, 38, shape),)))
    det = ViewportDetector().detect(frame.image)[0]
    cx = det.bbox.center.x - det.bbox.x
    cy = det.bbox.center.y - det.bbox.y
    assert not det.mask[cy, cx], f"{shape} is not actually concave; the test proves nothing"


def test_click_point_falls_back_to_distance_transform_when_erosion_empties() -> None:
    thin = np.zeros((21, 41), dtype=bool)
    thin[10, 5:36] = True  # a 1 px line: erosion annihilates it
    pt = click_point_for(thin, erode_iters=2)
    assert pt is not None and thin[pt[1], pt[0]]


def test_click_point_of_an_empty_mask_is_none() -> None:
    assert click_point_for(np.zeros((10, 10), dtype=bool)) is None


# --------------------------------------------------------------------------
# oid stability — the planner's only handle on the world
# --------------------------------------------------------------------------


def test_oid_is_stable_across_consecutive_frames() -> None:
    perceiver = PixelPerceiver()
    seen: list[int] = []
    previous: GameState | None = None
    for frame, _ in tracking_sequence(5, drift=3, add_at=None):
        state = perceiver.perceive(frame, previous)
        previous = state
        assert len(state.objects) == 1
        seen.append(state.objects[0].oid)
    assert len(set(seen)) == 1, f"oid churned across frames: {seen}"


def test_new_object_gets_a_new_oid_without_disturbing_the_old_one() -> None:
    perceiver = PixelPerceiver()
    previous: GameState | None = None
    first_oid: int | None = None
    for frame, _ in tracking_sequence(6, drift=3, add_at=3):
        state = perceiver.perceive(frame, previous)
        previous = state
        oids = sorted(o.oid for o in state.objects)
        if frame.seq < 3:
            assert len(oids) == 1
            first_oid = first_oid or oids[0]
            assert oids[0] == first_oid
        else:
            assert len(oids) == 2
            assert first_oid in oids
            assert max(oids) != first_oid


def test_tracker_holds_an_oid_through_a_dropped_detection() -> None:
    tracker = ObjectTracker(max_missing=2)
    det = Detection("tree", Rect(100, 100, 40, 40), Rect(100, 100, 40, 40).center, 0.9, 1600)
    oid = tracker.update([det])[0]
    assert tracker.update([]) == []  # one frame with nothing detected
    assert tracker.update([det]) == [oid], "a single missed frame must not renumber the world"


def test_tracker_retires_an_oid_that_stays_missing() -> None:
    tracker = ObjectTracker(max_missing=2)
    det = Detection("tree", Rect(100, 100, 40, 40), Rect(100, 100, 40, 40).center, 0.9, 1600)
    oid = tracker.update([det])[0]
    for _ in range(4):
        tracker.update([])
    assert tracker.update([det]) != [oid]


def test_tracker_never_associates_across_labels() -> None:
    tracker = ObjectTracker()
    box = Rect(100, 100, 40, 40)
    a = tracker.update([Detection("tree", box, box.center, 0.9, 1600)])[0]
    b = tracker.update([Detection("bank_booth", box, box.center, 0.9, 1600)])[0]
    assert a != b


def test_two_similar_objects_keep_distinct_oids() -> None:
    tracker = ObjectTracker()
    left = Rect(100, 100, 40, 40)
    right = Rect(300, 100, 40, 40)
    oids = tracker.update(
        [
            Detection("tree", left, left.center, 0.9, 1600),
            Detection("tree", right, right.center, 0.9, 1600),
        ]
    )
    assert len(set(oids)) == 2
    again = tracker.update(
        [
            Detection("tree", Rect(102, 100, 40, 40), Rect(102, 100, 40, 40).center, 0.9, 1600),
            Detection("tree", Rect(298, 100, 40, 40), Rect(298, 100, 40, 40).center, 0.9, 1600),
        ]
    )
    assert again == oids, "identities swapped between two similar objects"


# --------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------


def test_template_matcher_finds_a_registered_sprite(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene())
    slot = DEFAULT_LAYOUT.inventory_slots()[0]
    patch = frame.image[slot.y : slot.y + slot.h, slot.x : slot.x + slot.w]
    tm = TemplateMatcher()
    tm.add_template("item_slot", patch)
    hits = tm.find(frame.image, DEFAULT_LAYOUT.side_panel, threshold=0.95)
    assert hits and any(_iou(h.bbox, slot) > 0.8 for h in hits)


def test_template_matcher_is_inert_when_unconfigured(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene())
    assert TemplateMatcher().find(frame.image) == []


# --------------------------------------------------------------------------
# Inventory
# --------------------------------------------------------------------------


@pytest.mark.parametrize("used", [0, 1, 13, 27, 28])
def test_inventory_occupancy_matches_ground_truth(renderer: SyntheticRenderer, used: int) -> None:
    frame, gt = renderer.render(_scene(inventory_used=used))
    reading = InventoryReader().read(frame.image)
    assert reading.state.used == gt.inventory_used == used
    assert len(reading.state.slots) == 28


def test_is_full_is_exactly_right_at_the_boundary(renderer: SyntheticRenderer) -> None:
    # The whole woodcutting loop turns on this one boolean.
    for used, expected in ((27, False), (28, True)):
        frame, _ = renderer.render(_scene(inventory_used=used))
        state = InventoryReader().read(frame.image).state
        assert state.is_full is expected
        assert state.free == 28 - used


def test_inventory_occupancy_survives_scene_noise() -> None:
    r = SyntheticRenderer(difficulty=Difficulty.busy())
    for used in (0, 9, 28):
        frame, _ = r.render(_scene(inventory_used=used))
        assert InventoryReader().read(frame.image).state.used == used


def test_inventory_slot_rects_tile_the_grid() -> None:
    slots = DEFAULT_LAYOUT.inventory_slots()
    assert len(slots) == 28
    assert len({(s.x, s.y) for s in slots}) == 28
    for s in slots:
        assert s.clip_to(DEFAULT_LAYOUT.inventory_rect) == s


# --------------------------------------------------------------------------
# Dialogue
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec",
    [
        DialogueSpec("npc", "Woodsman", ("Hello there, adventurer.", "Care for an axe?")),
        DialogueSpec("options", options=("Yes please.", "No thanks.", "Tell me more.")),
        DialogueSpec(
            "level_up",
            body=(
                "Congratulations, you just advanced a Woodcutting level.",
                "Your Woodcutting level is now 42.",
            ),
        ),
        DialogueSpec("continue", body=("You have completed the tutorial.",)),
    ],
    ids=["npc", "options", "level_up", "continue"],
)
def test_dialogue_is_detected_and_classified(
    renderer: SyntheticRenderer, spec: DialogueSpec
) -> None:
    frame, gt = renderer.render(_scene(dialogue=spec))
    got = ChatboxReader().read(frame.image).dialogue
    assert got.is_open
    assert got.kind == gt.dialogue.kind
    assert got.speaker == gt.dialogue.speaker
    assert got.body == gt.dialogue.body
    assert got.options == gt.dialogue.options


def test_dialogue_options_are_tagged_as_selectable(renderer: SyntheticRenderer) -> None:
    spec = DialogueSpec("options", options=("Yes please.", "No thanks."))
    frame, _ = renderer.render(_scene(dialogue=spec))
    reading = ChatboxReader().read(frame.image)
    selectable = [ln.text for ln in reading.lines if ln.source == "dialogue_option"]
    assert selectable == ["Yes please.", "No thanks."]


def test_chat_log_is_not_reported_as_a_dialogue(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:4]))
    reading = ChatboxReader().read(frame.image)
    assert reading.mode == "log"
    assert not reading.dialogue.is_open
    assert all(ln.is_untrusted for ln in reading.lines)


def test_player_dialogue_is_distinguished_by_name(renderer: SyntheticRenderer) -> None:
    spec = DialogueSpec("npc", "Woodsman", ("Care for an axe?", "It is a good axe."))
    frame, _ = renderer.render(_scene(dialogue=spec))
    got = ChatboxReader(player_name="Woodsman").read(frame.image).dialogue
    assert got.kind == "player"


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------


def _run_animation(renderer: SyntheticRenderer, *, phase: bool, camera: bool) -> list[bool]:
    detector = AnimationDetector()
    out: list[bool] = []
    for i in range(6):
        frame, _ = renderer.render(
            _scene(
                seq=i,
                animate_phase=(i + 1) if phase else 0,
                camera_shift=(i * 9) if camera else 0,
            )
        )
        out.append(detector.update(frame.image).is_animating)
    return out


def test_animation_detects_chopping(renderer: SyntheticRenderer) -> None:
    readings = _run_animation(renderer, phase=True, camera=False)
    assert readings[0] is False, "the first frame has nothing to compare against"
    assert all(readings[1:]), readings


def test_idle_player_is_not_animating(renderer: SyntheticRenderer) -> None:
    assert not any(_run_animation(renderer, phase=False, camera=False))


def test_camera_motion_is_not_mistaken_for_animation(renderer: SyntheticRenderer) -> None:
    # The documented false positive, and the guard against it: churn in the
    # player box only counts when it beats churn in the scene around it.
    assert not any(_run_animation(renderer, phase=False, camera=True))


def test_camera_motion_is_reported_as_moving(renderer: SyntheticRenderer) -> None:
    detector = AnimationDetector()
    moving: list[bool] = []
    for i in range(5):
        frame, _ = renderer.render(_scene(seq=i, camera_shift=i * 9))
        moving.append(detector.update(frame.image).is_moving)
    assert any(moving)


def test_animation_confidence_is_zero_before_the_buffer_fills(
    renderer: SyntheticRenderer,
) -> None:
    detector = AnimationDetector()
    frame, _ = renderer.render(_scene())
    assert detector.update(frame.image).confidence == 0.0


# --------------------------------------------------------------------------
# The composed perceiver
# --------------------------------------------------------------------------


def test_pixel_perceiver_satisfies_the_protocol() -> None:
    assert isinstance(PixelPerceiver(), Perceiver)


def test_perceive_populates_every_part_of_gamestate(renderer: SyntheticRenderer) -> None:
    frame, _gt = renderer.render(
        _scene(chat=CHAT_CORPUS[:4], tooltip="Chop down Tree", inventory_used=11)
    )
    state = PixelPerceiver().perceive(frame)
    assert state.seq == frame.seq and state.captured_at == frame.captured_at
    assert state.inventory.used == 11
    assert len(state.objects) == 1
    assert {ln.source for ln in state.text} == {"chatbox", "tooltip"}
    assert state.minimap.bbox == DEFAULT_LAYOUT.minimap
    assert state.player.hitpoints == 31 and state.player.run_energy == 88
    assert state.perception_ms > 0.0


def test_object_lookup_helpers_work_on_the_produced_state(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene(trees=(TreeSpec("tree", 150, 150, 32),)))
    state = PixelPerceiver().perceive(frame)
    oid = state.objects[0].oid
    assert state.object_by_id(oid) is state.objects[0]
    assert state.objects_labelled("tree") == state.objects
    assert state.object_by_id(9999) is None


def test_per_stage_timings_are_recorded(renderer: SyntheticRenderer) -> None:
    perceiver = PixelPerceiver()
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:4]))
    state = perceiver.perceive(frame)
    assert set(perceiver.stage_ms) == {
        "chat",
        "tooltip",
        "objects",
        "inventory",
        "animation",
        "orbs",
    }
    assert all(v >= 0.0 for v in perceiver.stage_ms.values())
    # The total must account for the parts; slack covers the assembly around them.
    assert sum(perceiver.stage_ms.values()) <= state.perception_ms + 1e-6


def test_perception_fits_the_frame_budget(renderer: SyntheticRenderer) -> None:
    perceiver = PixelPerceiver()
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:6], tooltip="Chop down Tree"))
    for _ in range(3):  # warm caches; the fast loop is a steady-state measurement
        perceiver.perceive(frame)
    median = statistics.median(perceiver.perceive(frame).perception_ms for _ in range(9))
    assert median < PERCEPTION_BUDGET_MS, (
        f"{median:.1f} ms exceeds the {PERCEPTION_BUDGET_MS} ms budget"
    )


def test_confidence_is_computed_not_hardcoded(renderer: SyntheticRenderer) -> None:
    perceiver = PixelPerceiver()
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:4]))
    state = perceiver.perceive(frame)
    assert 0.0 < state.confidence <= 1.0
    assert set(perceiver.confidence_breakdown) >= {"ocr", "objects", "inventory", "layout"}


def test_confidence_falls_when_the_layout_does_not_fit_the_frame() -> None:
    # This is what wakes the slow loop: geometry mismatch invalidates every crop.
    renderer = SyntheticRenderer()
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:4]))
    good = PixelPerceiver().perceive(frame).confidence

    small = Frame(image=frame.image[:400, :600].copy(), captured_at=0.0, seq=1, backend="synthetic")
    bad = PixelPerceiver().perceive(small).confidence
    assert bad <= 0.4 < good


def test_confidence_reflects_unrecognised_text_colours(renderer: SyntheticRenderer) -> None:
    known, _ = renderer.render(_scene(chat=(("A known white line.", (255, 255, 255)),)))
    odd, _ = renderer.render(_scene(chat=(("An off-palette line.", (137, 71, 199)),)))
    p1, p2 = PixelPerceiver(), PixelPerceiver()
    p1.perceive(known)
    p2.perceive(odd)
    assert p2.confidence_breakdown["ocr_palette"] < p1.confidence_breakdown["ocr_palette"]


def test_continuity_term_reacts_to_an_implausible_jump(renderer: SyntheticRenderer) -> None:
    perceiver = PixelPerceiver()
    calm, _ = renderer.render(_scene(inventory_used=4))
    first = perceiver.perceive(calm)
    steady = perceiver.perceive(calm, first)
    jump, _ = renderer.render(_scene(inventory_used=26))
    jumped = perceiver.perceive(jump, first)
    assert perceiver.confidence_breakdown["continuity"] < 1.0
    assert jumped.confidence < steady.confidence


def test_reset_clears_cross_frame_state(renderer: SyntheticRenderer) -> None:
    perceiver = PixelPerceiver()
    frame, _ = renderer.render(_scene())
    perceiver.perceive(frame)
    assert perceiver.tracker.active_oids
    perceiver.reset()
    assert perceiver.tracker.active_oids == ()
    assert perceiver.animation.samples == 0


def test_perceive_is_repeatable_for_the_same_frame(renderer: SyntheticRenderer) -> None:
    frame, _ = renderer.render(_scene(chat=CHAT_CORPUS[:3], inventory_used=9))
    a = PixelPerceiver().perceive(frame)
    b = PixelPerceiver().perceive(frame)
    assert [ln.text for ln in a.text] == [ln.text for ln in b.text]
    assert a.inventory.used == b.inventory.used
    assert [(o.oid, o.bbox) for o in a.objects] == [(o.oid, o.bbox) for o in b.objects]


# --------------------------------------------------------------------------
# Layout profile
# --------------------------------------------------------------------------


def test_default_layout_is_self_consistent() -> None:
    assert DEFAULT_LAYOUT.validate() == []
    assert DEFAULT_LAYOUT.validate((503, 765, 3)) == []
    assert DEFAULT_LAYOUT.validate((480, 640, 3))


def test_layout_regions_carry_their_trust_label() -> None:
    sources = {r.name: r.source for r in DEFAULT_LAYOUT.text_regions()}
    assert sources["chatbox"] == "chatbox"
    assert sources["tooltip"] == "tooltip"


def test_layout_from_mapping_overrides_only_what_is_given() -> None:
    profile = LayoutProfile.from_mapping({"name": "custom", "chatbox": [1, 2, 3, 4]})
    assert profile.name == "custom"
    assert profile.chatbox == Rect(1, 2, 3, 4)
    assert profile.viewport == DEFAULT_LAYOUT.viewport


def test_layout_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "profile.toml"
    path.write_text('[layout]\nname = "from_toml"\ninv_cols = 4\nminimap = [1, 2, 3, 4]\n')
    profile = LayoutProfile.from_toml(path)
    assert profile.name == "from_toml"
    assert profile.minimap == Rect(1, 2, 3, 4)


def test_layout_validate_flags_a_broken_grid() -> None:
    assert LayoutProfile(inv_cols=5).validate()


def test_no_signature_is_defined_without_bounds() -> None:
    for sig in DEFAULT_SIGNATURES:
        assert sig.min_area > 0 and sig.max_area > sig.min_area
        assert 0 < sig.aspect_lo < sig.aspect_hi
