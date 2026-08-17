"""The calibration profile and the perception layout must not drift apart.

These tests exist because they already did. Perception carried its own rects
(minimap at y=8, chatbox 500x120) while the actuator's TOML carried different
ones (y=4, 506x130). Nothing failed, because perception reports absolute click
points that the actuator uses verbatim, so the two tables never had to agree
for a synthetic demo to run green. They would have had to agree the moment
anyone calibrated against a real client — and that failure would have
presented as "the agent clicks slightly off", i.e. as a perception bug.

So the assertions below are deliberately about *identity between two sources*
rather than about any particular coordinate being correct. The numbers are
still uncalibrated guesses; what is now guaranteed is that there is only one
copy of each guess.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gielinor.config import DEFAULT_PROFILE, Profile, load_profile
from gielinor.profiles import layout_for, layout_from_profile

if TYPE_CHECKING:
    from gielinor.perception.layout import LayoutProfile


@pytest.fixture
def profile() -> Profile:
    return load_profile(DEFAULT_PROFILE)


@pytest.fixture
def layout(profile: Profile) -> LayoutProfile:
    return layout_from_profile(profile)


@pytest.mark.parametrize("region", ["viewport", "chatbox", "chat_input", "minimap"])
def test_shared_rects_are_identical(profile: Profile, layout: LayoutProfile, region: str) -> None:
    assert getattr(layout, region) == getattr(profile, region).to_rect(), (
        f"{region} differs between the calibration profile and the derived layout"
    )


def test_client_size_matches(profile: Profile, layout: LayoutProfile) -> None:
    assert (layout.screen_w, layout.screen_h) == profile.client_size


def test_inventory_grid_steps_match_the_profile_pitch(
    profile: Profile, layout: LayoutProfile
) -> None:
    """Pitch, not cell size, is what must survive the projection.

    A slot rect that overshoots into the gutter is harmless for occupancy
    detection. A wrong *step* misreads every slot after the first.
    """
    grid = profile.inventory_grid
    assert (layout.inv_origin_x, layout.inv_origin_y) == (grid.origin_x, grid.origin_y)
    assert (layout.inv_cell_w, layout.inv_cell_h) == (grid.pitch_x, grid.pitch_y)
    assert layout.inv_cols * layout.inv_rows == 28


def test_every_slot_origin_agrees_with_the_profile(profile: Profile, layout: LayoutProfile) -> None:
    """All 28 slots, both sources, checked on the property that has to hold.

    Origins must match exactly, because both sides step by pitch. Sizes must
    NOT be asserted equal: the profile models the 36x32 item cell the client
    actually draws, while the layout deliberately uses the 42x36 pitch as its
    cell so slot rects tile without gaps. `layout_from_profile` documents why
    that asymmetry is the safe direction to lose information in.
    """
    slots = layout.inventory_slots()
    assert len(slots) == 28
    for i, slot in enumerate(slots):
        expected = profile.inventory_grid.slot_rect(i)
        assert (slot.x, slot.y) == (expected.x, expected.y), (
            f"slot {i} origin disagrees between profile and layout"
        )
        assert slot.w >= expected.w and slot.h >= expected.h, (
            f"slot {i} layout rect is smaller than the item cell it must contain"
        )


def test_perception_only_regions_come_from_the_profile(profile: Profile) -> None:
    """The [perception] TOML section must actually be read, not silently ignored."""
    extras = profile.perception
    assert extras.tooltip is not None, "profile is missing [perception].tooltip"
    layout = layout_from_profile(profile)
    assert layout.tooltip == extras.tooltip.to_rect()
    assert layout.side_panel == extras.side_panel.to_rect()  # type: ignore[union-attr]
    assert layout.hp_orb == extras.hp_orb.to_rect()  # type: ignore[union-attr]


def test_derived_layout_is_self_consistent(layout: LayoutProfile) -> None:
    """Perception's own validator must accept what we hand it."""
    assert layout.validate() == []


def test_all_regions_fit_inside_the_client(profile: Profile, layout: LayoutProfile) -> None:
    client = profile.viewport.to_rect()
    del client  # only used to prove the import path; real bound is client_w/h
    for name in ("viewport", "chatbox", "chat_input", "minimap", "side_panel", "tooltip"):
        r = getattr(layout, name)
        assert r.x >= 0 and r.y >= 0, f"{name} has a negative origin"
        assert r.x + r.w <= layout.screen_w, f"{name} overflows the client width"
        assert r.y + r.h <= layout.screen_h, f"{name} overflows the client height"


def test_layout_for_is_cached_but_clearable() -> None:
    a = layout_for(DEFAULT_PROFILE)
    b = layout_for(DEFAULT_PROFILE)
    assert a is b, "layout_for should cache; it is called per perceiver construction"
    layout_for.cache_clear()
    c = layout_for(DEFAULT_PROFILE)
    assert c is not a
    assert c == a


def test_perceiver_is_built_with_the_profile_layout() -> None:
    """The wiring, not just the projection: default_perceiver must use it."""
    from gielinor.runtime.loop import default_perceiver

    perceiver = default_perceiver()
    layout = getattr(perceiver, "layout", None)
    if layout is None:
        pytest.skip("perceiver exposes no .layout attribute to inspect")
    assert layout == layout_for(DEFAULT_PROFILE), (
        "default_perceiver built a perceiver on defaults instead of the calibration profile"
    )


def test_uncalibrated_profile_is_flagged(profile: Profile) -> None:
    """A guard on the guard: if this ever passes silently, someone set the flag
    without measuring. The demo profile is guesses and must say so."""
    assert profile.calibrated is False
