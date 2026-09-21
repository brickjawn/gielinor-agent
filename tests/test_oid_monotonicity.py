"""Identity invariants.

These are not tests of the tracker's accuracy. Accuracy is a Phase 1 measurement
against labelled frames and it will be imperfect. These test the property that
has to hold *regardless* of accuracy: an oid never refers to two different
things. A tracker that loses objects is a tracker that drops actions; a tracker
that confuses objects is a tracker that clicks the wrong one, and no amount of
downstream validation catches it.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from gielinor.identity import (
    DEFAULT_MAX_MISSES,
    IdentityRegistry,
    Observation,
    appearance,
    appearance_distance,
)
from gielinor.monotonic import NO_ID, MonotonicCounter

# --- a tiny synthetic world --------------------------------------------------

W, H = 320, 240
PALETTE = {
    "tree": (40, 110, 45),
    "oak": (70, 130, 60),
    "player": (190, 160, 120),
    "rock": (120, 120, 130),
}


def render(entities: dict[str, tuple[str, int, int, int, int]]) -> np.ndarray:
    """entities: ground_truth_id -> (label, x0, y0, x1, y1). Painted in order."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[:, :] = (25, 30, 25)
    for _gt, (label, x0, y0, x1, y1) in entities.items():
        frame[y0:y1, x0:x1] = PALETTE[label]
    return frame


def observe(
    frame: np.ndarray, entities: dict[str, tuple[str, int, int, int, int]]
) -> tuple[list[Observation], list[str]]:
    obs, truth = [], []
    for gt, (label, x0, y0, x1, y1) in entities.items():
        bbox = (x0, y0, x1, y1)
        obs.append(Observation(bbox=bbox, label=label, descriptor=appearance(frame, bbox)))
        truth.append(gt)
    return obs, truth


# --- the counter -------------------------------------------------------------


def test_counter_is_strictly_increasing_and_never_repeats():
    c = MonotonicCounter()
    seen = [c.next() for _ in range(10_000)]
    assert len(set(seen)) == len(seen)
    assert seen == sorted(seen)
    assert NO_ID not in seen
    assert c.issued == 10_000


def test_counter_reserves_zero():
    with pytest.raises(ValueError):
        MonotonicCounter(start=0)


def test_counter_is_thread_safe():
    c = MonotonicCounter()
    out: list[list[int]] = []
    lock = threading.Lock()

    def worker():
        mine = [c.next() for _ in range(2_000)]
        with lock:
            out.append(mine)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    flat = [v for sub in out for v in sub]
    assert len(set(flat)) == 8 * 2_000, "a duplicate id under contention is the whole bug"


def test_counter_exposes_no_way_to_return_a_value():
    forbidden = {"free", "release", "reset", "recycle", "give_back"}
    assert forbidden.isdisjoint(dir(MonotonicCounter))


# --- the registry ------------------------------------------------------------


def test_oid_is_stable_while_an_object_persists():
    reg = IdentityRegistry()
    ents = {"a": ("tree", 50, 50, 90, 110)}
    oids = []
    for f in range(20):
        # drift a couple of px per frame, as a rotating camera would
        x = 50 + f
        ents = {"a": ("tree", x, 50, x + 40, 110)}
        frame = render(ents)
        obs, _ = observe(frame, ents)
        tracks = reg.update(f, obs)
        assert len(tracks) == 1
        oids.append(tracks[0].oid)
    assert len(set(oids)) == 1


def test_retired_oid_is_never_reissued_to_a_new_object():
    """The dangerous false positive, in its simplest form.

    An object despawns. A different object appears in exactly the same place.
    If identity came from a per-frame index, or from any recycled pool, the new
    object inherits the old reference and every validity check downstream passes.
    """
    reg = IdentityRegistry()
    box = (100, 100, 140, 160)

    tree = {"a": ("tree", *box)}
    for f in range(3):
        reg.update(f, observe(render(tree), tree)[0])
    first = next(iter(reg.live))

    for f in range(3, 3 + DEFAULT_MAX_MISSES + 2):  # gone
        reg.update(f, [])
    assert not reg.is_live(first)
    assert reg.was_retired(first)

    rock = {"b": ("rock", *box)}
    f = 3 + DEFAULT_MAX_MISSES + 2
    tracks = reg.update(f, observe(render(rock), rock)[0])
    assert len(tracks) == 1
    assert tracks[0].oid != first
    assert tracks[0].oid > first


def test_same_label_same_place_after_despawn_still_gets_a_new_oid():
    """Harder than the previous test: the replacement is indistinguishable.

    Two trees look identical and occupy the same tile. Nothing in the pixels can
    tell them apart, so the only thing preventing a wrong binding is that the
    first oid was retired and cannot come back.
    """
    reg = IdentityRegistry()
    box = (60, 60, 100, 120)
    ents = {"a": ("tree", *box)}
    reg.update(0, observe(render(ents), ents)[0])
    first = next(iter(reg.live))

    for f in range(1, DEFAULT_MAX_MISSES + 3):
        reg.update(f, [])

    ents2 = {"b": ("tree", *box)}
    tracks = reg.update(DEFAULT_MAX_MISSES + 3, observe(render(ents2), ents2)[0])
    assert tracks[0].oid != first


def test_brief_occlusion_does_not_change_oid():
    reg = IdentityRegistry()
    ents = {"a": ("tree", 60, 60, 100, 120)}
    reg.update(0, observe(render(ents), ents)[0])
    oid = next(iter(reg.live))

    for f in range(1, DEFAULT_MAX_MISSES):  # hidden behind a player, but not aged out
        reg.update(f, [])
        assert reg.is_live(oid)

    tracks = reg.update(DEFAULT_MAX_MISSES, observe(render(ents), ents)[0])
    assert [t.oid for t in tracks] == [oid]
    assert tracks[0].occlusions == 1


def test_label_change_never_matches():
    reg = IdentityRegistry()
    box = (60, 60, 100, 120)
    ents = {"a": ("tree", *box)}
    reg.update(0, observe(render(ents), ents)[0])
    oid = next(iter(reg.live))

    other = {"b": ("player", *box)}
    tracks = reg.update(1, observe(render(other), other)[0])
    assert {t.oid for t in tracks} != {oid} or len(tracks) == 2
    assert all(t.oid != oid or t.label == "tree" for t in tracks)


def test_appearance_gate_rejects_a_different_looking_object_in_the_same_box():
    """Geometry alone would match these. Appearance is what stops it."""
    reg = IdentityRegistry()
    box = (60, 60, 100, 120)
    tree = {"a": ("tree", *box)}
    reg.update(0, observe(render(tree), tree)[0])
    oid = next(iter(reg.live))

    # same label, same box, but the pixels are now a player's palette
    frame = render({"b": ("player", *box)})
    obs = [Observation(bbox=box, label="tree", descriptor=appearance(frame, box))]
    tracks = reg.update(1, obs)
    new_ids = {t.oid for t in tracks} - {oid}
    assert new_ids, "a wholesale appearance change must produce a new identity"


def test_appearance_distance_bounds():
    a = appearance(render({"a": ("tree", 0, 0, 40, 40)}), (0, 0, 40, 40))
    b = appearance(render({"a": ("player", 0, 0, 40, 40)}), (0, 0, 40, 40))
    assert appearance_distance(a, a) == pytest.approx(0.0, abs=1e-6)
    assert appearance_distance(a, b) == pytest.approx(1.0, abs=1e-6)
    assert appearance_distance(a, np.zeros_like(a)) == 1.0  # degenerate never matches


# --- the global invariant ----------------------------------------------------


def test_no_oid_ever_refers_to_two_different_ground_truth_objects():
    """The invariant that actually matters, over a messy multi-object scenario.

    Objects move, cross each other, disappear and are replaced. Whatever the
    tracker's accuracy, the binding from oid to real object must be a function:
    one oid, at most one object, for all time. Losing track is allowed. Swapping
    is not.
    """
    rng = np.random.default_rng(0xC0FFEE)
    reg = IdentityRegistry()
    bindings: dict[int, set[str]] = {}

    live: dict[str, tuple[str, int, int, int, int]] = {
        "t1": ("tree", 40, 40, 80, 100),
        "t2": ("tree", 200, 40, 240, 100),
        "p1": ("player", 120, 120, 150, 170),
        "r1": ("rock", 30, 170, 70, 205),
    }
    next_gt = 2

    for f in range(120):
        # drift everything a little; players move faster
        for gt, (label, x0, y0, x1, y1) in list(live.items()):
            step = 6 if label == "player" else 2
            dx = int(rng.integers(-step, step + 1))
            dy = int(rng.integers(-step, step + 1))
            nx0, ny0 = np.clip(x0 + dx, 0, W - 45), np.clip(y0 + dy, 0, H - 45)
            live[gt] = (label, int(nx0), int(ny0), int(nx0 + (x1 - x0)), int(ny0 + (y1 - y0)))

        # churn: occasionally retire one and spawn a replacement of the same type
        if f % 17 == 16 and live:
            victim = sorted(live)[int(rng.integers(0, len(live)))]
            label = live.pop(victim)[0]
            gt = f"{label[0]}{next_gt}"
            next_gt += 1
            x = int(rng.integers(0, W - 45))
            y = int(rng.integers(0, H - 60))
            live[gt] = (label, x, y, x + 40, y + 55)

        frame = render(live)
        obs, truth = observe(frame, live)
        tracks = reg.update(f, obs)

        # rebuild the oid -> ground truth binding for this frame by exact bbox,
        # which is unambiguous here because the renderer and observer agree
        by_box = {o.bbox: t for o, t in zip(obs, truth, strict=True)}
        for t in tracks:
            gt = by_box.get(t.bbox)
            if gt is not None:
                bindings.setdefault(t.oid, set()).add(gt)

    offenders = {oid: gts for oid, gts in bindings.items() if len(gts) > 1}
    assert not offenders, f"oid(s) bound to more than one object: {offenders}"
    assert len(bindings) >= 4
