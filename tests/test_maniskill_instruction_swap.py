"""CPU checks of the instruction-swap scene ids, plans, outcomes and metrics (no simulator)."""
import numpy as np
import pytest
import torch

from fluffy_bassoon.rl import maniskill_instruction_swap as swap
from fluffy_bassoon.rl import maniskill_pi05 as ms


INFOS = (16, 0, 16, 0, 1, 0, 7, 4, 15, 0)
NAMES = [f"obj{i}" for i in range(25)]


def rlinf_decode(episode, infos=INFOS):
    """Verbatim arithmetic of RLinf PutOnPlateInScene25MultiCarrot._initialize_episode_pre."""
    lc, lc_offset, lo, lo_offset, lp, lp_offset, l1, l2, le, le_offset = infos
    carrot = episode // (le * lp * lo * l1 * l2) + lc_offset
    extra = (episode // (lp * lo * l1 * l2)) % le
    extra = (carrot + extra + 1) % lc + lc_offset
    return (carrot, extra, (episode // (lo * l1 * l2)) % lp, (episode // (l1 * l2)) % lo + lo_offset,
            (episode // l2) % l1, episode % l2)


def test_multi_episode_id_inverts_rlinf_arithmetic_and_is_a_bijection():
    lc, _, lo, _, lp, _, l1, l2, le, _ = INFOS
    seen = set()
    for slot0 in range(16):
        for slot1 in range(16):
            if slot0 == slot1:
                continue
            for overlay in (0, 7, 15):
                for pos in range(l1):
                    for quat in range(l2):
                        episode = swap.multi_episode_id(INFOS, slot0, slot1, overlay, pos, quat)
                        assert rlinf_decode(episode) == (slot0, slot1, 0, overlay, pos, quat)
                        assert swap.decode_multi_episode_id(INFOS, episode) == dict(
                            slot0=slot0, slot1=slot1, plate=0, overlay=overlay, position=pos, orientation=quat)
                        seen.add(episode)
    assert len(seen) == 16 * 15 * 3 * l1 * l2 and max(seen) < lc * le * lp * lo * l1 * l2


def test_multi_episode_id_rejects_bad_scenes():
    with pytest.raises(ValueError):
        swap.multi_episode_id(INFOS, 3, 3, 0, 0, 0)
    with pytest.raises(ValueError):
        swap.multi_episode_id(INFOS, 16, 3, 0, 0, 0)
    with pytest.raises(ValueError):
        swap.multi_episode_id((25, 0, 16, 0, 1, 0, 7, 4, 24, 0), 0, 1, 0, 0, 0)


def test_plan_balances_slot_order_and_blocks_mix_pairs():
    pairs = [(0, 9), (12, 13), (0, 13), (9, 12)]
    plan = swap.plan_multi_scenes(INFOS, pairs, 24, np.random.default_rng(3))
    assert plan == swap.plan_multi_scenes(INFOS, pairs, 24, np.random.default_rng(3))
    assert len(plan) == 96 and all(s["overlay"] < 16 for s in plan)
    for a, b in pairs:
        rows = [s for s in plan if s["pair"] == [a, b]]
        assert sum(s["slot0"] == a for s in rows) == sum(s["slot0"] == b for s in rows) == 12
    blocks = swap.interleave_blocks(plan, 32)
    assert len(blocks) == 3 and sum(len(b) for b in blocks) == 96
    for block in blocks:
        for a, b in pairs:
            rows = [s for s in block if s["pair"] == [a, b]]
            assert len(rows) == 8 and sum(s["slot0"] == a for s in rows) == 4
    with pytest.raises(ValueError):
        swap.plan_multi_scenes(INFOS, pairs, 3, np.random.default_rng(0))
    with pytest.raises(ValueError):
        swap.interleave_blocks(plan, 40)


def test_condition_instructions_follow_rlinf_template():
    scenes = [dict(slot0=0, slot1=9), dict(slot0=13, slot1=12)]
    names = ["carrot"] + [""] * 8 + ["toy bear", "", "", "banana", "hamburger"]
    assert swap.condition_instructions(scenes, names, "name_slot0") == ["put carrot on plate", "put hamburger on plate"]
    assert swap.condition_instructions(scenes, names, "name_slot1") == ["put toy bear on plate", "put banana on plate"]
    assert swap.condition_instructions(scenes, names, "vague") == ["put object on plate"] * 2
    with pytest.raises(ValueError):
        swap.condition_instructions(scenes, names, "other")


def fake_info(grasp, on, dist):
    return dict(swap_grasp=torch.tensor(grasp), swap_on_plate=torch.tensor(on),
                swap_plate_dist=torch.tensor(dist, dtype=torch.float32),
                success=torch.tensor([row[0] for row in on]))


def test_slot_outcomes_track_both_objects():
    tracker = swap.SlotOutcomes(2)
    for step in range(ms.STEPS):
        grasp = [[step >= 10, False], [False, step >= 30]]
        on = [[step >= 20, False], [False, 40 <= step < 50]]
        tracker.update(fake_info(grasp, on, [[.2 - step * .002, .3], [.25, .3 - step * .002]]))
    first, second = tracker.rows()
    assert first["slot0_grasped_once"] and first["slot0_first_grasp_step"] == 10
    assert first["slot0_on_plate_once"] and first["slot0_on_plate_end"] and first["slot0_first_on_plate_step"] == 20
    assert not first["slot1_grasped_once"] and first["slot1_first_grasp_step"] == -1
    assert second["slot1_on_plate_once"] and not second["slot1_on_plate_end"]
    assert second["slot1_plate_distance_start_m"] == pytest.approx(.3)
    assert second["slot1_plate_distance_min_m"] == pytest.approx(.3 - 79 * .002, abs=1e-6)
    assert tracker.slot0_success_mismatch == 0
    with pytest.raises(ValueError):
        tracker.update(fake_info([[True]], [[True]], [[.1]]))


def test_slot_outcomes_require_full_episode():
    tracker = swap.SlotOutcomes(1)
    tracker.update(fake_info([[False, False]], [[False, False]], [[.1, .2]]))
    with pytest.raises(RuntimeError):
        tracker.rows()


def outcome(slot0, slot1, placed=None, grasped=None, placed_step=20, grasp_step=10):
    row = dict(slot0=slot0, slot1=slot1)
    for s, obj in enumerate((slot0, slot1)):
        row[f"slot{s}_on_plate_once"] = placed == obj
        row[f"slot{s}_first_on_plate_step"] = placed_step if placed == obj else -1
        row[f"slot{s}_grasped_once"] = obj in (placed, grasped)
        row[f"slot{s}_first_grasp_step"] = grasp_step if obj in (placed, grasped) else -1
    return row


def test_chosen_object_prefers_placement_then_first_grasp():
    assert swap.chosen_object(outcome(0, 9, placed=9)) == 9
    assert swap.chosen_object(outcome(0, 9, grasped=0)) == 0
    assert swap.chosen_object(outcome(0, 9)) is None
    row = outcome(0, 9, grasped=0)
    row.update(slot1_grasped_once=True, slot1_first_grasp_step=4)
    assert swap.chosen_object(row) == 9


def multi_rows(policy, scenes):
    rows = []
    for key, (a, b) in enumerate(scenes):
        for condition in swap.CONDITIONS:
            named = {"name_slot0": a, "name_slot1": b, "vague": None}[condition]
            row = outcome(a, b, placed=policy(a, b, named))
            row.update(scene_key=str(key), condition=condition, pair=sorted([a, b]))
            rows.append(row)
    return rows


SCENES = [(0, 9), (9, 0)] * 5


def test_summary_of_a_perfect_instruction_follower_passes():
    rows = multi_rows(lambda a, b, named: named if named is not None else (a if hash((a, b)) % 2 else b), SCENES)
    single = [dict(object=o, success_once=True) for o in (0, 9) for _ in range(4)]
    metrics = swap.summarize(single, rows, NAMES)
    assert metrics["swap"]["follows_all"]["rate"] == 1 and metrics["swap"]["changes_all"]["rate"] == 1
    assert metrics["per_object"]["0"]["explicit_success"] == dict(rate=1.0, n=10, k=10)
    assert metrics["per_object"]["0"]["single_success"]["rate"] == 1
    criteria = metrics["criteria"]
    assert criteria["explicit_success_ge_60pct_each"]["passed"] and criteria["swap_follow_ge_80pct"]["passed"]
    assert metrics["vague"]["acted"] == 10


def test_summary_of_a_language_blind_policy_fails_swap_and_gap():
    rows = multi_rows(lambda a, b, named: a, SCENES)   # always places the slot-0 object
    metrics = swap.summarize([], rows, NAMES)
    assert metrics["swap"]["changes_all"]["rate"] == 0 and metrics["swap"]["follows_all"]["rate"] == 0
    assert metrics["per_object"]["0"]["explicit_success"]["rate"] == .5
    assert metrics["per_object"]["0"]["gap_explicit_minus_vague"] == 0
    assert metrics["vague"]["slot0_share_among_acted"] == 1
    assert not metrics["criteria"]["gap_ge_30_points_each"]["passed"]


def test_summary_ignores_incomplete_scenes():
    rows = multi_rows(lambda a, b, named: named, SCENES)[:-1]
    metrics = swap.summarize([], rows, NAMES)
    assert metrics["swap"]["pairs"] == len(SCENES) - 1 and metrics["incomplete_scenes"] == 1
