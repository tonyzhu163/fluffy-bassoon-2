"""Private balanced native plans must preserve cells, disjoint trials and exact budgets."""

from collections import Counter
from copy import deepcopy
import random

import pytest

from fluffy_bassoon.meta_rl.pi05_balanced_plan import (
    EVAL_THETAS, HELDOUT_OVERLAYS, OBJECTS, TRAIN_OVERLAYS, TRAIN_THETAS,
    make_balanced_history_manifest, make_heldout_probe_plan, make_teacher_probe_plan,
    manifest_sha256, plan_sha256, validate_balanced_history_manifest,
)


INFOS = (25, None, 21, None, 1, None, 32, 4)


def resign(manifest):
    for plan in manifest["plans"]:
        plan["plan_sha256"] = plan_sha256(plan)
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    return manifest


def test_default_covers_six_cells_in_both_explicit_history_splits():
    manifest = make_balanced_history_manifest(INFOS, seed=7)
    receipt = validate_balanced_history_manifest(manifest, expected_sha256=manifest["manifest_sha256"])
    assert receipt["train_trials"] == receipt["validation_trials"] == 6
    assert receipt["total_simulator_steps"] == 3840
    assert manifest["train_indices"] == [0] and manifest["validation_indices"] == [1]
    expected = Counter((theta, obj) for theta in TRAIN_THETAS for obj in OBJECTS)
    all_ids, all_seeds, all_scenes = [], [], []
    for index, split in enumerate(("train", "validation")):
        plan = manifest["plans"][index]
        assert plan["split"] == split and plan["metadata_is_private"]
        assert Counter(zip(plan["theta_degrees"], plan["object_indices"])) == expected
        assert plan["attempt_roles"] == ["support", "support", "support", "probe"]
        assert len(plan["reset_seeds"]) == 4
        for lane, obj in enumerate(plan["object_indices"]):
            assert all(attempt[lane]["object"] == obj for attempt in plan["scenes"])
        assert all(scene["overlay"] in TRAIN_OVERLAYS for batch in plan["scenes"] for scene in batch)
        assert all(seed % 2 == index for seed in plan["history_seeds"])
        all_ids.extend(plan["trial_ids"])
        all_seeds.extend(plan["history_seeds"])
        all_scenes.extend(scene["episode_id"] for batch in plan["scenes"] for scene in batch)
    assert len(set(all_ids)) == len(all_ids) == 12
    assert len(set(all_seeds)) == len(all_seeds) == 12
    assert len(set(all_scenes)) == len(all_scenes) == 48
    assert manifest["sample_budget"]["additional_label_motor_calls"] == 768


def test_reproducible_without_consuming_global_rng_and_repeat_budget_scales():
    before = random.getstate()
    left = make_balanced_history_manifest(INFOS, seed=19, trials_per_cell=2)
    assert random.getstate() == before
    right = make_balanced_history_manifest(INFOS, seed=19, trials_per_cell=2)
    assert left == right
    assert left != make_balanced_history_manifest(INFOS, seed=20, trials_per_cell=2)
    assert left["train_indices"] == [0, 1] and left["validation_indices"] == [2, 3]
    assert left["sample_budget"]["total_simulator_steps"] == 7680
    assert all(cell["trials"] == 2 for cells in left["coverage"].values() for cell in cells)


def test_hash_guards_and_signed_budget_and_split_tampering():
    original = make_balanced_history_manifest(INFOS)
    with pytest.raises(ValueError, match="hash differs"):
        validate_balanced_history_manifest(original, expected_sha256="0" * 64)
    changed = deepcopy(original)
    changed["plans"][0]["theta_degrees"][0] = 999
    with pytest.raises(ValueError, match="hash differs"):
        validate_balanced_history_manifest(changed)
    changed = deepcopy(original)
    changed["sample_budget"]["total_simulator_steps"] += 1
    with pytest.raises(ValueError, match="sample budget"):
        validate_balanced_history_manifest(resign(changed))
    changed = deepcopy(original)
    changed["validation_indices"] = [0]
    with pytest.raises(ValueError, match="batch indices"):
        validate_balanced_history_manifest(resign(changed))
    changed = deepcopy(original)
    changed["trial_sets"]["validation"] = changed["trial_sets"]["train"]
    with pytest.raises(ValueError, match="trial sets"):
        validate_balanced_history_manifest(resign(changed))


@pytest.mark.parametrize("change,match", [
    ("cell", "six theta/object cells"), ("history_seed", "seeds/trial identities"),
    ("scene", "Scene identity"), ("private", "attempt contract"),
])
def test_signed_manifest_rejects_unbalanced_or_leaking_histories(change, match):
    manifest = make_balanced_history_manifest(INFOS)
    train, validation = manifest["plans"]
    if change == "cell":
        train["theta_degrees"][0] = 0.
    elif change == "history_seed":
        validation["history_seeds"][0] = train["history_seeds"][0]
    elif change == "scene":
        validation["scenes"][0][0] = deepcopy(train["scenes"][0][0])
    else:
        train["metadata_is_private"] = False
    with pytest.raises(ValueError, match=match):
        validate_balanced_history_manifest(resign(manifest))


def test_support_probe_partners_and_fresh_replicates_are_explicit():
    plan = make_heldout_probe_plan(INFOS, seed=23, probes=2)
    support = plan["support_plan"]
    assert support["theta_degrees"] == [-30., 30., -30., 30.]
    assert set(support["theta_degrees"]) == set(EVAL_THETAS)
    assert support["object_indices"] == [0, 0, 1, 1]
    assert support["attempt_roles"] == ["support"] * 3
    assert plan["opposite_indices"] == [1, 0, 3, 2]
    support_ids = {scene["episode_id"] for batch in support["scenes"] for scene in batch}
    seen = set(support_ids)
    assert all(scene["overlay"] in TRAIN_OVERLAYS for batch in support["scenes"] for scene in batch)
    for probe in plan["probe_plans"]:
        assert probe["plan_sha256"] == plan_sha256(probe)
        assert all(scene["overlay"] in HELDOUT_OVERLAYS for scene in probe["scenes"])
        for lane, partner in enumerate(plan["opposite_indices"]):
            assert probe["scenes"][lane]["object"] == probe["scenes"][partner]["object"]
            assert support["theta_degrees"][lane] == -support["theta_degrees"][partner]
        ids = {scene["episode_id"] for scene in probe["scenes"]}
        assert not ids & seen
        seen.update(ids)
    assert len(set(plan["probe_seeds"])) == 2
    assert plan["plan_sha256"] == plan_sha256(plan)


def test_teacher_qualification_can_pair_all_cells_or_one_theta():
    all_cells = make_teacher_probe_plan(INFOS, seed=31)
    one_theta = make_teacher_probe_plan(INFOS, seed=31, theta_degrees=-45.)
    assert len(all_cells["scenes"]) == 1 and len(all_cells["scenes"][0]) == 6
    assert one_theta["theta_degrees"] == [-45., -45.]
    assert one_theta["object_indices"] == [0, 1]
    assert all(scene["overlay"] in HELDOUT_OVERLAYS for scene in all_cells["scenes"][0])
    assert all_cells["attempt_roles"] == ["probe"]
    assert all_cells["plan_sha256"] == plan_sha256(all_cells)


@pytest.mark.parametrize("kwargs", [dict(seed=-1), dict(seed=True), dict(seed=2**32),
                                    dict(lanes=2), dict(trials_per_cell=0), dict(trials_per_cell=True)])
def test_invalid_history_dimensions_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_balanced_history_manifest(INFOS, **kwargs)


def test_insufficient_distinct_scene_capacity_is_rejected():
    with pytest.raises(ValueError, match="Scene budget"):
        make_balanced_history_manifest((25, None, 21, None, 1, None, 1, 1))
