import copy

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_multitask import assert_context_equal, validate_manifest


def manifest():
    return dict(format="frozen_vla_multitask_v1", tasks=[
        dict(task_key=f"task-{i}", suite="libero_90", task_id=i,
             semantic_task=f"instruction-{i}", split=split)
        for i, split in enumerate(("train", "validation", "test"))])


def test_semantic_holdout_rejects_duplicate_tasks_across_suite_aliases():
    good = manifest()
    validate_manifest(good)
    bad = copy.deepcopy(good)
    bad["tasks"][-1]["semantic_task"] = bad["tasks"][0]["semantic_task"]
    bad["tasks"][-1]["suite"] = "libero_10"
    with pytest.raises(ValueError, match="semantic"):
        validate_manifest(bad)


def test_heldout_test_tasks_cannot_supply_supervised_labels():
    bad = manifest()
    bad["tasks"][-1]["datasets"] = dict(ad="test-labels.npz")
    with pytest.raises(ValueError, match="Held-out"):
        validate_manifest(bad)


def test_objective_comparison_allows_only_query_target_changes():
    row = {key: np.zeros((3, 2)) for key in
           ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")}
    ad, dpt = [dict(**row, target_noise=np.zeros((3, 2)))], [dict(**row, target_noise=np.ones((3, 2)))]
    assert_context_equal(ad, dpt)
    dpt[0]["executed_noise"] = np.ones((3, 2))
    with pytest.raises(ValueError, match="executed contexts"):
        assert_context_equal(ad, dpt)
