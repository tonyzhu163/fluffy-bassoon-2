"""A real changing source policy and same-context query labels stay distinct."""

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_rlinf_retry_corpus import (assemble_local_pilot, scenes,
    trial_arrays, validate_source_task)


def test_corpus_plan_has_six_disjoint_source_scenes_with_a_complete_episode_ceiling():
    plan = scenes(0)
    assert [row["initial_state"] for row in plan] == [0, 1, 2, 3, 36, 37]
    assert len(plan) * 4 * 400 == 9600
    assert {row["split"] for row in plan} == {"train", "validation"}


def test_paired_queries_do_not_relabel_executed_actions_or_feedback():
    rows = [dict(public_features=np.full(4104, i, np.float32),
        executed_noise=np.full(32, i / 10, np.float32), reward=float(-1 + i), done=True)
        for i in (0, 1)]
    paired = trial_arrays(rows, [True, True], [0, 20], np.ones((2, 32), np.float32))
    for field in ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps"):
        np.testing.assert_array_equal(paired["ad"][field], paired["dpt"][field])
    np.testing.assert_array_equal(paired["ad"]["target_noise"], paired["ad"]["executed_noise"])
    assert not np.array_equal(paired["dpt"]["target_noise"], paired["dpt"]["executed_noise"])


def test_source_corpus_rejects_nontraining_tasks_and_changed_bddl():
    task = dict(split="train", teacher_collection_allowed=True, source_state_indices=list(range(40)),
                suite="libero_90", task_id=47, bddl_sha256="expected", semantic_task="the_task")
    metadata = dict(suite="libero_90", task_id=47, bddl_sha256="expected", bddl_file="the_task.bddl",
                    horizon=400, fresh_environment_on_reset=True)
    validate_source_task(task, metadata)
    with pytest.raises(ValueError, match="training task"):
        validate_source_task(dict(task, split="test"), metadata)
    with pytest.raises(ValueError, match="exact frozen BDDL"):
        validate_source_task(task, dict(metadata, bddl_sha256="changed"))
    with pytest.raises(ValueError, match="exact frozen BDDL"):
        validate_source_task(task, dict(metadata, fresh_environment_on_reset=False))


def test_local_assembler_rejects_old_openpi_features_before_opening_datasets(tmp_path):
    with pytest.raises(ValueError, match="explicit shared RLinf-130"):
        assemble_local_pilot(dict(format="rlinf130_retry_corpus_index_v1",
            model_sha256="different", feature_contract=dict(kind="old-openpi")), tmp_path)
