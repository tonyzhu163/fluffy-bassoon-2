"""Source identity, train/validation isolation and data-arm compatibility."""

import json

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_rlinf_libero import FEATURE_KIND, MODEL_SHA256
from fluffy_bassoon.meta_rl.vla_rlinf_train_arm import _index, disjoint_identities, paired_contexts


def test_train_validation_cannot_share_trial_or_nominal_case_ids():
    train = [dict(trial_id="train", case_id="original:47:state0")]
    disjoint_identities(train, [dict(trial_id="validation", case_id="original:47:state36")])
    with pytest.raises(ValueError, match="trial_id"):
        disjoint_identities(train, [dict(trial_id="train", case_id="different")])
    with pytest.raises(ValueError, match="case_id"):
        disjoint_identities(train, [dict(trial_id="validation", case_id="original:47:state0")])


def test_plus_development_index_is_rejected_even_with_same_backbone(tmp_path):
    path = tmp_path / "index.json"
    value = dict(format="rlinf130_plus_source_corpus_index_v1", completed=True,
        model_sha256=MODEL_SHA256, feature_contract=dict(kind=FEATURE_KIND),
        global_task_split="train", trials=[{}], benchmark_kind="plus", role="development")
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="development/test"):
        _index(path, "plus")


def test_paired_contexts_require_bitwise_real_feedback_equality():
    trial = {key: np.zeros((2, 1), np.float32) for key in
             ("observations", "executed_noise", "rewards", "dones", "episode_starts", "teacher_steps")}
    paired_contexts([trial], [trial])
    changed = dict(trial, rewards=np.ones((2, 1), np.float32))
    with pytest.raises(ValueError, match="contexts differ"):
        paired_contexts([trial], [changed])
