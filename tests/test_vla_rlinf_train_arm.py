"""Source identity, train/validation isolation and data-arm compatibility."""

import json
import gzip
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_rlinf_libero import FEATURE_KIND, MODEL_SHA256
from fluffy_bassoon.meta_rl.vla_rlinf_train_arm import (_index, disjoint_identities,
    paired_contexts, training_replay)


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


def test_source_sac_rejects_validation_before_opening_any_episode():
    with pytest.raises(ValueError, match="Validation interactions"):
        training_replay(dict(training_trials=[dict(source_root="/tmp/unused",
            trial=dict(scene=dict(split="validation")))]))


def test_source_replay_checks_raw_interactions_against_student_contexts(tmp_path):
    from fluffy_bassoon.meta_rl.vla_history import file_sha256, save_dataset
    folder = tmp_path / "trial-000"
    folder.mkdir()
    arrays = dict(observations=np.zeros((2, 4104), np.float32), executed_noise=np.zeros((2, 32), np.float32),
        target_noise=np.zeros((2, 32), np.float32), rewards=np.array([-1., 0.], np.float32),
        dones=np.ones(2, bool), episode_starts=np.ones(2, bool), teacher_steps=np.array([0, 20], np.int64))
    dataset = tmp_path / "ad.npz"
    receipt = save_dataset(dataset, [arrays], objective="ad", metadata=dict(source_contract={"source": "declared"}))
    episodes = []
    for attempt in (1, 2):
        row = dict(public_features=np.zeros(4104, np.float32), executed_noise=np.zeros(32, np.float32),
            reward=float(arrays["rewards"][attempt - 1]), done=True, success=attempt == 2, primitive_steps=10,
            sac_observation=dict(pixels=np.zeros((64, 64, 3, 1), np.uint8), state=np.zeros((8, 1), np.float32)),
            sac_next_observation=dict(pixels=np.zeros((64, 64, 3, 1), np.uint8), state=np.zeros((8, 1), np.float32)))
        path = folder / f"episode-{attempt:03d}.pkl.gz"
        with gzip.open(path, "wb") as stream:
            pickle.dump(dict(rows=[row], summary=dict(initial_state=0, scene_seed=0)), stream)
        episodes.append(dict(attempt=attempt, primitive_steps=10, trace_sha256=file_sha256(path)))
    assembled = dict(training_trials=[dict(source_root=str(tmp_path),
        source_index="declared-index", source_index_sha256="declared-hash",
        datasets=dict(ad=dict(path=str(dataset), sha256=receipt["sha256"])),
        trial=dict(trial_id="train-0", scene=dict(split="train", initial_state=0, scene_seed=0),
            source_checkpoint="trial-000/source-recovery-004.pkl", episodes=episodes))],
        collection_cost=dict(training_primitive_steps=20))
    replay, provenance, steps, count = training_replay(assembled)
    assert len(replay) == 2 and steps == 20 and count == 2
    assert len(provenance) == 1 and replay[0]["actions"].shape == (1, 32)
    assert replay[0]["discount"] == np.float32(.999 ** 10)
