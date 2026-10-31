"""Independent same-scene source branches, real labels and cost accounting."""

from copy import deepcopy
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import load_dataset
from fluffy_bassoon.meta_rl.vla_retry_teacher import collect_retry_teachers


class FakeSource:
    instances = []

    def __init__(self, root, seed):
        self.updates, self.replay = 0, []
        self.agent = self
        self.instances.append(self)

    def restore(self, path, sampler, contract):
        self.updates = 10
        self.replay = [dict(prior=True)]
        return dict(episodes=2, primitive_steps=100)

    def action(self, observation):
        return np.full(2, self.updates / 100, np.float32)

    def eval_actions(self, observation):
        return self.action(observation)[None]

    def update(self, batch_size, sampler):
        self.updates += 1
        return {}

    def save(self, path, sampler, contract, **progress):
        with path.open("wb") as stream:
            pickle.dump(dict(updates=self.updates, replay=deepcopy(self.replay), progress=progress), stream)


def fake_episode(sim, motor, action, *, initial_state, scene_seed, chunk, bounded):
    observation = dict(pixels=np.full((2, 2, 3, 1), initial_state, np.uint8),
                       state=np.asarray([[scene_seed]], np.float32))
    rows = []
    for index in range(2):
        rows.append(dict(sac_observation=observation, sac_next_observation=observation,
            executed_noise=action(observation), reward=float(index - 1), done=index == 1,
            success=index == 1, primitive_steps=1, raw_rewards=[float(index)]))
    return rows, dict(primitive_steps=2, success=True,
        initial_observation_sha256=f"scene-{initial_state}-{scene_seed}")


def arguments(tmp_path):
    prior = tmp_path / "prior.pkl"
    prior.write_bytes(b"immutable shared prior")
    return dict(source_root=tmp_path, shared_prior_path=prior,
        prior_contract=dict(motor={"receipt_sha256": "a" * 64}, noise="bounded coordinate", source_commit="b" * 40),
        fixed_encoder=lambda obs: np.concatenate((obs["state"].reshape(-1),
                                                   np.asarray([obs["pixels"].mean()], np.float32))),
        scenes=[dict(trial_id="train:0", split="train", initial_state=0, scene_seed=1),
                dict(trial_id="validation:1", split="validation", initial_state=1, scene_seed=2)],
        attempts=2, batch_size=2, utd=1, chunk=1, output_dir=tmp_path / "retry",
        episode_horizon=4, max_primitive_steps=16, heldout_initial_states=[2, 3],
        learner_factory=FakeSource, episode_runner=fake_episode)


def test_independent_branches_and_factual_ad_vs_dpt_labels(tmp_path):
    FakeSource.instances.clear()
    paths, metadata = collect_retry_teachers(**arguments(tmp_path))
    assert len(FakeSource.instances) == 2
    assert [source.updates for source in FakeSource.instances] == [14, 14]
    assert [len(source.replay) for source in FakeSource.instances] == [5, 5]
    assert FakeSource.instances[0].replay is not FakeSource.instances[1].replay
    ad, manifest = load_dataset(paths["ad"]["train"])
    dpt, _ = load_dataset(paths["dpt"]["train"])
    assert np.array_equal(ad[0]["teacher_steps"], [10, 10, 12, 12])
    assert np.array_equal(ad[0]["episode_starts"], [True, False, True, False])
    assert np.array_equal(ad[0]["executed_noise"], ad[0]["target_noise"])
    assert np.array_equal(ad[0]["executed_noise"], dpt[0]["executed_noise"])
    assert np.array_equal(ad[0]["rewards"], [-1., 0., -1., 0.])
    assert np.all(dpt[0]["target_noise"] == np.float32(.14))
    assert not np.array_equal(dpt[0]["target_noise"], ad[0]["target_noise"])
    assert metadata["extra_source_primitive_steps"] == 8
    assert metadata["extra_source_updates"] == 8
    assert metadata["shared_prior_primitive_steps"] == 100
    assert metadata["total_source_primitive_steps"] == 108
    assert manifest["metadata"]["trial_ids"] == ["train:0"]
    assert metadata["scene_receipts"][0]["episodes"][0]["source_updates_before"] == 10


def test_budget_and_scene_split_rejected_before_collection(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["max_primitive_steps"] = 15
    with pytest.raises(ValueError, match="budget"):
        collect_retry_teachers(**kwargs)
    assert not kwargs["output_dir"].exists()
    kwargs["max_primitive_steps"] = 16
    kwargs["scenes"][1]["initial_state"] = 0
    with pytest.raises(ValueError, match="disjoint"):
        collect_retry_teachers(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_same_state_reset_must_reproduce_initial_observation(tmp_path):
    kwargs = arguments(tmp_path)
    calls = 0
    def changed_reset(*args, **kw):
        nonlocal calls
        rows, summary = fake_episode(*args, **kw)
        calls += 1
        summary["initial_observation_sha256"] += str(calls)
        return rows, summary
    kwargs["episode_runner"] = changed_reset
    with pytest.raises(ValueError, match="initial observation"):
        collect_retry_teachers(**kwargs)


def test_fixed_native_row_features_keep_their_explicit_contract(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["fixed_feature_contract"] = dict(kind="frozen native visual and instruction prefix", motor="fixed")
    kwargs["row_feature_encoder"] = lambda row: np.asarray([.25, .5, .75], np.float32)
    paths, metadata = collect_retry_teachers(**kwargs)
    data, manifest = load_dataset(paths["ad"]["train"])
    assert data[0]["observations"].shape == (4, 3)
    assert np.array_equal(data[0]["observations"][0], [.25, .5, .75])
    assert metadata["feature_contract"] == kwargs["fixed_feature_contract"]
    assert manifest["metadata"]["feature_contract"] == kwargs["fixed_feature_contract"]
