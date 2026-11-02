import gzip
import pickle

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig
from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.frozen_steering_history import (
    SteeringSource, load_episode, load_final_teacher, save_corpus, split_streams,
)
from fluffy_bassoon.meta_rl.vla_history import file_sha256, load_dataset, trial_tokens


def reference(path):
    return dict(path=str(path), sha256=file_sha256(path))


def source(arm="velocity"):
    interface = dict(kind=arm, steering_dim=32 if arm == "velocity" else 1152, fraction=.05)
    if arm == "kv_all_layers":
        interface["kv_layers"] = 18
    return SteeringSource(arm, "source", "runtime", "motor", "norm", ("A", "B"),
                          InterfaceSpec(**interface).record(), episodes_per_task_per_update=3, feature_dim=4)


def episode_file(root, spec, *, task="A", update=0, slot=0, success=True):
    directory = root / f"{task}-{update}-{slot}"
    directory.mkdir(parents=True)
    case = dict(task=task, slot=update * 3 + slot, native_seed=1000 + update * 10 + slot)
    chunks, logical = [], []
    for index in range(2):
        valid = index == 0 or not success
        row = dict(format="joint_steering_physical_chunk_v1", status="completed", phase="train",
            source_commit=spec.source_commit, runtime_commit=spec.runtime_commit,
            interface=spec.interface, policy_iteration=update, iteration=update + 1,
            case=case, chunk_index=index,
            features=np.full(4, 1. if task == "A" else 2., np.float32),
            control=np.full(spec.steering_dim, .2 + update / 10, np.float32),
            reward=float(success and index == 0), done=(index == 0 if success else index == 1),
            episode_start=index == 0, executed_controls=50, valid_ppo=valid,
            physical_done=index == 1, native_noise=np.zeros((50, 32), np.float32),
            first_success_step=25 if success else None)
        path = directory / f"chunk-{index}.pkl.gz"
        with gzip.open(path, "wb") as stream:
            pickle.dump(row, stream)
        chunks.append(reference(path))
        if valid:
            logical.append({key: row[key] for key in ("features", "control", "reward", "done",
                "episode_start", "executed_controls", "case", "chunk_index", "policy_iteration")})
    marker = directory / "physical.pt"
    torch.save({"actual": True}, marker)
    episode = dict(completed=True, case=case, valid_ppo_chunks=len(logical), physical_chunks=2,
        horizon=100, chunks=chunks, initial=reference(marker), final=reference(marker),
        success_once=success, first_success_step=25 if success else None)
    path = directory / "checkpoint.pt"
    torch.save(dict(format="joint_steering_episode_boundary_v1", source_commit=spec.source_commit,
        runtime_commit=spec.runtime_commit, episode=episode, records=logical), path)
    return reference(path)


def all_episodes(root, spec):
    return [load_episode(episode_file(root, spec, task=task, update=update, slot=slot,
                                     success=slot != 1), spec)
            for task in spec.tasks for update in (0, 1) for slot in range(3)]


def teacher_file(root, spec, *, interface=None):
    config = ResidualPPOConfig(spec.feature_dim, spec.steering_dim, width=8, layers=1)
    teacher = ResidualActorCritic(config)
    with torch.no_grad():
        teacher.actor[-1].bias.fill_(.75)
    from dataclasses import asdict
    path = root / "teacher.pt"
    constructor = {key: value for key, value in spec.interface.items()
                   if key in ("kind", "steering_dim", "fraction", "kv_layers")}
    torch.save(dict(format="joint_frozen_steering_boundary_v1", arm=spec.arm,
        spec=interface or constructor, policy_config=asdict(config), policy=teacher.state_dict(),
        iteration=2), path)
    return reference(path)


@pytest.mark.parametrize("arm", ["velocity", "kv_all_layers"])
def test_actual_full_physical_history_retains_suffix_and_first_reward(tmp_path, arm):
    spec = source(arm)
    record = load_episode(episode_file(tmp_path, spec), spec)
    trial = record["trial"]
    assert trial["executed_noise"].shape == (2, spec.steering_dim)
    np.testing.assert_array_equal(trial["rewards"], [1., 0.])
    np.testing.assert_array_equal(trial["dones"], [False, True])
    np.testing.assert_array_equal(trial["target_noise"], trial["executed_noise"])
    assert record["physical_episode"]["valid_ppo_chunks"] == 1


def test_hash_and_same_width_wrong_interface_are_rejected(tmp_path):
    spec = source()
    item = episode_file(tmp_path, spec)
    saved = torch.load(item["path"], weights_only=False)
    part = saved["episode"]["chunks"][0]
    with gzip.open(part["path"], "rb") as stream:
        physical = pickle.load(stream)
    physical["interface"] = dict(spec.interface, kind="conditioning_all")
    with gzip.open(part["path"], "wb") as stream:
        pickle.dump(physical, stream)
    with pytest.raises(ValueError, match="checksum"):
        load_episode(item, spec)
    saved["episode"]["chunks"][0] = reference(part["path"])
    torch.save(saved, item["path"])
    with pytest.raises(ValueError, match="interface"):
        load_episode(reference(item["path"]), spec)


def test_task_slot_split_preserves_versions_and_rejects_missing_or_duplicates(tmp_path):
    spec = source()
    episodes = all_episodes(tmp_path, spec)
    split = split_streams(episodes, spec, validation_slots=[2], expected_updates=(0, 1))
    assert [identity for identity, _ in split["train"]] == ["A:slot-000", "A:slot-001", "B:slot-000", "B:slot-001"]
    assert {identity for identity, _ in split["validation"]} == {"A:slot-002", "B:slot-002"}
    for identity, trial in split["train"] + split["validation"]:
        assert np.all(trial["observations"] == (1 if identity.startswith("A:") else 2))
        np.testing.assert_array_equal(trial["teacher_steps"], [0, 0, 1, 1])
    with pytest.raises(ValueError, match="Incomplete"):
        split_streams(episodes[:-1], spec, validation_slots=[2], expected_updates=(0, 1))
    with pytest.raises(ValueError, match="Duplicate"):
        split_streams(episodes + episodes[:1], spec, validation_slots=[2], expected_updates=(0, 1))


@pytest.mark.parametrize("arm", ["velocity", "kv_all_layers"])
def test_dpt_changes_only_query_targets_and_preserves_actual_history(tmp_path, arm):
    spec = source(arm)
    split = split_streams(all_episodes(tmp_path / "records", spec), spec,
                          validation_slots=[2], expected_updates=(0, 1))
    teacher = teacher_file(tmp_path, spec)
    save_corpus(tmp_path / "corpus", split, spec, teacher_reference=teacher,
                provenance=dict(producer_completion="qualified fixture"))
    for partition in ("train", "validation"):
        ad, am = load_dataset(tmp_path / "corpus" / f"ad-{partition}.npz")
        dpt, dm = load_dataset(tmp_path / "corpus" / f"dpt-{partition}.npz")
        assert am["metadata"]["trial_ids"] == dm["metadata"]["trial_ids"]
        assert am["noise_dim"] == dm["noise_dim"] == spec.steering_dim
        for actual, labeled in zip(ad, dpt, strict=True):
            for key in actual:
                if key != "target_noise":
                    np.testing.assert_array_equal(actual[key], labeled[key])
            np.testing.assert_allclose(labeled["target_noise"], .75)
            np.testing.assert_array_equal(trial_tokens(actual), trial_tokens(labeled))


def test_value_only_teacher_is_not_velocity_or_full_kv(tmp_path):
    spec = source()
    teacher = teacher_file(tmp_path, spec, interface=dict(kind="conditioning_all", steering_dim=32))
    with pytest.raises(ValueError, match="another steering interface"):
        load_final_teacher(teacher, spec)


def test_legacy_joint_noise_action_cannot_enter_steering_only_data(tmp_path):
    spec = source()
    teacher = teacher_file(tmp_path, spec)
    payload = torch.load(teacher["path"], weights_only=False)
    payload["policy_config"]["residual_dim"] = 1632
    torch.save(payload, teacher["path"])
    with pytest.raises(ValueError, match="steering-only"):
        load_final_teacher(reference(teacher["path"]), spec)


def test_source_mapping_and_final_teacher_version_cannot_be_reinterpreted(tmp_path):
    from dataclasses import replace
    spec = source("kv_all_layers")
    changed = dict(spec.interface, kv_map=dict(spec.interface["kv_map"], targets=["value"]))
    with pytest.raises(ValueError, match="mapping metadata"):
        replace(spec, interface=changed)
    streams = split_streams(all_episodes(tmp_path / "records", spec), spec,
                            validation_slots=[2], expected_updates=(0, 1))
    teacher = teacher_file(tmp_path, spec)
    saved = torch.load(teacher["path"], weights_only=False)
    saved["iteration"] = 1  # Behavior version1 is not the final post-update2 teacher.
    torch.save(saved, teacher["path"])
    with pytest.raises(ValueError, match="terminal source version"):
        save_corpus(tmp_path / "corpus", streams, spec, teacher_reference=reference(teacher["path"]),
                    provenance=dict(fixture=True))
