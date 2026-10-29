"""CND actual-action chronology and frozen-source contracts; allocated Linux only."""

from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl.vla_history import load_dataset, trial_tokens, validate_trial
from fluffy_bassoon.meta_rl.vla_noise_distillation import (NoiseSchedule,
    collect_noise_distillation, mix_noise)


def test_mixture_and_schedule_reach_random_and_teacher_endpoints():
    schedule = NoiseSchedule(5)
    assert [schedule.epsilon(i) for i in range(6)] == [1., .75, .5, .25, 0., 0.]
    expert, random = np.array([.8, -.8]), np.array([-.4, .4])
    np.testing.assert_array_equal(mix_noise(expert, random, 1.), random.astype(np.float32))
    np.testing.assert_array_equal(mix_noise(expert, random, 0.), expert.astype(np.float32))
    np.testing.assert_allclose(mix_noise(expert, random, .5), [.2, -.2])
    assert NoiseSchedule(5, 2).epsilon(1) == .75 ** 2
    with pytest.raises(ValueError):
        mix_noise(expert * 2, random, .5)


def fake_episode(sim, motor, choose, *, initial_state, scene_seed, chunk, bounded):
    assert bounded
    rows = []
    for i in range(2):
        action = choose(dict(index=i, scene=initial_state))
        rows.append(dict(executed_noise=action, reward=float(i), done=i == 1,
                         primitive_steps=1, public_features=np.array([initial_state, i, 1.], np.float32)))
    return rows, dict(primitive_steps=2, success=True,
                      initial_observation_sha256=f"{initial_state}:{scene_seed}")


def arguments(tmp_path):
    import hashlib
    checkpoint = tmp_path / "source.pkl"
    checkpoint.write_bytes(b"immutable qualified teacher")
    return dict(teacher_action=lambda obs: np.array([.8, -.8], np.float32),
        teacher_fingerprint=lambda: "fixed-state", teacher_checkpoint=checkpoint,
        teacher_steps=31, teacher_primitive_steps=100,
        teacher_evaluation=dict(successes=8, completed_episodes=10,
            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),
        motor_sha256="frozen-motor", feature_contract={"fixed": "native VLA features"},
        noise_contract="same bounded motor coordinates", scenes=[
            dict(trial_id="train:0", split="train", initial_state=0, scene_seed=7),
            dict(trial_id="validation:1", split="validation", initial_state=1, scene_seed=8)],
        output_dir=tmp_path / "collection", min_episodes=2, max_episodes=3,
        episode_horizon=2, chunk=1, anneal_decisions=(4, 4),
        anneal_powers=(1., 2.), seed=4, max_primitive_steps=12,
        heldout_initial_states=[2], noise_dim=2,
        sim=SimpleNamespace(metadata=dict(settle_steps=10)), episode_runner=fake_episode)


def test_collection_keeps_actual_mixtures_constant_source_and_no_schedule_inputs(tmp_path):
    kwargs = arguments(tmp_path)
    paths = collect_noise_distillation(**kwargs)
    train, manifest = load_dataset(paths["train"])
    data = train[0]
    assert manifest["objective"] == "cnd" and manifest["metadata"]["total_source_primitive_steps"] == 108
    assert manifest["metadata"]["collection_settling_steps"] == 40
    assert len(data["observations"]) == 4
    assert set(data["teacher_steps"]) == {31}
    np.testing.assert_array_equal(data["target_noise"], data["executed_noise"])
    np.testing.assert_array_equal(data["executed_noise"][-1], np.array([.8, -.8], np.float32))
    assert trial_tokens(data).shape == (4, 3 + 2 + 3)
    validate_trial(data, objective="cnd")
    with pytest.raises(ValueError, match="learning trajectory"):
        validate_trial(data, objective="ad")
    changed = dict(data, target_noise=data["target_noise"] + 1.)
    with pytest.raises(ValueError, match="actually executed"):
        validate_trial(changed, objective="cnd")


def test_fixed_teacher_competence_and_scene_splits_are_required(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["teacher_evaluation"]["successes"] = 4
    with pytest.raises(ValueError, match="measured robot competence"):
        collect_noise_distillation(**kwargs)
    kwargs["teacher_evaluation"]["successes"] = 8
    kwargs["heldout_initial_states"] = [0]
    with pytest.raises(ValueError, match="disjoint"):
        collect_noise_distillation(**kwargs)
    assert not kwargs["output_dir"].exists()


def test_short_episode_budget_does_not_export_fabricated_complete_schedule(tmp_path):
    kwargs = arguments(tmp_path)
    kwargs["anneal_decisions"] = (8, 8)
    with pytest.raises(RuntimeError, match="before epsilon reached zero"):
        collect_noise_distillation(**kwargs)
    assert not list(kwargs["output_dir"].glob("cnd-*.npz"))
    assert (kwargs["output_dir"] / "teacher-integrity.json").exists()


def test_teacher_mutation_is_rejected_even_after_real_collection(tmp_path):
    kwargs = arguments(tmp_path)
    calls = 0
    def fingerprint():
        nonlocal calls
        calls += 1
        return "before" if calls == 1 else "changed"
    kwargs["teacher_fingerprint"] = fingerprint
    with pytest.raises(RuntimeError, match="changed its fixed source"):
        collect_noise_distillation(**kwargs)
    assert not list(kwargs["output_dir"].glob("cnd-*.npz"))


def test_paired_dpt_export_only_changes_query_targets_without_extra_rollouts(tmp_path):
    import json
    kwargs = arguments(tmp_path)
    kwargs.update(export_paired_dpt=True, benchmark_contract=dict(suite="libero_90", task_id=57,
        semantic_task="cream_cheese_to_tray", bddl_sha256="b" * 64, teacher_contract_sha256="c" * 64))
    calls = []
    def counted_episode(*args, **kw):
        calls.append(kw["initial_state"])
        return fake_episode(*args, **kw)
    kwargs["episode_runner"] = counted_episode
    paths = collect_noise_distillation(**kwargs)
    assert set(paths) == {"train", "validation"}
    assert calls == [0, 0, 1, 1]
    paired = json.loads((kwargs["output_dir"] / "paired-collections.json").read_text())
    for split in ("train", "validation"):
        cnd, cm = load_dataset(paths[split])
        dpt, dm = load_dataset(paired["datasets"]["dpt"][split]["path"])
        assert cm["objective"] == "cnd" and dm["objective"] == "dpt"
        assert cm["metadata"]["benchmark_contract"] == kwargs["benchmark_contract"]
        for field in cnd[0]:
            if field != "target_noise":
                assert cnd[0][field].tobytes() == dpt[0][field].tobytes()
        np.testing.assert_array_equal(dpt[0]["target_noise"], np.tile([.8, -.8], (4, 1)).astype(np.float32))
        assert not np.array_equal(cnd[0]["target_noise"], dpt[0]["target_noise"])
