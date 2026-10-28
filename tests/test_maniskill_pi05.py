"""CPU checks of the ManiSkill pi0.5 rotation hook and scene planning (no simulator)."""
import numpy as np
import pytest
import torch

from fluffy_bassoon.rl import maniskill_pi05 as ms


INFOS = (25, 0, 21, 0, 1, 0, 6, 4)


def chunk(envs=3):
    torch.manual_seed(5)
    return torch.randn(envs, ms.CHUNK, 7)


def test_zero_rotation_is_identity():
    actions = chunk()
    torch.testing.assert_close(ms.rotate_translation(actions, [0., 0., 0.]), actions, rtol=0, atol=0)


def test_rotation_changes_only_xy_and_keeps_its_norm():
    actions = chunk()
    rotated = ms.rotate_translation(actions, [30., -45., 90.])
    torch.testing.assert_close(rotated[..., 2:], actions[..., 2:], rtol=0, atol=0)
    torch.testing.assert_close(rotated[..., :2].norm(dim=-1), actions[..., :2].norm(dim=-1))
    torch.testing.assert_close(rotated[2, :, 0], -actions[2, :, 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(rotated[2, :, 1], actions[2, :, 0], atol=1e-6, rtol=0)


def test_correction_and_hidden_rotation_compose_to_the_net_angle():
    actions, theta, phi = chunk(), [25., -60., 170.], [-10., 35., 20.]
    twice = ms.rotate_translation(ms.rotate_translation(actions, phi), theta)
    torch.testing.assert_close(twice, ms.rotate_translation(actions, np.add(theta, phi)), atol=1e-5, rtol=0)


def test_rotation_rejects_mismatched_angles():
    with pytest.raises(ValueError):
        ms.rotate_translation(chunk(), [0., 1.])
    with pytest.raises(ValueError):
        ms.rotate_translation(chunk(), [0., float("nan"), 1.])


def test_episode_id_matches_rlinf_mixed_radix_order():
    lo, lp, l1, l2 = 21, 1, 6, 4
    assert ms.episode_id(INFOS, 3, 7, 5, 2) == ((3*lp)*lo + 7)*l1*l2 + 5*l2 + 2
    ids = {ms.episode_id(INFOS, o, v, p, q) for o in range(25) for v in range(21) for p in range(6) for q in range(4)}
    assert ids == set(range(25*21*6*4))
    with pytest.raises(ValueError):
        ms.episode_id(INFOS, 25, 0, 0, 0)
    with pytest.raises(ValueError):
        ms.episode_id((24, 0, 21, 0, 1, 0, 6, 4), 0, 0, 0, 0)


def test_sampled_scenes_stay_in_training_overlays_and_are_seeded():
    first = ms.sample_scenes(INFOS, [0, 4, 15], 30, np.random.default_rng(9))
    again = ms.sample_scenes(INFOS, [0, 4, 15], 30, np.random.default_rng(9))
    assert first == again and len(first) == 90
    assert all(s["overlay"] < ms.TRAIN_OVERLAYS for s in first)
    assert [s["object"] for s in first] == [0]*30 + [4]*30 + [15]*30


def info(success, grasped, stable, distance):
    return dict(success=torch.tensor(success), is_src_obj_grasped=torch.tensor(grasped),
                consecutive_grasp=torch.tensor(stable), carrot_plate_dist=torch.tensor(distance, dtype=torch.float32))


def test_episode_outcomes_record_success_grasp_and_distance():
    outcomes = ms.EpisodeOutcomes(3)
    for step in range(ms.STEPS):
        # env 0: grasps at 10, holds from 14, succeeds at 30 and stays; env 1: grasps once at 20,
        # drops, never succeeds, ends far; env 2: nothing happens.
        distances = [.30 - step*.003 if step < 30 else .21, .25 - .001*min(step, 40) + .001*max(step-40, 0), .40]
        outcomes.update(info([step >= 30, False, False], [step >= 10, step == 20, False],
                             [step >= 14, False, False], distances))
    first, second, third = outcomes.rows()
    assert (first['success_once'], first['success_at_end'], first['first_success_step']) == (True, True, 30)
    assert (first['grasped_once'], first['stable_grasp_once'], first['first_grasp_step']) == (True, True, 10)
    assert first['object_plate_distance_end_m'] == pytest.approx(.21)
    assert first['object_plate_distance_min_m'] == pytest.approx(.21)
    assert (second['success_once'], second['first_success_step'], second['grasped_once']) == (False, -1, True)
    assert (second['stable_grasp_once'], second['first_grasp_step']) == (False, 20)
    assert second['object_plate_distance_min_m'] == pytest.approx(.21)
    assert second['object_plate_distance_end_m'] == pytest.approx(.249)
    assert (third['grasped_once'], third['first_grasp_step'], third['object_plate_distance_end_m']) == (False, -1, pytest.approx(.4))
    assert set(first) == set(ms.OUTCOME_FIELDS)


def test_success_at_end_uses_only_the_last_step():
    outcomes = ms.EpisodeOutcomes(1)
    for step in range(ms.STEPS):
        outcomes.update(info([step == 50], [True], [True], [.1]))
    (row,) = outcomes.rows()
    assert (row['success_once'], row['success_at_end'], row['first_success_step']) == (True, False, 50)


def test_episode_outcomes_reject_short_episodes_and_bad_infos():
    outcomes = ms.EpisodeOutcomes(2)
    outcomes.update(info([False, False], [False, False], [False, False], [.2, .3]))
    with pytest.raises(RuntimeError):
        outcomes.rows()
    with pytest.raises(ValueError):
        outcomes.update(info([False], [False], [False], [.2]))
    with pytest.raises(ValueError):
        outcomes.update(info([False, False], [False, False], [False, False], [.2, float('nan')]))
    with pytest.raises(KeyError):
        outcomes.update(dict(success=torch.tensor([False, False])))


def test_frozen_fingerprint_detects_parameter_and_scalar_buffer_changes():
    model = torch.nn.Linear(3, 2)
    model.register_buffer('counter', torch.tensor(0))
    initial = ms.model_fingerprint(model)
    assert ms.model_fingerprint(model) == initial
    model.counter.add_(1)
    assert ms.model_fingerprint(model) != initial
    model.counter.zero_()
    with torch.no_grad():
        model.weight[0, 0].add_(1)
    assert ms.model_fingerprint(model) != initial


def test_baige_boundary_rejects_dev_shell_and_accepts_owned_scheduler(monkeypatch):
    monkeypatch.setenv('FLUFFY_GPU_EXECUTION', 'baige')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES', 'GPU-00000001-0000-0000-0000-000000000001')
    monkeypatch.setenv('KUBERNETES_SERVICE_HOST', 'kube')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_STEP_ID', raising=False)
    monkeypatch.setattr('socket.gethostname', lambda: 'aibox-dev')
    with pytest.raises(ValueError):
        ms.execution_mode()
    monkeypatch.setattr('socket.gethostname', lambda: 'job-abc123-master-0')
    assert ms.execution_mode() == 'baige'
    assert ms.job_id() == 'job-abc123'
