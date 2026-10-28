"""Resolve the upstream placeholder to the native WidowX controller."""
import pytest

from fluffy_bassoon.rl.maniskill_pi05 import env_config


@pytest.mark.parametrize('placeholder', ['None', 'null'])
def test_native_controller_overrides_upstream_placeholder(tmp_path, monkeypatch, placeholder):
    monkeypatch.setenv('FLUFFY_GPU_EXECUTION', 'slurm')
    folder = tmp_path/'examples/embodiment/config/env'
    folder.mkdir(parents=True)
    (folder/'maniskill_put_on_plate_in_scene_25_main.yaml').write_text(
        'init_params:\n  control_mode: '+placeholder+'\n')
    cfg = env_config(tmp_path, 10, 3000300000)
    assert cfg.init_params.control_mode == 'arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos'
    assert cfg.init_params.max_episode_steps == cfg.max_episode_steps == 80
    assert cfg.init_params.obj_set == 'all' and cfg.total_num_envs == 10
    assert cfg.seed == 3000300000 and cfg.auto_reset is False
