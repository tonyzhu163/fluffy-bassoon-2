"""Numerical qualification of the released RoboCasa365 policy input contract.

Run on allocated Linux against the exact patched RLinf/OpenPI deployment.
No simulator, policy constructor, checkpoint or GPU is used by these cases.
"""
import numpy as np
import pytest


@pytest.fixture
def native_pipeline(monkeypatch, tmp_path):
    from openpi import transforms
    from openpi.models.pi0_config import Pi0Config
    from openpi.shared.normalize import NormStats
    from openpi.training.config import DataConfigFactory, ModelTransformFactory
    from rlinf.models.embodiment.openpi.dataconfig.robocasa_dataconfig import (
        LeRobotRobocasaDataConfig,
    )

    # No tokenizer download is needed to qualify the preceding data transforms.
    monkeypatch.setattr(ModelTransformFactory, "__call__", lambda *a: transforms.Group())
    mean = np.linspace(-0.4, 0.6, 32, dtype=np.float32)
    std = np.linspace(0.5, 1.5, 32, dtype=np.float32)
    stats = {
        "state": NormStats(mean=mean, std=std, q01=mean - 5, q99=mean + 9),
        "actions": NormStats(mean=mean + 0.2, std=std + 0.3, q01=mean - 8, q99=mean + 2),
    }
    monkeypatch.setattr(DataConfigFactory, "_load_norm_stats", lambda *a: stats)
    factory = LeRobotRobocasaDataConfig(
        repo_id="unused-numerical-fixture",
        state_space="16d",
        image_space="3views",
        action_space="12d",
        use_quantile_norm=False,
        pad_before_normalization=True,
    )
    cfg = factory.create(tmp_path, Pi0Config(pi05=True, action_horizon=50))
    return transforms, factory, cfg, stats


def raw_case():
    return {
        "observation/state": np.arange(25, dtype=np.float32),
        "observation/image": np.full((8, 9, 3), 31, np.uint8),
        "observation/wrist_image": np.full((8, 9, 3), 79, np.uint8),
        "observation/extra_view_image": np.full((8, 9, 3), 173, np.uint8),
        "prompt": "scrub the cutting board",
    }


def apply_inputs(cfg, case):
    for transform in cfg.data_transforms.inputs:
        case = transform(case)
    return case


def test_state_matches_published_relative_eef_base_gripper_order(native_pipeline):
    _, _, cfg, _ = native_pipeline
    actual = apply_inputs(cfg, raw_case())["state"]
    # Independent source contract: relative eef7, base7, gripper2, then zeros.
    expected = np.array(
        [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 7, 8]
        + [0] * 16, dtype=np.float32,
    )
    np.testing.assert_array_equal(actual, expected)


def test_padding_is_normalized_before_discrete_state_tokenization(native_pipeline):
    transforms, _, cfg, stats = native_pipeline
    case = apply_inputs(cfg, raw_case())
    normalized = transforms.Normalize(stats, use_quantiles=cfg.use_quantile_norm)(case)
    expected = (case["state"] - stats["state"].mean) / (stats["state"].std + 1e-6)
    np.testing.assert_allclose(normalized["state"], expected, rtol=0, atol=1e-6)
    assert normalized["state"].shape == (32,)
    # This catches padding after normalization and quantile substitution.
    assert np.any(normalized["state"][16:] != 0)
    assert cfg.use_quantile_norm is False


def test_three_camera_slots_preserve_reference_order(native_pipeline):
    _, _, cfg, _ = native_pipeline
    case = apply_inputs(cfg, raw_case())
    for key, value in [("base_0_rgb", 31), ("left_wrist_0_rgb", 79), ("right_wrist_0_rgb", 173)]:
        np.testing.assert_array_equal(case["image"][key], np.full((8, 9, 3), value, np.uint8))
        assert case["image_mask"][key]


def test_action_mean_std_output_keeps_native_eef_gripper_base_order(native_pipeline):
    transforms, _, cfg, stats = native_pipeline
    predicted = np.linspace(-0.7, 0.9, 50 * 32, dtype=np.float32).reshape(50, 32)
    case = transforms.Unnormalize(stats, use_quantiles=cfg.use_quantile_norm)(
        {"actions": predicted.copy(), "state": np.zeros(32, np.float32)}
    )
    for transform in cfg.data_transforms.outputs:
        case = transform(case)
    expected = predicted[:, :12] * (stats["actions"].std[:12] + 1e-6) + stats["actions"].mean[:12]
    np.testing.assert_allclose(case["actions"], expected, rtol=0, atol=1e-6)
    assert case["actions"].shape == (50, 12)


def test_native_score_controls_match_pandaomron_key_converter():
    from robocasa.wrappers.gym_wrapper import PandaOmronKeyConverter
    from rlinf.envs.sim.robocasa365.robocasa365_env import map_released_control_scores

    actions = np.arange(24, dtype=np.float32).reshape(2, 12) / 30
    actions[:, 6] = [0.49, 0.5]
    actions[:, 11] = [0.5, 0.49]
    original = actions.copy()
    mapped = map_released_control_scores(actions, dict(
        env_action_dim=12, binarize_gripper_control=True,
        binarize_base_mode_control=True,
    ))
    for row, actual in zip(original, mapped):
        ref = PandaOmronKeyConverter.unmap_action({
            "action.end_effector_position": row[:3],
            "action.end_effector_rotation": row[3:6],
            "action.gripper_close": row[6],
            "action.base_motion": row[7:11],
            "action.control_mode": row[11],
        })
        expected = np.r_[ref["robot0_right"], ref["robot0_right_gripper"],
                         ref["robot0_base"], ref["robot0_torso"], ref["robot0_base_mode"]]
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actions, original)
