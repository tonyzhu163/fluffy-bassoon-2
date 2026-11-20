"""Unit tests for Phase 3 held-out 6-DoF UR5e cross-morphology SysID."""

from pathlib import Path
import sys
import numpy as np
import pytest

from fluffy_bassoon.xemb.embodiment import (
    EmbodimentSpec,
    apply_embodiment_action,
    circular_yaw_error_deg,
    rot2,
    to_local_proprio,
)
from fluffy_bassoon.xemb.joint_audit import (
    evaluate_cross_morphology_joint_jacobian,
    fit_joint_jacobian,
    predict_yaw_from_joint_jacobian,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs/libero-xemb-ur5e-sysid-20261120-r1"))
import ur5e_motor  # noqa: E402


def test_joint_jacobian_same_arm_recovers_yaw_and_rejects_6dof_ur5e():
    rng = np.random.default_rng(7)
    j_true_7 = np.array(
        [
            [0.05, 0.01, 0.0],
            [-0.01, 0.04, 0.0],
            [0.0, 0.0, 0.05],
            [0.02, -0.01, 0.01],
            [0.01, 0.02, -0.01],
            [-0.01, 0.01, 0.02],
            [0.005, -0.005, 0.01],
        ],
        dtype=np.float64,
    )
    j_pinv_7 = np.linalg.pinv(j_true_7)

    # Nominal 7-DoF training data.
    n_train, k = 8, 3
    src_iss = rng.uniform(-0.6, 0.6, size=(n_train, k, 7))
    src_deef = src_iss[..., :3] * 0.05
    src_dq = src_deef @ j_pinv_7
    src_val = np.ones((n_train, k), dtype=bool)

    jac = fit_joint_jacobian(src_dq, src_deef, src_val, ridge=1e-10)
    assert jac.shape == (7, 3)

    # Same-arm 7-DoF test under rotated commands.
    yaws = [0.0, 90.0, 180.0, 45.0]
    tgt_iss = rng.uniform(-0.6, 0.6, size=(len(yaws), k, 7))
    tgt_deef_world = np.zeros((len(yaws), k, 3), dtype=np.float64)
    tgt_deef_local = tgt_iss[..., :3] * 0.05
    for i, yaw in enumerate(yaws):
        r = rot2(yaw).astype(np.float64)
        tgt_deef_world[i, :, :2] = tgt_deef_local[i, :, :2] @ r.T
        tgt_deef_world[i, :, 2] = tgt_deef_local[i, :, 2]
    tgt_dq_7 = tgt_deef_world @ j_pinv_7
    tgt_val = np.ones((len(yaws), k), dtype=bool)

    preds = predict_yaw_from_joint_jacobian(tgt_iss, tgt_dq_7, tgt_val, jac)
    for pred, true_yaw in zip(preds, yaws, strict=True):
        assert circular_yaw_error_deg(float(pred), true_yaw) < 1e-2

    # 6-DoF target arm (UR5e) must fail strict transfer with joint_dim_mismatch:7!=6,
    # and self-calibrated local Jacobian must stay yaw-blind (~0 deg) even at 180 deg.
    j_true_6 = rng.normal(scale=0.05, size=(6, 3))
    j_pinv_6 = np.linalg.pinv(j_true_6)
    tgt_dq_6 = tgt_deef_world @ j_pinv_6
    with pytest.raises(ValueError, match="Joint dimension mismatch"):
        predict_yaw_from_joint_jacobian(tgt_iss, tgt_dq_6, tgt_val, jac, pad_or_truncate=False)

    audit = evaluate_cross_morphology_joint_jacobian(
        src_dq, src_deef, src_val, tgt_iss, tgt_dq_6, tgt_deef_local, tgt_val
    )
    assert audit["source_dof"] == 7
    assert audit["target_dof"] == 6
    assert audit["applicable"] is False
    assert audit["reason"] == "joint_dim_mismatch:7!=6"
    assert audit["strict_yaw_deg"] is None
    assert len(audit["zero_padded_yaw_deg"]) == len(yaws)
    # Self-calibrated local Jacobian predicts ~0 deg for all yaws (including 180 deg).
    for yh in audit["self_calibrated_local_yaw_deg"]:
        assert circular_yaw_error_deg(float(yh), 0.0) < 1e-2


class _FakeUR5eSim:
    """Minimal deterministic 6-DoF UR5e simulator double for testing ur5e_motor.run_episode."""

    def __init__(self, horizon: int = 30):
        self.horizon = horizon
        self.spec = None
        self.anchor = np.array([0.0, 0.0, 0.9], dtype=np.float32)
        self.pos = None
        self.q = None
        self.jaw = None
        self.step_idx = 0
        self.phys_actions = []

    def _raw_obs(self):
        return dict(
            agentview_image=np.full((256, 256, 3), 120, dtype=np.uint8),
            robot0_eye_in_hand_image=np.full((256, 256, 3), 80, dtype=np.uint8),
            robot0_eef_pos=self.pos.copy(),
            robot0_eef_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            robot0_gripper_qpos=np.array([self.jaw / 2.0, -self.jaw / 2.0], dtype=np.float32),
            robot0_joint_pos=self.q.copy(),
            robot0_joint_vel=np.zeros((6,), dtype=np.float32),
        )

    def call(self, op, value=None):
        if op == "create":
            from fluffy_bassoon.xemb.embodiment import get_embodiment_spec

            self.spec = get_embodiment_spec(value["embodiment"])
            self.pos = self.anchor.copy()
            self.q = np.linspace(-0.5, 0.5, 6, dtype=np.float32)
            self.jaw = 0.08
            self.step_idx = 0
            self.phys_actions = []
            raw = self._raw_obs()
            local_raw = raw if self.spec.yaw_deg == 0 else to_local_proprio(raw, self.anchor, self.spec)
            create_spec = dict(max_steps=self.horizon, arm="UR5e")
            initial = dict(
                observation=local_raw,
                agentview_64=np.full((64, 64, 3), 120, dtype=np.uint8),
                anchor_eef_pos=self.anchor.copy(),
                language_instruction="put the bowl on the stove",
                model_identity=dict(arm="UR5e", model_class="MountedUR5e", joint_count=6),
                environment_reproducibility=dict(
                    scene_fingerprint="scene-sha",
                    initial_state_fingerprint="phys-sha",
                    perturbation_plan_fingerprint="plan-sha",
                ),
                action_schema=dict(low=[-1.0] * 7, high=[1.0] * 7),
                worker_step_index=0,
                max_steps=self.horizon,
                renderer="FAKE",
                renderer_vendor="FAKE",
            )
            return create_spec, initial

        if op == "chunk":
            low, high = np.full((7,), -1.0, dtype=np.float32), np.full((7,), 1.0, dtype=np.float32)
            positions, grippers, joints, rewards = [], [], [], []
            for issued in np.asarray(value, dtype=np.float32):
                phys = apply_embodiment_action(issued, self.spec, low, high)
                self.phys_actions.append(phys.copy())
                self.pos = self.pos + phys[:3] * np.float32(0.01)
                self.q = self.q + np.float32(0.01) * np.r_[phys[:3], phys[:3]]
                self.jaw = float(np.clip(self.jaw - 0.005 * float(phys[6]), 0.0, 0.08))
                self.step_idx += 1
                raw = self._raw_obs()
                local_raw = raw if self.spec.yaw_deg == 0 else to_local_proprio(raw, self.anchor, self.spec)
                positions.append(local_raw["robot0_eef_pos"].copy())
                grippers.append(raw["robot0_gripper_qpos"].copy())
                joints.append(raw["robot0_joint_pos"].copy())
                rewards.append(0.0)
                if self.step_idx >= self.horizon:
                    break
            return dict(
                observation=local_raw,
                agentview_64=np.full((64, 64, 3), 120, dtype=np.uint8),
                positions=positions,
                gripper_positions=grippers,
                joint_positions=joints,
                rewards=rewards,
                steps=len(rewards),
                success=False,
                any_success=False,
                terminated=False,
                truncated=self.step_idx >= self.horizon,
                language_instruction="put the bowl on the stove",
            )
        raise ValueError(op)


class _FakePolicy:
    def infer(self, item, noise=None):
        chunk = np.tile(np.array([0.4, 0.2, 0.0, 0.0, 0.0, 0.0, -0.5], dtype=np.float32), (10, 1))
        return dict(actions=chunk)


def test_run_episode_6dof_retarget_and_trace(monkeypatch):
    def _fake_to_openpi(raw, prompt):
        return {
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros((8,), dtype=np.float32),
            "prompt": prompt,
        }

    monkeypatch.setattr(ur5e_motor, "to_openpi_observation", _fake_to_openpi)
    sim = _FakeUR5eSim(horizon=30)
    policy = _FakePolicy()
    case = dict(arm="UR5e", suite="libero_goal", task_id=1, variant="BASE", init_state_id=13, seed=42)

    nom_spec = EmbodimentSpec("fam_nom", 0.0, 1.0, 1, "familiar")
    comp_spec = EmbodimentSpec("comp_180_lo_inv", 180.0, 0.65, -1, "compositional_ood")

    rec_nom, trace_nom = ur5e_motor.run_episode(
        sim, policy, case, nom_spec, record_chunks=True, max_record_chunks=6
    )
    nom_phys = np.stack(sim.phys_actions)

    rec_ora, trace_ora = ur5e_motor.run_episode(
        sim,
        policy,
        case,
        comp_spec,
        yaw_hat=180.0,
        scale_hat=0.65,
        grip_hat=-1,
        record_chunks=True,
        max_record_chunks=6,
    )
    ora_phys = np.stack(sim.phys_actions)

    assert rec_nom["reset"] == rec_ora["reset"]
    assert rec_nom["reset"]["joint_count"] == 6
    assert trace_ora["delta_joint"].shape == (6, 6)
    assert int(trace_ora["joint_count"]) == 6
    assert np.allclose(nom_phys, ora_phys, atol=1e-5)
    assert rec_nom["prefix_xy_disp_15_mm"] > 50.0
    assert rec_nom["prefix_xy_disp_30_mm"] > rec_nom["prefix_xy_disp_15_mm"]


def test_luminance_flow_reader_is_rgb_permutation_invariant():
    import jax
    from fluffy_bassoon.xemb.readers import ReaderConfig, init_reader, predict_embodiment

    cfg = ReaderConfig(model_type="factored_equiv", max_chunks=8, cnn_width=32, mlp_width=64, visual_mode="luminance_flow")
    params = init_reader(jax.random.key(0), cfg)
    rng = np.random.default_rng(123)
    img_b = rng.integers(0, 256, size=(2, 3, 64, 64, 3), dtype=np.uint8)
    img_a = rng.integers(0, 256, size=(2, 3, 64, 64, 3), dtype=np.uint8)
    batch = dict(
        img_before=img_b,
        img_after=img_a,
        issued_mean=rng.uniform(-0.6, 0.6, size=(2, 3, 7)).astype(np.float32),
        delta_eef=rng.uniform(-0.04, 0.04, size=(2, 3, 3)).astype(np.float32),
        delta_jaw=rng.uniform(-0.02, 0.02, size=(2, 3, 2)).astype(np.float32),
        valid=np.ones((2, 3), dtype=bool),
    )
    # Permute RGB channels (simulating completely different arm color: red -> blue -> green).
    perm = [2, 0, 1]
    batch_perm = dict(batch, img_before=img_b[..., perm], img_after=img_a[..., perm])
    y1, s1, g1 = predict_embodiment(params, cfg, batch)
    y2, s2, g2 = predict_embodiment(params, cfg, batch_perm)
    assert np.allclose(y1, y2, atol=1e-4)
    assert np.allclose(s1, s2, atol=1e-5)
    assert np.array_equal(g1, g2)
