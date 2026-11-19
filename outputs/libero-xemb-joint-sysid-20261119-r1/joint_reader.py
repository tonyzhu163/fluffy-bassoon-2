"""Frozen pulse estimator for the first cross-embodiment reader pilot."""
import numpy as np

from fluffy_bassoon.xemb.readers import analytic_proprio_sysid, fit_nominal_gain


def wrap(angle):
    return float((angle + 180.0) % 360.0 - 180.0)


def alignment_deg(commands, motion):
    xy = np.asarray(commands, dtype=np.float64)[..., :2]
    dx = np.asarray(motion, dtype=np.float64)[..., :2]
    dot = float(np.sum(xy * dx))
    cross = float(np.sum(xy[..., 0] * dx[..., 1] - xy[..., 1] * dx[..., 0]))
    if np.hypot(dot, cross) <= 1e-10:
        raise ValueError("Insufficient planar response for yaw")
    return float(np.degrees(np.arctan2(cross, dot)))


def fit(nominal):
    q = np.concatenate([np.diff(t["q"].astype(np.float64), axis=0) for t in nominal])
    xyz = np.concatenate([t["delta_eef"] for t in nominal]).astype(np.float64)
    commands = np.concatenate([t["issued_mean"] for t in nominal])
    if q.shape != (12, 7) or xyz.shape != (12, 3):
        raise ValueError("Expected twelve nominal five-control pulses")
    singular = np.linalg.svd(q, compute_uv=False)
    ridge = 0.001 * singular[0] ** 2
    jacobian = np.linalg.solve(q.T @ q + ridge * np.eye(7), q.T @ xyz)
    residual = q @ jacobian - xyz
    gain = fit_nominal_gain(commands, xyz, np.ones(len(xyz), dtype=bool))
    return dict(jacobian_q_to_xyz=jacobian.tolist(), ridge_lambda=float(ridge),
                singular_values_q=singular.tolist(),
                rms_residual_mm_xyz=(1e3 * np.sqrt(np.mean(residual ** 2, axis=0))).tolist(),
                nominal_alignment_deg=alignment_deg(commands, xyz), nominal_gain=gain)


def joint_yaw(commands, q, calibration):
    delta_q = np.diff(np.asarray(q, dtype=np.float64), axis=0)
    xyz = delta_q @ np.asarray(calibration["jacobian_q_to_xyz"])
    raw = alignment_deg(commands, xyz)
    return wrap(raw - calibration["nominal_alignment_deg"]), raw, xyz.tolist()


def predict(trace, calibration):
    commands = trace["issued_mean"]
    joint, raw, xyz = joint_yaw(commands, trace["q"], calibration)
    wrist, scale, grip = analytic_proprio_sysid(
        commands, trace["delta_eef"], trace["delta_jaw"], np.ones(3, dtype=bool),
        nominal_gain_m_per_cmd=calibration["nominal_gain"],
    )
    return dict(yaw_deg=joint, xy_scale=scale, gripper_sign=grip,
                wrist_yaw_deg=wrist, wrist_referenced_yaw_deg=wrap(wrist - calibration["nominal_alignment_deg"]),
                joint_raw_yaw_deg=raw, predicted_world_delta_m=xyz)
