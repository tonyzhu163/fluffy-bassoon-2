"""Competitive SysID & uncalibrated visual-servoing baselines for cross-embodiment evaluation.

Includes:
- Dense optical flow (OpenCV DIS / Farneback) + camera projection + Procrustes (`dense_flow_procrustes`).
- ReGround-style target motor-babbling affine/Procrustes calibration (`reground_babbling_sysid`),
  following Siddique, Li & Cao (2026, arXiv:2611.20244), supporting matched (15-step) and
  privileged (1,000-step) babbling budgets.
- Classical online Broyden uncalibrated visual-servoing Jacobian estimator (`broyden_uvs_yaw`).
"""

from __future__ import annotations

import math
import numpy as np

from .readers import analytic_proprio_sysid, fit_visual_projection


def dense_flow_image_motion(
    before: np.ndarray,
    after: np.ndarray,
    valid: np.ndarray,
    *,
    method: str = "dis",
) -> np.ndarray:
    """Estimate per-chunk 2D image displacement (pixels) via dense optical flow.

    Uses OpenCV DIS (default, fast & robust) or Farneback dense optical flow, then
    computes a flow-magnitude-weighted average vector so static background pixels
    contribute negligible weight while the moving robot arm dominates.
    """
    import cv2

    before_arr = np.asarray(before)
    after_arr = np.asarray(after)
    val = np.asarray(valid, dtype=bool)
    motion = np.zeros((*val.shape, 2), dtype=np.float64)

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM) if method == "dis" else None

    for idx in zip(*np.nonzero(val), strict=True):
        a = before_arr[idx]
        b = after_arr[idx]
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if b.dtype != np.uint8:
            b = np.clip(b * (255.0 if b.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) if b.ndim == 3 else b
        if method == "dis":
            flow = dis.calc(ga, gb, None)
        elif method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 3, 11, 3, 5, 1.2, 0)
        else:
            raise ValueError(f"Unknown dense flow method {method!r}; expected 'dis' or 'farneback'")
        # Motion-magnitude weighting focuses on the moving end-effector/arm over static background.
        mag = np.linalg.norm(flow, axis=-1)
        # Keep top active motion region (> 25% of peak or > 0.15 px) to reject background jitter.
        thresh = max(0.15, 0.25 * float(mag.max()))
        w = np.where(mag >= thresh, mag * mag, 0.0)
        wsum = float(w.sum())
        if wsum > 1e-9:
            u = float(np.sum(w * flow[..., 0]) / wsum)
            v = float(np.sum(w * flow[..., 1]) / wsum)
            motion[idx] = (u, v)
    return motion


def compute_dis_flow_maps(
    before: np.ndarray,
    after: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Compute dense 2-channel DIS optical flow maps [..., H, W, 2] for neural reader inputs."""
    import cv2

    before_arr = np.asarray(before)
    after_arr = np.asarray(after)
    val = np.asarray(valid, dtype=bool)
    out = np.zeros((*val.shape, before_arr.shape[-3], before_arr.shape[-2], 2), dtype=np.float32)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

    for idx in zip(*np.nonzero(val), strict=True):
        a = before_arr[idx]
        b = after_arr[idx]
        if a.dtype != np.uint8:
            a = np.clip(a * (255.0 if a.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if b.dtype != np.uint8:
            b = np.clip(b * (255.0 if b.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY) if a.ndim == 3 else a
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY) if b.ndim == 3 else b
        flow = dis.calc(ga, gb, None).astype(np.float32)
        # Normalize flow by image dimension so features are scale-stable across resolutions.
        scale = max(float(ga.shape[0]), 1.0)
        out[idx] = np.clip(flow / (scale * 0.1), -5.0, 5.0)
    return out


def reground_babbling_sysid(
    issued_mean: np.ndarray,
    delta_eef: np.ndarray,
    delta_jaw: np.ndarray,
    valid: np.ndarray,
    *,
    nominal_gain_m_per_cmd: float = 0.01,
    ridge: float = 1e-4,
    max_chunks: int | None = None,
):
    """ReGround-style target motor-babbling affine calibration (Siddique et al., 2026).

    Fits a per-target ridge-regularized linear dynamics map `delta_eef[:2] ≈ M @ issued[:2]`
    from target babbling transitions (proprioceptive state changes), projects `M` to the
    nearest scaled SO(2) rotation `(yaw_hat, scale_hat)` via polar/SVD decomposition, and
    estimates gripper polarity from `(issued_g, delta_jaw)`.

    Under rotated local EEF proprioception (`delta_eef_local = R(-yaw) delta_eef_world`),
    this estimator structurally recovers `yaw_hat ≈ 0 deg` regardless of babbling budget,
    whereas gain and gripper polarity remain observable.
    """
    iss = np.asarray(issued_mean, dtype=np.float64)
    deef = np.asarray(delta_eef, dtype=np.float64)
    djaw = np.asarray(delta_jaw, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    unbatched = iss.ndim == 2
    if unbatched:
        iss, deef, djaw, val = iss[None], deef[None], djaw[None], val[None]

    if max_chunks is not None:
        val = val.copy()
        val[:, int(max_chunks) :] = False

    bsz = iss.shape[0]
    yaws = np.zeros((bsz,), dtype=np.float32)
    scales = np.ones((bsz,), dtype=np.float32)
    _, _, grip_signs = analytic_proprio_sysid(
        iss, deef, djaw, val, nominal_gain_m_per_cmd=nominal_gain_m_per_cmd
    )

    for b in range(bsz):
        mask = val[b]
        if not np.any(mask):
            continue
        a_xy = iss[b, mask, :2]
        y_xy = deef[b, mask, :2] / max(float(nominal_gain_m_per_cmd), 1e-8)
        # Ridge regression: M_T = (A^T A + ridge I)^{-1} A^T Y, so y ≈ a @ M_T = M @ a.
        ata = a_xy.T @ a_xy + float(ridge) * np.eye(2, dtype=np.float64)
        aty = a_xy.T @ y_xy
        m_mat = np.linalg.solve(ata, aty).T  # [2, 2]: y ≈ M @ a
        # Polar decomposition onto SO(2) * positive isotropic scale.
        u, s, vt = np.linalg.svd(m_mat)
        det_sign = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
        r_hat = u @ np.diag([1.0, det_sign]) @ vt
        yaw_rad = math.atan2(float(r_hat[1, 0]), float(r_hat[0, 0]))
        yaws[b] = float(np.degrees(yaw_rad))
        # Estimate isotropic scale across valid XYZ axes.
        a_xyz = iss[b, mask, :3]
        y_xyz = deef[b, mask, :3] / max(float(nominal_gain_m_per_cmd), 1e-8)
        y_rot = y_xyz.copy()
        y_rot[:, :2] = y_xyz[:, :2] @ r_hat  # undo R_hat
        num = float(np.sum(a_xyz * y_rot))
        den = float(np.sum(a_xyz * a_xyz))
        if den > 1e-8 and num > 0.0:
            scales[b] = float(np.clip(num / den, 0.25, 4.0))

    if unbatched:
        return float(yaws[0]), float(scales[0]), int(grip_signs[0])
    return yaws, scales, grip_signs


def broyden_uvs_yaw(
    issued: np.ndarray,
    motion: np.ndarray,
    local_dz: np.ndarray,
    valid: np.ndarray,
    projection: np.ndarray,
    *,
    damping: float = 1e-2,
    step_size: float = 0.85,
) -> np.ndarray:
    """Classical online Broyden Uncalibrated Visual Servoing (UVS) planar Jacobian estimator.

    Maintains an online 2x2 command-to-world-XY Jacobian `J` initialized at `I_2`, updating
    sequentially over valid chunks via damped Broyden rank-1 updates:
        J_{k+1} = J_k + alpha * ((v_k - J_k a_k) a_k^T) / (||a_k||^2 + damping)
    where `v_k` is the Z-parallax-compensated visual XY displacement, then projects `J_K`
    onto SO(2) via SVD to extract yaw.
    """
    iss = np.asarray(issued, dtype=np.float64)
    mot = np.asarray(motion, dtype=np.float64)
    dz = np.asarray(local_dz, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    proj = np.asarray(projection, dtype=np.float64)
    unbatched = iss.ndim == 2
    if unbatched:
        iss, mot, dz, val = iss[None], mot[None], dz[None], val[None]

    pinv_xy = np.linalg.pinv(proj[:2], rcond=1e-5)
    planar = mot - dz[..., None] * proj[2]
    xy_vis = planar @ pinv_xy

    bsz, k_max, _ = iss.shape
    yaws = np.zeros((bsz,), dtype=np.float32)

    for b in range(bsz):
        j_mat = np.eye(2, dtype=np.float64)
        used = 0
        for k in range(k_max):
            if not val[b, k]:
                continue
            a = iss[b, k, :2]
            v = xy_vis[b, k]
            anorm2 = float(np.dot(a, a))
            vnorm = float(np.linalg.norm(v))
            if anorm2 < 1e-6 or vnorm < 1e-7:
                continue
            # Normalize scale so Broyden estimates direction/rotation cleanly across gain shifts.
            a_unit = a / math.sqrt(anorm2)
            v_unit = v / vnorm
            err = v_unit - j_mat @ a_unit
            j_mat = j_mat + float(step_size) * np.outer(err, a_unit) / (1.0 + float(damping))
            used += 1
        if used > 0:
            u, _, vt = np.linalg.svd(j_mat)
            det_sign = 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0
            r_hat = u @ np.diag([1.0, det_sign]) @ vt
            yaws[b] = float(np.degrees(math.atan2(float(r_hat[1, 0]), float(r_hat[0, 0]))))

    return float(yaws[0]) if unbatched else yaws


def fit_dense_flow_projection(
    train_before: np.ndarray,
    train_after: np.ndarray,
    train_delta_eef: np.ndarray,
    train_valid: np.ndarray,
    *,
    method: str = "dis",
) -> np.ndarray:
    """Calibrate pixels-per-metre 3x2 visual projection on nominal source trials using dense flow."""
    mot = dense_flow_image_motion(train_before, train_after, train_valid, method=method)
    return fit_visual_projection(mot, train_delta_eef, train_valid)
