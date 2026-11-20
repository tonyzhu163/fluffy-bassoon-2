"""Cross-morphology linear joint-Jacobian audit for proprioceptive yaw SysID."""

from __future__ import annotations

import numpy as np


def fit_joint_jacobian(
    delta_q: np.ndarray,
    delta_eef: np.ndarray,
    valid: np.ndarray,
    ridge: float = 1e-6,
) -> np.ndarray:
    """Fit a linear Jacobian J of shape (D_src, 3) mapping delta_q to delta_eef over valid chunks."""
    dq = np.asarray(delta_q, dtype=np.float64)
    deef = np.asarray(delta_eef, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if dq.ndim < 2 or deef.ndim != dq.ndim or deef.shape[:-1] != dq.shape[:-1] or deef.shape[-1] != 3:
        raise ValueError(f"Mismatched shapes: delta_q={dq.shape}, delta_eef={deef.shape}")
    if mask.shape != dq.shape[:-1]:
        raise ValueError(f"Mismatched valid mask shape {mask.shape} for delta_q {dq.shape}")
    if not mask.any():
        raise ValueError("Cannot fit joint Jacobian with zero valid chunks")

    x = dq[mask]
    y = deef[mask]
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite values in joint Jacobian fitting data")
    if float(np.linalg.norm(x)) <= 1e-12:
        raise ValueError("Degenerate zero joint displacement in fitting data")

    d_src = x.shape[1]
    reg = float(ridge) * np.eye(d_src, dtype=np.float64)
    lhs = x.T @ x + reg
    rhs = x.T @ y
    jacobian = np.linalg.solve(lhs, rhs)
    if not np.isfinite(jacobian).all():
        raise ValueError("Nonfinite fitted joint Jacobian")
    return jacobian


def _planar_procrustes_yaw_deg(issued_xy: np.ndarray, pred_xy: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Align issued XY commands [B, K, 2] to predicted world XY motion [B, K, 2]."""
    w = np.asarray(valid, dtype=bool)
    a = np.where(w[..., None], np.asarray(issued_xy, dtype=np.float64), 0.0)
    v = np.where(w[..., None], np.asarray(pred_xy, dtype=np.float64), 0.0)
    dot = np.sum(a[..., 0] * v[..., 0] + a[..., 1] * v[..., 1], axis=-1)
    cross = np.sum(a[..., 0] * v[..., 1] - a[..., 1] * v[..., 0], axis=-1)
    norm = np.hypot(dot, cross)
    has_valid = w.any(axis=-1)
    yaw_deg = np.where(has_valid & (norm > 1e-10), np.degrees(np.arctan2(cross, dot)), 0.0)
    return yaw_deg.astype(np.float32)


def predict_yaw_from_joint_jacobian(
    issued: np.ndarray,
    delta_q: np.ndarray,
    valid: np.ndarray,
    jacobian: np.ndarray,
    *,
    pad_or_truncate: bool = False,
) -> np.ndarray:
    """Predict planar yaw (deg) for [B, K, ...] traces using a fitted source joint Jacobian."""
    iss = np.asarray(issued, dtype=np.float64)
    dq = np.asarray(delta_q, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    jac = np.asarray(jacobian, dtype=np.float64)

    if jac.ndim != 2 or jac.shape[1] != 3:
        raise ValueError(f"Expected jacobian shape (D_src, 3), got {jac.shape}")
    if iss.ndim != 3 or iss.shape[-1] != 7:
        raise ValueError(f"Expected issued shape (B, K, 7), got {iss.shape}")
    if dq.ndim != 3 or dq.shape[:2] != iss.shape[:2]:
        raise ValueError(f"Expected delta_q shape (B, K, D_tgt), got {dq.shape}")
    if val.shape != iss.shape[:2]:
        raise ValueError(f"Expected valid shape {iss.shape[:2]}, got {val.shape}")

    d_src = jac.shape[0]
    d_tgt = dq.shape[-1]
    if d_tgt != d_src:
        if not pad_or_truncate:
            raise ValueError(f"Joint dimension mismatch: source={d_src}, target={d_tgt}")
        if d_tgt < d_src:
            pad = np.zeros((*dq.shape[:2], d_src - d_tgt), dtype=np.float64)
            dq_aligned = np.concatenate((dq, pad), axis=-1)
        else:
            dq_aligned = dq[..., :d_src]
    else:
        dq_aligned = dq

    deef_hat = dq_aligned @ jac
    return _planar_procrustes_yaw_deg(iss[..., :2], deef_hat[..., :2], val)


def evaluate_cross_morphology_joint_jacobian(
    source_dq: np.ndarray,
    source_deef: np.ndarray,
    source_valid: np.ndarray,
    target_issued: np.ndarray,
    target_dq: np.ndarray,
    target_deef_local: np.ndarray,
    target_valid: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> dict:
    """Evaluate strict, zero-padded/truncated, and self-calibrated local joint-Jacobian controls."""
    jac_src = fit_joint_jacobian(source_dq, source_deef, source_valid, ridge=ridge)
    d_src = int(jac_src.shape[0])
    dq_tgt = np.asarray(target_dq, dtype=np.float64)
    iss_tgt = np.asarray(target_issued, dtype=np.float64)
    deef_loc = np.asarray(target_deef_local, dtype=np.float64)
    val_tgt = np.asarray(target_valid, dtype=bool)
    d_tgt = int(dq_tgt.shape[-1])

    applicable = d_tgt == d_src
    reason = None if applicable else f"joint_dim_mismatch:{d_src}!={d_tgt}"
    strict_yaw = (
        predict_yaw_from_joint_jacobian(iss_tgt, dq_tgt, val_tgt, jac_src, pad_or_truncate=False).tolist()
        if applicable
        else None
    )
    zero_padded_yaw = predict_yaw_from_joint_jacobian(
        iss_tgt, dq_tgt, val_tgt, jac_src, pad_or_truncate=True
    ).tolist()

    # Self-calibrated local Jacobian fitted per episode solely on that robot's own local proprioception.
    self_cal_local_yaw = []
    for i in range(dq_tgt.shape[0]):
        try:
            jac_local = fit_joint_jacobian(dq_tgt[i : i + 1], deef_loc[i : i + 1], val_tgt[i : i + 1], ridge=ridge)
            yh = predict_yaw_from_joint_jacobian(
                iss_tgt[i : i + 1], dq_tgt[i : i + 1], val_tgt[i : i + 1], jac_local, pad_or_truncate=False
            )
            self_cal_local_yaw.append(float(yh[0]))
        except ValueError:
            self_cal_local_yaw.append(0.0)

    return dict(
        source_dof=d_src,
        target_dof=d_tgt,
        applicable=applicable,
        reason=reason,
        strict_yaw_deg=strict_yaw,
        zero_padded_yaw_deg=zero_padded_yaw,
        self_calibrated_local_yaw_deg=self_cal_local_yaw,
    )
