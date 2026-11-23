"""Planar command-interface shifts beyond yaw, gain and gripper sign.

An interface maps an issued planar command to world motion as `R(yaw) @ shape`.
The robot reports proprioception in its own frame, rotated by `yaw`; `shape` is
a 2x2 map applied in that frame. The paper's family is `shape = gain * I`.

A support trace recorded under one interface is relabeled to another by remapping
its issued commands and the frame of its proprioception. Images are untouched:
the arm made the same physical motion, so the relabeled trace is what a robot with
the new interface records when the remapped commands are issued from the same state.
"""

from __future__ import annotations

import numpy as np

FAMILIES = ("family", "aniso", "shear", "swap", "general")
GAIN_RANGE = (0.65, 1.5)
ANISO_RATIO = 1.6
SHEAR_RANGE = 0.7


def rot2_batch(yaw_deg) -> np.ndarray:
    rad = np.deg2rad(np.asarray(yaw_deg, dtype=np.float64))
    c, s = np.cos(rad), np.sin(rad)
    return np.stack((np.stack((c, -s), axis=-1), np.stack((s, c), axis=-1)), axis=-2)


def sample_interfaces(rng: np.random.Generator, family: str, n: int) -> dict:
    """Draw n interfaces. `shape` is gain times a unit-area 2x2 map; `z_gain` equals gain."""
    if family not in FAMILIES:
        raise ValueError(family)
    gain = np.exp(rng.uniform(np.log(GAIN_RANGE[0]), np.log(GAIN_RANGE[1]), n))
    unit = np.broadcast_to(np.eye(2), (n, 2, 2)).copy()
    if family == "aniso":
        root = np.exp(0.5 * rng.uniform(-np.log(ANISO_RATIO), np.log(ANISO_RATIO), n))
        unit[:, 0, 0], unit[:, 1, 1] = root, 1.0 / root
    elif family == "shear":
        unit[:, 0, 1] = rng.uniform(-SHEAR_RANGE, SHEAR_RANGE, n)
    elif family == "swap":
        unit[:] = np.array([[0.0, 1.0], [1.0, 0.0]])
    elif family == "general":
        root = np.exp(0.5 * rng.uniform(-np.log(ANISO_RATIO), np.log(ANISO_RATIO), n))
        stretch = np.zeros((n, 2, 2))
        stretch[:, 0, 0] = root
        stretch[:, 1, 1] = rng.choice([-1.0, 1.0], n) / root
        unit = rot2_batch(rng.uniform(-180, 180, n)) @ stretch @ rot2_batch(rng.uniform(-180, 180, n))
    return dict(
        yaw_deg=rng.uniform(-180.0, 180.0, n),
        shape=gain[:, None, None] * unit,
        z_gain=gain,
        gripper_sign=rng.choice([-1, 1], n),
    )


def interface_matrix(iface: dict) -> np.ndarray:
    return rot2_batch(iface["yaw_deg"]) @ np.asarray(iface["shape"], dtype=np.float64)


def relabel(batch: dict, target: dict) -> dict:
    """Relabel traces recorded under yaw_deg/xy_scale/gripper_sign to the target interfaces."""
    yaw0 = np.asarray(batch["yaw_deg"], dtype=np.float64)
    gain0 = np.asarray(batch["xy_scale"], dtype=np.float64)
    grip0 = np.asarray(batch["gripper_sign"], dtype=np.float64)
    issued = np.asarray(batch["issued_mean"], dtype=np.float64)
    delta = np.asarray(batch["delta_eef"], dtype=np.float64)

    source = gain0[:, None, None] * rot2_batch(yaw0)
    target_mat = interface_matrix(target)
    remap = np.linalg.solve(target_mat, source)
    new_issued = issued.copy()
    new_issued[..., :2] = np.einsum("bij,bkj->bki", remap, issued[..., :2])
    new_issued[..., 2] = issued[..., 2] * (gain0 / np.asarray(target["z_gain"]))[:, None]
    new_issued[..., 6] = issued[..., 6] * (grip0 * np.asarray(target["gripper_sign"]))[:, None]

    frame = np.swapaxes(rot2_batch(target["yaw_deg"]), -1, -2) @ rot2_batch(yaw0)
    new_delta = delta.copy()
    new_delta[..., :2] = np.einsum("bij,bkj->bki", frame, delta[..., :2])

    out = dict(batch)
    out["issued_mean"] = new_issued.astype(np.float32)
    out["delta_eef"] = new_delta.astype(np.float32)
    out["yaw_deg"] = np.asarray(target["yaw_deg"], dtype=np.float32)
    out["xy_scale"] = np.sqrt(np.abs(np.linalg.det(target["shape"]))).astype(np.float32)
    out["gripper_sign"] = np.asarray(target["gripper_sign"], dtype=np.int32)
    out["op"] = target_mat.astype(np.float32)
    return out


def delay_commands(batch: dict, chunks: int = 1) -> dict:
    """Pure actuation delay: motion in chunk k was caused by the command issued `chunks` earlier."""
    valid = np.asarray(batch["valid"], dtype=bool)
    out = dict(batch)
    issued = np.zeros_like(batch["issued_mean"])
    issued[:, :-chunks] = batch["issued_mean"][:, chunks:]
    keep = valid.copy()
    keep[:, :-chunks] &= valid[:, chunks:]
    keep[:, -chunks:] = False
    out["issued_mean"] = np.where(keep[..., None], issued, 0.0).astype(np.float32)
    out["valid"] = keep
    return out


def procrustes_yaw_deg(source, vectors, weights) -> np.ndarray:
    """Rotation angle that best aligns source planar vectors to observed vectors."""
    ax, ay = source[..., 0], source[..., 1]
    vx, vy = vectors[..., 0], vectors[..., 1]
    dot = np.sum(weights * (ax * vx + ay * vy), axis=-1)
    cross = np.sum(weights * (ax * vy - ay * vx), axis=-1)
    return np.degrees(np.arctan2(cross, dot))


def ridge_linear_map(inputs, outputs, weights, prior, ridge: float) -> np.ndarray:
    """argmin_M sum_k w_k |y_k - M x_k|^2 + ridge * sum_k w_k |x_k|^2 / 2 * |M - prior|_F^2."""
    x = np.asarray(inputs, dtype=np.float64)
    y = np.asarray(outputs, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    xx = np.einsum("bk,bki,bkj->bij", w, x, x)
    yx = np.einsum("bk,bki,bkj->bij", w, y, x)
    lam = ridge * 0.5 * np.trace(xx, axis1=-2, axis2=-1)[:, None, None] + 1e-12
    eye = np.eye(2)[None]
    return (yx + lam * np.asarray(prior, dtype=np.float64)) @ np.linalg.inv(xx + lam * eye)


def operator_error(true_op, est_op) -> np.ndarray:
    """|A inv(A_hat) - I|_F / sqrt(2): the command map the frozen policy is left with after correction."""
    resid = np.asarray(true_op, dtype=np.float64) @ np.linalg.pinv(np.asarray(est_op, dtype=np.float64))
    return np.linalg.norm(resid - np.eye(2), axis=(-2, -1)) / np.sqrt(2.0)


def command_spread(commands, weights) -> np.ndarray:
    """Smallest over largest singular value of the weighted command set; 0 means one direction only."""
    x = np.asarray(commands, dtype=np.float64) * np.sqrt(np.asarray(weights, dtype=np.float64))[..., None]
    s = np.linalg.svd(x, compute_uv=False)
    return s[..., -1] / np.maximum(s[..., 0], 1e-12)
