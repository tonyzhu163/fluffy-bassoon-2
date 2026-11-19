"""Parametric cross-embodiment transforms, local proprioception, and probe schedules."""

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class EmbodimentSpec:
    """Persistent latent embodiment specification across attempts in a trial."""

    name: str
    yaw_deg: float
    xy_scale: float
    gripper_sign: int
    regime: str

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty str, got {self.name!r}")
        for field_name, value in (("yaw_deg", self.yaw_deg), ("xy_scale", self.xy_scale)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite numeric, got {value!r}")
        if self.xy_scale <= 0.0:
            raise ValueError(f"xy_scale must be > 0, got {self.xy_scale!r}")
        if type(self.gripper_sign) is not int or self.gripper_sign not in (-1, 1):
            raise ValueError(f"gripper_sign must be -1 or +1, got {self.gripper_sign!r}")
        if not isinstance(self.regime, str) or not self.regime:
            raise ValueError(f"regime must be a non-empty str, got {self.regime!r}")


TRAIN_EMBODIMENTS = (
    EmbodimentSpec("nom", 0.0, 1.0, 1, "train"),
    EmbodimentSpec("yaw90", 90.0, 1.0, 1, "train"),
    EmbodimentSpec("yaw180", 180.0, 1.0, 1, "train"),
    EmbodimentSpec("yawm90", -90.0, 1.0, 1, "train"),
    EmbodimentSpec("scale_lo", 0.0, 0.65, 1, "train"),
    EmbodimentSpec("scale_hi", 0.0, 1.55, 1, "train"),
    EmbodimentSpec("grip_inv", 0.0, 1.0, -1, "train"),
)

EVAL_EMBODIMENTS = (
    # Familiar single-factor + nominal embodiments
    EmbodimentSpec("fam_nom", 0.0, 1.0, 1, "familiar"),
    EmbodimentSpec("fam_yaw90", 90.0, 1.0, 1, "familiar"),
    EmbodimentSpec("fam_yaw180", 180.0, 1.0, 1, "familiar"),
    EmbodimentSpec("fam_grip_inv", 0.0, 1.0, -1, "familiar"),
    # Compositional OOD: every factor value seen in training, never combined
    EmbodimentSpec("comp_180_lo_inv", 180.0, 0.65, -1, "compositional_ood"),
    EmbodimentSpec("comp_m90_hi_inv", -90.0, 1.55, -1, "compositional_ood"),
    EmbodimentSpec("comp_90_hi_inv", 90.0, 1.55, -1, "compositional_ood"),
    EmbodimentSpec("comp_m90_lo_inv", -90.0, 0.65, -1, "compositional_ood"),
    # Continuous Angle OOD: oblique yaw angles and intermediate unseen scales
    EmbodimentSpec("cont_45_nom", 45.0, 1.0, 1, "continuous_angle_ood"),
    EmbodimentSpec("cont_135_075_inv", 135.0, 0.75, -1, "continuous_angle_ood"),
    EmbodimentSpec("cont_m135_135_inv", -135.0, 1.35, -1, "continuous_angle_ood"),
    EmbodimentSpec("cont_m45_075", -45.0, 0.75, 1, "continuous_angle_ood"),
)

EVAL_REGIMES = ("familiar", "compositional_ood", "continuous_angle_ood")

_EMBODIMENT_BY_NAME = {
    spec.name: spec for spec in (*TRAIN_EMBODIMENTS, *EVAL_EMBODIMENTS)
}

# Three 5-control chunks (15 controls total) probing +X, +Y, return/+Z, and gripper sign.
ACTIVE_PROBE_BURSTS = np.asarray(
    [
        np.tile(np.array([0.62, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)),
        np.tile(np.array([0.0, 0.62, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32), (5, 1)),
        np.tile(np.array([-0.35, -0.35, 0.45, 0.0, 0.0, 0.0, 1.0], dtype=np.float32), (5, 1)),
    ],
    dtype=np.float32,
)


def get_embodiment_spec(name: str) -> EmbodimentSpec:
    """Look up a preregistered training or evaluation embodiment by name."""
    if name not in _EMBODIMENT_BY_NAME:
        raise KeyError(f"Unknown embodiment {name!r}")
    return _EMBODIMENT_BY_NAME[name]


def rot2(yaw_deg: float) -> np.ndarray:
    """Return 2x2 planar rotation matrix R(yaw_deg)."""
    rad = math.radians(float(yaw_deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def circular_yaw_error_deg(pred_deg: float, true_deg: float) -> float:
    """Return absolute circular angular error in degrees within [0, 180]."""
    diff = (float(pred_deg) - float(true_deg) + 180.0) % 360.0 - 180.0
    return abs(diff)


def jaw_width_from_qpos(qpos: np.ndarray) -> np.ndarray:
    """Compute scalar jaw opening width from Panda two-finger gripper qpos."""
    arr = np.asarray(qpos, dtype=np.float32)
    return arr[..., 0] - arr[..., 1]


def apply_embodiment_action(
    issued: np.ndarray,
    spec: EmbodimentSpec,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Transform issued 7D control(s) into clipped physical simulator command(s)."""
    arr = np.asarray(issued, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(f"Expected trailing action dim 7, got {arr.shape}")
    phys = arr.copy()
    r = rot2(spec.yaw_deg)
    phys[..., 0:2] = (arr[..., 0:2] * np.float32(spec.xy_scale)) @ r.T
    phys[..., 2] = arr[..., 2] * np.float32(spec.xy_scale)
    phys[..., 6] = arr[..., 6] * np.float32(spec.gripper_sign)
    return np.clip(phys, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)).astype(np.float32)


def to_local_proprio(raw_obs: dict, anchor_eef_pos: np.ndarray, spec: EmbodimentSpec) -> dict:
    """Rotate robot0_eef_pos displacement from anchor into the robot's uncalibrated base frame.

    Frame-0 observations are bitwise identical across embodiments because eef_pos == anchor.
    For t >= 1, delta_eef_local[:2] is invariant to spec.yaw_deg.
    """
    anchor = np.asarray(anchor_eef_pos, dtype=np.float32).reshape(3)
    out = dict(raw_obs)
    eef = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    d_world = eef - anchor
    d_xy_local = d_world[:2] @ rot2(-spec.yaw_deg).T
    out["robot0_eef_pos"] = anchor + np.array([d_xy_local[0], d_xy_local[1], d_world[2]], dtype=np.float32)
    return out


def retarget_observation_for_motor(
    public_obs: dict,
    anchor_eef_pos: np.ndarray,
    yaw_deg_hat: float,
) -> dict:
    """Map public local-frame EEF position back to the canonical camera frame expected by π0.5."""
    anchor = np.asarray(anchor_eef_pos, dtype=np.float32).reshape(3)
    out = dict(public_obs)
    eef_local = np.asarray(public_obs["robot0_eef_pos"], dtype=np.float32).reshape(3)
    d_local = eef_local - anchor
    d_xy_world = d_local[:2] @ rot2(yaw_deg_hat).T
    out["robot0_eef_pos"] = anchor + np.array([d_xy_world[0], d_xy_world[1], d_local[2]], dtype=np.float32)
    return out


def retarget_action_from_motor(
    proposal: np.ndarray,
    yaw_deg_hat: float,
    xy_scale_hat: float,
    gripper_sign_hat: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    """Invert estimated embodiment transform on canonical π0.5 action proposal(s)."""
    arr = np.asarray(proposal, dtype=np.float32)
    if arr.shape[-1] != 7:
        raise ValueError(f"Expected trailing action dim 7, got {arr.shape}")
    scale_inv = np.float32(1.0 / float(np.clip(xy_scale_hat, 0.25, 4.0)))
    r_inv = rot2(-yaw_deg_hat)
    issued = arr.copy()
    issued[..., 0:2] = (arr[..., 0:2] @ r_inv.T) * scale_inv
    issued[..., 2] = arr[..., 2] * scale_inv
    issued[..., 6] = arr[..., 6] * np.float32(1 if int(gripper_sign_hat) >= 0 else -1)
    return np.clip(issued, np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)).astype(np.float32)


def downsample_agentview(raw_image: np.ndarray, size: int = 64) -> np.ndarray:
    """Orient agentview upright (matching OpenPI reference) and area-downsample to (size, size, 3) uint8."""
    arr = np.asarray(raw_image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {arr.shape}")
    upright = np.ascontiguousarray(arr[::-1, ::-1])
    h, w, _ = upright.shape
    if h == size and w == size:
        return upright.astype(np.uint8, copy=False)
    if h % size == 0 and w % size == 0:
        fh, fw = h // size, w // size
        pooled = upright.reshape(size, fh, size, fw, 3).mean(axis=(1, 3))
        return np.clip(np.rint(pooled), 0, 255).astype(np.uint8)
    y_idx = np.linspace(0, h - 1, size).astype(np.int32)
    x_idx = np.linspace(0, w - 1, size).astype(np.int32)
    return upright[np.ix_(y_idx, x_idx)].astype(np.uint8)
