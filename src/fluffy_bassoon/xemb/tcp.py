"""Physical MuJoCo tool-center (TCP) offset interventions and compensated controller."""

from dataclasses import dataclass
import math
import numpy as np


FINGER_COLLISION_SUFFIXES = (
    "finger1_collision",
    "finger2_collision",
    "finger1_pad_collision",
    "finger2_pad_collision",
)


@dataclass(frozen=True)
class TcpOffsetSpec:
    """Physical finger collision-geometry TCP offset in reset world coordinates."""

    name: str
    dx_m: float
    dy_m: float
    dz_m: float = 0.0
    regime: str = "train"

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be non-empty str, got {self.name!r}")
        for field_name, val in (("dx_m", self.dx_m), ("dy_m", self.dy_m), ("dz_m", self.dz_m)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{field_name} must be finite numeric, got {val!r}")
        if not isinstance(self.regime, str) or not self.regime:
            raise ValueError(f"regime must be non-empty str, got {self.regime!r}")

    @property
    def vector_m(self) -> np.ndarray:
        return np.array([self.dx_m, self.dy_m, self.dz_m], dtype=np.float64)


def make_tcp_catalogue(radius_m: float = 0.035):
    """Build training (single-axis + nominal) and evaluation (familiar + withheld diagonals) catalogues."""
    if isinstance(radius_m, bool) or not isinstance(radius_m, (int, float)) or not math.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError(f"radius_m must be finite and > 0, got {radius_m!r}")
    r = float(radius_m)
    d = round(r / math.sqrt(2.0), 4)
    train_specs = (
        TcpOffsetSpec("tcp_nom", 0.0, 0.0, 0.0, "train"),
        TcpOffsetSpec("tcp_px", r, 0.0, 0.0, "train"),
        TcpOffsetSpec("tcp_mx", -r, 0.0, 0.0, "train"),
        TcpOffsetSpec("tcp_py", 0.0, r, 0.0, "train"),
        TcpOffsetSpec("tcp_my", 0.0, -r, 0.0, "train"),
    )
    eval_specs = (
        TcpOffsetSpec("tcp_nom", 0.0, 0.0, 0.0, "familiar"),
        TcpOffsetSpec("tcp_px", r, 0.0, 0.0, "familiar"),
        TcpOffsetSpec("tcp_mx", -r, 0.0, 0.0, "familiar"),
        TcpOffsetSpec("tcp_py", 0.0, r, 0.0, "familiar"),
        TcpOffsetSpec("tcp_my", 0.0, -r, 0.0, "familiar"),
        TcpOffsetSpec("comp_pxpy", d, d, 0.0, "compositional_ood"),
        TcpOffsetSpec("comp_mxpy", -d, d, 0.0, "compositional_ood"),
    )
    return train_specs, eval_specs


def quat_xyzw_to_mat(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert unit quaternion [x, y, z, w] to 3x3 rotation matrix."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def world_tcp_offset_at_quat(quat0_xyzw: np.ndarray, quat_t_xyzw: np.ndarray, delta_world_0: np.ndarray) -> np.ndarray:
    """Rotate reset-frame TCP offset delta_world_0 by the public gripper orientation change."""
    r0 = quat_xyzw_to_mat(quat0_xyzw)
    rt = quat_xyzw_to_mat(quat_t_xyzw)
    d0 = np.asarray(delta_world_0, dtype=np.float64).reshape(3)
    return (rt @ (r0.T @ d0)).astype(np.float64)


def apply_mujoco_tcp_offset(world, dx_m: float, dy_m: float, dz_m: float = 0.0) -> dict:
    """Shift the four Panda finger collision geoms in MuJoCo after reset settling."""
    sim = world.sim
    model = sim.model
    names = [model.geom_id2name(i) or "" for i in range(model.ngeom)]
    geom_ids = []
    for suffix in FINGER_COLLISION_SUFFIXES:
        matches = [i for i, name in enumerate(names) if name.endswith("_" + suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected 1 geom matching _{suffix}, got {matches}")
        geom_ids.append(matches[0])

    pad1_id, pad2_id = geom_ids[2], geom_ids[3]
    before_xpos = np.asarray(sim.data.geom_xpos[geom_ids], dtype=np.float64).copy()
    before_gap = float(np.linalg.norm(before_xpos[2] - before_xpos[3]))
    before_ncon = int(sim.data.ncon)

    delta_world = np.array([float(dx_m), float(dy_m), float(dz_m)], dtype=np.float64)
    if np.linalg.norm(delta_world) > 0.0:
        for gid in geom_ids:
            bid = int(model.geom_bodyid[gid])
            body_rmat = np.asarray(sim.data.body_xmat[bid], dtype=np.float64).reshape(3, 3)
            model.geom_pos[gid] = np.asarray(model.geom_pos[gid], dtype=np.float64) + body_rmat.T @ delta_world
        sim.forward()

    after_xpos = np.asarray(sim.data.geom_xpos[geom_ids], dtype=np.float64).copy()
    after_gap = float(np.linalg.norm(after_xpos[2] - after_xpos[3]))
    after_ncon = int(sim.data.ncon)
    shifts = after_xpos - before_xpos
    max_shift_err_m = float(np.max(np.linalg.norm(shifts - delta_world[None], axis=1)))
    pad_gap_err_m = abs(after_gap - before_gap)

    if max_shift_err_m >= 1e-6:
        raise RuntimeError(f"MuJoCo TCP shift error too large: {max_shift_err_m:.3e} m")
    if pad_gap_err_m >= 1e-6:
        raise RuntimeError(f"Pad gap changed after TCP shift: {pad_gap_err_m:.3e} m")
    if after_ncon != before_ncon:
        raise RuntimeError(f"TCP shift introduced reset contacts: before={before_ncon}, after={after_ncon}")

    return dict(
        geom_names=[names[gid] for gid in geom_ids],
        pad_names=[names[pad1_id], names[pad2_id]],
        delta_world_m=delta_world.tolist(),
        max_shift_err_m=max_shift_err_m,
        pad_gap_err_m=pad_gap_err_m,
        before_ncon=before_ncon,
        after_ncon=after_ncon,
    )


class TcpCompensatedController:
    """Public-observation controller wrapping frozen π0.5 proposals with physical TCP compensation."""

    def __init__(
        self,
        quat0_xyzw: np.ndarray,
        dx_hat_m: float,
        dy_hat_m: float,
        *,
        compensate_release: bool = False,
    ):
        self.quat0 = np.asarray(quat0_xyzw, dtype=np.float64).reshape(4).copy()
        self.dx_hat = float(dx_hat_m)
        self.dy_hat = float(dy_hat_m)
        self.compensate_release = bool(compensate_release)
        self.active = math.hypot(self.dx_hat, self.dy_hat) >= 1e-4

        self.phase = "nominal"
        self.close_streak = 0
        self.release_streak = 0
        self.rearm_open_steps = 0
        self.post_grasp_steps = 0
        self.grasp_done_once = False
        self.release_done_for_grasp = False

        self._step_in_phase = 0
        self._p_target = np.zeros(3, dtype=np.float64)
        self._delta_at_grasp = np.zeros(3, dtype=np.float64)
        self._release_target_xy = np.zeros(2, dtype=np.float64)
        self.grasp_trigger_count = 0
        self.release_trigger_count = 0

    @property
    def needs_motor_proposal(self) -> bool:
        return self.phase == "nominal"

    def _delta_t(self, quat_t: np.ndarray) -> np.ndarray:
        return world_tcp_offset_at_quat(
            self.quat0,
            quat_t,
            np.array([self.dx_hat, self.dy_hat, 0.0], dtype=np.float64),
        )

    def _enter_phase(self, name: str):
        self.phase = name
        self._step_in_phase = 0

    def next_action(
        self,
        eef_pos: np.ndarray,
        eef_quat_xyzw: np.ndarray,
        gripper_qpos: np.ndarray,
        motor_proposal: np.ndarray = None,
    ) -> np.ndarray:
        """Return bounded-scale 7D action given public proprioception and optional π0.5 proposal."""
        cur_pos = np.asarray(eef_pos, dtype=np.float64).reshape(3)
        cur_quat = np.asarray(eef_quat_xyzw, dtype=np.float64).reshape(4)
        jaw = float(gripper_qpos[0] - gripper_qpos[1])

        if not self.active:
            if motor_proposal is None:
                raise ValueError("motor_proposal required in nominal mode")
            return np.asarray(motor_proposal, dtype=np.float32).copy()

        if self.phase == "nominal":
            if motor_proposal is None:
                raise ValueError("motor_proposal required when phase == 'nominal'")
            prop = np.asarray(motor_proposal, dtype=np.float32).copy()
            if self.grasp_done_once:
                self.post_grasp_steps += 1
                if jaw > 0.06:
                    self.rearm_open_steps += 1
                else:
                    self.rearm_open_steps = 0

            can_trigger_grasp = (not self.grasp_done_once) or (self.rearm_open_steps >= 30)
            if can_trigger_grasp and prop[6] > 0.0 and jaw > 0.05:
                self.close_streak += 1
            else:
                self.close_streak = 0

            if (
                self.compensate_release
                and self.grasp_done_once
                and (not self.release_done_for_grasp)
                and self.post_grasp_steps >= 15
                and prop[6] < 0.0
                and jaw < 0.055
            ):
                self.release_streak += 1
            else:
                self.release_streak = 0

            if self.close_streak >= 2:
                self.close_streak = 0
                self.release_streak = 0
                self.grasp_done_once = True
                self.release_done_for_grasp = False
                self.post_grasp_steps = 0
                self.rearm_open_steps = 0
                self.grasp_trigger_count += 1
                self._delta_at_grasp = self._delta_t(cur_quat)
                self._p_target = cur_pos - self._delta_at_grasp
                self._enter_phase("lift_clear")
            elif self.release_streak >= 2:
                self.release_streak = 0
                self.close_streak = 0
                self.release_done_for_grasp = True
                self.release_trigger_count += 1
                delta_rel = self._delta_t(cur_quat)
                self._release_target_xy = cur_pos[:2] - delta_rel[:2]
                self._enter_phase("release_shift")
            else:
                return prop

        # Execute active scripted TCP maneuver phase.
        if self.phase == "lift_clear":
            self._step_in_phase += 1
            if self._step_in_phase >= 4:
                self._enter_phase("shift_xy")
            return np.array([0.0, 0.0, 0.65, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        if self.phase == "shift_xy":
            err_xy = self._p_target[:2] - cur_pos[:2]
            if float(np.linalg.norm(err_xy)) < 0.003 or self._step_in_phase >= 12:
                self._enter_phase("descend")
            else:
                self._step_in_phase += 1
                cmd_xy = np.clip(12.0 * err_xy, -0.8, 0.8)
                return np.array([cmd_xy[0], cmd_xy[1], 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        if self.phase == "descend":
            err = self._p_target - cur_pos
            if float(np.linalg.norm(err)) < 0.004 or self._step_in_phase >= 8:
                self._enter_phase("close_jaw")
            else:
                self._step_in_phase += 1
                cmd = np.clip(12.0 * err, -0.8, 0.8)
                return np.array([cmd[0], cmd[1], cmd[2], 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        if self.phase == "close_jaw":
            self._step_in_phase += 1
            if self._step_in_phase >= 10:
                self._enter_phase("lift_obj")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if self.phase == "lift_obj":
            self._step_in_phase += 1
            if self._step_in_phase >= 5:
                self._enter_phase("nominal")
            return np.array([0.0, 0.0, 0.65, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if self.phase == "release_shift":
            err_xy = self._release_target_xy - cur_pos[:2]
            if float(np.linalg.norm(err_xy)) < 0.003 or self._step_in_phase >= 10:
                self._enter_phase("release_open")
            else:
                self._step_in_phase += 1
                cmd_xy = np.clip(12.0 * err_xy, -0.8, 0.8)
                return np.array([cmd_xy[0], cmd_xy[1], 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        if self.phase == "release_open":
            self._step_in_phase += 1
            if self._step_in_phase >= 6:
                self._enter_phase("nominal")
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

        raise RuntimeError(f"Unexpected controller phase {self.phase!r}")
