"""Fixed CAD rig. Deployment path takes only target task-space pose and jaw."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from libero.libero import get_libero_path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "appearance"))
from render import CAM, render, reset


class PandaRig:
    def __init__(self):
        # Public installation from a disjoint LIBERO-Spatial task, independent
        # of the ten LIBERO-Goal queries. Retain fixed table, mount and lighting.
        base = Path(get_libero_path("bddl_files")) / "libero_spatial"
        self.bddl = base / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"
        self.env = reset(self.bddl, "Panda", 201)
        self.sim = self.env.sim
        self.robot = self.env.robots[0]
        self.idx = self.robot._ref_joint_pos_indexes
        self.rest = self.robot.robot_model.init_qpos.copy()
        ctx = self.sim._render_context_offscreen
        ctx.vopt.geomgroup[5] = 0
        ctx.vopt.sitegroup[:] = 0
        for geom, body in enumerate(self.sim.model.geom_bodyid):
            name = self.sim.model.body_id2name(body)
            if name not in ("world", "table") and not name.startswith(("robot0_", "gripper0_", "mount0_")):
                self.sim.model.geom_group[geom] = 5

    def installation(self):
        s = self.sim
        cam = s.model.camera_name2id(CAM)
        return dict(fixed_scene=self.bddl.name, seed=201, rest_q=self.rest.tolist(),
                    rest_hand_quat_xyzw=np.roll(s.data.get_body_xquat(self.robot.robot_model.eef_name), -1).tolist(),
                    camera_xyz=s.data.cam_xpos[cam].tolist(), camera_matrix=s.data.cam_xmat[cam].tolist(),
                    camera_fovy_deg=float(s.model.cam_fovy[cam]),
                    base_xyz=s.data.get_body_xpos("robot0_base").tolist(),
                    base_matrix=s.data.get_body_xmat("robot0_base").tolist(),
                    light_pos=s.data.light_xpos.tolist(), light_dir=s.data.light_xdir.tolist())

    def set_pose(self, q, grip):
        self.sim.data.qpos[self.idx] = q
        self.sim.data.qpos[self.robot._ref_gripper_joint_pos_indexes] = grip
        self.sim.forward()

    def infer(self, pos, quat_xyzw, grip):
        desired = Rotation.from_quat(quat_xyzw)

        def residual(q):
            self.set_pose(q, grip)
            tcp = self.sim.data.site_xpos[self.robot.eef_site_id]
            hand = self.sim.data.get_body_xmat(self.robot.robot_model.eef_name)
            delta = (desired.inv() * Rotation.from_matrix(hand)).as_rotvec()
            return np.r_[tcp - pos, delta]

        bounds = self.sim.model.jnt_range[self.robot._ref_joint_indexes]
        # Seven joints, six end-effector constraints. Resolve redundancy by
        # minimum squared displacement from the fixed manufacturer/model rest.
        fit = minimize(lambda q: 0.5 * np.sum((q - self.rest) ** 2), self.rest.copy(),
                       jac=lambda q: q - self.rest, method="SLSQP", bounds=bounds,
                       constraints=[dict(type="eq", fun=residual)], options=dict(ftol=1e-12, maxiter=150))
        err = residual(fit.x)
        if not fit.success or np.max(np.abs(err)) > 1e-8:
            raise ValueError(f"{fit.message}; tool residual {err}")
        rgb, mask = render(self.env)
        return dict(rgb=rgb, mask=mask, q=fit.x, tool_error=err)

    def close(self):
        self.env.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("public", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    rig = PandaRig()
    try:
        (args.output / "installation.json").write_text(json.dumps(rig.installation(), indent=2) + "\n")
        files = sorted(args.public.glob("*.npz"))
        assert len(files) == 10
        for path in files:
            # Native image, wrist image, and task are unnecessary for IK/render.
            with np.load(path) as data:
                sprite = rig.infer(data["pos"], data["quat"], data["grip"])
            np.savez_compressed(args.output / path.name, **sprite)
    finally:
        rig.close()
