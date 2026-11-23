"""Qualify mounted-arm encoder FK on archived closed-loop support, using software EGL."""
import sys
from pathlib import Path

import numpy as np

from qualify import HERE, require, sources, rows, sha, load, json, os

sys.path.insert(0, str(HERE.parent / "symmeta-camera-command-20261123-r1"))
from world_control import GripSite, align


class MountedSite(GripSite):
    """Use the established scratch-data site evaluation with an arm-specific chain."""
    def __init__(self, arm, identity):
        import mujoco

        sim = arm._env.sim
        robot = arm._env.robots[0]
        self.model = sim.model._model
        self.site = robot.eef_site_id
        self.indices = list(robot._ref_joint_pos_indexes)
        chain = []
        body = int(self.model.site_bodyid[self.site])
        while body:
            require(self.model.body_mocapid[body] < 0, "Mocap ancestor of grip site")
            start = int(self.model.body_jntadr[body])
            chain[:0] = list(range(start, start + int(self.model.body_jntnum[body])))
            body = int(self.model.body_parentid[body])
        names = [self.model.joint(j).name for j in chain]
        require(names == identity["joint_names"] and len(names) == identity["joint_count"]
                and list(self.model.jnt_qposadr[chain]) == self.indices
                and np.all(self.model.jnt_type[chain] == mujoco.mjtJoint.mjJNT_HINGE)
                and self.model.site(self.site).name == identity["tcp_site"], (identity, names))
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:] = sim.data.qpos
        self.data.mocap_pos[:] = sim.data._data.mocap_pos
        self.data.mocap_quat[:] = sim.data._data.mocap_quat

    def endpoints(self, q0, dq):
        return np.stack([self.position(q0 + d) for d in
                         np.vstack([np.zeros_like(q0), np.cumsum(dq, axis=0, dtype=np.float64)])])


def motor_state(raw, canonical):
    q = np.asarray(raw["robot0_eef_quat_xyzw"], dtype=np.float64)
    flip = bool(canonical and abs(q[3]) < .85 and q[0] < 0)
    if flip:
        q = -q
    w = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - w*w)
    rot = 2 * np.arccos(w) * q[:3] / den if den > 1e-10 else np.zeros(3)
    return np.concatenate((raw["robot0_eef_pos"], rot, raw["robot0_gripper_qpos"])).astype(np.float32), flip


def tape(path, dof, slots):
    with np.load(path, allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    shapes = dict(img_before=(slots,64,64,3), img_after=(slots,64,64,3), issued_mean=(slots,7),
                  delta_joint=(slots,dof), delta_eef=(slots,3), delta_jaw=(slots,2), frame0_state=(8,))
    require(all(t[k].shape == s and np.isfinite(t[k]).all() for k,s in shapes.items()), path)
    require(t["joint_count"].item() == dof and t["valid"].dtype == bool
            and np.array_equal(t["valid"], np.arange(slots) < 3)
            and np.array_equal(t["frame0_img"], t["img_before"][0])
            and np.array_equal(t["img_before"][1:3], t["img_after"][:2])
            and np.max(np.abs(t["issued_mean"])) <= 1, path)
    return t


def replay(arm, first, active):
    from fluffy_bassoon.xemb.embodiment import ACTIVE_PROBE_BURSTS

    np.testing.assert_array_equal(ACTIVE_PROBE_BURSTS.mean(axis=1), active["issued_mean"])
    raw = first["observation"]
    qs = [raw["robot0_joint_pos"].copy()]
    ps = [raw["robot0_eef_pos"].copy()]
    n = 0
    for burst in ACTIVE_PROBE_BURSTS:
        for a in burst:
            step = arm.step(a)
            n += 1
            require(step["worker_step_index"] == n and not (step["terminated"] or step["truncated"]), n)
        qs.append(step["observation"]["robot0_joint_pos"].copy())
        ps.append(step["observation"]["robot0_eef_pos"].copy())
    dq = np.diff(qs, axis=0)
    dp = np.diff(ps, axis=0)
    np.testing.assert_allclose(dq, active["delta_joint"], atol=2e-6, rtol=0)
    np.testing.assert_allclose(dp, active["delta_eef"], atol=1e-6, rtol=0)
    return dict(controls=n, joint_max_abs_rad=float(np.max(np.abs(dq-active["delta_joint"]))),
                tcp_max_abs_m=float(np.max(np.abs(dp-active["delta_eef"]))))


def main(panel, reader_file, output):
    from collect_ur5e import issued_exposure, CHUNK_SIZE
    from arm_env import ArmEnv
    from fluffy_bassoon.xemb.embodiment import circular_yaw_error_deg

    ref = load(reader_file)
    require(ref["panel"] == panel and ref["pin"] == info["pin"] and len(ref["rows"]) == 96, reader_file)
    receipts, estimates = [], []
    for shard in range(8):
        manifest = list(rows(archive, shard))
        first_row = ref["rows"][12 * shard]
        suite, tid = first_row["suite"], first_row["task_id"]
        with ArmEnv(info["arm"], suite, tid) as arm:
            initial = arm.reset(init_state_id=first_row["support_layout"], seed=first_row["support_seed"])
            raw = initial["observation"]
            q0 = raw["robot0_joint_pos"].copy()
            state_a, _ = motor_state(raw, False)
            state_p, flip = motor_state(raw, info["policy"] == "pi0")
            identity = initial["model_identity"]
            site = MountedSite(arm, identity)
            m0 = manifest[0][0]
            require(m0["embodiment"] == "fam_nom", m0)
            nom = tape(m0["active_npz"], len(q0), 3)
            active_receipt = replay(arm, initial, nom)
            repro = initial["environment_reproducibility"]
            receipts.append(dict(shard=shard, suite=suite, task_id=tid, q0=q0.tolist(), identity=identity,
                frame0_active=state_a.tolist(), frame0_passive=state_p.tolist(), quaternion_flipped=flip,
                released_row=repro["released_row_fingerprint"], actual_physics=repro["initial_state_fingerprint"],
                archived_physics=m0["physics"], bitwise_reset=repro["initial_state_fingerprint"] == m0["physics"],
                active_replay=active_receipt, native_horizon=initial["max_steps"],
                support_layout=first_row["support_layout"], support_seed=first_row["support_seed"]))
            for m, hist in manifest:
                r = ref["rows"][hist["index"]]
                a = tape(m["active_npz"], len(q0), 3)
                p = tape(m["passive_npz"], len(q0), 24)
                require(m["physics"] == m0["physics"] == r["physics"]
                        and sha(Path(m["passive_npz"])) == r["support_sha256"]
                        and sha(Path(m["active_npz"])) == r["active_sha256"], m)
                np.testing.assert_array_equal(nom["frame0_img"], a["frame0_img"])
                np.testing.assert_array_equal(a["frame0_img"], p["frame0_img"])
                np.testing.assert_allclose(a["frame0_state"], state_a, atol=2e-6, rtol=0)
                np.testing.assert_allclose(p["frame0_state"], state_p, atol=2e-6, rtol=0)
                mask = p["valid"]
                p["chunk_steps"] = np.where(mask, CHUNK_SIZE, 0).astype(np.int32)
                exposure = issued_exposure(p)[mask]
                dp = np.diff(site.endpoints(q0, p["delta_joint"][mask]), axis=0)
                fit = align(exposure, dp)
                yaw = fit["yaw_hat_deg"]
                require(yaw is not None and np.isfinite(yaw), r["index"])
                native = p["delta_eef"][mask]
                residual = np.abs(np.linalg.norm(native[:,:2],axis=1) - np.linalg.norm(dp[:,:2],axis=1))
                estimates.append(dict(**r, fk_yaw_hat=yaw, fk_available=True, reader_available=True,
                    native_reader_qualified=r["native_match"], issued_xy=exposure[:,:2].tolist(),
                    world_delta_xy_m=dp[:,:2].tolist(),
                    fk_yaw_error_deg=circular_yaw_error_deg(yaw, hist["true_yaw"]),
                    reader_yaw_error_deg=circular_yaw_error_deg(r["yaw_hat"], hist["true_yaw"]),
                    regime=hist["regime"], shard=shard, aligned_correlation=fit["aligned_correlation"],
                    world_xy_path_mm=fit["world_xy_path_mm"], xy_length_max_error_mm=float(1000*max(residual)),
                    z_max_error_mm=float(1000*max(abs(native[:,2]-dp[:,2])))))
        print(panel, "FK", shard, len(estimates), flush=True)
    with output.open("x") as f:
        json.dump(dict(**info, checkpoint_sha256=ref["checkpoint_sha256"], resets=receipts, rows=estimates,
                       fk_qualified=len(estimates)==96, native_reader_qualified=ref["qualified"],
                       qualified=len(estimates)==96 and ref["qualified"]), f, allow_nan=False)


if __name__ == "__main__":
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("LIBGL_ALWAYS_SOFTWARE") == "1",
            "CPU and software GL required")
    panel, reader_file, output = sys.argv[1:]
    archive, src, info = sources(panel)
    main(panel, Path(reader_file), Path(output))
