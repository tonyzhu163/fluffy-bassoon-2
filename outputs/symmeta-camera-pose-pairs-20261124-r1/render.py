"""One physics trajectory per trace with same-clock renders. Historical orbits retain their original files."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from geometry import camera, dump, pair, project

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "symmeta-camera-command-20261123-r1"))
from replay_worker import OrbitalViews
from world_control import ARCHIVE, SOURCE, STORAGE, file_sha, motor_state, panel, reset, setup

ROOT = STORAGE / "outputs" / HERE.name
ORIGINAL = json.loads((HERE / "original-index.json").read_text())
SOURCE_MANIFEST = Path(ORIGINAL["source_manifest"])
VIEWS = ("nominal", "orbit-", "orbit+", "roll-", "roll+")


def rz(angle):
    a = np.deg2rad(angle)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


class Views(OrbitalViews):
    def __init__(self, arm, first, source):
        super().__init__(arm, first, [0])
        m = self.sim.model._model
        self.orbit, self.roll = (15, 10) if source else (30, 20)
        self.fovy = float(m.cam_fovy[self.cam])
        assert self.direction == 1 and not m.cam_orthographic[self.cam] and not m.cam_sensorsize[self.cam].any()
        self.marker_checks = []
        self.validate = False

    def extrinsics(self, name, pos, mat):
        if name == "nominal":
            return pos, mat
        sign = 1 if name.endswith("+") else -1
        if name.startswith("orbit"):
            rot = rz(sign * self.orbit)
            return self.pivot + rot @ (pos - self.pivot), rot @ mat
        # Positive angle about local +Z, i.e. backward along the optical axis.
        return pos, mat @ rz(sign * self.roll)

    def check_marks(self, world, calib):
        """Independently rasterize known bright spheres onto a separate temporary visual scene."""
        import mujoco
        ctx = self.sim._render_context_offscreen
        viewport = mujoco.MjrRect(0, 0, 256, 256)
        # Reuse current camera scene, removing occluders only in this diagnostic render.
        # Restore the original scene by the ordinary renderer on its next call.
        ctx.scn.ngeom = 0
        for point in world:
            geom = ctx.scn.geoms[ctx.scn.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([.006, .006, .006]), point, np.eye(3).ravel(),
                                np.array([1, 0, 1, 1], np.float32))
            geom.emission = 1
            ctx.scn.ngeom += 1
        mujoco.mjr_render(viewport, ctx.scn, ctx.con)
        img = ctx.read_pixels(256, 256)[::-1, ::-1]
        mask = (img[..., 0] > 180) & (img[..., 2] > 180) & (img[..., 1] < 80)
        from scipy.ndimage import label, center_of_mass
        ids, n = label(mask)
        if n != len(world):
            raise ValueError(f"Independent mark count {n} != {len(world)}")
        measured = (np.asarray(center_of_mass(mask, ids, np.arange(1, n+1)))[:, ::-1]+.5)/4-.5
        expected = project(calib["P"], world)
        from scipy.optimize import linear_sum_assignment
        dist = np.linalg.norm(expected[:, None] - measured[None], axis=-1)
        a, b = linear_sum_assignment(dist)
        if np.max(dist[a,b]) >= .5:  # nearest pooled pixel must agree to within a half pixel
            raise ValueError(f"Recorded-image projection failed: expected {expected}, observed {measured}")
        return dict(world_m=world.tolist(), predicted_px=expected[a].tolist(), measured_px=measured[b].tolist(),
                    errors_px=dist[a,b].tolist())

    def record(self, nominal, settings):
        import mujoco
        from embodiment import downsample_agentview
        sim = self.sim
        model, data = sim.model._model, sim.data._data
        flag = mujoco.mjtState.mjSTATE_INTEGRATION
        states = [np.empty(mujoco.mj_stateSize(model, flag)) for _ in range(2)]
        mujoco.mj_getState(model, data, states[0], flag)
        pos = data.cam_xpos[self.cam].copy()
        mat = data.cam_xmat[self.cam].reshape(3, 3).copy()
        imgs, cams = {}, {}
        # Noncollinear diagnostic landmarks, fixed in world coordinates across views.
        world = self.pivot + np.array([[-.07, -.03, .015], [.05, -.05, .045], [.025, .07, -.02]])
        try:
            for name in VIEWS:
                p, r = self.extrinsics(name, pos, mat)
                data.cam_xpos[self.cam] = p
                data.cam_xmat[self.cam] = r.ravel()
                rgb = nominal if name == "nominal" else self.render(**settings)
                imgs[name] = downsample_agentview(rgb[::self.direction], 64)
                cams[name] = camera(p, r, self.fovy)
                if self.validate:
                    # Populate scene also when nominal sensor was cached.
                    self.render(**settings)
                    self.marker_checks.append(dict(view=name, **self.check_marks(world, cams[name])))
        finally:
            data.cam_xpos[self.cam] = pos
            data.cam_xmat[self.cam] = mat.ravel()
        mujoco.mj_getState(model, data, states[1], flag)
        np.testing.assert_array_equal(*states)
        assert all(not np.array_equal(imgs["nominal"], imgs[k]) for k in VIEWS[1:])
        imgs[0] = imgs["nominal"]  # historical match() confirms actual native frame and TCP together
        return dict(images=imgs, camera=cams, world_xyz_m=data.site_xpos[self.robot.eef_site_id].copy(),
                    time_s=float(data.time))


def cases(mode, shards):
    if mode == "source":
        assert file_sha(SOURCE_MANIFEST) == ORIGINAL["source_manifest_sha256"]
        trials = json.loads(SOURCE_MANIFEST.read_text())["trials"]
        for body in ("Panda", "Sawyer"):
            sub = [t for t in trials if t["arm"] == body and t["split"] == "train"]
            first = sub[0]
            for t in sub:
                if all(t[k] == first[k] for k in ("suite", "task_id", "init_state_id", "seed")) and t["embodiment"] in ("nom", "yaw90", "yaw180", "yawm90"):
                    yield body, {k:t[k] for k in ("shard","suite","task_id","init_state_id","seed")}, dict(
                        index=t["index"], embodiment=t["embodiment"], physics=t["physics"], passive=SOURCE_MANIFEST.parent / t["passive_path"])
    else:
        for case, entries in panel(SOURCE, ARCHIVE, shards):
            for e in entries[8:12]:
                yield "UR5e", case, e


def collect(folder, body, case, entry, historical):
    from arm_env import ArmEnv
    from embodiment import apply_embodiment_action, get_embodiment_spec, jaw_width_from_qpos, to_local_proprio
    with np.load(entry["passive"], allow_pickle=False) as z:
        archive = {k:z[k] for k in z}
    assert not archive["valid"][3:].any() and archive["valid"][:3].all()
    spec = get_embodiment_spec(entry["embodiment"])
    issued = np.repeat(archive["issued_mean"][:3], 5, axis=0)
    physical = apply_embodiment_action(issued, spec, -np.ones(7), np.ones(7))
    source = body != "UR5e"
    uid = f"{body}-{entry['index']:04d}"
    with ArmEnv(body, case["suite"], case["task_id"]) as arm:
        first = reset(arm, case)
        obs = first["observation"]
        np.testing.assert_allclose(motor_state(obs), archive["frame0_state"], atol=2e-6, rtol=0)
        cams = Views(arm, first, source)
        cams.validate = True
        frames = [cams.frame0(obs)]
        cams.validate = False
        anchor = obs["robot0_eef_pos"].copy()
        local, joints, jaws = [anchor], [obs["robot0_joint_pos"].copy()], [jaw_width_from_qpos(obs["robot0_gripper_qpos"])]
        for i, cmd in enumerate(physical, 1):
            if i % 5 == 0:
                step, frame = cams.boundary_step(arm, cmd)
                frames.append(frame)
                o = step["observation"]
                local.append(to_local_proprio(o, anchor, spec)["robot0_eef_pos"] if spec.yaw_deg else o["robot0_eef_pos"])
                joints.append(o["robot0_joint_pos"].copy())
                jaws.append(jaw_width_from_qpos(o["robot0_gripper_qpos"]))
            else:
                step = arm.step(cmd)
            assert step["worker_step_index"] == i and (i == 15 or not (step["terminated"] or step["truncated"]))
        shared = dict(delta_eef=np.diff(local,axis=0).astype(np.float32), delta_joint=np.diff(joints,axis=0),
                      delta_jaw=np.stack([np.diff(jaws),jaws[1:]],axis=1).astype(np.float32), issued_mean=archive["issued_mean"][:3],
                      chunk_steps=np.full(3,5,np.int32),valid=np.ones(3,bool),frame0_state=motor_state(obs),
                      joint_count=np.int32(len(joints[0])))
        fp = first["environment_reproducibility"]["initial_state_fingerprint"]
        z = float(anchor[2])
        views = {}
        for name in VIEWS:
            cam = frames[0]["camera"][name]
            assert all(f["camera"][name] == cam for f in frames)
            imgs = np.stack([f["images"][name] for f in frames])
            angles = {"nominal":0,"orbit-":-cams.orbit,"orbit+":cams.orbit,"roll-":-cams.roll,"roll+":cams.roll}
            angle = angles[name]
            arrays = dict(**shared, frame0_img=imgs[0], img_before=imgs[:-1], img_after=imgs[1:])
            if historical is not None and not name.startswith("roll"):
                old = historical["views"][str(angle)]
                path = Path(old["path"])
                assert file_sha(path) == old["sha256"]
                with np.load(path,allow_pickle=False) as orig:
                    for k, v in arrays.items():
                        np.testing.assert_array_equal(v, orig[k], err_msg=f"{uid}/{name}/{k}")
                for b in historical["camera_extrinsics"]:
                    oldcam = b[str(angle)]
                    np.testing.assert_array_equal(cam["center_m"], oldcam["center_m"])
                    np.testing.assert_array_equal(cam["gl_to_world"], oldcam["camera_to_world"])
            else:
                path = folder / uid / f"{name}.npz"
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
            views[name] = dict(passive_npz=str(path), sha256=file_sha(path), angle_deg=angle, **cam,
                               **pair(frames[0]["camera"]["nominal"],cam,z))
        row = dict(id=uid,index=entry["index"],arm=body,role="source" if source else "heldout",**case,
                   native_control_id=spec.name, control_yaw_deg=spec.yaw_deg, xy_scale=spec.xy_scale,gripper_sign=spec.gripper_sign,
                   plane_z_m=z,physics=fp, input_physics=entry["physics"],archive_npz=str(entry["passive"]),
                   archive_sha256=file_sha(entry["passive"]),replay="archived_chunk_mean_held_for_five",
                   original_orbit=historical is not None,frame_steps=[0,5,10,15],frame_times_s=[f["time_s"] for f in frames],views=views)
        diag = dict(id=uid,landmarks=cams.marker_checks,world_tcp_m=[f["world_xyz_m"].tolist() for f in frames],
                    physical_commands=physical.tolist(), issued_commands=issued.tolist(),
                    archive_joint_delta_max_error_rad=float(np.abs(shared["delta_joint"]-archive["delta_joint"][:3,:len(joints[0])]).max()),
                    archive_final_nominal_rgb_mae=float(np.abs(frames[-1]["images"]["nominal"].astype(float)-archive["img_after"][2]).mean()))
        if body == "UR5e":
            from world_control import GripSite
            archived = GripSite(arm).archived_positions(joints[0],archive["delta_joint"][:3])
            diag["archive_endpoint_distances_mm"] = (1000*np.linalg.norm(np.array(diag["world_tcp_m"])-archived,axis=1)).tolist()
        return row, diag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode",choices=["source","target"])
    ap.add_argument("out",type=Path)
    ap.add_argument("shards",type=int,nargs="*")
    args=ap.parse_args()
    assert os.sched_getaffinity(0) <= {44,45,46} and not os.environ.get("CUDA_VISIBLE_DEVICES")
    setup(SOURCE)
    hist = {r["index"]:r for r in ORIGINAL["rows"]}
    args.out.mkdir(parents=True,exist_ok=False)
    rows,diagnostics=[],[]
    for body,case,e in cases(args.mode,args.shards):
        row,diag=collect(args.out,body,case,e,hist.get(e["index"]) if body=="UR5e" else None)
        rows.append(row)
        diagnostics.append(diag)
        dump(args.out/"diagnostics.json",diagnostics)
        dump(args.out/"bank.json",dict(rows=rows))
        print(row["id"],row["native_control_id"],len(rows),flush=True)


if __name__=="__main__":
    main()
