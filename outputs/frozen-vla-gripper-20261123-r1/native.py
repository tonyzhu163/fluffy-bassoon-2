"""Probe mounted native actuators, then compare TCP and actual pad trajectories on a frozen tape."""
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import sys

import mujoco
import numpy as np
from OpenGL import GL

import archive
from arm_env import ArmEnv
from sim_reset import fresh_kinematics

ARMS = ("Panda", "Sawyer", "Kinova3", "IIWA", "UR5e")
SOURCE = (archive.BASE / "libero-retry-headroom-20261119-r1/paired01/episodes"
          / "libero_10-3-r0-BASE/default")
SCHEDULE = [1.]*12 + [0.]*4 + [-1.]*12


class Instrument:
    """Observe native actuator targets and physical contacts at every physics integration."""
    def __init__(self, session):
        self.env = session._env
        self.robot = self.env.robots[0]
        self.sim = self.env.sim
        self.m = self.sim.model._model
        self.scratch = mujoco.MjData(self.m)
        self.hand = self.m.body(self.robot.robot_model.eef_name).id
        self.site = self.robot.eef_site_id
        self.pads = [self.m.geom(f"gripper0_finger{i}_pad_collision").id for i in (1, 2)]
        self.geoms = set(self.pads + [self.m.geom(f"gripper0_finger{i}_collision").id for i in (1, 2)])
        self.act = [self.m.actuator(x).id for x in self.robot.gripper.actuators]
        self.fq = self.robot._ref_gripper_joint_pos_indexes
        self.joints = [self.m.joint(x).id for x in self.robot.gripper_joints]
        self.contacts = set()
        self.substeps = []
        self.formats = []
        original_format = self.robot.gripper.format_action
        self.format_file = Path(inspect.getsourcefile(original_format))
        self.grip_file = Path(inspect.getsourcefile(self.robot.grip_action))

        def format_action(action):
            value = original_format(action)
            self.formats.append((float(action[0]), value.copy()))
            return value

        self.robot.gripper.format_action = format_action
        original_step = self.sim.step

        def step():
            self.substeps.append(self.sim.data.ctrl[self.act].copy())
            original_step()
            for c in self.sim.data._data.contact:
                if c.efc_address < 0:
                    continue
                for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if a in self.geoms and b not in self.geoms:
                        self.contacts.add((self.m.geom(a).name, self.m.geom(b).name))

        self.sim.step = step
        # Images are unnecessary for fixed-action replay. Keep other native clocks and physics.
        for name, sensor in self.env.env._observables.items():
            if name.endswith("_image"):
                sensor.set_enabled(False)
                sensor.set_active(False)

    def body_offsets(self, canonical=False):
        d = self.scratch
        fresh_kinematics(self.env, d)
        if canonical:
            # Equal and opposite slides; remove asymmetry due to the current grasp/load.
            d.qpos[self.fq] = [.02, -.02]
            mujoco.mj_kinematics(self.m, d)
        center = d.geom_xpos[self.pads].mean(axis=0)
        tcp = d.site_xpos[self.site]
        r = d.site_xmat[self.site].reshape(3,3)
        h = d.xmat[self.hand].reshape(3,3)
        return dict(tcp=tcp.copy(), center=center, local=r.T@(center-tcp), hand_local=h.T@(center-tcp),
                    hand_to_tcp=h.T@(tcp-d.xpos[self.hand]), hand_to_site=h.T@r, world_R=r.copy())

    def identity(self):
        m, j, a = self.m, self.joints, self.act
        calibrated = self.body_offsets(canonical=True)
        return dict(xml=archive.sha(Path(self.robot.gripper.file)), formatter_sha256=archive.sha(self.format_file),
                    grip_action_sha256=archive.sha(self.grip_file), formatter_name=self.robot.gripper.__class__.__name__,
                    speed=self.robot.gripper.speed, normalized_dof=self.robot.gripper.dof,
                    joints=[m.joint(i).name for i in j], joint_type=m.jnt_type[j].tolist(),
                    joint_axis=m.jnt_axis[j].tolist(), joint_range=m.jnt_range[j].tolist(),
                    actuator_names=[m.actuator(i).name for i in a], actuator_joints=m.actuator_trnid[a,0].tolist(),
                    joint_ids=j, ctrlrange=m.actuator_ctrlrange[a].tolist(), force_range=m.actuator_forcerange[a].tolist(),
                    gain=m.actuator_gainprm[a].tolist(), bias=m.actuator_biasprm[a].tolist(),
                    finger_damping=m.dof_damping[m.jnt_dofadr[j]].tolist(), pad_halfsize_m=m.geom_size[self.pads].tolist(),
                    canonical_pad_offset_tcp_m=calibrated["local"].tolist(),
                    canonical_pad_offset_hand_m=calibrated["hand_local"].tolist(),
                    hand_to_tcp_m=calibrated["hand_to_tcp"].tolist(), hand_to_site_R=calibrated["hand_to_site"].tolist(),
                    integration_dt=m.opt.timestep)

    def execute(self, issued):
        self.contacts.clear()
        self.substeps.clear()
        self.formats.clear()
        before = self.robot.gripper.current_action.copy()
        raw, reward, done, _ = self.env.step(issued.copy())
        # Verify every inner application, including clipping at each limit and zero hold.
        n = len(self.substeps)
        assert n > 0 and len(self.formats) == n
        assert all(command == issued[6] for command, _ in self.formats)
        expected = np.clip(before + np.arange(1,n+1)[:,None]*self.robot.gripper.speed
                           * np.sign(issued[6])*np.array([-1.,1.]), -1., 1.)
        np.testing.assert_allclose(expected, [x[1] for x in self.formats], rtol=0, atol=1e-14)
        limits = self.m.actuator_ctrlrange[self.act]
        controls = limits.mean(axis=1) + .5*(limits[:,1]-limits[:,0])*expected
        np.testing.assert_allclose(controls, self.substeps, rtol=0, atol=1e-14)
        geom = self.body_offsets()
        self.public_quat = np.asarray(raw["robot0_eef_quat"], dtype=np.float32).copy()
        q = raw["robot0_gripper_qpos"]
        return dict(q=q.tolist(), jaw_mm=float((q[0]-q[1])*1000), ctrl=self.substeps[-1].tolist(),
                    substeps=n, contacts=sorted(self.contacts), reward=float(reward),
                    success=bool(self.env.check_success()), done=bool(done), public_tcp=raw["robot0_eef_pos"].tolist(),
                    **{k:v.tolist() for k,v in geom.items() if k in ("local", "center", "tcp", "world_R")})


def pulse(out, arm, magnitude):
    with ArmEnv(arm, "libero_goal", 3) as session:
        first = session.reset(11, 31810311)
        renderer = GL.glGetString(GL.GL_RENDERER).decode()
        assert "llvmpipe" in renderer.lower()
        meter = Instrument(session)
        header = meter.identity()
        assert header["actuator_joints"] == header["joint_ids"]
        trace = []
        for sign in SCHEDULE:
            a = np.zeros(7, dtype=np.float32)
            a[-1] = sign*magnitude
            row = meter.execute(a)
            assert not row["done"] and not row["success"]
            trace.append(dict(command=float(a[-1]), **row))
        record = dict(arm=arm, magnitude=magnitude, initial_q=first["observation"]["robot0_gripper_qpos"].tolist(),
                      release=first["environment_reproducibility"], model=first["model_identity"],
                      renderer=renderer, transduction=header, trace=trace)
        archive.write(out/f"pulse-{arm}-{magnitude:g}.json", record)
        return record


def probe_all(out):
    profiles, canonical = {}, {}
    scene = None
    summary = {}
    for arm in ARMS:
        a, b = [pulse(out, arm, mag) for mag in (1., .125)]
        assert a["release"]["initial_state_fingerprint"] == b["release"]["initial_state_fingerprint"]
        if scene is None:
            scene = a["release"]["scene_fingerprint"]
        assert scene == a["release"]["scene_fingerprint"]
        at, bt = a["trace"], b["trace"]
        for field in ("q", "ctrl", "contacts", "reward", "public_tcp", "world_R", "center"):
            assert [x[field] for x in at] == [x[field] for x in bt], (arm, field)
        profiles[arm] = np.array([x["jaw_mm"] for x in at])
        canonical[arm] = np.array(a["transduction"]["canonical_pad_offset_tcp_m"])
        hits = np.flatnonzero(profiles[arm] <= 40)
        reopen = np.flatnonzero(profiles[arm][16:] >= 40)
        summary[arm] = dict(first_half_stroke_close_control=int(hits[0]+1) if len(hits) else None,
                            first_half_stroke_reopen_control=int(reopen[0]+1) if len(reopen) else None,
                            min_width_mm=profiles[arm].min(), max_width_mm=profiles[arm].max(),
                            max_jaw_vs_panda_mm=np.abs(profiles[arm]-profiles["Panda"]).max(),
                            max_same_sign_magnitude_jaw_difference_mm=0.,
                            pad_offset_vs_panda_mm=(1000*(canonical[arm]-canonical["Panda"])).tolist(),
                            canonical_pad_offset_m=canonical[arm].tolist(),
                            substeps_per_control=sorted({x["substeps"] for x in at}),
                            external_finger_contacts=sum(bool(x["contacts"]) for x in at))
        print(arm, json.dumps(summary[arm]), flush=True)
    archive.write(out/"pulse-summary.json", summary)
    return canonical["Panda"]


def replay(out, arm, case, commands, source_offset, metadata):
    with ArmEnv(arm, case["suite"], case["task_id"]) as session:
        first = session.reset(case["init_state_id"], case["seed"])
        meter = Instrument(session)
        scene = first["environment_reproducibility"]
        bowl = meter.env.env.obj_body_id["akita_black_bowl_1"]
        trace = []
        for a in commands:
            row = meter.execute(np.clip(a, -1, 1).astype(np.float32))
            row["bowl"] = meter.scratch.xpos[bowl].tolist()
            row["source_offset_error_mm"] = float(np.linalg.norm(np.array(row["local"])-source_offset)*1000)
            trace.append(row)
            if row["done"] or row["success"]:
                break
        record = dict(case=case, arm=arm, reset=scene, model=first["model_identity"], source=metadata,
                      trace=trace, emitted_controls=len(trace), native_success=trace[-1]["success"],
                      native_return=sum(x["reward"] for x in trace))
        archive.write(out / f"drawer-{arm}.json", record)
        return record


def drawer(out, source_offset):
    paths = [SOURCE/"public.npz", SOURCE/"result.json"]
    hashes = [archive.sha(p) for p in paths]
    assert hashes == ["c264a0db031867c7cf62461feba68be6d73a5e5459f1deb9b4ffc584e1379928",
                      "d5900f6c7ab12bfb3619cecf4e466d30271dcf99ac6f9b53c26432238d6b714e"]
    metadata = dict(paths=[str(p) for p in paths], sha256=hashes,
                    collector_pin="c45aa965db73ad0644382da578ff03a30b2083d7")
    assert (SOURCE.parents[2]/"launch.txt").read_text().split()[0] == metadata["collector_pin"]
    orig = json.loads(paths[1].read_text())
    with np.load(paths[0]) as tape:
        commands, positions, rewards = [tape[k].copy() for k in ("issued","positions","rewards")]
    assert len(commands) == orig["controls"] == 206
    first_close = int(np.flatnonzero(commands[:,6]>0)[0])
    report = {}
    for arm in ARMS:
        rec = replay(out, arm, orig["case"], commands, source_offset, metadata)
        rr = rec["trace"]
        if arm == "Panda":
            ref = rr
            scene = rec["reset"]["scene_fingerprint"]
            diff = np.abs(np.array([x["public_tcp"] for x in rr]) - positions[:len(rr)])
            report["archive_source_check"] = dict(reset_matches=rec["reset"]["initial_state_fingerprint"]==orig["identity"]["physics"],
                      success_matches=rec["native_success"]==orig["success"],
                      native_reward_tape_equal=np.array_equal([r["reward"] for r in rr],rewards),
                      maximum_abs_public_tcp_difference_m=float(diff.max()), max_input_clip=float(np.abs(commands-np.clip(commands,-1,1)).max()))
            assert report["archive_source_check"]["success_matches"] and report["archive_source_check"]["native_reward_tape_equal"]
        assert rec["reset"]["scene_fingerprint"] == scene
        row = rr[first_close-1]  # state immediately before first positive command
        srow = ref[first_close-1]
        def gap(field):
            return float(np.linalg.norm(np.subtract(row[field],srow[field]))*1000)
        obj_relative = np.subtract(row["center"],row["bowl"])-np.subtract(srow["center"],srow["bowl"])
        id_err = 1000*np.linalg.norm([r["local"] for r in rr],axis=1)
        offset_err = np.array([r["source_offset_error_mm"] for r in rr])
        bowl_contacts = [i+1 for i,r in enumerate(rr) if any("akita_black_bowl" in b for a,b in r["contacts"])]
        report[arm] = dict(success=rec["native_success"], native_return=rec["native_return"], controls=len(rr),
                first_close_control=first_close+1, preclose_tcp_vs_panda_mm=gap("tcp"), preclose_pad_vs_panda_mm=gap("center"),
                preclose_bowl_relative_pad_vs_panda_mm=(1000*obj_relative).tolist(),
                first_finger_bowl_contact=bowl_contacts[0] if bowl_contacts else None, bowl_contact_controls=len(bowl_contacts),
                site_as_midpoint_mean_error_mm=float(id_err.mean()), panda_offset_mean_error_mm=float(offset_err.mean()),
                panda_offset_max_error_mm=float(offset_err.max()))
        print("drawer", arm, json.dumps(report[arm]), flush=True)
    archive.write(out/"drawer-summary.json",report)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    assert len(os.sched_getaffinity(0)) <= 2
    archive.run(out)
    offset = probe_all(out)
    drawer(out, offset)
