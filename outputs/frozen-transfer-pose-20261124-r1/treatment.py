"""Native fixed-tape comparison with a causal wrist-only residual and an equal-magnitude opposite sham."""
import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from geometry import Geometry
from panel import cases, key
from replay import digest, save
from solve import Problem, SCALE, SPEED


def wrist(g, cmd, public_q, public_p, travel):
    """The decoder receives no scene, contact, task, time, label or reward observations."""
    zero = np.zeros(3)
    if not np.any(cmd[3:6]):
        return zero
    q=np.array(public_q,dtype=float)
    g.d.qpos[g.adr]=q
    mujoco.mj_kinematics(g.m,g.d)
    R=g.d.site_xmat[g.site].reshape(3,3).copy()
    goal_R=Rotation.from_rotvec(SCALE[3:]*cmd[3:6]).as_matrix() @ R
    problem=Problem(g,np.array(public_p)+SCALE[:3]*cmd[:3],goal_R)
    lo,hi=np.maximum(q-travel,g.lo),np.minimum(q+travel,g.hi)
    if np.any(hi<=lo):
        return zero
    fit=problem.poses(q,lo,hi,None,False)
    if np.array_equal(fit["full_q"],fit["free_q"]):
        return zero
    problem.value(fit["free_q"])
    free=g.d.site_xmat[g.site].reshape(3,3)
    desired=Rotation.from_matrix(free@R.T).as_rotvec()/SCALE[3:]
    delta=desired-cmd[3:6]
    use=np.abs(delta)>0
    alpha=min(1., np.min((1.-np.abs(cmd[3:6][use]))/np.abs(delta[use]))) if np.any(use) else 0.
    return alpha*delta


def run(spec, out, source):
    arm=spec["arm"]
    name=f"{key(spec)}-{arm}"
    rp=source / f"{name}.npz"
    meta=json.loads(rp.with_suffix(".json").read_bytes())
    tape=Path(meta["tape"])
    assert digest(tape)==meta["tape_sha256"] and digest(rp)==meta["data_sha256"]
    original=np.concatenate([np.array(c["cmd"],dtype=np.float32) for c in json.loads(tape.read_bytes())["chunks"]])
    original.setflags(write=False)
    assert original.shape==(meta["native_controls"],7)
    earlier=np.load(rp,allow_pickle=False)
    cases_out={}
    for method in ("raw","adaptive","sham"):
        with ArmEnv(arm,spec["suite"],spec["task_id"]) as env:
            first=env.reset(spec["init_state_id"],spec["seed"])
            assert first["environment_reproducibility"]==meta["reset"]
            obs=first["observation"]
            g=Geometry(env._env)
            # Scratch FK for deployment uses the known model and encoder vector only.
            g.d=mujoco.MjData(g.m)
            travel=np.array(SPEED[arm])*meta["sample_dt"]
            robot=env._env.robots[0]
            ctrl=robot.controller
            pad_ids=[g.m.geom(f"gripper0_finger{i}_pad_collision").id for i in (1,2)]
            snapshot=mujoco.MjData(g.m)

            def at_endpoint():
                snapshot.qpos[:]=env._env.sim.data.qpos
                mujoco.mj_kinematics(g.m,snapshot)
                return (snapshot.site_xpos[g.site].copy(),snapshot.site_xmat[g.site].reshape(3,3).copy(),
                        snapshot.geom_xpos[pad_ids].mean(axis=0))

            is_drawer=spec["suite"]=="libero_goal" and spec["task_id"]==3
            if is_drawer:
                drawer_adr=g.m.joint("wooden_cabinet_1_top_level").qposadr[0]
                bowl_id=g.m.body("akita_black_bowl_1_main").id
                drawer_init=float(env._env.sim.data.qpos[drawer_adr])
                at_endpoint()
                bowl_init=float(snapshot.xpos[bowl_id,2])
            rows=[]
            blank=np.zeros((256,256,3),np.uint8)
            with patch.object(env._env.sim,"render",return_value=blank):
                for i,cmd in enumerate(original):
                    delta=np.zeros(3)
                    if method=="adaptive":
                        delta=wrist(g,cmd,obs["robot0_joint_pos"],obs["robot0_eef_pos"],travel)
                    if method=="sham" and i<len(cases_out["adaptive"]["delta"]):
                        delta=-cases_out["adaptive"]["delta"][i]
                    applied=cmd.copy()
                    applied[3:6]=cmd[3:6].astype(float)+delta
                    assert np.max(np.abs(applied))<=1 and np.array_equal(applied[[0,1,2,6]],cmd[[0,1,2,6]])
                    before,rot0,pad0=at_endpoint()
                    raw_R=(Rotation.from_rotvec(cmd[3:6]*SCALE[3:]).as_matrix()@rot0
                           if np.any(cmd[3:6]) else ctrl.goal_ori.copy())
                    step=env.step(applied)
                    obs=step["observation"]
                    after,rot1,pad1=at_endpoint()
                    raw_goal=before+cmd[:3]*SCALE[:3]
                    assert np.linalg.norm(raw_goal-ctrl.goal_pos)<1e-7
                    goal_pad=raw_goal+raw_R@(rot0.T@(pad0-before))
                    rows.append(dict(cmd=cmd,applied=applied,delta=delta,actual_delta=applied[3:6].astype(float)-cmd[3:6],
                                     before_p=before,before_R=rot0,before_pad=pad0,raw_p=raw_goal,raw_R=raw_R,
                                     raw_pad=goal_pad,issued_R=ctrl.goal_ori.copy(),end_R=rot1,end_p=after,end_pad=pad1,
                                     reward=step["reward"],end_q=snapshot.qpos[g.adr].copy()))
                    if is_drawer:
                        # Offline scorer only. The motion decoder above never reads these bodies or contacts.
                        snapshot.mocap_pos[:]=env._env.sim.data.mocap_pos
                        snapshot.mocap_quat[:]=env._env.sim.data.mocap_quat
                        mujoco.mj_fwdPosition(g.m,snapshot)
                        touched=set()
                        for contact in snapshot.contact:
                            if contact.dist>=0 or contact.exclude:
                                continue
                            a,b=g.geoms[contact.geom1],g.geoms[contact.geom2]
                            if a[0]=="finger":
                                touched.add(b[1])
                            if b[0]=="finger":
                                touched.add(a[1])
                        rows[-1].update(drawer_travel_m=snapshot.qpos[drawer_adr]-drawer_init,
                            bowl_lift_m=snapshot.xpos[bowl_id,2]-bowl_init,
                            finger_drawer="wooden_cabinet_1_cabinet_top" in touched,
                            finger_bowl="akita_black_bowl_1_main" in touched, finger_plate="plate_1_main" in touched)
                    if step["terminated"] or step["truncated"]:
                        break
            arr={k:np.array([v[k] for v in rows]) for k in rows[0]}
            dest=out / f"{name}-{method}.npz"
            np.savez_compressed(dest,**arr)
            info=dict(case=spec,method=method,source_pin=os.environ["POSE_CODE"],source_tape_sha256=meta["tape_sha256"],
                      original_tape_length=len(original),steps=len(rows),reset_sha=meta["reset"]["initial_state_fingerprint"],
                      success=bool(step["success"]),native_return=float(arr["reward"].sum()),
                      terminated=bool(step["terminated"]),truncated=bool(step["truncated"]),data_sha256=digest(dest),
                      script_sha256=digest(Path(__file__)),affinity=sorted(os.sched_getaffinity(0)))
            if method=="raw":
                assert len(rows)==len(earlier["cmd"]) and np.array_equal(arr["applied"],earlier["cmd"])
                assert np.allclose(arr["end_p"],earlier["end_p"],rtol=0,atol=1e-12)
                assert np.allclose(arr["end_R"],earlier["end_R"],rtol=0,atol=1e-12)
                assert np.allclose(arr["end_pad"],earlier["end_pad"],rtol=0,atol=1e-12)
                assert np.array_equal(arr["reward"],earlier["native_reward"])
            if method=="sham":
                common=min(len(arr["delta"]),len(cases_out["adaptive"]["delta"]))
                assert np.array_equal(arr["delta"][:common],-cases_out["adaptive"]["delta"][:common])
                err=np.max(abs(arr["actual_delta"][:common]+cases_out["adaptive"]["actual_delta"][:common]))
                assert err < 1.2e-7
                info.update(matched_common=common,max_mirror_roundoff=float(err))
            save(dest.with_suffix(".json"),info)
            cases_out[method]=arr
            print(name,method,len(rows),info["native_return"],flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and set(os.sched_getaffinity(0))<={46,47}
    for spec in cases(args.arm):
        run(spec,args.out,Path(os.environ["POSE_REPLAY"]))
