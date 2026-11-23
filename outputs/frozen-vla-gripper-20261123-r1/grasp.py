"""Source geometry bracket using the public hand frame and the actual first closing proposal."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from libero_remote.environment.libero import LiberoEnvironmentAdapter

import archive
from arm_env import ArmEnv, CAMERAS
from native import Instrument, SOURCE

MODES = ("identity", "hold", "toward_finger1", "toward_finger2")
PAIRS = [
    (SOURCE, "c264a0db031867c7cf62461feba68be6d73a5e5459f1deb9b4ffc584e1379928",
     "d5900f6c7ab12bfb3619cecf4e466d30271dcf99ac6f9b53c26432238d6b714e"),
    (SOURCE.parents[1]/"libero_10-3-r1-BASE/default", "9e4bfbb704578f5a1b6959fdf1a5509cb8287bcb80e689823da5a3bb357faa9e",
     "d004e19c47cd51a19220f78ca679b6f30b52a09b3ae69557674ee322ec050800"),
]


def public_axis(quat):
    # Both signed slides run along hand-local +Y, checked against Panda CAD below.
    return Rotation.from_quat(quat).as_matrix()[:,1]


def grasp_goal(mode, tcp, quat, q):
    sign = {"toward_finger1":1., "toward_finger2":-1., "hold":0.}[mode]
    return np.array(tcp) + sign * .25 * (q[0]-q[1]) * public_axis(quat)


def hold_action(goal, tcp, last_grip):
    a = np.zeros(7, dtype=np.float32)
    a[:3] = np.clip((goal-np.asarray(tcp))/.05, -1, 1)
    a[-1] = last_grip
    return a


def digest(x):
    return hashlib.sha256(str((x.shape,x.dtype.str)).encode()+x.tobytes()).hexdigest()


def release_check(session, first, case, orig):
    released = LiberoEnvironmentAdapter()
    try:
        fresh = released.create(dict(**case, _bddl_path=str(session.bddl), camera_names=list(CAMERAS),
                                    camera_height=256,camera_width=256,control_mode="relative",
                                    max_steps=first["max_steps"]))
        current = {k:digest(v) for k,v in first["observation"].items()}
        independent = {k:digest(v) for k,v in fresh["observation"].items()}
        return dict(current_crossarm=first["environment_reproducibility"]["initial_state_fingerprint"],
                    current_release=fresh["environment_reproducibility"]["initial_state_fingerprint"],
                    original=orig["identity"]["physics"],
                    current_wrapper_vs_release_public={k:current[k]==independent[k] for k in current},
                    current_release_vs_archived_public={k:independent[k]==orig["identity"]["public"][k] for k in current})
    finally:
        released.close()


def branch(out, arm, mode, orig, commands, ref_positions):
    case, obs = orig["case"], None
    layout = case["init_state_id"]
    with ArmEnv(arm,case["suite"],case["task_id"]) as s:
        first = s.reset(layout,case["seed"])
        parity = release_check(s, first, case, orig) if arm=="Panda" and mode=="identity" else None
        reset = first["environment_reproducibility"]
        meter = Instrument(s)
        # Audit using compiled kinematics on Panda only. Never supply a target axis oracle to the adapter.
        if arm=="Panda":
            meter.body_offsets()
            np.testing.assert_allclose(meter.scratch.xmat[meter.hand].reshape(3,3).T @ meter.scratch.xaxis[meter.joints[0]],
                                       [0,1,0], atol=1e-12)
        o = first["observation"]
        obs = dict(tcp=o["robot0_eef_pos"], q=o["robot0_gripper_qpos"], quat=o["robot0_eef_quat_xyzw"])
        rows, native_rewards = [], []
        tape_i, trigger = 0, None
        last_grip = -1.
        def issue(a):
            nonlocal obs
            r = meter.execute(a)
            obs = dict(tcp=r["public_tcp"],q=r["q"],
                       quat=meter.public_quat)
            native_rewards.append(r["reward"])
            rows.append(dict(a=a.tolist(), tcp=obs["tcp"], center=r["center"], q=r["q"],
                             contacts=r["contacts"], success=r["success"], done=r["done"], reward=r["reward"]))
            return r["done"] or r["success"] or len(rows)>=first["max_steps"]
        while tape_i<len(commands):
            proposed = np.clip(commands[tape_i],-1,1).astype(np.float32)
            if trigger is None and proposed[6]>0:
                assert tape_i>0 and last_grip<0
                trigger = dict(at=tape_i, prefix_hash=digest(s._env.get_sim_state()),
                               prefix_rows_sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                               axis=public_axis(obs["quat"]).tolist(), start_center=rows[-1]["center"],
                               opening_mm=1000*float(np.subtract(*obs["q"])))
                if mode!="identity":
                    goal=grasp_goal(mode,obs["tcp"],obs["quat"],obs["q"])
                    for _ in range(5):
                        if issue(hold_action(goal,obs["tcp"],last_grip)):
                            break
                    trigger["goal"]=goal.tolist()
                trigger["before_close_center"]=rows[-1]["center"]
                if rows[-1]["done"] or rows[-1]["success"] or len(rows)>=first["max_steps"]:
                    break
            tape_i+=1
            last_grip=float(proposed[-1])
            if issue(proposed):
                break
        assert trigger is not None
        err = None
        if arm=="Panda" and mode=="identity":
            err=float(np.abs(np.array([r["tcp"] for r in rows])-ref_positions[:len(rows)]).max())
            assert bool(rows[-1]["success"])==orig["success"] and len(rows)==orig["controls"]
        record=dict(arm=arm,case=case,mode=mode,trigger=trigger,source_parity=parity,
                    max_source_abs_public_tcp_error_m=err, reset=reset["initial_state_fingerprint"],
                    scene=reset["scene_fingerprint"],success=bool(rows[-1]["success"]),native_return=sum(native_rewards),
                    controls=len(rows),tape_commands=tape_i,horizon=first["max_steps"],rows=rows)
        archive.write(out/f"fork-{arm}-{layout}-{mode}.json",record)
        return record


def main(out):
    summaries=[]
    for folder, tape_hash, meta_hash in PAIRS:
        meta, tape=folder/"result.json",folder/"public.npz"
        assert (archive.sha(tape),archive.sha(meta))==(tape_hash,meta_hash)
        orig=json.loads(meta.read_bytes())
        with np.load(tape) as f:
            commands,positions,rewards=[f[k].copy() for k in ("issued","positions","rewards")]
        for arm in ("Panda","Sawyer"):
            rec={mode:branch(out,arm,mode,orig,commands,positions) for mode in MODES}
            baseline=rec["identity"]
            assert all(baseline[k]==r[k] for r in rec.values() for k in ("scene","reset"))
            assert all(baseline["trigger"][k]==r["trigger"][k] for r in rec.values()
                       for k in ("at","prefix_hash","prefix_rows_sha256"))
            if arm=="Panda":
                np.testing.assert_array_equal([r["reward"] for r in baseline["rows"]],rewards)
            stationary=rec["hold"]["trigger"]
            cell=dict(arm=arm,layout=orig["case"]["init_state_id"],seed=orig["case"]["seed"],
                      tape_sha256=tape_hash,meta_sha256=meta_hash,source_parity=baseline["source_parity"],
                      max_source_abs_public_tcp_error_m=baseline["max_source_abs_public_tcp_error_m"],
                      trigger_control=stationary["at"]+1,opening_mm=stationary["opening_mm"],modes={})
            for mode,r in rec.items():
                tr=r["trigger"]
                diff=np.subtract(tr["before_close_center"],stationary["before_close_center"])
                cell["modes"][mode]=dict(native_success=r["success"],native_return=r["native_return"],controls=r["controls"],
                     axis_shift_vs_hold_mm=1000*float(diff@np.array(tr["axis"])),
                     first_bowl_contact=next((i+1 for i,row in enumerate(r["rows"])
                                    if any("akita_black_bowl" in b for a,b in row["contacts"])),None))
            summaries.append(cell)
            print(json.dumps(cell),flush=True)
    scores={m:sum(s["modes"][m]["native_success"] for s in summaries) for m in MODES}
    winners=[m for m in MODES[2:] if scores[m]>max(scores[k] for k in MODES if k!=m)]
    archive.write(out/"fork-summary.json",dict(source=summaries,scores=scores,selected=winners[0] if winners else "identity"))
    print("source scores",scores,"advance",winners,flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]))
