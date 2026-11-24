"""Re-evaluate all stored joint solutions; report by tape, arm, case and physical event prefix."""
import argparse
import csv
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from arm_env import ArmEnv
from geometry import Geometry
from panel import cases, key


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def norm(x):
    return np.linalg.norm(x,axis=-1)


def put_stats(row,k,x):
    row.update({f"{k}_{suffix}":float(v) for suffix,v in (
        ("mean",np.mean(x)),("median",np.median(x)),("p95",np.quantile(x,.95)),("max",np.max(x)))})


def serialize(path,rows):
    with path.open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def readback(args):
    replay,solve,out=args.replay,args.solve,args.out
    tape_rows, stat_rows, replay_rows, donor_rows=[],[],[],[]
    largest_fk_diff=0.
    inputs={}
    for model_arm in ("Kinova3","Panda"):
        for spec in cases(model_arm):
            case=key(spec)
            with ArmEnv(model_arm,spec["suite"],spec["task_id"]) as env:
                env.reset(spec["init_state_id"], spec["seed"])
                g=Geometry(env._env)
                for tape_arm in ("Kinova3","Panda"):
                    name=f"{case}-{tape_arm}"
                    rp=replay / f"{name}.npz"
                    sp=solve / f"{name}-on-{model_arm}.npz"
                    m=json.loads(rp.with_suffix(".json").read_bytes())
                    s=json.loads(sp.with_suffix(".json").read_bytes())
                    assert sha(rp)==m["data_sha256"]==s["replay_sha256"] and sha(sp)==s["data_sha256"]
                    inputs[sp.name]=s["data_sha256"]
                    d=np.load(rp,allow_pickle=False)
                    f=np.load(sp,allow_pickle=False)
                    n=len(d["p"])
                    same=model_arm==tape_arm
                    pin_local=np.einsum("nji,nj->ni",d["R"], d["pad"]-d["p"])
                    fields=["static_full","static_free"]+(["rate_full","rate_free","rate_matched"] if same else [])
                    pinch={}
                    for field in fields:
                        q=f[field+"_q"]
                        assert np.all(q>=g.lo-1e-10) and np.all(q<=g.hi+1e-10)
                        if field.startswith("rate"):
                            assert np.all(np.abs(q-d["q"]) <= s["dt_s"]*np.array(s["speed_rad_s"])+1e-10)
                        p_err=[]
                        a_err=[]
                        pad_err=[]
                        for i,qi in enumerate(q):
                            g.d.qpos[g.adr]=qi
                            mujoco.mj_kinematics(g.m,g.d)
                            rot=g.d.site_xmat[g.site].reshape(3,3)
                            pos=g.d.site_xpos[g.site]
                            p_err.append(norm(pos-d["goal_p"][i]))
                            a_err.append(Rotation.from_matrix(rot.T@d["goal_R"][i]).magnitude())
                            pad_err.append(norm(pos+rot@pin_local[i]-d["goal_pad"][i]))
                        diff=np.max(abs(np.column_stack((p_err,a_err))-f[field]))
                        assert diff < 1e-9
                        largest_fk_diff=max(largest_fk_diff,float(diff))
                        pinch[field]=np.array(pad_err)
                    accepted=(f["static_full"][:,0]<=.001)&(f["static_full"][:,1]<=.01)
                    excluded=f["chain_lb_at_0p01_m"]>.001
                    assert not np.any(excluded & accepted)
                    static=dict(tape=name, model=model_arm, replay_controls=n, witnesses_reset_convention=int(sum(accepted)),
                                excluded_reset_convention=int(sum(excluded)),unresolved=int(sum(~accepted & ~excluded)),
                                first_excluded=int(np.argmax(excluded)+1) if np.any(excluded) else "",
                                last_excluded=int(np.flatnonzero(excluded)[-1]+1) if np.any(excluded) else "",
                                max_lower_bound_mm=float(1e3*max(f["chain_lb_at_0p01_m"])))
                    for name_field in ["static_full","static_free"]:
                        put_stats(static,name_field+"_mm",1000*f[name_field][:,0])
                        put_stats(static,name_field+"_deg",np.rad2deg(f[name_field][:,1]))
                        put_stats(static,name_field+"_pad_mm",1000*pinch[name_field])
                    tape_rows.append(static)
                    if not same:
                        continue
                    obs_delta=d["recorded_tcp_delta_m"]
                    f_delta=d["recorded_finger_max_delta_m"]
                    r_delta=d["recorded_chunk_angle_rad"]
                    drift=(obs_delta>.001)|(f_delta>.001)|(r_delta>.01)
                    actual_angle=Rotation.from_matrix(d["end_R"]@d["R"].transpose(0,2,1)).magnitude()
                    goal_angle=Rotation.from_matrix(d["goal_R"]@d["R"].transpose(0,2,1)).magnitude()
                    eef_err=norm(d["end_p"]-d["goal_p"])
                    ang_err=norm(f["native_error"][:,3:])
                    replay_row=dict(tape=name,source_steps=m["native_controls"],controls=n,source_success=m["recorded_success"],
                        replay_success=m["success"],terminal_flags_equal=m["terminal_flags_equal"],
                        reward_tape_equal=m["reward_tape_equal"],first_divergence=int(np.argmax(drift)+1) if drift.any() else "",
                        tcp_drift_max_mm=1000*max(obs_delta),finger_drift_max_mm=1000*max(f_delta),
                        boundary_hand_drift_max_deg=np.rad2deg(np.nanmax(r_delta)),reconstructed_p_max_m=max(d["pos_goal_reconstruction_m"]),
                        reconstructed_R_max_rad=max(d["ori_goal_reconstruction_rad"]),jacobian_max_diff_scaled=s["jac_max_diff_scaled"],
                        settled_hash_equal=m["reset"]["initial_state_fingerprint"]==m["recorded_settled_hash"],
                        held_orientation_controls=int(np.sum(np.all(d["cmd"][:,3:6]==0,axis=1))),
                        max_public_joint_vs_boundary_rad=np.max(abs(d["public_q"]-d["q"])),
                        max_public_tcp_vs_boundary_mm=np.max(norm(d["public_p"]-d["p"]))*1000)
                    replay_rows.append(replay_row)
                    groups=[("whole",n)]
                    if spec["suite"]=="libero_goal" and spec["task_id"]==3:
                        groups.extend([("first_interaction_prefix",123 if spec["init_state_id"]==14 else 67)])
                        if spec["init_state_id"]==15:
                            groups.append(("shared_trajectory_prefix",117))
                        k=spec["init_state_id"]
                        demo_path=Path(__file__).resolve().parents[1]/f"frozen-vla-reachability-20261123-r1/evidence/goal3-{k}.json.gz"
                        demo=json.loads(gzip.decompress(demo_path.read_bytes()))["demonstrations"][0]
                        tgt=np.array(next(w["goal"] for w in demo["waypoints"] if w["phase"]=="last_touch" and "bowl" in w["body"]))
                        distance=norm(d["goal_p"]-tgt[:3,3])
                        angular=Rotation.from_matrix(d["goal_R"]@tgt[:3,:3].T).magnitude()
                        i=distance.argmin()
                        donor_rows.append(dict(tape=name,donor=demo["demo"],closest_step=i+1,world_distance_mm=1000*distance[i],
                                               angular_difference_deg=np.rad2deg(angular[i]),
                                               closest_goal_xyz_m=json.dumps(d["goal_p"][i].tolist()),
                                               staged_donor_xyz_m=json.dumps(tgt[:3,3].tolist())))
                    for group,stop in groups:
                        idx=slice(0,stop)
                        full=f["rate_full"][idx]
                        free=f["rate_free"][idx]
                        match=f["rate_matched"][idx]
                        native=ang_err[idx]
                        gain=np.maximum(full[:,0]-free[:,0],0)
                        matched_gain=np.maximum(match[:,0]-free[:,0],0)
                        maximum=int(np.argmax(gain))
                        row=dict(tape=name,window=group,controls=stop,step_max_rate_position_gain=maximum+1,
                                 full_exact_attitude_count=int(np.sum(full[:,1]<=1e-6)),
                                 matched_angle_valid_count=int(np.sum(match[:,1]<=native+1e-6)),
                                 rate_gain_at_peak_mm=1000*gain[maximum],tcp_tracking_at_peak_mm=1000*eef_err[maximum],
                                 free_wrist_cost_at_peak_deg=np.rad2deg(free[maximum,1]),
                                 goal_pad_at_peak_m=json.dumps(d["goal_pad"][maximum].tolist()),
                                 actual_pad_at_peak_m=json.dumps(d["end_pad"][maximum].tolist()),
                                 frac_native_joint_travel_over_reference=np.mean(f["native_max_speed_fraction"][idx]>1),
                                 min_compiled_margin_rad=min(f["native_joint_margin_rad"][idx]))
                        for name_field,series in {
                            "requested_step_mm":1000*norm(d["goal_p"]-d["p"]), "requested_step_deg":np.rad2deg(goal_angle),
                            "actual_step_mm":1000*norm(d["end_p"]-d["p"]), "actual_step_deg":np.rad2deg(actual_angle),
                            "tracking_mm":1000*eef_err,"tracking_deg":np.rad2deg(ang_err),
                            "pinch_tracking_mm":1000*norm(d["goal_pad"]-d["end_pad"]),
                            "static_full_mm":1000*f["static_full"][:,0],"rate_full_mm":1000*f["rate_full"][:,0],
                            "rate_free_mm":1000*f["rate_free"][:,0],"rate_matched_mm":1000*f["rate_matched"][:,0],
                            "rate_full_deg":np.rad2deg(f["rate_full"][:,1]), "rate_free_deg":np.rad2deg(f["rate_free"][:,1]),
                            "free_pinchoffset_mm":1000*pinch["rate_free"], "full_pinchoffset_mm":1000*pinch["rate_full"],
                            "static_linear_prediction_mm":1000*norm(f["static_linear_and_fk"][:,:3]),
                            "static_linear_witness_fk_mm":1000*norm(f["static_linear_and_fk"][:,6:9]),
                            "rate_linear_prediction_mm":1000*norm(f["rate_linear_and_fk"][:,:3]),
                            "rate_linear_witness_fk_mm":1000*norm(f["rate_linear_and_fk"][:,6:9]),
                        }.items():
                            put_stats(row,name_field,series[idx])
                        put_stats(row,"rate_full_to_free_gain_mm",1000*gain)
                        put_stats(row,"rate_matched_to_free_gain_mm",1000*matched_gain)
                        stat_rows.append(row)
    serialize(out/"static.csv",tape_rows)
    serialize(out/"tracking.csv",stat_rows)
    serialize(out/"qualification.csv",replay_rows)
    serialize(out/"donor-distance.csv",donor_rows)
    receipt=dict(script_sha256=sha(Path(__file__)),source_revision=subprocess.check_output(
        ["git","-C",os.environ["POSE_CODE"],"rev-parse","HEAD"],text=True).strip(),fk_repeat_max_abs=largest_fk_diff,
        input_solution_sha256=inputs,affinity=sorted(os.sched_getaffinity(0)),cuda_visible=os.environ["CUDA_VISIBLE_DEVICES"])
    (out/"readback.json").write_text(json.dumps(receipt,indent=2)+"\n")


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for k in ("replay","solve","out"):
        ap.add_argument(k,type=Path)
    args=ap.parse_args()
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and set(os.sched_getaffinity(0))<={46,47}
    readback(args)
