"""Consume a released Kinova3 full-head panel in a fresh, parent-owned native job."""
import argparse
import gzip
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

import numpy as np

from bank import CASES_SHA, HERE, SOURCE, cases, sha
NATIVE = HERE.parent/"symmeta-camera-frame-native-20261124-r1"
sys.path.insert(0,str(NATIVE))
from protocol import MANIFEST_SHA, RELEASE_PIN, atomic, checked, digest, encode, parse, release_rows
from native import ObservedQuery
from paired_inputs import native
from score import commands, fp
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
import ur5e_sim_worker

ARMS = ("A","B","frozen_warp","shipped","true_map")
HEADS = ("yaw_deg","xy_scale","gripper_sign")


def released_sites(release, release_sha, resets_path, resets_sha):
    packet = parse(checked(release,release_sha))
    bank = parse(checked(Path(packet["bank"]),packet["bank_sha256"]))
    assert bank["cases_sha256"] == packet["cases_sha256"] == CASES_SHA and packet["seed"] == 41
    n = len(packet["rows"])
    cases_by_id = {c["index"]:c for c in cases() if n==64 or c["slot"]>=8}
    physical = {r["index"]:r for r in bank["rows"]}
    indices = sorted(cases_by_id)
    assert n in (32,64) and sorted(physical)==indices==[r["index"] for r in packet["rows"]]
    pred = parse(gzip.decompress(checked(Path(packet["predictions"]),packet["predictions_sha256"])))
    p = {(r["index"],r["model"]):r for r in pred if r["seed"]==41 and r["calibration"]=="true"}
    assert len(pred)==n*7*3 and len(p)==4*n
    fit = Path(packet["fit"])
    seal = parse(checked(fit/"sealed.json",packet["seal_sha256"]))
    assert seal["steps"]==packet["selected_step"] and packet["source_fit_revision"]==seal["revision"]
    assert seal["steps"]==int(min(seal["source_B_mean"],key=seal["source_B_mean"].get))
    hist = HERE.parent/"symmeta-perceiver-retry-20261123-r1/evidence/train02pro/checkpoints.sha256"
    shipped_sha, = (x.split()[0] for x in hist.read_text().splitlines() if x.endswith("seed41.msgpack"))
    hashes = packet["checkpoint_sha256"]
    for model in ARMS[:-1]:
        if model in ("A","B"):
            name = f"{model}-seed41-step{seal['steps']}.msgpack"
            assert hashes[model] == seal["checkpoint_sha256"][name] == sha(fit/name)
        else:
            assert hashes[model] == shipped_sha
    refs = release_rows()
    resets = parse(checked(resets_path,resets_sha))
    by_reset = {(r["shard"],r["reset"]["init_state_id"]):r for r in resets}
    assert len(by_reset)==len(resets)==16
    sites = []
    for r in packet["rows"]:
        i = r["index"]
        c, b = cases_by_id[i], physical[i]
        assert all(r[k]==v for k,v in c.items())
        assert set(r["corrections"])==set(ARMS) and r["calibration_physics"]==b["physics"]
        assert c["support_sha256"]==b["archive_sha256"]==r["source_sha256"]==sha(b["archive_npz"])
        key = (r["suite"],r["task_id"],r["query_layout"])
        ref = refs[key]
        qr = by_reset[r["shard"],r["query_layout"]]
        q = qr["reset"]
        assert (q["suite"],q["task_id"],q["init_state_id"])==key and q["seed"]==r["query_seed"]
        assert q["released_row_fingerprint"]==ref["released_row_fingerprint"] and q["bddl_sha256"]==ref["bddl_sha256"]
        assert qr["model"]["arm"]=="Kinova3" and qr["model"]["joint_count"]==7 and ref["horizon"]==qr["horizon"]
        spec = get_embodiment_spec(r["embodiment"])
        truth = dict(yaw_deg=spec.yaw_deg,xy_scale=spec.xy_scale,gripper_sign=spec.gripper_sign)
        assert r["corrections"]["true_map"]==truth
        v = b["views"][b["assigned_view"]]
        assert v["sha256"]==sha(v["passive_npz"]) and v["angle_deg"]==r["angle_deg"]
        with np.load(v["passive_npz"]) as z:
            obs_hash = digest(np.concatenate([z["frame0_img"][None],z["img_after"]]).tobytes())
            sensors = {k:digest(z[k].tobytes()) for k in ("issued_mean","delta_eef","delta_jaw","valid")}
        h_hash = digest(np.asarray(v["H_ref_to_view"],np.float64).tobytes())
        for model in ARMS[:-1]:
            m = p[i,model]
            assert m["source_sha256"]==r["support_sha256"] and m["native_view_sha256"]==v["sha256"]
            assert m["observed_sha256"]==r["observed_sha256"]==obs_hash and m["sensor_sha256"]==sensors
            assert m["view"]==b["assigned_view"] and m["calibration_sha256"]==r["calibration_sha256"]==h_hash
            assert m["checkpoint_sha256"]==hashes[model] and r["corrections"][model]=={k:m[k] for k in HEADS}
            assert np.isfinite(m["yaw_deg"]) and np.isfinite(m["xy_scale"]) and m["xy_scale"]>0 and m["gripper_sign"] in (-1,1)
        query = dict(arm="Kinova3",suite=r["suite"],task_id=r["task_id"],variant="BASE",
                     init_state_id=r["query_layout"],seed=r["query_seed"])
        sites.append(dict(r,query=query,reference=ref,model=qr["model"],prompt=qr["prompt"],true_map=truth))
    return sites


def audited_episode(ev, sim, motor, row, model, resets):
    c = row["corrections"][model]
    interface = ObservedQuery(sim,motor,ev,row,c)
    spec = get_embodiment_spec(row["embodiment"])
    rec = ev.run_paired_query(ev.run_episode, interface, interface, row["query"], spec, arm_name=model,
              key=row["index"], resets=resets, expected_prompt=row["prompt"],support_physics=row["calibration_physics"],
              create_attempts=3, yaw_hat=c["yaw_deg"],scale_hat=c["xy_scale"],grip_hat=c["gripper_sign"])
    first = interface.first
    env, reset = first["environment_reproducibility"],rec["reset"]
    assert first["model_identity"]==row["model"] and reset["arm"]=="Kinova3" and reset["joint_count"]==7
    assert reset["worker_step_index"]==0 and reset["renderer_vendor"]=="NVIDIA Corporation"
    assert env["source_pin"]==RELEASE_PIN and env["manifest_sha256"]==MANIFEST_SHA
    assert env["perturbation_plan"]["case_id"]=="BASE" and env["seed"]==row["query_seed"] and env["init_state_id"]==row["query_layout"]
    assert all(env[k]==row["reference"][k] for k in ("bddl_sha256","released_row_fingerprint"))
    assert env["initial_state_fingerprint"]==reset["physics"] not in (row["calibration_physics"],row["physics"])
    assert rec["inferred"]==c and reset["prompt"]==rec["released_prompt"]==row["prompt"]
    horizon, n = reset["horizon"],rec["controls"]
    assert horizon==row["reference"]["horizon"] and 0<n<=horizon and reset["noise_seed"]==row["noise_seed"]
    noise = np.random.default_rng(row["noise_seed"]).standard_normal(((horizon+4)//5,10,32)).astype(np.float32)
    assert reset["noise_sha256"]==fp(noise) and len(interface.chunks)==(n+4)//5
    rewards = []
    for j, block in enumerate(interface.chunks):
        raw = np.asarray(block["raw"],np.float32)
        assert raw.shape==(10,7) and np.isfinite(raw).all() and block["noise"]==fp(noise[j])
        if j==0:
            assert fp(raw)==rec["opening_proposal_sha256"] and block["motor_inputs"]==reset["motor_frame0"]
        steps = block["steps"]
        count = min(5,n-j*5)
        issued, actual = commands(raw[:5],c,row["true_map"])
        np.testing.assert_array_equal(issued[:count],np.asarray(block["issued"],np.float32))
        np.testing.assert_array_equal(actual[:count],np.asarray([s["action"] for s in steps],np.float32))
        assert len(steps)==count
        for step in steps:
            rewards.append(step["reward"])
            assert step["index"]==len(rewards) and type(step["success"]) is bool and step["reward"]==int(step["success"])
            assert step["terminated"] is step["success"] and step["truncated"] is (len(rewards)==horizon and not step["success"])
            assert len(rewards)==n or not (step["terminated"] or step["truncated"])
    assert all(steps[-1][k]==rec[k] for k in ("success","terminated","truncated"))
    assert rec["terminated"] != rec["truncated"] and rewards==rec["native_rewards"] and sum(rewards)==rec["native_return"]==int(rec["success"])
    return dict(native=rec,env=env,body=first["model_identity"],chunks=interface.chunks)


def summarize(sites, records):
    assert len(records)==len(sites) and len({r["index"] for r in records})==len(sites)
    def metric(group):
        a = np.array([[r["success"][m] for m in ARMS] for r in group],int)
        return dict(starts=len(group), successes=dict(zip(ARMS,a.sum(axis=0).tolist())),
                    A_only=int(sum(a[:,0]>a[:,1])),B_only=int(sum(a[:,1]>a[:,0])))
    tasks = sorted({r["shard"] for r in records})
    grouped = [[r for r in records if r["shard"]==t] for t in tasks]
    by_task = [metric(g) for g in grouped]
    rates = np.array([[v["successes"][m]/v["starts"] for m in ARMS] for v in by_task])
    draws = np.random.default_rng(1124).integers(len(tasks),size=(20000,len(tasks)))
    contrasts = {}
    for j,m in enumerate(ARMS[:2]):
        for k,other in enumerate(ARMS):
            if j==k:
                continue
            delta = 100*(rates[:,j]-rates[:,k])
            contrasts[f"{m} minus {other}"] = dict(pp=float(delta.mean()),per_task_pp=delta.tolist(),
                              ci95_pp=np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist())
    def corr(x,y):
        return float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else None
    yaw_reduction = [np.mean([r["yaw_error_B_minus_A"] for r in g]) for g in grouped]
    return dict(**metric(records), task_order=[f"{g[0]['suite']}/{g[0]['task_id']}" for g in grouped],
                per_task=by_task,paired_contrasts=contrasts,correlation_task_A_B=corr(rates[:,0],rates[:,1]),
                correlation_task_yaw_reduction_success_gain=corr(yaw_reduction,rates[:,0]-rates[:,1]),
                strata={k:{str(v):metric([r for r in records if r[k]==v]) for v in {r[k] for r in records}}
                        for k in ("regime","angle_deg")}, paired_rows=records)


def main(a):
    sites = released_sites(a.release,a.release_sha,a.resets,a.resets_sha)
    atomic(a.out/"sites.json",encode(sites))
    if a.mode=="prepare":
        return
    import jax
    import ur5e_motor
    from paired_inputs import check_tree
    from bank import SOURCE_PIN
    check_tree(SOURCE,SOURCE_PIN)
    ev = native()
    cfg = parse((HERE.parent/"symmeta-fk-crossarm-20261123-r1/QUERY_CONFIG.json").read_bytes())["panels"]["gpu06kinova3"]
    assert cfg["source_pin"]==SOURCE_PIN and ur5e_motor.OPENPI_CKPT==Path(cfg["checkpoint"])
    checked(ur5e_motor.OPENPI_CKPT/"transport-manifest.json",cfg["weight_transport_manifest_sha256"])
    ready = parse(checked(ur5e_motor.OPENPI_CKPT/"READY.json","ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170"))
    assert ready["status"]=="verified" and ready["checkpoint_path"]==str(ur5e_motor.OPENPI_CKPT)
    gpu = parse((a.out/"gpu.json").read_bytes())
    assert gpu["hostname"]==socket.gethostname() and gpu["status"]=="passed" and gpu["purpose"]=="evaluation"
    assert len(gpu["devices"])==1 and gpu["allocation_type"]=="baige_container" and "RTX 4090" in gpu["devices"][0]["cuda_name"].upper()
    assert len(jax.devices())==1 and jax.devices()[0].platform=="gpu"
    motor = ev.load_frozen_policy("pi05")
    assert (motor._model.action_horizon,motor._model.action_dim)==(10,32)
    with patch.dict(os.environ,{"CUDA_VISIBLE_DEVICES":""}), patch.object(ur5e_sim_worker,"__file__",str(NATIVE/"audited_worker.py")):
        sim = ev.UR5eSimulator()
    records = []
    with sim:
        previous = None
        for r in sites:
            if r["shard"] != previous:
                sim.call("create",dict(r["query"],embodiment=r["embodiment"]))
                previous = r["shard"]
            order = list(ARMS)
            if (r["shard"]+r["slot"])%2:
                order[0],order[1] = order[1],order[0]
            arms, resets = {}, {}
            for name in order:
                obj = arms[name] = audited_episode(ev,sim,motor,r,name,resets)
                ref = arms[order[0]]
                assert obj["native"]["reset"]==ref["native"]["reset"] and obj["env"]==ref["env"] and obj["chunks"][0]["raw"]==ref["chunks"][0]["raw"]
            path = a.out/"pairs"/f"{r['index']:02d}.json.gz"
            data = dict(release_sha256=a.release_sha,index=r["index"],execution_order=order,arms=arms)
            atomic(path,gzip.compress(encode(data),compresslevel=4,mtime=0))
            wrap = lambda m:abs((r["corrections"][m]["yaw_deg"]-r["true_map"]["yaw_deg"]+180)%360-180)
            records.append(dict(**{k:r[k] for k in ("shard","index","suite","task_id","regime","angle_deg")},
                         success={m:arms[m]["native"]["success"] for m in ARMS},pair_sha256=sha(path),
                         yaw_error_B_minus_A=wrap("B")-wrap("A")))
            print("verified complete five-arm set",r["index"],flush=True)
    atomic(a.out/"native-results.json",encode(summarize(sites,records)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=("prepare","gpu"))
    p.add_argument("out",type=Path)
    p.add_argument("release",type=Path)
    p.add_argument("release_sha")
    p.add_argument("resets",type=Path)
    p.add_argument("resets_sha")
    main(p.parse_args())
