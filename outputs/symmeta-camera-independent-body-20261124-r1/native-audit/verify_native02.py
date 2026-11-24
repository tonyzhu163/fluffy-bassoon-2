"""Read original Baige Kinova3 native02 tapes. Run with python -B - first|final on core 37."""
import csv
import gzip
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

BASE = Path("/mowenjie/fluffy-bassoon-2")
STUDY = "symmeta-camera-independent-body-20261124-r1"
STORE = BASE / "outputs" / STUDY
NATIVE = STORE / "native02"
PIN = "d2b307bc4ca8c0cf723b3aaba31755d14a2f8303"
SOURCE = BASE / "setups/symmeta-perceiver-followups-20261123-r1/src-d2b307bc4ca8"
RELEASE = "50e320040e06c70d2b0924f14afc24f0608e5951bfc7bcec37f5781384c7820f"
RESETS = "cb4309ddea85347bf06798cfd8156222a827f94bc8a0d06153ed157f2fa69ec4"
CASES = "7a7f704bab9cc9d33ce6c1fbeb7e9722f22ed72553c67cac8303582a22769795"
ARMS = ("A", "B", "frozen_warp", "shipped", "true_map")
HEADS = ("yaw_deg", "xy_scale", "gripper_sign")
FIELDS = ("observation/image", "observation/wrist_image", "observation/state")
PAIRS = (("A","B"), ("frozen_warp","B"), ("A","frozen_warp"), ("A","shipped"),
         ("B","shipped"), ("A","true_map"), ("B","true_map"), ("true_map","shipped"))
sha = lambda b: hashlib.sha256(b).hexdigest()


def read(path, expected=None):
    path = Path(path)
    raw = path.read_bytes()
    if expected:
        assert sha(raw) == expected, path
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def fingerprint(a):
    return sha(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def rotation_transpose(yaw):
    ang = math.radians(yaw)
    c, s = math.cos(ang), math.sin(ang)
    return np.array([[c, s], [-s, c]], dtype=np.float32)


def controls(raw, estimate, truth):
    x = np.clip(raw[:5], -1, 1)
    gain_inv = np.float32(1 / min(4, max(0.25, estimate["xy_scale"])))
    inv = x.copy()
    inv[:, :2] = (x[:, :2] @ rotation_transpose(-estimate["yaw_deg"])) * gain_inv
    inv[:, 2] *= gain_inv
    inv[:, 6] *= estimate["gripper_sign"]
    issued = np.clip(inv, -1, 1)
    actual = issued.copy()
    rot = rotation_transpose(truth["yaw_deg"])
    for i, v in enumerate(issued):
        actual[i, :2] = (v[:2] * np.float32(truth["xy_scale"])) @ rot
    actual[:, 2] *= np.float32(truth["xy_scale"])
    actual[:, 6] *= truth["gripper_sign"]
    return issued, np.clip(actual, -1, 1)


def check_support(row, bank, predictions, ckpts):
    angle = row["angle_deg"]
    view = "orbit+" if angle > 0 else "orbit-"
    assert abs(angle) == 36 and bank["assigned_view"] == view
    v = bank["views"][view]
    assert v["angle_deg"] == angle and bank["arm"] == "Kinova3"
    assert bank["archive_sha256"] == row["source_sha256"] == row["support_sha256"] == sha(Path(bank["archive_npz"]).read_bytes())
    assert bank["source_collector_pin"] == "465f33dc657ceab341dbceaf6b98355b94810059"
    assert bank["physics"] == row["calibration_physics"] and bank["input_physics"] == row["physics"]
    assert bank["init_state_id"] == row["support_layout"] == 11 and bank["seed"] == row["support_seed"]
    assert bank["native_control_id"] == row["embodiment"]
    true = dict(zip(HEADS, (bank["control_yaw_deg"], bank["xy_scale"], bank["gripper_sign"])))
    assert row["corrections"]["true_map"] == true
    assert sha(Path(v["passive_npz"]).read_bytes()) == v["sha256"]
    with np.load(v["passive_npz"], allow_pickle=False) as z:
        frames = np.concatenate((z["frame0_img"][None], z["img_after"]))
        assert frames.shape == (4,64,64,3) and np.array_equal(frames[:-1],z["img_before"])
        assert int(z["joint_count"]) == 7 and z["delta_joint"].shape == (3,7)
        assert sha(frames.tobytes()) == row["observed_sha256"]
        sensors = {k:sha(z[k].tobytes()) for k in ("issued_mean","delta_eef","delta_jaw","valid")}
    assert sha(np.asarray(v["H_ref_to_view"],np.float64).tobytes()) == row["calibration_sha256"]
    models = {m:predictions[row["index"], m] for m in ARMS[:-1]}
    for name, pred in models.items():
        assert all(pred[k] == row[k] for k in ("suite","task_id","angle_deg","embodiment","regime","observed_sha256","calibration_sha256","source_sha256"))
        assert pred["checkpoint_sha256"] == ckpts[name] and pred["view"] == view
        assert pred["sensor_sha256"] == sensors and pred["native_view_sha256"] == v["sha256"]
        assert row["corrections"][name] == {k:pred[k] for k in HEADS}
        assert pred["xy_scale"] > 0 and pred["gripper_sign"] in (-1,1)
    for a,b in (("A","frozen_warp"), ("B","shipped")):
        for k in ("img_before","img_after",*sensors):
            assert models[a]["feature_sha256"][k] == models[b]["feature_sha256"][k]


def check_group(row, cpu_ref, first):
    i = row["index"]
    path = NATIVE / "pairs" / f"{i:02}.json.gz"
    raw_bytes = path.read_bytes()
    pair = json.loads(gzip.decompress(raw_bytes))
    order = list(ARMS)
    if (row["shard"] + row["slot"]) % 2:
        order[0], order[1] = order[1], order[0]
    assert (pair["index"], pair["release_sha256"], pair["execution_order"]) == (i, RELEASE, order)
    arms = pair["arms"]
    assert set(arms) == set(ARMS)
    opening = arms["A"]["native"]["reset"]
    assert opening["noise_seed"] == row["noise_seed"] == row["query_seed"] + 817117
    horizon = cpu_ref["horizon"]
    noise = np.random.default_rng(row["noise_seed"]).standard_normal(((horizon+4)//5,10,32)).astype("f4")
    assert fingerprint(noise) == opening["noise_sha256"]
    true = row["corrections"]["true_map"]
    result = {k:row[k] for k in ("index","suite","task_id","shard","slot","regime","angle_deg","query_layout")}
    result.update(tape_sha256=sha(raw_bytes), physics=opening["physics"], scene=opening["scene_fingerprint"],
                  opening_proposal_sha256=arms["A"]["native"]["opening_proposal_sha256"],
                  cpu_exact_physics=(opening["physics"] == cpu_ref["reset"]["initial_state_fingerprint"]),
                  score01_seed41_yaw_B_minus_A_deg=0, arms={})
    if first:
        result.update(file=str(path), motor_inputs=opening["motor_frame0"], noise_sha256=opening["noise_sha256"],
                      corrections=row["corrections"], joint_names=arms["A"]["body"]["joint_names"])
    for name in ARMS:
        ep = arms[name]
        n, env, body = ep["native"], ep["env"], ep["body"]
        assert n["reset"] == opening and env == arms["A"]["env"] and body == cpu_ref["model"]
        assert len(set(body["joint_names"])) == len(body["joint_links"]) == body["joint_count"] == opening["joint_count"] == 7
        assert body["joint_names"] == [f"robot0_Actuator{x}" for x in range(1,8)]
        assert body["model_class"] == opening["model_class"] == "MountedKinova3" and body["arm"] == opening["arm"] == "Kinova3"
        assert len(env["alignment"]["qpos"]) == 7 and body["cameras"] == ["agentview","robot0_eye_in_hand"]
        assert {"robot0_joint_pos","robot0_joint_vel","agentview_image","robot0_eye_in_hand_image"} <= set(opening["public"])
        assert opening["renderer_vendor"] == "NVIDIA Corporation" and opening["worker_step_index"] == 0
        assert env["initial_state_fingerprint"] == opening["physics"] and opening["physics"] not in (row["physics"],row["calibration_physics"])
        assert env["scene_fingerprint"] == opening["scene_fingerprint"] and env["perturbation_plan"]["case_id"] == "BASE"
        for k in ("suite","task_id","seed","init_state_id","source_pin","manifest_sha256","bddl_sha256","released_row_fingerprint","scene_fingerprint","perturbation_plan_fingerprint"):
            assert env[k] == cpu_ref["reset"][k], (i,name,k)
        assert env["seed"] == row["query_seed"] and env["init_state_id"] == row["query_layout"] and env["suite"] == row["suite"] and env["task_id"] == row["task_id"]
        assert opening["horizon"] == horizon and n["released_prompt"] == opening["prompt"] == cpu_ref["prompt"]
        assert n["inferred"] == row["corrections"][name]
        chunks = ep["chunks"]
        steps = n["controls"]
        assert 0 < steps <= horizon and len(chunks) == (steps+4)//5
        reward_series = []
        for j, chunk in enumerate(chunks):
            raw = np.asarray(chunk["raw"],np.float32)
            assert set(chunk["motor_inputs"]) == set(FIELDS) and all(len(h)==64 for h in chunk["motor_inputs"].values())
            assert np.isfinite(raw).all() and raw.shape == (10,7) and chunk["noise"] == fingerprint(noise[j])
            if j == 0:
                assert chunk["motor_inputs"] == opening["motor_frame0"]
                assert np.array_equal(raw, np.asarray(arms["A"]["chunks"][0]["raw"],np.float32))
                assert fingerprint(raw) == n["opening_proposal_sha256"] == result["opening_proposal_sha256"]
            issued, physical = controls(raw, n["inferred"], true)
            count = min(5, steps - 5*j)
            receipts = chunk["steps"]
            assert len(receipts) == count
            assert np.array_equal(issued[:count], np.asarray(chunk["issued"],np.float32)), (i,name,j,"issued")
            assert np.array_equal(physical[:count], np.asarray([s["action"] for s in receipts],np.float32)), (i,name,j,"env")
            for receipt in receipts:
                t = len(reward_series)+1
                assert receipt["index"] == t
                assert type(receipt["success"]) is bool and receipt["reward"] == int(receipt["success"])
                assert receipt["terminated"] is receipt["success"]
                assert receipt["truncated"] is (t == horizon and not receipt["success"])
                assert (receipt["terminated"] or receipt["truncated"]) == (t == steps)
                reward_series.append(receipt["reward"])
        last = chunks[-1]["steps"][-1]
        assert all(n[k] is last[k] for k in ("terminated","truncated","success"))
        assert n["native_rewards"] == reward_series and sum(reward_series) == n["native_return"] == int(n["success"])
        a = result["arms"][name] = dict(success=n["success"],controls=steps,calls=len(chunks))
        if first:
            c0 = chunks[0]
            a.update(first_raw=c0["raw"][0],first_issued=c0["issued"][0],first_physical=c0["steps"][0]["action"],
                     native_return=n["native_return"],terminated=n["terminated"],truncated=n["truncated"])
    yaw = lambda m: abs((row["corrections"][m]["yaw_deg"]-true["yaw_deg"]+180)%360-180)
    result["score01_seed41_yaw_B_minus_A_deg"] = yaw("B")-yaw("A")
    first_step = {m:arms[m]["chunks"][0]["steps"][0]["action"] for m in ARMS}
    result["first_physical_action_changed"] = {f"{a} minus {b}":first_step[a] != first_step[b] for a,b in PAIRS}
    return result


def main(mode):
    assert mode in ("first","final","intact") and os.sched_getaffinity(0)=={37} and os.environ["CUDA_VISIBLE_DEVICES"]==""
    cases_path = SOURCE / "outputs" / STUDY / "cases.csv"
    assert sha(cases_path.read_bytes()) == CASES
    with cases_path.open() as f:
        contract = {int(c["index"]):c for c in csv.DictReader(f)}
    launch = (NATIVE / "launch.txt").read_text().split()
    assert launch[:4] == [PIN, "job-nn97ohkc3wf8-master-0", PIN, "native02"]
    assert launch[4:8] == [str(STORE/"score01/full-head-release.json"), RELEASE, str(STORE/"cpu01/query-resets.json"), RESETS]
    release = read(launch[4], RELEASE)
    assert release["seed"] == 41 and release["cases_sha256"] == CASES and release["selected_step"] == 5000
    assert release["bank"] == str(STORE/"cpu01/bank64.json") and release["predictions"] == str(STORE/"score01/predictions.json.gz")
    assert release["bank_sha256"] == "8639817ce59b79ecfe26564e94c0bb61352a3a5942b13a5c8edd999a6c5d6e60"
    seal = read(Path(release["fit"])/"sealed.json",release["seal_sha256"])
    assert seal["steps"] == release["selected_step"] and seal["revision"] == release["source_fit_revision"]
    predictions = read(release["predictions"], release["predictions_sha256"])
    chosen = {(r["index"],r["model"]):r for r in predictions if r["seed"] == 41 and r["calibration"] == "true"}
    assert len({(r["index"],r["model"],r["seed"],r["calibration"]) for r in predictions})==len(predictions)==1344 and len(chosen)==256
    ckpts = release["checkpoint_sha256"]
    assert ckpts["frozen_warp"] == ckpts["shipped"]
    for m in ARMS[:-1]:
        p = Path(chosen[4,m]["checkpoint_path"])
        assert sha(p.read_bytes()) == ckpts[m]
        if m in ("A","B"):
            assert p.name == f"{m}-seed41-step5000.msgpack" and seal["checkpoint_sha256"][p.name] == ckpts[m]
    b = read(release["bank"],release["bank_sha256"])
    assert b["cases_sha256"] == CASES
    bank = {r["index"]:r for r in b["rows"]}
    rows = {r["index"]:r for r in release["rows"]}
    cpu = {(r["shard"],r["reset"]["init_state_id"]):r for r in read(launch[6], RESETS)}
    assert len(cpu)==16 and len(contract)==len(bank)==len(rows)==64 and contract.keys()==bank.keys()==rows.keys()
    gpu = read(NATIVE/"gpu.json")
    assert gpu["hostname"] == launch[1] and gpu["purpose"] == "evaluation" and gpu["status"] == "passed"
    assert len(gpu["devices"]) == 1 and "RTX 4090" in gpu["devices"][0]["cuda_name"]
    paths = sorted((NATIVE/"pairs").glob("*.json.gz"),key=lambda p:p.stat().st_mtime_ns)
    if mode == "first":
        paths = paths[:1]
        assert paths[0].name == "04.json.gz"
    elif mode == "final":
        assert (NATIVE/"exit.txt").read_text().split()[0]=="0"
        assert {p.name for p in paths} == {f"{i:02}.json.gz" for i in contract} and len(paths)==64
    checked = []
    for path in sorted(paths):
        i = int(path.name.split(".")[0])
        row = rows[i]
        assert all(str(row[k]) == v for k,v in contract[i].items())
        check_support(row, bank[i], chosen, ckpts)
        checked.append(check_group(row, cpu[row["shard"],row["query_layout"]],mode=="first"))
    report = dict(mode=mode, checked_at=datetime.now().astimezone().isoformat(), audit_host=os.uname().nodename,
                  cpu=37, numpy=np.__version__, blas=os.getenv("OPENBLAS_CORETYPE","auto"), pin=PIN, job=launch[1],
                  source=str(NATIVE), release=RELEASE, resets=RESETS, cases=CASES, score01_seed=41,
                  checkpoint_sha256=ckpts, groups=len(checked), episodes=5*len(checked), records=checked)
    if mode=="final":
        report.update(summarize(checked))
        yaw_report = read(STORE/"score01/offline.json")["all"]
        arm_yaw = {}
        for m in ARMS[:-1]:
            by_seed = {}
            for seed in (41,42,43):
                preds = [r for r in predictions if r["seed"]==seed and r["calibration"]=="true" and r["model"]==m]
                assert len(preds)==len(rows)
                errors = [abs((p["yaw_deg"]-rows[p["index"]]["corrections"]["true_map"]["yaw_deg"]+180)%360-180) for p in preds]
                by_seed[seed] = float(np.mean(errors))
                assert math.isclose(by_seed[seed],yaw_report[f"{m}/true"]["yaw_mae_deg"]["per_seed"][str(seed)],abs_tol=1e-12)
            arm_yaw[m] = dict(per_seed=by_seed,mean_deg=float(np.mean(list(by_seed.values()))))
        report["score01_offline_yaw_mae_deg"] = arm_yaw
    print(json.dumps(report,indent=2,allow_nan=False))


def summarize(rows):
    def counts(rs):
        return {m:sum(r["arms"][m]["success"] for r in rs) for m in ARMS}

    def discord(rs, a, b):
        outcomes = Counter((r["arms"][a]["success"],r["arms"][b]["success"]) for r in rs)
        return dict(left_only=outcomes[True,False], right_only=outcomes[False,True],
                    both=outcomes[True,True], neither=outcomes[False,False])

    tasks = sorted({r["shard"] for r in rows})
    groups = [[r for r in rows if r["shard"]==t] for t in tasks]
    assert len(groups)==8 and all(len(g)==8 for g in groups)
    assert all(len({(r["suite"],r["task_id"]) for r in g})==1 for g in groups)
    assert all(len({r["query_layout"] for r in g})==2 for g in groups)
    assert all(sorted(Counter((r["angle_deg"],r["regime"]) for r in g).values())==[2]*4 for g in groups)
    draws = np.random.default_rng(1124).integers(8,size=(20000,8))
    task_rows = [dict(task=f"{g[0]['suite']}/{g[0]['task_id']}", successes=counts(g),
                      score01_seed41_yaw_B_minus_A_deg=float(np.mean([r["score01_seed41_yaw_B_minus_A_deg"] for r in g])),
                      discordants={}) for g in groups]
    comparisons = {}
    official = read(NATIVE/"native-results.json")
    assert official["starts"] == len(rows) and official["successes"] == counts(rows)
    assert official["task_order"] == [r["task"] for r in task_rows]
    published = {r["index"]:r for r in official["paired_rows"]}
    assert len(published)==len(official["paired_rows"])==len(rows)
    for r in rows:
        p = published[r["index"]]
        assert p["pair_sha256"] == r["tape_sha256"] and p["success"] == {m:r["arms"][m]["success"] for m in ARMS}
        assert p["yaw_error_B_minus_A"] == r["score01_seed41_yaw_B_minus_A_deg"]
    for a,b in PAIRS:
        key = f"{a} minus {b}"
        effect = []
        for g,t in zip(groups,task_rows):
            d = discord(g,a,b)
            delta = 100*(d["left_only"]-d["right_only"])/len(g)
            t["discordants"][key] = dict(**d,pp=delta)
            effect.append(delta)
        arr = np.asarray(effect)
        ci = np.quantile(arr[draws].mean(axis=1),[.025,.975]).tolist()
        comparisons[key] = dict(**discord(rows,a,b),pp=float(arr.mean()),ci95_pp=ci,
                                task_directions=[int(np.sign(delta)) for delta in effect],
                                first_physical_action_changed=sum(r["first_physical_action_changed"][key] for r in rows))
        for match,sign in ((key,1),(f"{b} minus {a}",-1)):
            off = official["paired_contrasts"].get(match)
            if off:
                assert np.array_equal(arr,sign*np.array(off["per_task_pp"]))
                assert np.array_equal(ci,off["ci95_pp"] if sign==1 else [-v for v in off["ci95_pp"][::-1]])
    rates = np.asarray([[t["successes"][m]/8 for m in ("A","B")] for t in task_rows])
    corr = lambda x,y: float(np.corrcoef(x,y)[0,1]) if np.ptp(x)>0 and np.ptp(y)>0 else None
    return dict(successes=counts(rows),comparisons=comparisons,tasks=task_rows,
                controls=sum(r["arms"][m]["controls"] for r in rows for m in ARMS),
                policy_calls=sum(r["arms"][m]["calls"] for r in rows for m in ARMS),
                cpu01_exact_physics_groups=sum(r["cpu_exact_physics"] for r in rows),
                native_A_B_task_rate_r=corr(rates[:,0],rates[:,1]),
                score01_seed41_yaw_vs_native02_task_r=corr([t["score01_seed41_yaw_B_minus_A_deg"] for t in task_rows],rates[:,0]-rates[:,1]),
                by_stratum={k:{str(v):counts([r for r in rows if r[k]==v]) for v in sorted({r[k] for r in rows})} for k in ("regime","angle_deg")},
                native_result_sha256=sha((NATIVE/"native-results.json").read_bytes()))

if __name__=="__main__":
    main(sys.argv[1])
