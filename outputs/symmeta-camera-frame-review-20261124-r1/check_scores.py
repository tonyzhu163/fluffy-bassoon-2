"""Read scored rows, rebuild input identities and compute paired whole-task estimates."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
C = 31.5
SEEDS = (41, 42, 43)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def digest(arr):
    return sha(np.ascontiguousarray(arr).tobytes())


def signed(v):
    return (v + 180) % 360 - 180


def absolute(row):
    return abs(signed(row["yaw_deg"] - row["true_yaw_deg"]))


def load(path):
    return json.loads(path.read_bytes())


def read_tape(path, require_counts=True):
    with np.load(path, allow_pickle=False) as z:
        before, after = z["img_before"][:3], z["img_after"][:3]
        assert np.array_equal(before[0], z["frame0_img"])
        assert np.array_equal(after[:2], before[1:])
        assert z["valid"][:3].all() and not z["valid"][3:].any()
        if require_counts or "chunk_steps" in z:
            assert list(z["chunk_steps"][:3]) == [5,5,5]
        else:
            # The legacy official archive records three valid slots; the collector fixes CHUNK_SIZE=5.
            assert list(z["valid"]) == [True,True,True]
        controls = {k: z[k][:3] for k in ("issued_mean","delta_eef","delta_jaw","valid")}
    return np.concatenate((before[:1], after)), controls


def digital_observation(f, view):
    angle = int(view[4:]) if view.startswith("roll") else 0
    if view.startswith("plane"):
        # Recover registered synthetic plane transform, using source positive focal lengths.
        # Keep independent construction and tolerate floating point expression order on H.
        orbit = int(view[5:])
        def plane(a):
            a = np.deg2rad(a)
            center = np.array([np.cos(a), np.sin(a), 1.])/np.sqrt(2)
            forward = -center
            right = np.cross(forward, [0,0,1])
            right /= np.linalg.norm(right)
            rot = np.stack([right, np.cross(forward,right), forward])
            focal = 32/np.tan(np.pi / 8)
            k = np.array([[focal,0,C],[0,focal,C],[0,0,1.]])
            return k @ np.column_stack((rot[:,:2],-rot@center))
        h = np.eye(3) @ plane(orbit) @ np.linalg.inv(plane(0))
    else:
        h = np.row_stack((cv2.getRotationMatrix2D((C,C),angle,1.), [0,0,1]))
    h = np.eye(3) @ h
    h /= h[2,2]
    if not view.startswith("plane") and angle % 90 == 0:
        return np.ascontiguousarray(np.rot90(f, angle//90, axes=(1,2))), h
    def warp(x, border):
        return cv2.warpPerspective(x, h, (64,64), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=border)
    valid = warp(np.ones((64,64), dtype="f4"), 0) >= 1.
    observed = np.stack([warp(x, (127,127,127)) for x in f])
    observed[:,~valid] = 127
    return observed, h


def paired(rows):
    effects = defaultdict(lambda: defaultdict(list))
    groups = defaultdict(dict)
    all_rows = {(r["index"],r["view"],r["seed"],r["model"],r["calibration"]):r for r in rows}
    assert len(rows) == len(all_rows)
    for r in rows:
        i,v,s,m,cal = (r[k] for k in ("index","view","seed","model","calibration"))
        if cal == "true":
            groups[i,v,s][m] = r
        if cal == "identity_token":
            assert all(r[k] == all_rows[i,"nominal",s,"A","true"][k] for k in ("yaw_deg","xy_scale","gripper_sign"))
    for (i,v,s), arms in groups.items():
        a,b = arms["A"],arms["B"]
        for name in ("observed_sha256","calibration_sha256","native_view_sha256","source_sha256","true_yaw_deg","true_gain","true_sign","suite","task_id","embodiment"):
            assert len({r[name] for r in arms.values()}) == 1
        if a["panel"] == "digital" and a["regime"] == "familiar":
            continue
        notes = [v]
        if a["panel"] == "digital" and v in ("roll+20","roll-20","roll+45","roll-45"):
            notes += ["four_primary_rolls"]
        if a["panel"] == "physical" and v in ("orbit+","orbit-"):
            notes += ["both_orbits"]
        for group in notes:
            task = a["suite"] + "/" + str(a["task_id"])
            key = (task,s)
            stats = {"A_minus_B":absolute(a)-absolute(b), "A":absolute(a), "B":absolute(b),
                     "shipped":absolute(arms["shipped"])}
            if "shipped_corrected" in arms:
                stats["shipped_corrected"] = absolute(arms["shipped_corrected"])
            for arm in ("A","B"):
                r = arms[arm]
                stats[arm+"_cost_vs_nominal"] = absolute(r)-absolute(all_rows[i,"nominal",s,arm,"true"])
                if (i,v,s,arm,"opposite") in all_rows:
                    wrong = all_rows[i,v,s,arm,"opposite"]
                    assert r["observed_sha256"]==wrong["observed_sha256"]
                    assert r["calibration_sha256"]!=wrong["calibration_sha256"]
                    stats[arm+"_wrong_cost"] = absolute(wrong)-absolute(r)
            for name, value in stats.items():
                effects[group,name][key].append(value)
    out = {}
    for (group,name),cells in effects.items():
        tasks = sorted({k[0] for k in cells})
        means = np.array([[np.mean(cells[t,s]) for s in SEEDS] for t in tasks])
        tm = means.mean(axis=1)
        rng = np.random.default_rng(1124)
        sample = rng.choice(tm, (20000,len(tm)), replace=True).mean(axis=1)
        value = dict(mean=float(tm.mean()), interval=np.quantile(sample,[.025,.975]).tolist())
        if name=="A_minus_B":
            value.update(per_task=dict(zip(tasks,tm.tolist())), per_seed=means.mean(axis=0).tolist())
        out.setdefault(group,{})[name]=value
    return out


def check(pred, bank_path):
    with gzip.open(pred,"rt") as f:
        rows = json.load(f)
    physical = bank_path is not None
    bank = {} if not physical else {r["index"]:r for r in load(bank_path)["rows"]
                                    if (r["arm"],r["role"]) == ("UR5e","heldout")}
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["index"]].append(r)
    if physical:
        assert set(by_id)==set(bank)
    else:
        assert set(by_id)==set(range(96))
    distinct = set()
    raw_views = 0
    h_checked = 0
    for i, cases in by_id.items():
        first = cases[0]
        if physical:
            item = bank[i]
            ref = Path(item["views"]["nominal"]["passive_npz"])
        else:
            ref = ROOT / f"libero-xemb-ur5e-scale-20261120-r1/gpu09test/test_live/shard-{i//12}/support/{i%12:02d}-{first['embodiment']}-passive.npz"
        orig, controls = read_tape(ref, require_counts=physical)
        assert len({r["source_sha256"] for r in cases})==1
        source_sha = first["source_sha256"]
        if physical:
            assert item["archive_sha256"] == source_sha
            assert all((r["suite"],r["task_id"],r["embodiment"],r["true_yaw_deg"],r["true_gain"],r["true_sign"]) ==
                       (item["suite"],item["task_id"],item["native_control_id"],item["control_yaw_deg"],item["xy_scale"],item["gripper_sign"]) for r in cases)
        else:
            assert sha(ref.read_bytes())==source_sha
        shard = i//12
        release = "gpu04s2agoal6" if shard==3 else "gpu03s2a"
        official = load(ROOT/f"xemb-icml-campaign-20261121-r1/{release}/live/shard-{shard}/live-shard.json")["rows"][i%12]
        for r in cases:
            assert (r["suite"],r["task_id"],r["embodiment"],r["regime"])==tuple(official[k] for k in ("suite","task_id","embodiment","regime"))
            np.testing.assert_allclose((r["true_yaw_deg"],r["true_gain"],r["true_sign"]),
                                       [official[k] for k in ("true_yaw","true_scale","true_grip")],rtol=1e-7)
        views = defaultdict(list)
        for r in cases:
            views[r["view"]].append(r)
            distinct.add((r["model"],r["seed"],r["checkpoint_sha256"]))
        for v, cells in views.items():
            if physical:
                mv = item["views"][v]
                observed, alt = read_tape(Path(mv["passive_npz"]))
                with np.load(mv["passive_npz"],allow_pickle=False) as z:
                    assert np.float32(item["plane_z_m"])==z["frame0_state"][2]
                assert all(np.array_equal(controls[k],alt[k]) for k in controls)
                truth = np.asarray(mv["H_ref_to_view"])
                true_hash = sha(Path(mv["passive_npz"]).read_bytes())
                oppname = v if v=="nominal" else v[:-1]+("+" if v.endswith("-") else "-")
                other = np.asarray(item["views"][oppname]["H_ref_to_view"])
            else:
                observed,truth = digital_observation(orig,v)
                true_hash = source_sha
                alt_name = v.replace("+","-") if "+" in v else v.replace("-","+")
                _,other = digital_observation(orig,alt_name)
            obs_hash = digest(observed)
            assert all(r["observed_sha256"] == obs_hash and r["native_view_sha256"]==true_hash for r in cells), (i,v)
            for r in cells:
                expected = other if r["calibration"]=="opposite" else truth
                assert r["calibration_sha256"]==digest(expected), (i,v,r["calibration"])
                h_checked += 1
            for s in SEEDS:
                for arm in ("A","B"):
                    assert sum((r["model"],r["seed"],r["calibration"])==(arm,s,"true") for r in cells)==1
            raw_views += 1
    result=paired(rows)
    published=load(pred.with_name(("physical" if physical else "digital")+"-summary.json"))
    comparison=published["all" if physical else "ood"]["views"]
    agreements={}
    for view,vals in result.items():
        group="primary_nonquarter" if view=="four_primary_rolls" else view
        if group not in comparison:
            continue
        other=comparison[group]["A_minus_B"]
        v=vals["A_minus_B"]
        np.testing.assert_allclose([v["mean"],*v["interval"]],[other["mean"],*other["ci95"]],rtol=1e-9,atol=1e-9)
        agreements[group]=True
    return dict(agrees_with_reported_paired_intervals=agreements,prediction_sha256=sha(pred.read_bytes()), scored_rows=len(rows), base_supports=len(by_id),
                observed_views_verified=raw_views, calibrated_hashes_verified=h_checked,
                checkpoints=sorted(distinct), comparison=result)


if __name__=="__main__":
    arg=argparse.ArgumentParser()
    arg.add_argument("predictions",type=Path)
    arg.add_argument("--bank",type=Path)
    opt=arg.parse_args()
    cv2.setNumThreads(1)
    print(json.dumps(check(opt.predictions,opt.bank),indent=2))
