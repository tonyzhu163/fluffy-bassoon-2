"""Read complete native forks and reduce correlated windows to paired task contrasts."""
import argparse
import csv
from itertools import product
from pathlib import Path

import numpy as np

from diagnostic import MODES, QUERIES, TARGETS, read, write


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def row_for(ref, raw, r, model):
    start = ref["trigger_start"]
    assert r["case"] == ref["case"] and r["scene"] == raw["scene"] == ref["scene"]
    for k in ("reset", "before_window", "prefix_xyz_hash", "prefix_actions_hash", "trigger_start"):
        assert r[k] == raw[k], k
    assert r["files"] == ref["files"] and r["n"] == len(r["rewards"]) == len(r["issued"])
    assert r["n"] <= r["horizon"] and r["native_return"] in (0,1)
    assert sum(r["rewards"]) == r["native_return"] and (not r["native_return"] or r["terminal"])
    assert r["n"] == r["tape_length"] or r["terminal"] or r["exhausted_horizon"]
    case = r["case"]
    z = np.asarray(r["xyz_m"])[:, 2]
    p = np.asarray(ref["xyz_m"])[:, 2]
    rawz = np.asarray(raw["xyz_m"])[:, 2]
    row = dict(suite=case["suite"], task=case["task_id"], layout=case["init_state_id"], seed=case["seed"],
               arm=r["arm"], mode=r["mode"], start_0based=start, controls=r["n"],
               native=int(r["native_return"]), native_horizon=r["horizon"], tape_censored=r["tape_censored"],
               terminal=r["terminal"], clipping=r["clip_count"], tape_length=r["tape_length"],
               panda_archive_max_abs_mm=ref["archive_panda_max_abs_mm"], panda_reset_matched=ref["archive_reset"]==ref["reset"])
    if start is None or min(r["n"],raw["n"],ref["n"])<start+10:
        return dict(**row, complete=False)
    sl = slice(start, start+10)
    vz, vp, vr = [1000*np.diff(v) for v in (z,p,rawz)]
    for name, q in [("panda",ref),("raw",raw),("branch",r)]:
        events = q["finger_contacts"][sl]
        row[name+"_contact_controls"] = sum(bool(h) for h in events)
        row[name+"_contact_geoms"] = "|".join(sorted({v for h in events for v in h}))
    corr = np.asarray(r["derivative"])
    window_u = np.asarray(r["issued"])[sl, 2]
    orig = np.asarray(raw["issued"])[sl, 2]
    pred = {"identity":vr[sl], "static":model["g0"]*vr[sl],
            "timing":model["g"]*vr[sl]+model["b"]*np.diff(vr[start-1:start+10])}
    acc = np.diff(vr[start-1:start+10]).reshape(2,5)
    pred["shifted"] = model["g"]*vr[sl]+model["b"]*np.roll(acc,2,axis=1).reshape(-1)
    return dict(**row, complete=True, v_rmse_mm=rms(vz[sl]-vp[sl]),
                anchored_z_rmse_mm=rms(1000*((z[start+1:start+11]-z[start])-(p[start+1:start+11]-p[start]))),
                prefix_z_gap_mm=float(1000*(z[start]-p[start])), forecast_rmse_mm=rms(pred[r["mode"]]-vp[sl]),
                changed_actions=int(np.sum(window_u!=orig)), z_change_rms=rms(window_u-orig), differential_rms=rms(corr))


def task_intervals(rows, key, arms=TARGETS, contacts=False):
    blocks, coverage = [], []
    for suite, t in QUERIES:
        block = []
        for arm in arms:
            for layout in (14,15):
                cell = [r for r in rows if (r["suite"],r["task"],r["arm"],r["layout"]) == (suite,t,arm,layout)]
                if key != "native" and not all(r["complete"] for r in cell):
                    continue
                if contacts and not all(r["complete"] and r["raw_contact_controls"] and r["panda_contact_controls"] for r in cell):
                    continue
                by = {r["mode"]:r[key] for r in cell}
                block.append([by[m] for m in MODES])
        coverage.append(len(block))
        if block:
            blocks.append(np.mean(block,axis=0))
    if not blocks:
        return dict(coverage=coverage)
    scores = np.asarray(blocks)
    stats = dict(coverage=coverage, task_condition_means=scores.tolist(), conditions=dict(zip(MODES,scores.mean(axis=0))))
    remap = np.asarray(list(product(range(len(scores)),repeat=len(scores))))
    stats["timing_minus"]={}
    for mi,m in enumerate(MODES):
        if m=="timing":
            continue
        delta = scores[:,2]-scores[:,mi]
        boot = delta[remap].mean(axis=1)
        stats["timing_minus"][m]=dict(mean=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),
                                      task_effects=delta.tolist())
    return stats


def score(base):
    root = base/"test"
    model = read(base/"fit/model.json")
    read(root/"completed.json")
    rows, donor = [], []
    for suite,t in QUERIES:
        for layout in (14,15):
            stem = f"test-{suite}-{t}-{layout}"
            p = read(root/f"{stem}-Panda-identity.json")
            archive = read(Path(next(iter(p["files"]))))
            positions = np.concatenate([c["xyz"] for c in archive["chunks"]])
            tick = p["trigger_start"]
            raw_z = np.asarray(p["xyz_m"])[1:,2]
            native_reward_equal = p["rewards"] == archive["record"]["native_rewards"]
            if tick is not None and tick + 10 <= min(len(positions),len(raw_z)):
                win = slice(tick - 1, tick + 10)
                donor_v = np.diff(positions[win,2])
                replay_v = np.diff(raw_z[win])
                window_reproduction = 1000 * rms(donor_v-replay_v)
            else:
                window_reproduction = None
            donor.append(dict(suite=suite,task=t,layout=layout, n=p["n"], tape_length=p["tape_length"],
                              donor_window_v_replay_rmse_mm=window_reproduction,
                              native_reward_tape_equal=native_reward_equal,
                              archive_native_return=archive["record"]["native_return"],
                              actual=p["native_return"], reset_match=p["archive_reset"]==p["reset"],
                              replay_mm=p["archive_panda_max_abs_mm"], files=p["files"]))
            for arm in TARGETS:
                group = [read(root / f"{stem}-{arm}-{m}.json") for m in MODES]
                raw, _, timing, shifted = group
                if timing["trigger_start"] is not None and min(timing["n"],shifted["n"])>=timing["trigger_start"]+10:
                    np.testing.assert_array_equal(np.roll(np.reshape(timing["derivative"],(2,5)),2,axis=1).ravel(),
                                                  shifted["derivative"])
                for r in group:
                    rows.append(row_for(p, raw, r, model))
    assert len(rows)==48 and len(donor)==6
    summary = {}
    for name, arm in (("pooled",TARGETS),*((a,(a,)) for a in TARGETS)):
        summary[name]={key:task_intervals(rows,key,arm) for key in ("v_rmse_mm","anchored_z_rmse_mm","forecast_rmse_mm","native")}
        summary[name]["contact_v_rmse_mm"] = task_intervals(rows, "v_rmse_mm", arm, contacts=True)
    def pairs(m):
        lookup={(r["suite"],r["task"],r["layout"],r["arm"],r["mode"]):r for r in rows}
        ref=[lookup[(*k[:-1],m)] for k in lookup if k[-1]=="timing"]
        cand=[r for k,r in lookup.items() if k[-1]=="timing"]
        return dict(gains=sum(b["native"]>a["native"] for a,b in zip(ref,cand)),
                    losses=sum(b["native"]<a["native"] for a,b in zip(ref,cand)))
    write(base/"scores.json",dict(model=model,donors=donor,summary=summary,native_timing_vs={m:pairs(m) for m in MODES if m!="timing"}))
    cols=list(dict.fromkeys(k for r in rows for k in r))
    with (base/"pairs.csv").open("x") as f:
        w=csv.DictWriter(f,cols)
        w.writeheader()
        w.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("base",type=Path)
    score(p.parse_args().base)
