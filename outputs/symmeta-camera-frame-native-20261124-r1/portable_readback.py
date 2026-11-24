"""Portable CPU readback of immutable native pairs. The producer checks exact actions in its own runtime."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def digest(b):
    return hashlib.sha256(b).hexdigest()


def fp(a):
    a = np.ascontiguousarray(a)
    return digest(repr((a.shape, a.dtype.str)).encode() + a.tobytes())


def check(value, reason):
    if not value:
        raise ValueError(reason)


def planar(values, degrees, factor, grip):
    # Explicit scalar channels, float64 references. Both original input arrays are float32.
    th = math.radians(degrees)
    cs,sn = math.cos(th),math.sin(th)
    v = np.asarray(values,dtype=np.float64)
    w = v.copy()
    w[:,0] = factor*(cs*v[:,0]-sn*v[:,1])
    w[:,1] = factor*(sn*v[:,0]+cs*v[:,1])
    w[:,2] *= factor
    w[:,6] *= grip
    return np.clip(w,-1,1)


def read_episode(r, name, obj):
    c, truth = r["corrections"][name], r["true_map"]
    rec = obj["native"]
    reset = rec["reset"]
    ref = r["reference"]
    horizon = ref["horizon"]
    n = rec["controls"]
    check(rec["inferred"] == c and rec["released_prompt"] == reset["prompt"] == r["prompt"], "decode or prompt")
    check(reset["horizon"] == horizon and 0 < n <= horizon and len(obj["chunks"]) == (n+4)//5, "duration")
    check(reset["arm"] == "UR5e" and reset["joint_count"] == 6 and reset["renderer_vendor"] == "NVIDIA Corporation"
          and reset["worker_step_index"] == 0 and reset["action_schema"] == r["action_schema"], "native setup")
    q = r["query"]
    env = obj["env"]
    check(all(env[k] == q[k] for k in ("suite","task_id","init_state_id","seed"))
          and env["released_row_fingerprint"] == ref["released_row_fingerprint"]
          and env["initial_state_fingerprint"] == reset["physics"] and reset["physics"] not in
          (r["support_physics"],r["calibration_physics"]), "released query")
    noise_seed = q["seed"]+817117
    noise = np.random.default_rng(noise_seed).standard_normal(((horizon+4)//5,10,32)).astype("f4")
    check(reset["noise_seed"]==noise_seed and reset["noise_sha256"]==fp(noise), "query noise")
    inverse = 1/float(np.clip(c["xy_scale"],0.25,4))
    forward = truth["xy_scale"]
    worst = [0.,0.]
    # Unit roundoff u. Per XY component there are at most two coefficient/gain
    # roundings, two products, one sum, and one scale operation. The input is
    # bounded by one and |sin|,|cos| <= 1, so the absolute dot terms sum to <= 2.
    # Clipping is nonexpansive. Test forward against the actual transmitted
    # float32 wire command to avoid accumulating the inverse residual twice.
    u = 2**-24
    gamma6 = 6*u/(1-6*u)
    limits = [2*gamma6*inverse,2*gamma6*forward]
    rewards = []
    for i, block in enumerate(obj["chunks"]):
        p = np.asarray(block["raw"],dtype="f4")
        check(p.shape==(10,7) and np.isfinite(p).all() and block["noise"]==fp(noise[i]), "proposal")
        if i==0:
            check(fp(p)==rec["opening_proposal_sha256"] and block["motor_inputs"]==reset["motor_frame0"], "opening")
        steps = block["steps"]
        count = min(5,n-5*i)
        wire = np.asarray(block["issued"],dtype=np.float32)
        actual = np.asarray([s["action"] for s in steps],dtype=np.float32)
        check(wire.shape==actual.shape==(count,7) and len(steps)==count, "control boundary shape")
        expected_wire = planar(np.clip(p[:count],-1,1),-c["yaw_deg"],inverse,c["gripper_sign"])
        expected_phys = planar(wire,truth["yaw_deg"],forward,truth["gripper_sign"])
        for j,(got,want) in enumerate(((wire,expected_wire),(actual,expected_phys))):
            residual=float(np.abs(got[:,:3]-want[:,:3]).max())
            check(np.array_equal(got[:,3:],want[:,3:]) and residual<=limits[j], "control decode outside float32 bound")
            worst[j]=max(worst[j],residual)
        for s in steps:
            rewards.append(s["reward"])
            check(s["index"] == len(rewards) and s["reward"] == int(s["success"]) and s["terminated"] is s["success"]
                  and s["truncated"] is (len(rewards)==horizon and not s["success"])
                  and (len(rewards)==n or not (s["terminated"] or s["truncated"])), "native predicates")
    check(all(rec[k] == steps[-1][k] for k in ("success","terminated","truncated"))
          and rec["truncated"] != rec["terminated"] and rec["native_rewards"] == rewards
          and rec["native_return"] == sum(rewards) == int(rec["success"]), "native reward and endpoint")
    return dict(controls=n,success=int(rec["success"]),queries=len(obj["chunks"]),max_action_error=worst)


def main(out,first):
    panel=(out/"panel.json").read_bytes()
    check(panel==(out.parent/"prepare01/panel.json").read_bytes(),"GPU/CPU panel mismatch")
    p=json.loads(panel)
    check(not p["cpu_fixture"] and len(p["rows"])==64, "production panel")
    rows=p["rows"][:1] if first else p["rows"]
    files={out/"pairs"/f"{r['index']:02d}.json.gz" for r in rows}
    if not first:
        check(files==set((out/"pairs").glob("*")), "incomplete or extra outcomes")
    receipts=[]
    for r in rows:
        path=out/"pairs"/f"{r['index']:02d}.json.gz"
        raw=path.read_bytes()
        record=json.loads(gzip.decompress(raw))
        arms=record["arms"]
        check(record["panel_sha256"]==digest(panel) and record["index"]==r["index"] and
              set(arms)==set(r["corrections"])=={"A","B","shipped","true_map"}, "arm coverage")
        anchor=arms["A"]
        for a in arms.values():
            check(a["native"]["reset"]==anchor["native"]["reset"] and a["body"]==anchor["body"]
                  and a["env"]==anchor["env"] and a["chunks"][0]["raw"]==anchor["chunks"][0]["raw"], "unpaired start")
        receipts.append(dict(index=r["index"],suite=r["suite"],task_id=r["task_id"],file_sha256=digest(raw),
                             reset_physics=anchor["native"]["reset"]["physics"],noise=anchor["native"]["reset"]["noise_sha256"],
                             opening=anchor["native"]["opening_proposal_sha256"],
                             arms={n:read_episode(r,n,a) for n,a in arms.items()}))
    result=dict(pin=p["pin"],panel_sha256=digest(panel),pairs=len(receipts),first_only=first,
                controls=sum(a["controls"] for r in receipts for a in r["arms"].values()),
                policy_calls=sum(a["queries"] for r in receipts for a in r["arms"].values()),
                max_portable_action_error=[max(a["max_action_error"][i] for r in receipts for a in r["arms"].values())
                                           for i in range(2)],rows=receipts)
    if not first:
        native=json.loads((out/"native-results.json").read_bytes())
        study=native["studies"]["physical"]
        counts=Counter()
        for r in receipts:
            counts.update({n:a["success"] for n,a in r["arms"].items()})
        check(dict(counts)==study["successes"],"success summary")
        wins=sum(r["arms"]["A"]["success"]>r["arms"]["B"]["success"] for r in receipts)
        losses=sum(r["arms"]["A"]["success"]<r["arms"]["B"]["success"] for r in receipts)
        check((wins,losses)==(study["a_only"],study["b_only"]), "discordant counts")
        official = native["paired_rows"]["physical"]
        effects=[]
        for t in sorted({(r["suite"],r["task_id"]) for r in receipts}):
            group=[r for r in receipts if (r["suite"],r["task_id"])==t]
            check(len(group)==8, "task panel coverage")
            effects.append(sum(r["arms"]["A"]["success"]-r["arms"]["B"]["success"] for r in group))
        check(len(effects)==8, "task cluster count")
        # Sum eight draws from the empirical eight-task distribution by integer convolution.
        counts_per_draw = np.bincount(np.asarray(effects)+8,minlength=17)
        hist=np.array([1],dtype=np.int64)
        for _ in effects:
            hist=np.convolve(hist,counts_per_draw)
        check(int(hist.sum())==8**8, "bootstrap mass")
        cdf=hist.cumsum()
        cuts=[(int(np.searchsorted(cdf,8**8*q))-64)*100/64 for q in (.025,.975)]
        check(cuts==study["task_cluster_ci95_pp"] and
              100*sum(effects)/64==study["task_weighted_success_gain_pp"],"primary task interval")
        check(len(official)==len(receipts) and all(
              r["index"]==e["index"] and {n:a["success"] for n,a in r["arms"].items()}==e["success"]
              for r,e in zip(receipts,official)), "native labels on matched rows")
        result.update(successes=dict(counts),a_only=wins,b_only=losses,task_interval_pp=cuts)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("out",type=Path)
    p.add_argument("--first",action="store_true")
    args=p.parse_args()
    main(args.out,args.first)
