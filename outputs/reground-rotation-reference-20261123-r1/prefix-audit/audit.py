"""Read native NPZs, CSVs, generation receipts and sealed CPU checkpoints. Run from audit folder."""
from collections import Counter, defaultdict
import csv
import gzip
from hashlib import sha1, sha256
import json
import os
from pathlib import Path
import sys

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
FIELDS = {"X": 11, "A": 10, "A_vla": 10, "speed": 10}
METHODS = ("base", "symmeta")
PINS = ("16a98486f560676b9036d1a6d067b9dd51a5917a", "7152f13e34c6c115d808aea0e5d08aca74eac14b")
STORE = {"base": "reference-native", "symmeta": "symmeta-native"}
READ = lambda path: json.loads(path.read_text())


def write(name, data):
    (OUT/name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(value):
    return sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def seed_for(i, r):
    return int(sha1(f"{i}:{r}".encode()).hexdigest()[:8], 16)


def key_of(row):
    return row["task"], int(row["init_id"]), int(row["rep"])


def read_arm(method, plan):
    records = {}
    for folder in sorted((ROOT/STORE[method]/"shards").glob(f"seed*/rot_z/90/{method}/shard*")):
        if not folder.is_dir():
            continue
        seed, shard = int(folder.parents[3].name[4:]), int(folder.name[5:])
        manifest = READ(folder/"manifest.json")
        assert (manifest["git_sha"], manifest["vla_revision"]) == PINS and not manifest["vla_compiled"]
        assert manifest["cell"] == "rot_z/90" and manifest["adapt_seed"] == seed
        with (folder/"results.csv").open() as f:
            rows = list(csv.DictReader(f))
        ordered = list(map(key_of, rows))
        assert ordered == plan[seed][16*shard:16*(shard+1)]
        costs = {} if method == "base" else {key_of(c): c for c in READ(folder/"calibration.json")}
        seen = []
        for batch, path in enumerate(sorted((folder/"traces").glob("*.npz"))):
            with np.load(path, allow_pickle=False) as raw:
                ks = [k.rsplit("__",1)[0] for k in raw.files if k.endswith("__A_vla")]
                batch_keys = [(t, int(i), int(r)) for t,i,r in (k.split("__") for k in ks)]
                assert batch_keys == ordered[4*batch:4*batch+4] and len(batch_keys) == 4
                seen.extend(batch_keys)
                for lane, (prefix, key) in enumerate(zip(ks, batch_keys)):
                    row = rows[4*batch+lane]
                    assert row["method"] == method and row["policy"] == "pi05" and row["shift"] == "rot_z"
                    assert float(row["level"]) == 90 and row["split"] == "test" and int(row["adapt_seed"]) == seed
                    noise = seed_for(key[1], key[2])
                    assert int(row["noise_seed"]) == noise
                    arrays = {name: raw[f"{prefix}__{name}"] for name in FIELDS}
                    n = int(row["steps"])
                    for name, a in arrays.items():
                        assert a.dtype == np.float32 and len(a) == n + (name == "X") and np.isfinite(a).all()
                    opening = {name: a[:FIELDS[name]].copy() for name,a in arrays.items()}
                    issued = opening["A_vla"].copy()
                    issued[:,:6] = np.clip(issued[:,:6], -1, 1)
                    assert digest(issued) == digest(opening["A"])
                    assert int(row["speed_viol"]) == int((arrays["speed"] > 0.27572885155677795).sum())
                    rec = dict(task=key[0],start=key[1],rep=key[2],seed=seed,shard=shard,batch=batch,
                               slot=lane,batch_keys=batch_keys,rank=int(manifest["argv"].split("--rank ")[1].split()[0]),
                               egl=manifest["egl_device"],host=manifest["host"],pid=manifest["pid"],shard_time=manifest["time"],
                               trace=str(path.relative_to(ROOT)),noise_seed=noise,
                               success=int(row["success"]),unsafe=int(row["unsafe"]),steps=n,
                               opening=opening,full={k:digest(a) for k,a in arrays.items()})
                    if costs:
                        c = costs[key]
                        assert c["adapt_seed"] == seed and c["noise_seed"] == noise and c["scene_seed"] == 0
                        assert (c["native_resets"],c["extra_resets"],c["settle_controls"],c["probe_controls"]) == (1,0,10,0)
                        assert c["offline_target_transitions"] == 0
                        q = c["queries"][0]
                        assert [t["start_control"] for t in c["queries"]] == list(range(0,n,10))
                        assert [t["chunk_seed"] for t in c["queries"]] == [seed_for(noise,k) for k in range((n+9)//10)]
                        assert q["estimate"] == [0,1,1] and q["policy_eef"] == q["public_eef"]
                        assert digest(np.array(q["public_eef"], np.float32)) == digest(opening["X"][0,:3])
                        u, v = c["updates"][:2]
                        assert (u["end_control"],v["end_control"]) == (5,10) and u["next_adapter"] == [0,1,1]
                        for j, update in enumerate((u,v)):
                            assert np.array_equal(opening["A"][5*j:5*j+5].mean(0),update["issued_mean"])
                        assert c["first_changed_control"] is None or c["first_changed_control"] >= 10
                        rec.update(q0=q, camera=u["before_sha256"], camera64=u["before_64_sha256"],
                                   reset_jaw=u["before_jaw_qpos"],first_changed=c["first_changed_control"],
                                   camera5=u["after_sha256"],camera10=v["after_sha256"],
                                   next_adapter5=u["next_adapter"],next_adapter10=v["next_adapter"])
                    assert (seed,*key) not in records
                    records[seed,*key] = rec
        assert seen == ordered
    assert len(records) == 1400
    return records


def diff(a,b):
    # Keep native float32 arithmetic for parity with original diagnostic. Use uint32 for strict bits.
    changed = (a.view(np.uint32) != b.view(np.uint32)).reshape(len(a),-1)
    idx = np.flatnonzero(changed.any(axis=1))
    if not len(idx):
        return None
    d = np.abs(a-b)
    i = int(idx[0])
    return dict(first=i, first_max=float(d[i].max()),max=float(d.max()), scalars=int(changed.sum()),
                per_index_max=d.reshape(len(a),-1).max(1).tolist())


def main():
    with gzip.open(ROOT/"src-23a6641562a4d833837f27ab0535a20ad94e8ec8/outputs"/ROOT.name/"evidence/rotation-jobs.json.gz","rt") as f:
        base_jobs = json.load(f)
    with gzip.open(ROOT/"src-cab855a03c8ff1a3f1926513369099bf11c4b859/outputs"/ROOT.name/"evidence/symmeta-jobs.json.gz","rt") as f:
        stream_jobs = json.load(f)
    assert len(base_jobs) == len(stream_jobs) == 5
    for a,b in zip(base_jobs,stream_jobs):
        assert {k:v for k,v in a.items() if k not in ("cfg","method")} == {k:v for k,v in b.items() if k not in ("cfg","method")}
    plan = {j["seed"]:sorted(map(tuple,j["episodes"])) for j in base_jobs}
    records = {m:read_arm(m, plan) for m in METHODS}
    base, stream = records.values()
    assert base.keys() == stream.keys()
    mismatches, all_rows, failures = [], [], set()
    for key,a in sorted(base.items()):
        s = stream[key]
        assert all(a[k] == s[k] for k in ("noise_seed","batch_keys","shard","batch","slot"))
        assert digest(a["opening"]["X"][0]) == digest(s["opening"]["X"][0])
        differences = {f:d for f in FIELDS if (d:=diff(a["opening"][f],s["opening"][f]))}
        if differences:
            failures.add(key)
            mismatches.append(dict(seed=key[0],task=key[1],start=key[2],rep=key[3],fields=differences,
                                   base_action0=a["opening"]["A_vla"][0].tolist(),stream_action0=s["opening"]["A_vla"][0].tolist(),
                                   base_speed0=float(a["opening"]["speed"][0]),stream_speed0=float(s["opening"]["speed"][0])))
        for m,r in zip(METHODS,(a,s)):
            all_rows.append(dict(method=m,**{k:v for k,v in r.items() if k!="opening"},
                                 reset_X=digest(r["opening"]["X"][0]),prefix={k:digest(v) for k,v in r["opening"].items()}))
    write("mismatches.json", mismatches)
    with gzip.open(OUT/"assignment-receipts.jsonl.gz","wt") as f:
        for r in all_rows:
            f.write(json.dumps(r, separators=(",",":"))+"\n")
    out = {}
    for method, grid in records.items():
        varying = defaultdict(list)
        pairs = Counter()
        for k in plan[0]:
            reps = [grid[(i,*k)] for i in range(5)]
            fields = {f:[digest(r["opening"][f]) for r in reps] for f in FIELDS}
            fields.update({f"full_{f}":[r["full"][f] for r in reps] for f in FIELDS})
            fields["reset_X"] = [digest(r["opening"]["X"][0]) for r in reps]
            if method == "symmeta":
                fields.update({f:[r[f] for r in reps] for f in ("camera","camera64","camera5","camera10")})
                fields.update({f:[r["q0"][f] for r in reps] for f in ("proposal_sha256","public_eef","policy_eef")})
            for f,values in fields.items():
                groups = defaultdict(list)
                for i,v in enumerate(values): groups[str(v)].append(i)
                if len(groups)>1:
                    varying[f].append(dict(task=k[0],start=k[1],groups=list(groups.values())))
                pairs[f] += sum(a != b for i,a in enumerate(values) for b in values[:i])
        out[method] = dict(varying=dict(varying),unequal_within_arm_pairs_of_2800=pairs)
    parent = READ(ROOT/"stream-prefix-diagnostics.json")
    canonical = {(r["seed"],r["task"],r["start"],r["field"]):(r["first"],r["first_abs_error"],r["max_abs_error"]) for r in parent}
    mine = {(r["seed"],r["task"],r["start"],f):(d["first"],d["first_max"],d["max"]) for r in mismatches for f,d in r["fields"].items()}
    assert canonical == mine
    write("self-comparisons.json",out)
    cases = {k[1:] for k in failures}
    write("failed-start-repetitions.json",[r for r in all_rows if (r["task"],r["start"],r["rep"]) in cases])
    # Cross-check all full published arrays for every lane in every affected native Base batch.
    sys.path.insert(0,str(ROOT/"upstream"))
    import torch
    checkpoints = []
    for seed,task,start,rep in sorted(failures):
        rec = base[(seed,task,start,rep)]
        sub = Path(rec["trace"]).parent.parent.relative_to("reference-native/shards")
        path = ROOT/"reference-native/batches"/sub/f"batch{rec['batch']:04}.pt"
        if str(path.relative_to(ROOT)) in [r["file"] for r in checkpoints]: continue
        saved = torch.load(path,map_location="cpu",weights_only=False)
        assert (saved["upstream"],saved["policy"]) == PINS
        assert saved["episodes"] == [(*k,0) for k in rec["batch_keys"]]
        for key,t in zip(rec["batch_keys"],saved["result"].traces,strict=True):
            assert key == (t["task"], t["init_id"],t["rep"])
            assert all(digest(t[f]) == base[seed,*key]["full"][f] for f in FIELDS)
        checkpoints.append(dict(file=str(path.relative_to(ROOT)),sha256=sha256(path.read_bytes()).hexdigest(),episodes=saved["episodes"],
                                verified_fields=list(FIELDS)))
    write("checkpoints.json",checkpoints)
    print("exact opening:",1400-len(failures),"of 1400. Parent diagnostic identical. Pre-publication Base batches:",len(checkpoints))
    for m,v in out.items():
        print(m,{f:len(g) for f,g in v["varying"].items()})

if __name__ == "__main__":
    main()
