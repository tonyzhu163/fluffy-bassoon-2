"""Join CPU audit with actual per-process batch history and hardware receipts."""
from collections import defaultdict, Counter
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
read = lambda path: json.loads(path.read_text())

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2)+"\n")


def main():
    rows = [json.loads(l) for l in gzip.open(OUT/"assignment-receipts.jsonl.gz","rt")]
    failed = {(x["seed"],x["task"],x["start"]) for x in read(OUT/"mismatches.json")}
    key = lambda x:(x["seed"],x["task"],x["start"])
    arm_summary={}
    histories=[]
    repair=[]
    for method,sub,attempt in [("base","reference-native","base01"),("symmeta","symmeta-native","stream01")]:
        run=ROOT/sub/"attempts"/attempt
        allocation, reservation=read(run/"gpu.json"),read(run/"reservation.json")
        devices=[{k:d[k] for k in ("allocation_ordinal","cuda_name","uuid","pci_bus_id","cuda_total_bytes")}
                 for d in allocation["devices"]]
        selected=[r for r in rows if r["method"]==method]
        order_by_rank=defaultdict(list)
        for rank in range(4):
            if method=="base":
                entries=run/f"base-gpu{rank}.jsonl"
                published=[json.loads(l)["unit"] for l in entries.open() if json.loads(l)["event"]=="published"]
            else:
                published=[l for l in (run/f"stream-gpu{rank}.log").open() if l.startswith("published ")]
            order_by_rank[rank]=[tuple(map(int,re.search(r"seed(\d+)/rot_z/90/(?:symmeta|base)/shard(\d+)",s).groups())) for s in published]
        assert sum(map(len,order_by_rank.values()))==90
        for rank,order in order_by_rank.items():
            h=[[] for _ in range(4)]
            for seed,shard in order:
                sh=sorted([r for r in selected if (r["seed"],r["shard"])==(seed,shard)],key=lambda r:(r["batch"],r["slot"]))
                for r in sh:
                    assert r["rank"]==rank
                    slot=r["slot"]
                    entry=(r["task"],r["start"])
                    if h[slot] and h[slot][-1][0]!=r["task"]:
                        h[slot]=[]
                    if (r["task"]=="10_3" and 44<=r["start"]<=49) or (r["task"],r["start"])==("spatial_0",48):
                        histories.append(dict(method=method,seed=seed,shard=shard,batch=r["batch"],slot=slot,rank=rank,
                                              task=r["task"],start=r["start"],prior_same_task_ids=[v[1] for v in h[slot]],
                                              uuid=devices[rank]["uuid"],file=r["trace"]))
                    h[slot].append(entry)
                    if method=="base" and r["slot"]==0 and any((seed,t,i) in failed for t,i,_ in r["batch_keys"]):
                        repair.append(dict(seed=seed,shard=shard,batch=r["batch"],keys=r["batch_keys"],
                                     archived_base_npz=r["trace"],target_prefix_by_lane=[
                                         x["prefix"] for x in sh if x["batch"]==r["batch"]]))
        patterns={}
        for metric in ("success","unsafe"):
            groups=defaultdict(list)
            for r in selected:
                groups[(r["task"],r["seed"])].append(r)
            scores=[]
            for forced in (None,0,1):
                vals=[np.mean([forced if forced is not None and key(r) in failed else r[metric] for r in v]) for v in groups.values()]
                scores.append(dict(iqm=float(np.mean(np.sort(vals)[10:-10])),mean=float(np.mean(vals))))
            patterns[metric]=dict(observed=scores[0],seven_set_zero=scores[1],seven_set_one=scores[2],
                                  all_count=sum(r[metric] for r in selected),flagged_count=sum(r[metric] for r in selected if key(r) in failed))
        counts=Counter(str(r.get("first_changed")) for r in selected)
        arm_summary[method]=dict(host=allocation["hostname"],devices=devices,reservation=reservation,
                                 source=str(run.relative_to(ROOT)),exit=read(run/"exit.json"),
                                 strict_load_receipts=[sum("All keys loaded successfully!" in l for l in
                                     (run/f"{'base' if method=='base' else 'stream'}-gpu{i}.log").open()) for i in range(4)],
                                 declared_first_changed=counts if method=="symmeta" else None,
                                 effect_bounds=patterns)
    assert len(repair)==3
    write("batch-context.json",sorted(histories,key=lambda r:(r["task"],r["start"],r["method"],r["seed"])))
    write("target-batches.json",sorted(repair,key=lambda r:(r["seed"],r["batch"])))
    write("hardware-and-bounds.json",dict(arms=arm_summary,shared_env_qualification=read(ROOT/"package-check.json"),
                                    dev_smoke=read(ROOT/"dev-gpu-smoke.json"),
                                    selected_packages=[s for s in (ROOT/"resolved.txt").read_text().splitlines()
                                       if s.split("==")[0] in ("torch","torchvision","transformers","lerobot","mujoco","hf-libero","robosuite","numpy")],
                                    versions_note="Shared lock identified by the per-attempt hash. Neither job probe records the loaded NVIDIA driver version."))
    full=read(OUT/"self-comparisons.json")
    late=[]
    for method in ("base","symmeta"):
        for item in full[method]["varying"]["full_X"]:
            ids=[g[0] for g in item["groups"]]
            pair=[next(r for r in rows if r["method"]==method and r["task"]==item["task"] and r["start"]==item["start"] and r["seed"]==i) for i in ids]
            zkey=f'{item["task"]}__{item["start"]}__0__'
            with np.load(ROOT/pair[0]["trace"]) as za, np.load(ROOT/pair[1]["trace"]) as zb:
                diffs={}
                for f in ("X","A","A_vla","speed"):
                    a,b=za[zkey+f],zb[zkey+f]
                    n=min(len(a),len(b))
                    delta=(a[:n].view(np.uint32)!=b[:n].view(np.uint32)).reshape(n,-1).any(1)
                    idx=np.flatnonzero(delta)
                    diffs[f]=int(idx[0]) if len(idx) else None
            late.append(dict(method=method,**item,first_different=diffs,
                             outcome=[{k:r[k] for k in ("seed","success","unsafe","steps")} for r in pair]))
    write("full-trajectory-variation.json",late)

if __name__=="__main__": main()
