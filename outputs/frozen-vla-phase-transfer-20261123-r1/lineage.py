"""Summarize source and target provenance, native outcomes and reversals from full-control physics."""
import argparse
import csv
import gzip
import json
import os
import socket
from pathlib import Path

import numpy as np
import torch

from common import save, sha


def audit(run, groups):
    read = lambda p: json.loads(p.read_bytes())
    split = read(run/"source01/split.json")
    bank = np.asarray(torch.load(split["bank_path"],map_location="cpu",weights_only=False))
    rows = [14,15,16,17,30,31,32,33,36,37,38,39]
    xyz = np.array([np.arange(x,x+3) for x in (10,17,24,31)])
    records, layout_hash, robustness = [], {}, {}
    for group in groups:
        folder = run/(group+"01")
        scored = read(run/f"evidence01/{group}.json")["cases"]
        with gzip.open(run/f"evidence01/{group}.csv.gz","rt") as handle:
            points = list(csv.DictReader(handle))
        cluster = {}
        for name in scored:
            v = read(folder/f"lineage-{name}.json")
            t = read(folder/f"truth-{name}.json")
            q = np.asarray(t["q"])
            after = np.maximum.accumulate(q < -0.14)
            recross = np.flatnonzero(after & (q >= -0.14))
            reset = v["reset"]
            row = v["layout"]
            assert layout_hash.setdefault(row,reset["initial_state_fingerprint"]) == reset["initial_state_fingerprint"]
            assert reset["seed"]==v["seed"] and reset["init_state_id"]==row
            reference = {"original_controls":v["source_controls"],"original_success":v["source_native_success"],
                         "source_settled_reset":v["source_settled_reset"]}
            if v["matched_original"]:
                reference.update(reward_tape_equal=v["reward_tape_equal"],tcp_max_m=max(v["tcp_error_m"]),
                                 finger_max_m=max(v["finger_max_error_m"]),rotation_max_rad=max(v["rotation_boundary_error_rad"]))
            records.append(dict(group=group,episode=name,seed=v["seed"],controls=v["controls"],
                                terminal_native_success=t["success"],native_return=sum(t["rewards"]),
                                input=v["input_path"],input_sha256=v["sha256"],donor=v["donor"],
                                cpu_settled_reset=reset["initial_state_fingerprint"],scene=reset["scene_fingerprint"],
                                released_row=reset["released_row_fingerprint"],bddl_sha256=reset["bddl_sha256"],
                                perturbation=reset["perturbation_plan_fingerprint"],original_comparison=reference,
                                final_q=float(q[-1]),min_q=float(min(q)),
                                after_but_closed_count=len(recross),first_reclose=int(recross[0]) if len(recross) else None))
            eligible = [p for p in points if p["case"]==name and int(p["t"]) < v["controls"]]
            scores = [np.mean([(float(p[k])-int(p["ever_open"]))**2 for p in eligible]) for k in ("visual","clock")]
            cluster.setdefault(row,[]).append(scores)
        robustness[group] = np.mean([np.mean(v,axis=0) for v in cluster.values()],axis=0).tolist()
    assert len(set(layout_hash.values())) == len(layout_hash)
    return dict(runtime=read(run/"known01/runtime.json"),reported_by=socket.gethostname(),
                cpu_affinity=sorted(os.sched_getaffinity(0)),source_original_sha256=split["original_sha256"],
                source_original_size=(run/"source.h5").stat().st_size,model_sha256=sha(run/"model.json"),
                min_fit_vs_held_or_registered_max_object_distance_m=min(
                    split["initial_scene_audit"][str(i)]["max_object_separation_m"] for i in split["fit"]),
                min_distinct_registered_row_max_object_distance_m=min(
                    float(np.linalg.norm(bank[a,xyz]-bank[b,xyz],axis=-1).max()) for a in rows for b in rows if a<b),
                next_action_only_brier_visual_clock=robustness,recordings=records)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run",type=Path)
    p.add_argument("groups",nargs="+")
    a = p.parse_args()
    save(a.run/"evidence01/lineage.json",audit(a.run,a.groups))
