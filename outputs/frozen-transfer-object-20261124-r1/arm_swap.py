"""Secondary control: the eight IIWA tapes under the same released scenes on UR5e."""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from capture import MULTI, record, _probe_egl
from score import METHODS, PAIRS, SCORES, paired_windows, predict, summarize, dump


def capture(root):
    cpus = sorted(os.sched_getaffinity(0))
    renderer = _probe_egl(True)
    assert len(cpus) <=2 and "llvmpipe" in renderer[1].lower() and not os.environ["CUDA_VISIBLE_DEVICES"]
    cv2.setNumThreads(1)
    dump(root/"runtime-arm-swap.json",dict(cpus=cpus,renderer=renderer))
    donors = sorted((MULTI/"gpu01-IIWA").glob("*-14-raw.json"))
    assert len(donors)==8
    for path in donors:
        spec = json.loads(path.read_bytes())["case"]
        record(root/"arm_swap", path,spec,14,"arm_swap", actuated_arm="UR5e")


def rows(path):
    def read(field):
        return json.loads(path.with_name(path.name.replace("lineage",field)).read_bytes())
    lineage = read("lineage")
    spec = lineage["spec"]
    task = f"{spec['suite']}/{spec['task_id']}"
    results = []
    for pub,truth in zip(read("public"), read("truth"), strict=True):
        assert pub["control"] == truth["control"]
        if truth["y"] is not None:
            results.append(dict(**pub,task=task,y=truth["y"],arm=spec["arm"],donor=spec["init_state_id"],
                                original=lineage["split"]=="target"))
    return lineage,pd.DataFrame(results)


def score(root):
    models = json.loads((root/"analysis/models.json").read_bytes())
    complete = []
    valid = []
    finished = sorted((root/"arm_swap").glob("*.lineage.json"))
    assert len(finished)==8
    for path in finished:
        other,b = rows(path)
        own,a = rows(root/"target"/path.name.replace("UR5e-","IIWA-"))
        assert other["model"]["arm"]=="UR5e" and own["model"]["arm"]=="IIWA"
        assert own["spec"]==other["spec"] and own["seed"]==other["seed"] and own["input_sha256"]==other["input_sha256"]
        for k in ("bddl_sha256","released_row_fingerprint","scene_fingerprint","perturbation_plan_fingerprint"):
            assert own["reset"][k] == other["reset"][k]
        task=f"{own['spec']['suite']}/{own['spec']['task_id']}"
        valid.append(dict(task=task,sha256=own["input_sha256"],scene=own["reset"]["scene_fingerprint"],
                          row=own["reset"]["released_row_fingerprint"],seed=own["seed"],IIWA_controls=own["cpu_controls"],
                          UR5e_controls=other["cpu_controls"],IIWA_success=own["cpu_success"],UR5e_success=other["cpu_success"],
                          IIWA_eligible=len(a),UR5e_eligible=len(b),UR5e_censored=other["censored"]))
        for frame in (a,b):
            if frame.empty:
                continue
            for m in METHODS:
                frame[m]=predict(models[task][m],frame)
            complete.append(frame)
    df=pd.concat(complete,ignore_index=True)
    pairs=paired_windows(df)
    variables=["rank_"+m for m in SCORES]+["loss_"+m for m in SCORES]+["delta_"+m for m in SCORES[:-1]]
    pairs.to_csv(root/"analysis/arm-swap-opposing.csv.gz",index=False)
    df.to_csv(root/"analysis/arm-swap-predictions.csv.gz",index=False)
    dump(root/"analysis/arm-swap.json",dict(validation=valid,opposing=len(pairs),bouts=pairs.bout.nunique(),
         IIWA_carries=int(pairs.y_own.sum()),UR5e_carries=int((1-pairs.y_own).sum()),
         tasks=pairs.task.nunique(),primary=summarize(pairs,variables),by_bout=summarize(pairs,variables,True)))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("root",type=Path)
    p.add_argument("mode",choices=["capture","score"])
    a=p.parse_args()
    (capture if a.mode=="capture" else score)(a.root)
