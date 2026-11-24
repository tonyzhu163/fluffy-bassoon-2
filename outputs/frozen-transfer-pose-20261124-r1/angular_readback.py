"""Evaluate existing positive hinge-chain certificates at the actual measured dynamic wrist error."""
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main(root,out):
    reports=sorted((Path(__file__).resolve().parents[1]/"frozen-vla-reachability-20261123-r1/evidence").glob("*-readback.json.gz"))
    rows=[]
    for model_arm in ("Kinova3","Panda"):
        offsets=[np.zeros(7)]
        levers=[None]
        for report in reports:
            for r in json.loads(gzip.decompress(report.read_bytes()))["rows"]:
                if r["arm"]==model_arm and "bound" in r:
                    b=r["bound"]
                    offsets.append(b["axis_offsets_m"])
                    levers.append(b["rotation_slack_m"]/(2*np.sin(.01/2)))
        for path in sorted(root.glob(f"*-{model_arm}-on-{model_arm}.npz")):
            meta=json.loads(path.with_suffix(".json").read_bytes())
            assert np.array_equal(meta["chain_offsets_m"],offsets)
            d=np.load(path,allow_pickle=False)
            base=d["chain_lb_at_0p01_m"]
            worst=int(np.argmax(base))
            a=np.linalg.norm(d["native_error"][:,3:],axis=1)
            p=np.linalg.norm(d["native_error"][:,:3],axis=1)
            lb=np.zeros(len(a))
            for i,k in enumerate(d["chain_index"]):
                if base[i]>0 and levers[k] is not None:
                    # Exact same attaining hinge-point certificate; positive pre-clipping value known.
                    lb[i]=max(0.,base[i]-2*levers[k]*(np.sin(a[i]/2)-np.sin(.01/2)))
                elif a[i]<=.01:
                    lb[i]=base[i]
            gp=int(np.argmax(lb))
            # The measured endpoint witnesses this pose error whenever its joints stayed inside the box;
            # the chain bound itself holds at every hinge angle including a soft-limit overshoot.
            assert np.all(lb<=p+1e-8)
            rows.append(dict(tape=path.stem,controls=len(a),positive_gt_1mm_at_reset_angle=int(sum(base>.001)),
                             positive_gt_1mm_at_observed_wrist_error=int(sum(lb>.001)),max_matched_lower_mm=1000*lb[gp],
                             step_max_matched_lower=int(gp+1) if lb[gp]>0 else "",
                             observed_position_error_here_mm=1000*p[gp] if lb[gp]>0 else "",
                             observed_wrist_error_here_deg=np.rad2deg(a[gp]) if lb[gp]>0 else "",
                             step_max_reset_lower=worst+1 if base[worst]>0 else "",
                             free_tilt_here_deg=np.rad2deg(d["static_free"][worst,1]) if base[worst]>0 else "",
                             free_tcp_error_here_mm=1000*d["static_free"][worst,0] if base[worst]>0 else ""))
    with (out/"angular-bound.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out/"angular-revision.txt").write_text(subprocess.check_output(
        ["git","-C",Path(__file__).parent,"rev-parse","HEAD"],text=True))


if __name__=="__main__":
    assert not os.environ["CUDA_VISIBLE_DEVICES"] and set(os.sched_getaffinity(0))<={46,47}
    main(Path(sys.argv[1]), Path(sys.argv[2]))
