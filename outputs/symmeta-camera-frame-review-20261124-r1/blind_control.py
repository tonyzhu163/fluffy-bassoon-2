"""Match C to B on source exposures, chosen budget and original target observations."""
import gzip
import hashlib
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT=Path("/mowenjie/fluffy-bassoon-2/outputs/symmeta-camera-frame-reader-20261124-r1")
blind=ROOT/"blind01"
main=ROOT/"fit01"
def read(path):
    return json.loads(path.read_bytes())
def rows(path):
    return json.loads(gzip.decompress(path.read_bytes()))
fit=read(main/"sealed.json")
control=read(blind/"sealed.json")
assert (control["steps"],control["training_exposures"],control["params_per_arm"])==(fit["steps"],fit["training_exposures"],fit["params_per_arm"])
identity=hashlib.sha256(np.eye(3).tobytes()).hexdigest()
out={}
for panel, folder, poses in (("digital",main,("roll-20","roll+20","roll-45","roll+45")),("physical",ROOT/"phyGpu01",("orbit-","orbit+"))):
    b={(r["index"],r["seed"],r["view"]):r for r in rows(folder/f"{panel}-predictions.json.gz") if r["model"]=="B" and r["calibration"]=="true"}
    c=rows(blind/f"{panel}-C-predictions.json.gz")
    assert len(c)==len(b)
    task=defaultdict(list)
    for r in c:
        orig=b[r["index"],r["seed"],r["view"]]
        assert r["calibration_sha256"]==identity and r["calibration"]=="unsupplied" and r["model"]=="C"
        assert all(r[k]==orig[k] for k in ("true_yaw_deg","true_gain","true_sign","native_view_sha256","observed_sha256","source_sha256"))
        if r["regime"]!="familiar" and r["view"] in poses:
            def err(x):
                return abs((x["yaw_deg"]-x["true_yaw_deg"]+180)%360-180)
            task[r["task"]].append((err(r),err(orig)))
    means=np.stack([np.mean(task[t],0) for t in sorted(task)])
    diff=means[:,0]-means[:,1]
    boot=np.random.default_rng(1124).choice(diff,(20000,len(diff))).mean(1)
    out[panel]=dict(total_C_rows=len(c),tasks=len(diff),C_B=means.mean(0).tolist(),
                    C_minus_B=float(diff.mean()),paired_interval=np.quantile(boot,[.025,.975]).tolist())
print(json.dumps(out,indent=2))
