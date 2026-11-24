"""Read actual movement on controls 16–20, the first intervention on common prefixes."""
import json
import sys
from pathlib import Path
import numpy as np

def read(path): return json.loads(path.read_bytes())

def main(root):
    out=root/"gpu01"
    src=root/"src-522ccf30d18f96f4ad49104c9c862096a9db1652/outputs/frozen-vla-tracking-feedback-20261123-r1"
    m=read(src/"source-model.json")
    b=np.array(m["prior_matrix"])
    c=np.array(m["source_fit"]["command_bias"])
    rows=[]
    for raw in sorted(out.glob("*-raw.json")):
        data=read(raw)
        origin=data["chunks"]
        cmd=[np.asarray(s["cmd"],float) for s in origin[:4]]
        assert all(s.shape==(5,7) for s in cmd)
        u=[s[:,:3].mean(axis=0) for s in cmd]
        expected_prior=np.r_[u[3],u[2]] @ b
        expected=expected_prior-np.r_[1,u[3],u[2],u[1]] @ c
        errors={}
        for method in ("raw","feedback","stale"):
            d=read(raw.with_name(raw.name.replace("-raw.json",f"-{method}.json")))
            assert d["chunks"][:3]==origin[:3]
            s=d["chunks"][3]
            issued=np.array(s["cmd"])
            proposed=issued if method=="raw" else issued-np.array(d["corrections"][3])
            # The recorded action and correction were rounded independently to float32.
            max_diff=float(np.abs(proposed-cmd[3]).max())
            assert max_diff <= 2*np.finfo(np.float32).eps
            achieved=np.asarray(s["xyz"][-1])-s["before"]
            errors[method]=dict(source_forecast_error_mm=float(1000*np.linalg.norm(expected-achieved)),
                                raw_prior_error_mm=float(1000*np.linalg.norm(expected_prior-achieved)),
                                max_original_action_roundtrip_diff=max_diff)
        rows.append(dict(case=data["case"],errors=errors))
    return rows


if __name__=="__main__": print(json.dumps(main(Path(sys.argv[1])),indent=2))
