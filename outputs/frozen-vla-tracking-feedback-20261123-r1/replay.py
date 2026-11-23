"""Frozen-VLA trajectory replay at the native five-control boundary, no task reward input."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/mowenjie/fluffy-bassoon-2/outputs")
PILOT = ROOT / "libero-crossarm-20261120-r1/gpu01/pilot"
NATIVE = ROOT / "frozen-vla-native-transfer-20261123-r1/gpu01"
PRIOR = ROOT / "frozen-vla-motion-models-20261123-r1/cpu01/source-fit.json"


def lag(x, k=1):
    return np.concatenate((np.zeros_like(x[:k]), x[:-k]))


def snapshot(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load():
    rows = []
    for pool in ("joint_queries", "source_gate", "evaluation"):
        for p in sorted((PILOT / pool).glob("*/*/unadapted/tape.npz")):
            episode = json.loads((p.parent / "episode.json").read_bytes())
            c = episode["case"]
            with np.load(p, allow_pickle=False) as t:
                cmd, response = t["commands"][:, :3], t["motion"][:, :3]
                if not np.array_equal(t["commands"], t["proposals"]):
                    raise ValueError(f"Intervention detected in {p}")
            n = len(cmd) // 5 * 5
            rows.append(dict(arm=c["body"], task=f"{c['suite']}/{c['task_id']}", start=c["init_state_id"],
                             path=snapshot(p), u=cmd[:n].reshape(-1,5,3).mean(axis=1),
                             y=response[:n].reshape(-1,5,3).sum(axis=1)))
    for p in sorted(NATIVE.glob("*-Kinova3.json")):
        d = json.loads(p.read_bytes())
        c, chunks = d["case"], [j for j in d["chunks"] if len(j["cmd"]) == 5]
        rows.append(dict(arm=c["arm"], task=f"{c['suite']}/{c['task_id']}", start=c["init_state_id"],
                         path=snapshot(p), u=np.array([np.mean(j["cmd"], axis=0)[:3] for j in chunks]),
                         y=np.array([np.subtract(j["xyz"][-1],j["before"]) for j in chunks])))
    return rows


def parts(tape, prior):
    u, y = tape["u"], tape["y"]
    x = np.column_stack([u,lag(u)])
    # Source-response error, taking issued controls and measured motion from the same chunk.
    residual = x @ prior - y
    z = np.column_stack([np.ones(len(u)),u,lag(u),lag(u,2)])
    return z, residual


def fit(rows, prior):
    pairs = [parts(r,prior) for r in rows]
    z, e = [np.concatenate([p[i] for p in pairs]) for i in (0,1)]
    base, _, rank, _ = np.linalg.lstsq(z,e,rcond=None)
    if rank != z.shape[1]:
        raise ValueError("Source command-only design lacks rank")
    remainders = [out - inp @ base for inp, out in pairs]
    prev = np.concatenate([r[:-1] for r in remainders])
    current = np.concatenate([r[1:] for r in remainders])
    rho = float(np.sum(current * prev) / np.sum(prev ** 2))
    return dict(command_bias=base.tolist(), rho=rho)


def score(row, prior, model, noise):
    z, e = parts(row, prior)
    r = e - z @ model["command_bias"]
    actual, last = r[3:], r[2:-1]
    stale = r[:-3]
    inv = np.linalg.solve(prior[:3] @ prior[:3].T + noise*np.eye(3), prior[:3]).T
    prediction = model["rho"]*last
    cand = prediction @ inv
    norm = np.linalg.norm
    u=row["u"][3:]
    m = np.minimum(1., norm(u,axis=1)/np.maximum(norm(cand,axis=1),1e-12))
    bounded = np.clip(u + m[:,None]*cand,-1,1)-u
    # Held-out replay estimates prediction accuracy; projected response only describes direction.
    def err(a):
        return 1000*float(norm(a,axis=1).mean())
    return dict(chunks=len(r), raw_mm=err(e[3:]), cmd_mm=err(actual), online_mm=err(actual-prediction),
                stale_mm=err(actual-model["rho"]*stale), reverse_mm=err(actual+prediction),
                predictor_positive_dot_fraction=float(np.mean(np.sum(actual*prediction,axis=1)>0)),
                typical_correction_norm=float(norm(bounded,axis=1).mean()),
                cases_capped_by_pending_command=int((m<1.).sum()), measured_displacement_mm=err(row["y"][3:]),
                projected_compensation_mm=err(bounded @ prior[:3]))


def main(out):
    sourcefit = json.loads(PRIOR.read_bytes())
    prior=np.array(sourcefit["prior"])
    rows=load()
    source=[r for r in rows if r["arm"] in ("Panda","Sawyer")]
    if sorted([r["arm"] for r in source]).count("Panda") != 5 or len(source) != 10:
        raise ValueError("Missing source trajectories")
    fit_all = fit(source,prior)
    models = {arm:fit([r for r in source if r["arm"]!=arm],prior) for arm in ("Panda","Sawyer")}
    scores=[]
    for row in rows:
        recorded = {k:v for k,v in row.items() if k not in ("u","y")}
        scores.append(dict(**recorded, **score(row,prior,models.get(row["arm"],fit_all),sourcefit["residual_power"])))
    artifact=dict(prior=snapshot(PRIOR), prior_matrix=prior.tolist(), source_residual_power=sourcefit["residual_power"],
                  source_fit=fit_all, leave_arm_out_source=models, rows=scores)
    out.write_text(json.dumps(artifact,indent=2,allow_nan=False)+"\n")
    for arm in ("Panda","Sawyer","UR5e","Kinova3"):
        select=[r for r in scores if r["arm"]==arm]
        print(arm,len(select),sum(r["chunks"] for r in select),
              {key:round(float(np.mean([r[key] for r in select])),3) for key in ("raw_mm","cmd_mm","online_mm","stale_mm","reverse_mm","predictor_positive_dot_fraction","typical_correction_norm")})
    print("rho",fit_all["rho"],{k:v["rho"] for k,v in models.items()})


if __name__=="__main__":
    main(Path(sys.argv[1]))
