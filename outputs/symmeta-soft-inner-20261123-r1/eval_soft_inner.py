"""Fit PR #8's identity rule on source-arm traces and score frozen UR5e predictions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.wider_family import operator_error, relabel, rot2_batch, sample_interfaces

HERE = Path(__file__).resolve().parent
EXPORT = HERE.parent / "symmeta-perceiver-retry-20261123-r1"
SHARED = Path("/mowenjie/fluffy-bassoon-2/outputs")
FIT = SHARED / EXPORT.name / "train02pro"
TARGET_SHA = "c8420c2cb7d0cb38e6f0d73b3a20ce47997dbef7657b652d45eaaa8988eb6c0b"
SEEDS = (41, 42, 43)
GATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
ID = np.eye(2)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def inventory():
    locks = json.loads((EXPORT / "export-inputs.json").read_text())["sha256"]
    files = {SHARED / k: v for k, v in locks.items()}
    for line in (EXPORT / "evidence/train02pro/checkpoints.sha256").read_text().splitlines():
        expected, rel = line.split()
        files[FIT / rel] = expected
    for path, expected in files.items():
        if digest(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    return files


def mask(estimate, threshold):
    if threshold == "always":
        return np.ones(len(estimate), dtype=bool)
    distance = np.linalg.norm(estimate - ID, axis=(-2, -1)) / np.sqrt(2)
    return distance < threshold


def choose(source_predictions):
    scores = []
    for threshold in (*GATES, "always"):
        losses = []
        for truth, estimate in source_predictions:
            leave = mask(estimate, threshold)
            losses.append(float(np.mean(operator_error(truth, np.where(leave[:, None, None], ID, estimate)))))
        scores.append(dict(threshold=threshold, nominal=losses[0], shifted=losses[1], mean=sum(losses)/2))
    return min(scores, key=lambda r: r["mean"])["threshold"], scores


def fit(out, files):
    import jax
    from eval_wider_family import CHUNK_KEYS, LABEL_KEYS, predict_ops
    from eval_stage2a_live_ur5e import load_v2_checkpoint
    if jax.default_backend() != "gpu" or len(jax.devices()) != 1:
        raise RuntimeError("Expected one allocated JAX GPU")
    groups = []
    for file in sorted(p for p in files if p.suffix == ".npz"):
        with np.load(file, allow_pickle=False) as data:
            groups.append({k: data[k][:, :3] if k in CHUNK_KEYS else data[k] for k in (*CHUNK_KEYS, *LABEL_KEYS)})
    train = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
    n = len(train["valid"])
    if n != 1466 or train["valid"].shape != (1466, 3):
        raise ValueError("Unexpected source membership or support length")
    nominal = dict(yaw_deg=np.zeros(n), shape=np.broadcast_to(ID,(n,2,2)), z_gain=np.ones(n), gripper_sign=np.ones(n))
    source = [relabel(train, t) for t in (nominal, sample_interfaces(np.random.default_rng(20261124),"family",n))]
    result = []
    for seed in SEEDS:
        for name, stem in [("symmeta","equiv_v2_dis_dz"),("tuned","perceiver_cont_warm")]:
            path, = [p for p in files if p.name == f"{stem}-passive_attempt1-seed{seed}.msgpack"]
            params,cfg,_ = load_v2_checkpoint(path)
            estimates = [(batch["op"],predict_ops(params,"factors",cfg,batch)[0]) for batch in source]
            threshold,scores = choose(estimates)
            result.append(dict(reader=name,seed=seed,checkpoint_sha256=files[path],threshold=threshold,source_scores=scores))
            print(f"[fitted] {name} seed={seed} threshold={threshold}",flush=True)
    dump(out / "source-rules.json", dict(source_trace_count=n,input_hashes={str(k):v for k,v in files.items()},rules=result))


def score(out):
    rules = json.loads((out / "source-rules.json").read_text())["rules"]
    path = FIT / "predictions.json"
    if digest(path) != TARGET_SHA:
        raise ValueError("Frozen native-reader prediction export differs")
    exported = [r for r in json.loads(path.read_text())["rows"] if r["panel"] == "gpu09test"]
    if {(r["seed"],r["index"]) for r in exported} != {(s,i) for s in SEEDS for i in range(96)} or len(exported)!=288:
        raise ValueError("Missing or duplicate UR5e assignments")
    rows = []
    for rule in rules:
        name,seed=rule["reader"],rule["seed"]
        for r in (v for v in exported if v["seed"]==seed):
            spec = get_embodiment_spec(r["embodiment"])
            truth = spec.xy_scale * rot2_batch([spec.yaw_deg])[0]
            pred = r[name]
            est = pred["xy_scale"] * rot2_batch([pred["yaw_deg"]])[0]
            leave = bool(mask(est[None],rule["threshold"])[0])
            if not np.isfinite(est).all():
                raise ValueError("Nonfinite estimate")
            row={k:r[k] for k in ("index","suite","task_id","embodiment","regime","tape_sha256","seed")}
            row.update(reader=name,nominal=r["embodiment"]=="fam_nom",leave_alone=leave,grip_true=spec.gripper_sign,
                       grip_raw=pred["gripper_sign"],grip_gated=1 if leave else pred["gripper_sign"],
                       raw_yaw=pred["yaw_deg"],raw_gain=pred["xy_scale"],raw_error=float(operator_error(truth,est)),
                       gated_error=float(operator_error(truth,ID if leave else est)),identity_error=float(operator_error(truth,ID)))
            rows.append(row)
    groups = {}
    for name in ("symmeta","tuned"):
        groups[name]={}
        for condition in ("nominal","shifted"):
            g=[r for r in rows if r["reader"]==name and r["nominal"]==(condition=="nominal")]
            tasks=sorted({(r["suite"],r["task_id"]) for r in g})
            sizes=Counter((r["suite"],r["task_id"]) for r in g)
            if len(tasks)!=8 or set(sizes.values())!={3 if condition=="nominal" else 33}:
                raise ValueError("Unbalanced paired tasks")
            differences=np.array([np.mean([r["gated_error"]-r["raw_error"] for r in g if (r["suite"],r["task_id"])==t]) for t in tasks])
            boot=np.random.default_rng(20261123).choice(differences,(10000,8),replace=True).mean(1)
            groups[name][condition]=dict(assignments=len(g),unique_tapes=len({r["tape_sha256"] for r in g}),
                applications=sum(not r["leave_alone"] for r in g),abstentions=sum(r["leave_alone"] for r in g),
                raw_error=np.mean([r["raw_error"] for r in g]),gated_error=np.mean([r["gated_error"] for r in g]),
                identity_error=np.mean([r["identity_error"] for r in g]),gated_minus_raw_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                raw_wrong_gripper=sum(r["grip_true"]!=r["grip_raw"] for r in g),
                gated_wrong_gripper=sum(r["grip_true"]!=r["grip_gated"] for r in g))
    dump(out/"target-cases.json",rows)
    dump(out/"soft-inner-summary.json",dict(predictions_sha256=TARGET_SHA,source_rules_sha256=digest(out/"source-rules.json"),results=groups))


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode",choices=("preflight","run"))
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    files=inventory()
    if args.mode=="preflight":
        eye=np.repeat(ID[None],3,axis=0)
        tilted=np.stack([ID,ID*2,ID*.5])
        assert mask(tilted,.1).tolist()==[True,False,False]
        assert choose([(eye,tilted),(tilted,tilted)])[0]==0.0
        assert digest(FIT/"predictions.json")==TARGET_SHA
        print(f"Verified {len(files)} source inputs/checkpoints, frozen target export and gate controls.")
    else:
        fit(args.output,files)
        score(args.output)
