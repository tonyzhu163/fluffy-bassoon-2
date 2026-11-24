"""Label-blind replay of the published NPZ keyed by case/seed, using compact retained sequences."""
import argparse
from pathlib import Path

import jax
import numpy as np

from eval_stage2a_live_ur5e import load_v2_checkpoint
from eval_wider_family import CHUNK_KEYS
from fluffy_bassoon.xemb.equiv_v2 import predict_embodiment_v2
from source import read_json, require, sha, write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("data", type=Path)
    ap.add_argument("checkpoints", type=Path)
    ap.add_argument("prediction", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    require(jax.default_backend() == "cpu", "Independent CPU replay")
    seal = read_json(args.data/"source-seal.json")
    receipt = read_json(args.prediction/"prediction-receipt.json")
    require(receipt["data_seal"] == sha(args.data/"source-seal.json"), "Source seal mismatch")
    require(receipt["checkpoints"] == sha(args.checkpoints/"checkpoints.json"), "Checkpoint manifest mismatch")
    pred_path = args.prediction/"readers.npz"
    require(sha(pred_path) == receipt["reader_sha256"], "Prediction digest mismatch")
    with np.load(pred_path, allow_pickle=False) as z:
        published = {k: z[k] for k in z.files}
    query = {(str(case), int(seed)): i for i,(case,seed) in enumerate(zip(published["case_id"], published["seed"], strict=True))}
    require(len(query) == 2016, "Duplicate case/seed or incomplete grid")
    ck = read_json(args.checkpoints/"checkpoints.json")
    results = []
    read_rows = 0
    for part in ("eval", "cal"):
        path = args.data / f"reader-{part}.npz"
        require(sha(path) == seal["sha256"][path.name], path)
        with np.load(path, allow_pickle=False) as z:
            source = {k: z[k] for k in z.files}
        for seed in (43, 41, 42):
            take = np.asarray([query[str(case), seed] for case in source["case_id"]])
            read_rows += len(take)
            for k in ("split", "physical_trace_id", "task", "arm", "valid"):
                np.testing.assert_array_equal(published[k][take], source[k])
            np.testing.assert_array_equal(published["cmd"][take], source["issued_mean"][..., :2])
            for method, axis in (("geom",1), ("tuned",0)):
                record, = [r for r in ck["rows"] if r["seed"] == seed and r["model"] == method]
                ckpath = args.checkpoints / record["path"]
                require(sha(ckpath) == record["sha256"], ckpath)
                p, cfg, meta = load_v2_checkpoint(ckpath)
                require(meta["revision"] == receipt["revision"] and meta["seed"] == seed, ckpath)
                for omit in (2, 0, 1, None):
                    want = (published["full_yaw"][take,axis] if omit is None
                            else published["loo_yaw"][take,omit,axis])
                    keep = [j for j in range(3) if j != omit]
                    params = p
                    if method == "tuned" and omit is not None:
                        att = p["perceiver"]
                        params = dict(p, perceiver=dict(att, pos=att["pos"][:,keep,:]))
                    preds = []
                    for start in range(0, len(take), 32):
                        batch = {k: source[k][start:start+32][:,keep] for k in CHUNK_KEYS}
                        preds.append(predict_embodiment_v2(params,cfg,batch))
                    y, g, s = (np.concatenate(v) for v in zip(*preds,strict=True))
                    err = np.abs((y.astype(float) - want + 180.) % 360. - 180.)
                    row = dict(split=part, seed=seed, method=method, omit=omit, n=len(want),
                               max_abs_yaw_difference_deg=float(err.max()), exact_yaws=int(np.sum(y==want)))
                    # Two sequence lengths and three batch shapes can change float32 contraction order.
                    # This is the same unit-circle precision check used by the pinned masking audit.
                    cy = np.exp(1j*np.deg2rad(y.astype(float)))
                    cwant = np.exp(1j*np.deg2rad(want.astype(float)))
                    np.testing.assert_allclose(cy,cwant,atol=2e-4,rtol=0)
                    if method == "tuned" and omit is None:
                        expected = published["tuned_gain"][take]
                        np.testing.assert_allclose(g,expected,atol=2e-6,rtol=2e-4)
                        np.testing.assert_array_equal(s,published["tuned_sign"][take])
                        row.update(max_gain_difference=float(np.max(np.abs(g-expected))), sign_exact=len(s))
                    results.append(row)
    require(read_rows == len(query), "Incomplete replay")
    write_json(args.out, dict(revision=seal["revision"], prediction_sha256=receipt["reader_sha256"],
               trace_seed_rows=read_rows, scalar_yaw_checks=sum(r["n"] for r in results),
               max_difference_deg=max(r["max_abs_yaw_difference_deg"] for r in results),
               exact_yaws=sum(r["exact_yaws"] for r in results), results=results))


if __name__ == "__main__":
    main()
