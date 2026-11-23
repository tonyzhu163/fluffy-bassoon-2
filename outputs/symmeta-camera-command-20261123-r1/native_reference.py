"""Independently replay the original Stage 2A native inference after a CPU/GPU readback difference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def distance(a, b):
    return abs((a - b + 180) % 360 - 180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sha = subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip()
    selected = {"437da7f677632e9495c813aaa52071e207ba834e": {0, 1, 2, 4, 5, 6, 7},
                "ff2ddf0ab03e6fbe243979556d8d8f2ebb9bc29c": {3}}[sha]
    sys.path[:0] = [str(args.source / "src"), str(args.source / "outputs/xemb-icml-campaign-20261121-r1"),
                    str(args.source / "outputs/libero-crossarm-20261120-r1")]
    import cv2
    import jax
    import eval_stage2a_live_ur5e as published
    assert jax.default_backend() == "cpu"
    cv2.setNumThreads(1)
    scores = json.loads(args.scores.read_text())
    params, cfg, _ = published.load_v2_checkpoint(Path(scores["sources"]["checkpoint"]["path"]))
    projection = np.array(scores["sources"]["calibration"]["projection"])
    references = {(row["id"], row["roll_deg"]): row["yaw_deg"] for row in scores["rows"]}
    rows = []
    for support in scores["supports"]:
        case = support["case"]
        if case["shard"] not in selected:
            continue
        with np.load(case["passive_npz"], allow_pickle=False) as z:
            batch = {key: z[key][None] for key in z.files}
        for angle in (0, 90, -90):
            inputs = dict(batch)
            inputs["chunk_steps"] = np.where(inputs["valid"], published.CHUNK_SIZE, 0).astype(np.int32)
            for key in ("img_before", "img_after", "frame0_img"):
                inputs[key] = np.rot90(inputs[key], angle // 90, axes=(-3, -2)).copy()
            before, after, valid = (inputs[k] for k in ("img_before", "img_after", "valid"))
            inputs["dis_flow"] = published.compute_dis_flow_maps(before, after, valid)
            yaw = float(published.predict_embodiment_v2(params, cfg, inputs)[0][0])
            hs = float(published.analytic_visual_yaw(published.issued_exposure(inputs),
                       published.image_motion(before, after, valid), inputs["delta_eef"][..., 2], valid, projection)[0])
            pred = {"model_native": yaw, "hs_native": hs}
            new = references[case["id"], angle]
            rows.append(dict(id=case["id"], angle=angle, clipped_trace_native_yaw_deg=pred,
                             vs_new_cpu_deg={k:distance(v, new[k]) for k,v in pred.items()},
                             vs_archived_gpu_deg={k: distance(v, case["reference_yaw_deg"][k]) for k,v in pred.items()}
                             if angle == 0 else None))
    maxima = {m: max(r["vs_new_cpu_deg"][m] for r in rows) for m in ("model_native", "hs_native")}
    audit = dict(source=sha, n=len(rows), max_disagreement_same_backend_deg=maxima, rows=rows)
    with args.output.open("x") as f:
        json.dump(audit, f, allow_nan=False, indent=2)
        f.write("\n")
    print(sha, len(rows), maxima, flush=True)
    # This tests whether changing inference implementations on the same backend changes the result.
    if max(maxima.values()) > 1.e-4:
        raise ValueError("Historical and new native CPU paths disagree")


if __name__ == "__main__":
    main()
