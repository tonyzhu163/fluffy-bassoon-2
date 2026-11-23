"""Export unrounded seed-41 yaw, gain and grip from each original passive support."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import cv2
import jax

HERE = Path(__file__).resolve().parent
CAMERA = HERE.parent / "symmeta-camera-command-20261123-r1"
sys.path.insert(0, str(CAMERA))
from score import CHECKPOINT_SHA, compute_dis_flow_maps, frozen_inputs, load_prefix, official_panel, predict_embodiment_v2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("Expected the qualified CPU reader runtime")
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    params, cfg, _, _ = frozen_inputs(args.data_root)
    cases, _ = official_panel(args.data_root)
    with gzip.open(CAMERA / "evidence/roll.json.gz", "rt") as f:
        support_hashes = {s["case"]["id"]: s["npz"]["sha256"] for s in json.load(f)["supports"]}
    rows = []
    for case in cases:
        batch, tape, _ = load_prefix(dict(case, sha256=support_hashes[case["id"]]), official=True)
        batch["dis_flow"] = compute_dis_flow_maps(batch["img_before"], batch["img_after"], batch["valid"])
        yaw, gain, grip = predict_embodiment_v2(params, cfg, batch)
        rows.append(dict(id=case["id"], index=case["index"], support_sha256=tape["sha256"],
                         yaw_hat=float(yaw[0]), scale_hat=float(gain[0]), grip_hat=int(grip[0])))
    with args.output.open("x") as f:
        json.dump(dict(checkpoint_sha256=CHECKPOINT_SHA, backend=jax.default_backend(), rows=rows),
                  f, allow_nan=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
