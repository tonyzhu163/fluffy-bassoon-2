"""LIBERO-Pro2.0 Visual Disturbance Stress Test on the 96 Official Test Queries (6-DoF UR5e).

Evaluates our frozen 3-seed checkpoints and classical optical flow baselines (no retraining) under:
1. clean: original official test frames
2. exposure_flicker: realistic auto-exposure brightness step (+18% / -15%) between before & after frames
3. moving_distractor: a moving high-contrast object sliding across the background edge of the workspace
4. camera_jitter: 2-pixel camera vibration shift + mild sensor noise (sigma=6/255)
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.embodiment import get_embodiment_spec
from fluffy_bassoon.xemb.readers import analytic_visual_yaw, farneback_image_motion, image_motion
from collect_ur5e import CHUNK_SIZE, issued_exposure
from eval_live_ur5e import infer_one
from phase_a_hybrid import load_tagged_checkpoint
from ur5e_motor import write_json

SEEDS = (41, 42, 43)


def yaw_err(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _load_npz(path_str: str):
    with np.load(path_str) as z:
        d = {k: np.asarray(z[k])[None] for k in z.files}
    d["chunk_steps"] = np.where(d["valid"], CHUNK_SIZE, 0).astype(np.int32)
    return d


def perturb_batch(batch: dict, kind: str, rng: np.random.Generator) -> dict:
    if kind == "clean":
        return batch
    out = {k: np.array(v, copy=True) for k, v in batch.items()}
    ib = out["img_before"].astype(np.float32)
    ia = out["img_after"].astype(np.float32)
    b, k, h, w, c = ib.shape

    if kind == "exposure_flicker":
        # Realistic auto-exposure gain/bias step on img_after only (violates brightness constancy)
        for ki in range(k):
            gain = float(rng.choice([0.82, 1.18]))
            bias = float(rng.uniform(-10.0, 10.0))
            ia[0, ki] = np.clip(ia[0, ki] * gain + bias, 0.0, 255.0)
    elif kind == "moving_distractor":
        # High-contrast textured 14x14 block sliding horizontally across top background band (rows 2..16)
        patch = rng.integers(30, 240, size=(14, 14, 3), dtype=np.int32).astype(np.float32)
        for ki in range(k):
            x0 = int(rng.integers(4, w - 26))
            dx = int(rng.choice([-8, 8]))
            x1 = int(np.clip(x0 + dx, 0, w - 14))
            ib[0, ki, 2:16, x0 : x0 + 14] = patch
            ia[0, ki, 2:16, x1 : x1 + 14] = patch
    elif kind == "camera_jitter":
        # 2-pixel camera vibration shift on img_after + mild Gaussian sensor noise (sigma=6)
        for ki in range(k):
            sx = int(rng.choice([-2, 2]))
            sy = int(rng.choice([-2, 2]))
            ia[0, ki] = np.roll(np.roll(ia[0, ki], sx, axis=1), sy, axis=0)
            ib[0, ki] = np.clip(ib[0, ki] + rng.normal(0.0, 6.0, size=(h, w, c)), 0.0, 255.0)
            ia[0, ki] = np.clip(ia[0, ki] + rng.normal(0.0, 6.0, size=(h, w, c)), 0.0, 255.0)

    out["img_before"] = ib.astype(np.uint8)
    out["img_after"] = ia.astype(np.uint8)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--baselines-dir", type=Path, required=True)
    parser.add_argument("--support-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = json.loads((args.train_dir / "offline-audit.json").read_text())
    archive_sha = audit["archive_sha256"]
    steps = int(audit["steps"])
    vis_active_proj = np.asarray(audit["modes"]["active_probe"]["analytic_visual_sysid"]["projection"], dtype=np.float32)
    vis_passive_proj = np.asarray(audit["modes"]["passive_attempt1"]["analytic_visual_sysid"]["projection"], dtype=np.float32)
    fb_proj = np.asarray(json.loads((args.baselines_dir / "farneback-calib.json").read_text())["projection"], dtype=np.float32)

    ckpt_dir = args.train_dir / "checkpoints"
    base_ckpt_dir = args.baselines_dir / "checkpoints"
    readers = {}
    for s in SEEDS:
        readers[("ours_active", s)] = load_tagged_checkpoint(ckpt_dir, "", "active_probe", "factored_equiv", s, archive_sha, steps)
        readers[("ours_passive", s)] = load_tagged_checkpoint(ckpt_dir, "rgb-", "passive_attempt1", "factored_equiv", s, archive_sha, steps)
        readers[("yang_aug", s)] = load_tagged_checkpoint(base_ckpt_dir, "aug-", "active_probe", "joint_transformer", s, archive_sha, 6000)

    # Load all 96 official test support entries across shards 0..7
    items = []
    for sh in range(8):
        meta = json.loads((args.support_dir / f"shard-{sh}" / "support-meta.json").read_text())
        for row in meta:
            spec = get_embodiment_spec(row["embodiment"])
            items.append((spec, _load_npz(row["active_npz"]), _load_npz(row["passive_npz"])))

    conditions = ("clean", "exposure_flicker", "moving_distractor", "camera_jitter")
    methods = ("ours_active", "ours_passive", "global_grad_flow_passive", "global_grad_flow_active", "dense_lk_flow_passive", "yang_aug")
    results = {}
    for cond in conditions:
        rng = np.random.default_rng(20261122 + conditions.index(cond))
        errs = {m: [] for m in methods}
        errs_ood = {m: [] for m in methods}
        for spec, act_b, pas_b in items:
            ab = perturb_batch(act_b, cond, rng)
            pb = perturb_batch(pas_b, cond, rng)

            # Classical optical flow
            a_iss = issued_exposure(ab)
            p_iss = issued_exposure(pb)
            y_ga = float(analytic_visual_yaw(a_iss, image_motion(ab["img_before"], ab["img_after"], ab["valid"]), ab["delta_eef"][..., 2], ab["valid"], vis_active_proj)[0])
            y_gp = float(analytic_visual_yaw(p_iss, image_motion(pb["img_before"], pb["img_after"], pb["valid"]), pb["delta_eef"][..., 2], pb["valid"], vis_passive_proj)[0])
            y_lk = float(analytic_visual_yaw(p_iss, farneback_image_motion(pb["img_before"], pb["img_after"], pb["valid"]), pb["delta_eef"][..., 2], pb["valid"], fb_proj)[0])

            for m, yv in (("global_grad_flow_active", y_ga), ("global_grad_flow_passive", y_gp), ("dense_lk_flow_passive", y_lk)):
                e = yaw_err(yv, spec.yaw_deg)
                errs[m].append(e)
                if spec.regime != "familiar":
                    errs_ood[m].append(e)

            # 3-seed learned readers
            for m, b_in in (("ours_active", ab), ("ours_passive", pb), ("yang_aug", ab)):
                for s in SEEDS:
                    yh, _, _ = infer_one(*readers[(m, s)], b_in)
                    e = yaw_err(yh, spec.yaw_deg)
                    errs[m].append(e)
                    if spec.regime != "familiar":
                        errs_ood[m].append(e)

        results[cond] = {
            m: dict(overall=round(float(np.mean(errs[m])), 2), unseen=round(float(np.mean(errs_ood[m])), 2))
            for m in methods
        }
        print(f"=== {cond} ===", json.dumps(results[cond]), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, dict(n_queries=len(items), seeds=list(SEEDS), conditions=results))


if __name__ == "__main__":
    main()
