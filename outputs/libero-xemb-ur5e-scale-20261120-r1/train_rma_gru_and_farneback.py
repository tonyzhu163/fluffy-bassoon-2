"""Train 3-seed RMA-GRU baseline and calibrate OpenCV Farneback dense optical flow on 1,344 7-DoF Panda/Sawyer trials."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.readers import (
    analytic_visual_yaw,
    farneback_image_motion,
    fit_visual_projection,
)
from collect_scale import validate_scale_archive
from collect_ur5e import issued_exposure
from train_scale import train_one
from train_ur5e_readers import aggregate_seeds, audit_model, load_mode_split, make_donor_permutation
from ur5e_motor import write_json


def circular_yaw_err(pred_deg, true_deg):
    d = (np.asarray(pred_deg) - np.asarray(true_deg) + 180.0) % 360.0 - 180.0
    return np.abs(d)


def summarize_yaw_by_regime(y_pred, dev_data, dev_rows):
    out = {}
    for reg in ("familiar", "compositional_ood", "continuous_angle_ood", "overall"):
        idx = [i for i, r in enumerate(dev_rows) if reg == "overall" or r["regime"] == reg]
        errs = circular_yaw_err(y_pred[idx], dev_data["yaw_deg"][idx])
        out[reg] = round(float(np.mean(errs)), 2)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cnn-width", type=int, default=96)
    parser.add_argument("--mlp-width", type=int, default=192)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    args = parser.parse_args()

    raw_archive = args.archive.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_scale_archive(archive)
    archive_dir = args.archive.parent
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"

    # 1. Calibrate OpenCV Farneback dense optical flow on 7-DoF Panda/Sawyer active_probe and evaluate on 96 UR5e dev.
    act_train, _ = load_mode_split(archive_dir, archive["trials"], "active_probe", "train")
    act_dev, act_dev_rows = load_mode_split(archive_dir, archive["trials"], "active_probe", "dev")
    pas_dev, pas_dev_rows = load_mode_split(archive_dir, archive["trials"], "passive_attempt1", "dev")

    print("Computing Farneback dense optical flow on train active_probe...", flush=True)
    tr_fb = farneback_image_motion(act_train["img_before"], act_train["img_after"], act_train["valid"])
    tr_iss = issued_exposure(act_train)
    fb_proj_info = fit_visual_projection(
        tr_iss, tr_fb, act_train["delta_eef"][..., 2], act_train["valid"], act_train["yaw_deg"], act_train["xy_scale"]
    )
    fb_proj = np.asarray(fb_proj_info["projection"], dtype=np.float32)

    dev_act_fb = farneback_image_motion(act_dev["img_before"], act_dev["img_after"], act_dev["valid"])
    y_fb_act = analytic_visual_yaw(issued_exposure(act_dev), dev_act_fb, act_dev["delta_eef"][..., 2], act_dev["valid"], fb_proj)
    dev_pas_fb = farneback_image_motion(pas_dev["img_before"], pas_dev["img_after"], pas_dev["valid"])
    y_fb_pas = analytic_visual_yaw(issued_exposure(pas_dev), dev_pas_fb, pas_dev["delta_eef"][..., 2], pas_dev["valid"], fb_proj)

    fb_summary = dict(
        projection=fb_proj.tolist(),
        available=bool(fb_proj_info.get("available", True)),
        dev_active_yaw_err=summarize_yaw_by_regime(y_fb_act, act_dev, act_dev_rows),
        dev_passive_yaw_err=summarize_yaw_by_regime(y_fb_pas, pas_dev, pas_dev_rows),
    )
    write_json(args.output / "farneback-calib.json", fb_summary)
    print("FARNEBACK CALIB:", json.dumps(fb_summary), flush=True)

    # 2. Train 3 seeds of RMA-GRU (active_probe, luminance_flow — identical inputs & steps to joint_transformer).
    donor_perm = make_donor_permutation(act_dev_rows, np.random.default_rng(20261120))
    seed_retained, seed_cleared, seed_swapped, curves = [], [], [], []
    for seed in seeds:
        params, cfg, info = train_one(
            "active_probe",
            "rma_gru",
            seed,
            act_train,
            args.steps,
            args.batch_size,
            0.05,
            ckpt_dir,
            archive_sha,
            visual_mode="luminance_flow",
            cnn_width=args.cnn_width,
            mlp_width=args.mlp_width,
            tag_prefix="",
        )
        aud = audit_model(params, cfg, act_dev, act_dev_rows, donor_perm)
        seed_retained.append(aud["retained"])
        seed_cleared.append(aud["cleared"])
        seed_swapped.append(aud["swapped_donor"])
        curves.append(info)
        print(
            json.dumps(
                dict(
                    model="rma_gru",
                    seed=seed,
                    overall_yaw=round(aud["retained"]["overall"]["yaw_err_deg"], 2),
                    fam_yaw=round(aud["retained"]["familiar"]["yaw_err_deg"], 2),
                    comp_yaw=round(aud["retained"]["compositional_ood"]["yaw_err_deg"], 2),
                    cont_yaw=round(aud["retained"]["continuous_angle_ood"]["yaw_err_deg"], 2),
                )
            ),
            flush=True,
        )

    rma_out = dict(
        archive_sha256=archive_sha,
        seeds=list(seeds),
        farneback=fb_summary,
        rma_gru=dict(
            visual_mode="luminance_flow",
            retained=aggregate_seeds(seed_retained),
            cleared=aggregate_seeds(seed_cleared),
            swapped_donor=aggregate_seeds(seed_swapped),
            training=curves,
        ),
    )
    write_json(args.output / "baselines-audit.json", rma_out)
    print("DONE BASELINES TRAINING", flush=True)


if __name__ == "__main__":
    main()
