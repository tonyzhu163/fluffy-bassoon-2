"""Train 3-seed RMA-GRU, 3-seed Steel-Manned Yang et al. (Transformer + Exact Continuous Augmentation),
and calibrate pure-NumPy dense Lucas-Kanade optical flow on 1,344 7-DoF Panda/Sawyer trials."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.readers import (
    HISTORY_FIELDS,
    ReaderConfig,
    analytic_visual_yaw,
    farneback_image_motion,
    fit_visual_projection,
    init_reader,
    reader_loss,
)
from collect_scale import validate_scale_archive
from collect_ur5e import issued_exposure
from train_scale import save_checkpoint, train_one
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


def augment_batch_exact_physics(batch_np: dict, rng: np.random.Generator) -> dict:
    """Physically exact continuous rotation + scale + gripper-sign augmentation.

    Rotating issued XY and local delta_eef XY by R(dtheta) while shifting yaw_deg -> yaw_deg - dtheta
    preserves world physical motion and camera images identically.
    Scaling issued XY by 1/m while scaling xy_scale -> xy_scale * m preserves physical motion identically.
    Flipping issued gripper by f in {-1, +1} while flipping gripper_sign -> gripper_sign * f preserves jaw motion identically.
    """
    out = {k: np.array(v, copy=True) for k, v in batch_np.items()}
    bsz = len(out["yaw_deg"])
    dtheta = rng.uniform(-180.0, 180.0, size=(bsz,)).astype(np.float32)
    rad = np.radians(dtheta)
    c = np.cos(rad)[:, None]
    s = np.sin(rad)[:, None]

    # Rotate issued_mean[..., :2] and delta_eef[..., :2] by R(dtheta)
    ix, iy = out["issued_mean"][..., 0].copy(), out["issued_mean"][..., 1].copy()
    out["issued_mean"][..., 0] = c * ix - s * iy
    out["issued_mean"][..., 1] = s * ix + c * iy
    dx, dy = out["delta_eef"][..., 0].copy(), out["delta_eef"][..., 1].copy()
    out["delta_eef"][..., 0] = c * dx - s * dy
    out["delta_eef"][..., 1] = s * dx + c * dy
    out["yaw_deg"] = ((out["yaw_deg"] - dtheta + 180.0) % 360.0 - 180.0).astype(np.float32)

    # Continuous scale multiplier m in [0.65, 1.55]
    m = np.exp(rng.uniform(np.log(0.65), np.log(1.55), size=(bsz,))).astype(np.float32)
    out["issued_mean"][..., :2] = out["issued_mean"][..., :2] / m[:, None, None]
    out["xy_scale"] = (out["xy_scale"] * m).astype(np.float32)

    # Random gripper sign flip f in {-1, +1}
    f = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=(bsz,))
    out["issued_mean"][..., 6] = out["issued_mean"][..., 6] * f[:, None]
    out["gripper_sign"] = (out["gripper_sign"] * f).astype(np.int32)
    return out


def train_transformer_aug(
    seed: int,
    train_data: dict,
    steps: int,
    batch_size: int,
    ckpt_dir: Path,
    archive_sha: str,
    cnn_width: int = 96,
    mlp_width: int = 192,
):
    max_chunks = int(train_data["valid"].shape[1])
    cfg = ReaderConfig(
        model_type="joint_transformer",
        max_chunks=max(max_chunks, 32),
        cnn_width=cnn_width,
        mlp_width=mlp_width,
        visual_mode="luminance_flow",
    )
    params = init_reader(jax.random.key(seed + 1000), cfg)
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=steps, alpha=0.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(p, os_state, batch):
        (loss, metrics), grads = jax.value_and_grad(lambda pp: reader_loss(pp, cfg, batch), has_aux=True)(p)
        updates, os_state = optimizer.update(grads, os_state, p)
        return optax.apply_updates(p, updates), os_state, loss, metrics

    rng = np.random.default_rng(seed + 1000)
    train_sub = {k: train_data[k] for k in (*HISTORY_FIELDS, "yaw_deg", "xy_scale", "gripper_sign")}
    n_train = len(train_sub["yaw_deg"])
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(n_train, size=batch_size)
        raw_batch = {k: v[idx] for k, v in train_sub.items()}
        aug_batch = augment_batch_exact_physics(raw_batch, rng)
        batch = {k: jnp.asarray(v) for k, v in aug_batch.items()}
        params, opt_state, loss, metrics = update_step(params, opt_state, batch)
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            curve.append(dict(step=step, loss=float(loss), yaw_loss=float(metrics["yaw_loss"])))

    ckpt_path = ckpt_dir / f"aug-active_probe-joint_transformer-seed{seed}.msgpack"
    save_checkpoint(ckpt_path, params, opt_state, steps, seed, "active_probe", cfg, 0.05, archive_sha, rng)
    return params, cfg, dict(seed=seed, seconds=round(time.monotonic() - started, 2), checkpoint=str(ckpt_path), curve=curve)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--aug-steps", type=int, default=6000)
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

    act_train, _ = load_mode_split(archive_dir, archive["trials"], "active_probe", "train")
    act_dev, act_dev_rows = load_mode_split(archive_dir, archive["trials"], "active_probe", "dev")
    pas_dev, pas_dev_rows = load_mode_split(archive_dir, archive["trials"], "passive_attempt1", "dev")

    # 1. Calibrate pure-NumPy dense windowed Lucas-Kanade optical flow on nominal (yaw=0) 7-DoF Panda/Sawyer traces
    print("Computing dense windowed Lucas-Kanade optical flow on train active_probe...", flush=True)
    nom_mask = np.abs(act_train["yaw_deg"]) < 1e-3
    tr_fb = farneback_image_motion(act_train["img_before"][nom_mask], act_train["img_after"][nom_mask], act_train["valid"][nom_mask])
    fb_proj = np.asarray(
        fit_visual_projection(tr_fb, act_train["delta_eef"][nom_mask], act_train["valid"][nom_mask]),
        dtype=np.float32,
    )

    dev_act_fb = farneback_image_motion(act_dev["img_before"], act_dev["img_after"], act_dev["valid"])
    y_fb_act = analytic_visual_yaw(issued_exposure(act_dev), dev_act_fb, act_dev["delta_eef"][..., 2], act_dev["valid"], fb_proj)
    dev_pas_fb = farneback_image_motion(pas_dev["img_before"], pas_dev["img_after"], pas_dev["valid"])
    y_fb_pas = analytic_visual_yaw(issued_exposure(pas_dev), dev_pas_fb, pas_dev["delta_eef"][..., 2], pas_dev["valid"], fb_proj)

    fb_summary = dict(
        method="dense_windowed_lucas_kanade",
        projection=fb_proj.tolist(),
        available=True,
        dev_active_yaw_err=summarize_yaw_by_regime(y_fb_act, act_dev, act_dev_rows),
        dev_passive_yaw_err=summarize_yaw_by_regime(y_fb_pas, pas_dev, pas_dev_rows),
    )
    write_json(args.output / "farneback-calib.json", fb_summary)
    print("DENSE LK FLOW CALIB:", json.dumps(fb_summary), flush=True)

    # 2. Train 3 seeds of RMA-GRU (unaugmented)
    donor_perm = make_donor_permutation(act_dev_rows, np.random.default_rng(20261120))
    gru_ret, gru_clr, gru_swp, gru_curves = [], [], [], []
    for seed in seeds:
        params, cfg, info = train_one(
            "active_probe", "rma_gru", seed, act_train, args.steps, args.batch_size,
            0.05, ckpt_dir, archive_sha, visual_mode="luminance_flow",
            cnn_width=args.cnn_width, mlp_width=args.mlp_width, tag_prefix="",
        )
        aud = audit_model(params, cfg, act_dev, act_dev_rows, donor_perm)
        gru_ret.append(aud["retained"])
        gru_clr.append(aud["cleared"])
        gru_swp.append(aud["swapped_donor"])
        gru_curves.append(info)
        print(
            json.dumps(dict(
                model="rma_gru", seed=seed,
                overall_yaw=round(aud["retained"]["overall"]["yaw_err_deg"], 2),
                fam_yaw=round(aud["retained"]["familiar"]["yaw_err_deg"], 2),
                comp_yaw=round(aud["retained"]["compositional_ood"]["yaw_err_deg"], 2),
                cont_yaw=round(aud["retained"]["continuous_angle_ood"]["yaw_err_deg"], 2),
            )),
            flush=True,
        )

    # 3. Train 3 seeds of Steel-Manned Yang et al.: Transformer + Exact Continuous Augmentation (6,000 steps!)
    aug_ret, aug_clr, aug_swp, aug_curves = [], [], [], []
    for seed in seeds:
        params, cfg, info = train_transformer_aug(
            seed, act_train, args.aug_steps, args.batch_size, ckpt_dir, archive_sha,
            cnn_width=args.cnn_width, mlp_width=args.mlp_width,
        )
        aud = audit_model(params, cfg, act_dev, act_dev_rows, donor_perm)
        aug_ret.append(aud["retained"])
        aug_clr.append(aud["cleared"])
        aug_swp.append(aud["swapped_donor"])
        aug_curves.append(info)
        print(
            json.dumps(dict(
                model="joint_transformer_aug", seed=seed,
                overall_yaw=round(aud["retained"]["overall"]["yaw_err_deg"], 2),
                fam_yaw=round(aud["retained"]["familiar"]["yaw_err_deg"], 2),
                comp_yaw=round(aud["retained"]["compositional_ood"]["yaw_err_deg"], 2),
                cont_yaw=round(aud["retained"]["continuous_angle_ood"]["yaw_err_deg"], 2),
            )),
            flush=True,
        )

    out = dict(
        archive_sha256=archive_sha,
        seeds=list(seeds),
        aug_steps=args.aug_steps,
        dense_lk_flow=fb_summary,
        rma_gru=dict(retained=aggregate_seeds(gru_ret), cleared=aggregate_seeds(gru_clr), swapped_donor=aggregate_seeds(gru_swp), training=gru_curves),
        joint_transformer_aug=dict(retained=aggregate_seeds(aug_ret), cleared=aggregate_seeds(aug_clr), swapped_donor=aggregate_seeds(aug_swp), training=aug_curves),
    )
    write_json(args.output / "baselines-audit.json", out)
    print("DONE BASELINES TRAINING (RMA-GRU + STEEL-MANNED YANG TRANSFORMER AUG + DENSE LK FLOW)", flush=True)


if __name__ == "__main__":
    main()
