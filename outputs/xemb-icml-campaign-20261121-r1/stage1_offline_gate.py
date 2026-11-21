"""Stage 1 Gate: Benchmark V2 equivariant readers + full competitive baselines on 24-task archive.

Supports:
- `--analytic-only`: CPU-only evaluation of ReGround babbling, Horn-Schunck analytic visual,
  OpenCV DIS dense flow + Procrustes, and online Broyden UVS on the existing 1,344-trial
  7-DoF Panda/Sawyer -> 96-trial held-out 6-DoF UR5e archive (0 GPUh).
- Full GPU mode: trains 3-seed `factored_equiv_v2` (dis_lumaflow+dz, lumaflow+dz, lumaflow_nodz),
  `factored_no_equiv_v2`, and `perceiver_icba` (Yang, Finn & Sadigh style), and audits all
  models on held-out 6-DoF UR5e across familiar / compositional OOD / continuous-angle OOD
  + cleared / swapped donor controls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
SCALE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-scale-20261120-r1"
for d in (BASE_DIR, SCALE_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from fluffy_bassoon.xemb.baselines import (
    broyden_uvs_yaw,
    compute_dis_flow_maps,
    dense_flow_image_motion,
    fit_dense_flow_projection,
    reground_babbling_sysid,
)
from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, circular_yaw_error_deg
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    init_reader_v2,
    predict_embodiment_v2,
    reader_loss_v2,
)
from fluffy_bassoon.xemb.readers import (
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
)
from train_ur5e_readers import load_mode_split, make_donor_permutation
from ur5e_motor import write_json


def score_predictions(rows, y_true, s_true, g_true, y_hat, s_hat, g_hat):
    out = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        if reg == "overall":
            idx = list(range(len(rows)))
        elif reg == "ood_combined":
            idx = [i for i, r in enumerate(rows) if r["regime"] in ("compositional_ood", "continuous_angle_ood")]
        else:
            idx = [i for i, r in enumerate(rows) if r["regime"] == reg]
        if not idx:
            continue
        y_err = [circular_yaw_error_deg(float(y_hat[i]), float(y_true[i])) for i in idx]
        s_rel = [abs(float(s_hat[i]) - float(s_true[i])) / max(float(s_true[i]), 1e-6) for i in idx]
        g_acc = [int(int(g_hat[i]) == int(g_true[i])) for i in idx]
        all_ok = [
            int(ye <= 22.5 and sr <= 0.25 and ga == 1)
            for ye, sr, ga in zip(y_err, s_rel, g_acc, strict=True)
        ]
        out[reg] = dict(
            n=len(idx),
            yaw_err_deg=round(float(np.mean(y_err)), 2),
            yaw_med_deg=round(float(np.median(y_err)), 2),
            scale_rel_err=round(float(np.mean(s_rel)), 4),
            grip_acc=round(float(np.mean(g_acc)), 4),
            all_correct=round(float(np.mean(all_ok)), 4),
        )
    return out


def run_analytic_baselines(train_data, train_rows, dev_data, dev_rows, clean_hs_proj=None, clean_dis_proj=None):
    nom_idx = np.array([i for i, r in enumerate(train_rows) if r["embodiment"] == "nom"], dtype=np.int64)
    nom_gain = fit_nominal_gain(
        train_data["issued_mean"][nom_idx],
        train_data["delta_eef"][nom_idx],
        train_data["valid"][nom_idx],
    )
    y_t, s_t, g_t = dev_data["yaw_deg"], dev_data["xy_scale"], dev_data["gripper_sign"]

    # 1. Analytic proprioception (yaw-blind)
    yp, sp, gp = analytic_proprio_sysid(
        dev_data["issued_mean"], dev_data["delta_eef"], dev_data["delta_jaw"], dev_data["valid"],
        nominal_gain_m_per_cmd=nom_gain,
    )
    # 2. ReGround motor-babbling affine/Procrustes baseline (Siddique et al., 2026)
    yr, sr, gr = reground_babbling_sysid(
        dev_data["issued_mean"], dev_data["delta_eef"], dev_data["delta_jaw"], dev_data["valid"],
        nominal_gain_m_per_cmd=nom_gain,
    )

    # 3. Horn-Schunck brightness constancy visual SysID + Procrustes
    if clean_hs_proj is None:
        tr_hs = image_motion(train_data["img_before"][nom_idx], train_data["img_after"][nom_idx], train_data["valid"][nom_idx])
        hs_proj = fit_visual_projection(tr_hs, train_data["delta_eef"][nom_idx], train_data["valid"][nom_idx])
    else:
        hs_proj = clean_hs_proj
    dev_hs = image_motion(dev_data["img_before"], dev_data["img_after"], dev_data["valid"])
    y_hs = analytic_visual_yaw(dev_data["issued_mean"], dev_hs, dev_data["delta_eef"][..., 2], dev_data["valid"], hs_proj)

    # 4. OpenCV DIS dense optical flow + Procrustes
    if clean_dis_proj is None:
        dis_proj = fit_dense_flow_projection(
            train_data["img_before"][nom_idx],
            train_data["img_after"][nom_idx],
            train_data["delta_eef"][nom_idx],
            train_data["valid"][nom_idx],
            method="dis",
        )
    else:
        dis_proj = clean_dis_proj
    dev_dis = dense_flow_image_motion(dev_data["img_before"], dev_data["img_after"], dev_data["valid"], method="dis")
    y_dis = analytic_visual_yaw(dev_data["issued_mean"], dev_dis, dev_data["delta_eef"][..., 2], dev_data["valid"], dis_proj)

    # 5. Online Broyden UVS Jacobian estimator (both Horn-Schunck and DIS motion)
    y_broy_hs = broyden_uvs_yaw(dev_data["issued_mean"], dev_hs, dev_data["delta_eef"][..., 2], dev_data["valid"], hs_proj)
    y_broy_dis = broyden_uvs_yaw(dev_data["issued_mean"], dev_dis, dev_data["delta_eef"][..., 2], dev_data["valid"], dis_proj)

    results = dict(
        nominal_gain=float(nom_gain),
        analytic_proprio_sysid=score_predictions(dev_rows, y_t, s_t, g_t, yp, sp, gp),
        reground_babbling=score_predictions(dev_rows, y_t, s_t, g_t, yr, sr, gr),
        analytic_visual_hs=score_predictions(dev_rows, y_t, s_t, g_t, y_hs, sp, gp),
        dense_flow_dis_procrustes=score_predictions(dev_rows, y_t, s_t, g_t, y_dis, sp, gp),
        broyden_uvs_hs=score_predictions(dev_rows, y_t, s_t, g_t, y_broy_hs, sp, gp),
        broyden_uvs_dis=score_predictions(dev_rows, y_t, s_t, g_t, y_broy_dis, sp, gp),
    )
    return results, hs_proj, dis_proj, nom_gain


def train_and_audit_v2(
    mode: str,
    tag: str,
    cfg: ReaderV2Config,
    train_data: dict,
    dev_data: dict,
    dev_rows: list,
    donor_perm: np.ndarray,
    *,
    seeds: tuple[int, ...],
    steps: int,
    batch_size: int,
    ckpt_dir: Path,
    nominal_gain: float,
    archive_sha: str,
):
    from flax import serialization
    import jax
    import jax.numpy as jnp
    import optax

    fields = ["img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid"]
    if cfg.visual_mode == "dis_lumaflow":
        fields.append("dis_flow")

    tr_sub = {k: train_data[k] for k in (*fields, "yaw_deg", "xy_scale", "gripper_sign")}
    dv_in = {k: dev_data[k] for k in fields}
    dv_clr = dict(dv_in, valid=np.zeros_like(dv_in["valid"]))
    dv_swp = {k: dv_in[k][donor_perm] for k in fields}
    y_t, s_t, g_t = dev_data["yaw_deg"], dev_data["xy_scale"], dev_data["gripper_sign"]
    n_train = len(tr_sub["yaw_deg"])

    seed_results = []
    for seed in seeds:
        params = init_reader_v2(jax.random.key(seed), cfg)
        schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=steps, alpha=0.05)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
        opt_state = optimizer.init(params)

        @jax.jit
        def update_step(p, os_state, batch):
            (loss, metrics), grads = jax.value_and_grad(lambda pp: reader_loss_v2(pp, cfg, batch), has_aux=True)(p)
            updates, os_state = optimizer.update(grads, os_state, p)
            return optax.apply_updates(p, updates), os_state, loss, metrics

        rng = np.random.default_rng(seed)
        t0 = time.monotonic()
        for step in range(1, steps + 1):
            idx = rng.integers(n_train, size=batch_size)
            batch = {k: jnp.asarray(v[idx]) for k, v in tr_sub.items()}
            params, opt_state, loss, _ = update_step(params, opt_state, batch)
            if step == steps and not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite loss for {tag}/{mode}/seed{seed}")

        yh, sh, gh = predict_embodiment_v2(params, cfg, dv_in)
        yc, sc, gc = predict_embodiment_v2(params, cfg, dv_clr)
        ys, ss, gs = predict_embodiment_v2(params, cfg, dv_swp)

        ckpt_path = ckpt_dir / f"{tag}-{mode}-seed{seed}.msgpack"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(
            params=params,
            reader_v2_config=asdict(cfg),
            mode=mode,
            tag=tag,
            seed=seed,
            steps=steps,
            nominal_gain=float(nominal_gain),
            archive_sha256=archive_sha,
        )
        tmp = ckpt_path.with_suffix(".tmp")
        tmp.write_bytes(serialization.to_bytes(payload))
        tmp.replace(ckpt_path)

        seed_results.append(
            dict(
                seed=seed,
                seconds=round(time.monotonic() - t0, 2),
                matched=score_predictions(dev_rows, y_t, s_t, g_t, yh, sh, gh),
                cleared=score_predictions(dev_rows, y_t, s_t, g_t, yc, sc, gc),
                swapped=score_predictions(dev_rows, y_t, s_t, g_t, ys, ss, gs),
                checkpoint=str(ckpt_path),
            )
        )

    agg = {}
    for reg in (*EVAL_REGIMES, "ood_combined", "overall"):
        agg[reg] = dict(
            yaw_err_deg=round(float(np.mean([r["matched"][reg]["yaw_err_deg"] for r in seed_results])), 2),
            all_correct=round(float(np.mean([r["matched"][reg]["all_correct"] for r in seed_results])), 4),
        )
    agg["cleared_yaw_deg"] = round(float(np.mean([r["cleared"]["overall"]["yaw_err_deg"] for r in seed_results])), 2)
    agg["swapped_yaw_deg"] = round(float(np.mean([r["swapped"]["overall"]["yaw_err_deg"] for r in seed_results])), 2)
    return dict(config=asdict(cfg), aggregate=agg, seeds=seed_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analytic-only", action="store_true")
    parser.add_argument("--steps", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--only-models", type=str, default="")
    args = parser.parse_args()

    raw = args.archive.read_bytes()
    archive_sha = hashlib.sha256(raw).hexdigest()
    archive = json.loads(raw)
    archive_dir = args.archive.parent
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    report = dict(archive=str(args.archive), archive_sha256=archive_sha, analytic_only=args.analytic_only, modes={})
    clean_hs_proj = None
    clean_dis_proj = None

    for mode in archive["modes"]:
        t0 = time.monotonic()
        train_data, train_rows = load_mode_split(archive_dir, archive["trials"], mode, "train")
        dev_data, dev_rows = load_mode_split(archive_dir, archive["trials"], mode, "dev")
        donor_perm = make_donor_permutation(dev_rows, np.random.default_rng(20261121))

        an, hs_proj, dis_proj, nom_gain = run_analytic_baselines(
            train_data,
            train_rows,
            dev_data,
            dev_rows,
            clean_hs_proj=(clean_hs_proj if mode == "passive_attempt1" else None),
            clean_dis_proj=(clean_dis_proj if mode == "passive_attempt1" else None),
        )
        if mode == "active_probe":
            clean_hs_proj, clean_dis_proj = hs_proj, dis_proj

        print(f"=== MODE {mode} ANALYTIC BASELINES ({time.monotonic()-t0:.1f}s) ===", flush=True)
        for name in ("analytic_proprio_sysid", "reground_babbling", "analytic_visual_hs", "dense_flow_dis_procrustes", "broyden_uvs_hs", "broyden_uvs_dis"):
            ov = an[name]["overall"]
            ood = an[name]["ood_combined"]
            print(f"  {name:28s}: overall={ov['yaw_err_deg']:6.2f} deg (ok={ov['all_correct']:.3f}) | ood={ood['yaw_err_deg']:6.2f} deg", flush=True)

        models_out = {}
        if not args.analytic_only:
            t_flow = time.monotonic()
            train_data["dis_flow"] = compute_dis_flow_maps(train_data["img_before"], train_data["img_after"], train_data["valid"])
            dev_data["dis_flow"] = compute_dis_flow_maps(dev_data["img_before"], dev_data["img_after"], dev_data["valid"])
            print(f"Computed DIS optical-flow maps for {mode} in {time.monotonic()-t_flow:.1f}s", flush=True)

            max_chunks = int(train_data["valid"].shape[1])
            specs = [
                ("equiv_flow_homog_dz", ReaderV2Config(model_type="equiv_flow_homography", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32), use_dz_parallax=True)),
                ("equiv_flow_homog_nodz", ReaderV2Config(model_type="equiv_flow_homography", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32), use_dz_parallax=False)),
                ("equiv_v2_dis_dz", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32), use_dz_parallax=True)),
                ("equiv_v2_dis_nodz", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32), use_dz_parallax=False)),
                ("equiv_v2_luma_dz", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="luminance_flow", max_chunks=max(max_chunks, 32), use_dz_parallax=True)),
                ("equiv_v2_rgb_dz", ReaderV2Config(model_type="factored_equiv_v2", visual_mode="rgb", max_chunks=max(max_chunks, 32), use_dz_parallax=True)),
                ("no_equiv_v2_dis", ReaderV2Config(model_type="factored_no_equiv_v2", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32))),
                ("perceiver_icba_dis", ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=max(max_chunks, 32))),
            ]
            only = {s.strip() for s in args.only_models.split(",") if s.strip()}
            if only:
                specs = [(t, c) for t, c in specs if t in only]
            for tag, cfg in specs:
                res = train_and_audit_v2(
                    mode, tag, cfg, train_data, dev_data, dev_rows, donor_perm,
                    seeds=seeds, steps=args.steps, batch_size=args.batch_size,
                    ckpt_dir=args.output / "checkpoints", nominal_gain=nom_gain, archive_sha=archive_sha,
                )
                models_out[tag] = res
                ag = res["aggregate"]
                print(
                    f"  {tag:24s}: overall={ag['overall']['yaw_err_deg']:6.2f} deg (ok={ag['overall']['all_correct']:.3f}) | "
                    f"fam={ag['familiar']['yaw_err_deg']:5.2f} comp={ag['compositional_ood']['yaw_err_deg']:5.2f} cont={ag['continuous_angle_ood']['yaw_err_deg']:5.2f} | "
                    f"clr={ag['cleared_yaw_deg']:5.1f} swp={ag['swapped_yaw_deg']:5.1f}",
                    flush=True,
                )

        report["modes"][mode] = dict(analytic=an, models=models_out)

    write_json(args.output / "stage1-offline-report.json", report)
    print(f"Wrote {args.output / 'stage1-offline-report.json'}", flush=True)


if __name__ == "__main__":
    main()
