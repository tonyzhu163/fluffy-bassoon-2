"""Train cross-embodiment history readers across 3 seeds and produce Idea 3 offline audit."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, circular_yaw_error_deg
from fluffy_bassoon.xemb.readers import (
    MODEL_NAMES,
    ReaderConfig,
    analytic_proprio_sysid,
    apply_reader,
    fit_nominal_gain,
    init_reader,
    predict_embodiment,
    reader_loss,
)
from motor_iface import write_json


SEEDS = (41, 42, 43)


def load_mode_split(archive_dir: Path, trials: list, mode: str, split_name: str):
    key_name = "active_path" if mode == "active_probe" else "passive_path"
    subset = [t for t in trials if t["split"] == split_name]
    if not subset:
        raise ValueError(f"No trials for split={split_name!r}")

    ib, ia, im, de, dj, val, f0i, f0s = [], [], [], [], [], [], [], []
    yaws, scales, grips = [], [], []
    for row in subset:
        with np.load(archive_dir / row[key_name], allow_pickle=False) as npz:
            ib.append(npz["img_before"].astype(np.float32) / 255.0)
            ia.append(npz["img_after"].astype(np.float32) / 255.0)
            im.append(npz["issued_mean"].astype(np.float32))
            de.append(npz["delta_eef"].astype(np.float32))
            dj.append(npz["delta_jaw"].astype(np.float32))
            val.append(npz["valid"].astype(bool))
            f0i.append(npz["frame0_img"].astype(np.float32) / 255.0)
            f0s.append(npz["frame0_state"].astype(np.float32))
        yaws.append(float(row["yaw_deg"]))
        scales.append(float(row["xy_scale"]))
        grips.append(int(row["gripper_sign"]))

    return (
        dict(
            img_before=np.stack(ib),
            img_after=np.stack(ia),
            issued_mean=np.stack(im),
            delta_eef=np.stack(de),
            delta_jaw=np.stack(dj),
            valid=np.stack(val),
            frame0_img=np.stack(f0i),
            frame0_state=np.stack(f0s),
            yaw_deg=np.asarray(yaws, dtype=np.float32),
            xy_scale=np.asarray(scales, dtype=np.float32),
            gripper_sign=np.asarray(grips, dtype=np.int32),
        ),
        subset,
    )


def score_predictions(rows: list, yaw_hat, scale_hat, grip_hat):
    yaw_errs = np.array([circular_yaw_error_deg(yh, r["yaw_deg"]) for yh, r in zip(yaw_hat, rows)], dtype=np.float64)
    scale_errs = np.array(
        [abs(float(sh) - float(r["xy_scale"])) / float(r["xy_scale"]) for sh, r in zip(scale_hat, rows)],
        dtype=np.float64,
    )
    grip_ok = np.array([int(gh) == int(r["gripper_sign"]) for gh, r in zip(grip_hat, rows)], dtype=bool)
    all_ok = (yaw_errs < 15.0) & (scale_errs < 0.20) & grip_ok

    out = {}
    for regime in (*EVAL_REGIMES, "overall"):
        mask = np.ones(len(rows), dtype=bool) if regime == "overall" else np.array([r["regime"] == regime for r in rows])
        if not mask.any():
            continue
        out[regime] = dict(
            trials=int(mask.sum()),
            yaw_err_deg=float(yaw_errs[mask].mean()),
            scale_rel_err=float(scale_errs[mask].mean()),
            grip_acc=float(grip_ok[mask].mean()),
            all_correct=float(all_ok[mask].mean()),
        )
    return out


def make_donor_permutation(rows: list, rng: np.random.Generator) -> np.ndarray:
    """Select same-task mismatched-embodiment donor indices deterministically."""
    perm = np.arange(len(rows))
    for i, row in enumerate(rows):
        same_task = [
            j
            for j, cand in enumerate(rows)
            if cand["suite"] == row["suite"]
            and cand["task_id"] == row["task_id"]
            and cand["embodiment"] != row["embodiment"]
        ]
        if same_task:
            perm[i] = int(same_task[int(rng.integers(len(same_task)))])
    return perm


def predict_in_batches(params, cfg: ReaderConfig, data: dict, batch_size: int = 32):
    n = len(data["yaw_deg"])
    y_all, s_all, g_all = [], [], []
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sl = {k: v[start:end] for k, v in data.items()}
        yh, sh, gh = predict_embodiment(params, cfg, sl)
        y_all.append(yh)
        s_all.append(sh)
        g_all.append(gh)
    return np.concatenate(y_all), np.concatenate(s_all), np.concatenate(g_all)


def aggregate_seeds(seed_reports: list) -> dict:
    regimes = seed_reports[0].keys()
    summary = {}
    for reg in regimes:
        summary[reg] = dict(trials=seed_reports[0][reg]["trials"])
        for metric in ("yaw_err_deg", "scale_rel_err", "grip_acc", "all_correct"):
            vals = np.array([rep[reg][metric] for rep in seed_reports], dtype=np.float64)
            summary[reg][metric] = dict(mean=float(vals.mean()), std=float(vals.std(ddof=0)), values=vals.tolist())
    return summary


def save_checkpoint(path: Path, params, opt_state, step: int, seed: int, mode: str, cfg: ReaderConfig, nominal_gain: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=params,
        optimizer=opt_state,
        step=step,
        seed=seed,
        mode=mode,
        reader_config=asdict(cfg),
        nominal_gain=float(nominal_gain),
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(serialization.to_bytes(payload))
    tmp.replace(path)


def train_one(mode: str, model_type: str, seed: int, train_data: dict, dev_data: dict, dev_rows: list, steps: int, batch_size: int, nominal_gain: float, ckpt_dir: Path):
    max_chunks = int(train_data["valid"].shape[1])
    cfg = ReaderConfig(model_type=model_type, max_chunks=max(max_chunks, 32))
    params = init_reader(jax.random.key(seed), cfg)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(1e-3, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(p, os_state, batch):
        (loss, metrics), grads = jax.value_and_grad(lambda pp: reader_loss(pp, cfg, batch), has_aux=True)(p)
        updates, os_state = optimizer.update(grads, os_state, p)
        return optax.apply_updates(p, updates), os_state, loss, metrics

    rng = np.random.default_rng(seed)
    n_train = len(train_data["yaw_deg"])
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(n_train, size=batch_size)
        batch = {k: jnp.asarray(v[idx]) for k, v in train_data.items()}
        params, opt_state, loss, metrics = update_step(params, opt_state, batch)
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            curve.append(
                dict(
                    step=step,
                    loss=float(loss),
                    yaw_loss=float(metrics["yaw_loss"]),
                    scale_loss=float(metrics["scale_loss"]),
                    grip_loss=float(metrics["grip_loss"]),
                )
            )

    ckpt_path = ckpt_dir / f"{mode}-{model_type}-seed{seed}.msgpack"
    save_checkpoint(ckpt_path, params, opt_state, steps, seed, mode, cfg, nominal_gain)

    y_ret, s_ret, g_ret = predict_in_batches(params, cfg, dev_data)
    cleared_data = dict(dev_data, valid=np.zeros_like(dev_data["valid"]))
    y_clr, s_clr, g_clr = predict_in_batches(params, cfg, cleared_data)

    donor_perm = make_donor_permutation(dev_rows, np.random.default_rng(seed + 5000))
    donor_data = dict(dev_data)
    for field in ("img_before", "img_after", "issued_mean", "delta_eef", "delta_jaw", "valid"):
        donor_data[field] = dev_data[field][donor_perm]
    y_don, s_don, g_don = predict_in_batches(params, cfg, donor_data)

    diagnostics = {}
    if model_type == "factored_equiv":
        # Save sample visual vectors on continuous angle OOD for debugging if needed.
        from fluffy_bassoon.xemb.readers import _apply_cnn, _apply_mlp, _vis_inputs

        vis_in, b, k = _vis_inputs(jnp.asarray(dev_data["img_before"]), jnp.asarray(dev_data["img_after"]))
        vis = _apply_cnn(params["cnn"], vis_in).reshape(b, k, cfg.cnn_width)
        vec_out = np.asarray(_apply_mlp(params["vis_vec"], vis))
        diagnostics["sample_cont_ood"] = [
            dict(
                embodiment=dev_rows[i]["embodiment"],
                true_yaw=float(dev_rows[i]["yaw_deg"]),
                pred_yaw=float(y_ret[i]),
                v_hat_first_chunks=vec_out[i, :3, :2].tolist(),
                issued_xy_first_chunks=dev_data["issued_mean"][i, :3, :2].tolist(),
            )
            for i in range(len(dev_rows))
            if dev_rows[i]["regime"] == "continuous_angle_ood"
        ][:4]

    return dict(
        seed=seed,
        seconds=round(time.monotonic() - started, 3),
        checkpoint=str(ckpt_path),
        curve=curve,
        retained=score_predictions(dev_rows, y_ret, s_ret, g_ret),
        cleared=score_predictions(dev_rows, y_clr, s_clr, g_clr),
        swapped_donor=score_predictions(dev_rows, y_don, s_don, g_don),
        diagnostics=diagnostics,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--gate-yaw-deg", type=float, default=None)
    args = parser.parse_args()

    archive = json.loads(args.archive.read_text())
    archive_dir = args.archive.parent
    trials = archive["trials"]
    modes = tuple(archive["modes"])
    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    report = dict(archive=str(args.archive), steps=args.steps, batch_size=args.batch_size, seeds=list(seeds), modes={})

    for mode in modes:
        train_data, train_rows = load_mode_split(archive_dir, trials, mode, "train")
        dev_data, dev_rows = load_mode_split(archive_dir, trials, mode, "dev")

        nom_mask = np.array([r["embodiment"] == "nom" for r in train_rows], dtype=bool)
        if not nom_mask.any():
            raise ValueError("Need 'nom' trials in train split to calibrate nominal gain")
        nominal_gain = fit_nominal_gain(
            train_data["issued_mean"][nom_mask],
            train_data["delta_eef"][nom_mask],
            train_data["valid"][nom_mask],
        )

        # Analytic proprioceptive baseline.
        ya, sa, ga = analytic_proprio_sysid(
            dev_data["issued_mean"],
            dev_data["delta_eef"],
            dev_data["delta_jaw"],
            dev_data["valid"],
            nominal_gain_m_per_cmd=nominal_gain,
        )
        yc, sc, gc = analytic_proprio_sysid(
            dev_data["issued_mean"],
            dev_data["delta_eef"],
            dev_data["delta_jaw"],
            np.zeros_like(dev_data["valid"]),
            nominal_gain_m_per_cmd=nominal_gain,
        )
        analytic_rep = dict(
            nominal_gain=float(nominal_gain),
            retained=score_predictions(dev_rows, ya, sa, ga),
            cleared=score_predictions(dev_rows, yc, sc, gc),
        )

        models_rep = {}
        for model_type in MODEL_NAMES:
            per_seed = []
            for seed in seeds:
                res = train_one(
                    mode,
                    model_type,
                    seed,
                    train_data,
                    dev_data,
                    dev_rows,
                    args.steps,
                    args.batch_size,
                    nominal_gain,
                    ckpt_dir,
                )
                per_seed.append(res)
                print(
                    json.dumps(
                        dict(
                            mode=mode,
                            model=model_type,
                            seed=seed,
                            seconds=res["seconds"],
                            overall=res["retained"]["overall"],
                            comp_ood=res["retained"].get("compositional_ood"),
                            cont_ood=res["retained"].get("continuous_angle_ood"),
                        )
                    ),
                    flush=True,
                )
            models_rep[model_type] = dict(
                retained=aggregate_seeds([s["retained"] for s in per_seed]),
                cleared=aggregate_seeds([s["cleared"] for s in per_seed]),
                swapped_donor=aggregate_seeds([s["swapped_donor"] for s in per_seed]),
                seeds=per_seed,
            )

        report["modes"][mode] = dict(
            nominal_gain=float(nominal_gain),
            train_trials=len(train_rows),
            dev_trials=len(dev_rows),
            analytic_proprio_sysid=analytic_rep,
            models=models_rep,
        )

    gate_passed = True
    gate_details = {}
    if "active_probe" in report["modes"] and "continuous_angle_ood" in report["modes"]["active_probe"]["models"]["factored_equiv"]["retained"]:
        cont_yaw = report["modes"]["active_probe"]["models"]["factored_equiv"]["retained"]["continuous_angle_ood"]["yaw_err_deg"]["mean"]
        gate_details["active_probe_cont_angle_yaw_err_deg"] = cont_yaw
        if args.gate_yaw_deg is not None:
            gate_passed = cont_yaw < float(args.gate_yaw_deg)
    report["offline_gate_passed"] = gate_passed
    report["offline_gate_details"] = gate_details

    write_json(args.output / "offline-audit.json", report)
    print(json.dumps(dict(offline_gate_passed=gate_passed, gate_details=gate_details), indent=2), flush=True)
    if args.gate_yaw_deg is not None and not gate_passed:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
