"""Train cross-embodiment history readers across 3 seeds and produce Idea 3 offline audit."""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
import sys
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, circular_yaw_error_deg
from fluffy_bassoon.xemb.readers import (
    FRAME0_FIELDS,
    HISTORY_FIELDS,
    MODEL_NAMES,
    ReaderConfig,
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_visual_projection,
    image_motion,
    fit_nominal_gain,
    init_reader,
    predict_embodiment,
    reader_loss,
    visual_motion_vectors,
)
from collect import SUPPORT_BUDGET, issued_exposure, read_support, validate_archive
from motor_iface import write_json


SEEDS = (41, 42, 43)


def load_mode_split(archive_dir: Path, trials: list, mode: str, split_name: str):
    subset = [r for r in trials if r["split"] == split_name]
    tapes = [read_support(archive_dir, row, mode) for row in subset]
    data = {k: np.stack([t[k] for t in tapes]) for k in tapes[0]}
    for label in ("yaw_deg", "xy_scale", "gripper_sign"):
        data[label] = np.array([r[label] for r in subset], dtype=np.float32)
    return data, subset


def score_predictions(rows: list, yaw_hat, scale_hat, grip_hat):
    yaw_errs = np.array([circular_yaw_error_deg(yh, r["yaw_deg"]) for yh, r in zip(yaw_hat, rows, strict=True)], dtype=np.float64)
    scale_errs = np.array(
        [abs(float(sh) - float(r["xy_scale"])) / float(r["xy_scale"]) for sh, r in zip(scale_hat, rows, strict=True)],
        dtype=np.float64,
    )
    grip_ok = np.array([int(gh) == int(r["gripper_sign"]) for gh, r in zip(grip_hat, rows, strict=True)], dtype=bool)
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
    groups = {}
    for i, row in enumerate(rows):
        group = tuple(row[k] for k in ("suite", "task_id", "init_state_id", "seed"))
        groups.setdefault(group, []).append(i)
    perm = np.empty(len(rows), dtype=int)
    for indices in groups.values():
        if len({rows[i]["embodiment"] for i in indices}) != len(indices) or len(indices) < 2:
            raise ValueError("Swapping requires distinct factors on a matched reset")
        order = rng.permutation(indices)
        perm[order] = np.roll(order, 1)
    return perm


def predict_in_batches(params, cfg: ReaderConfig, data: dict, batch_size: int = 32):
    n = len(data["yaw_deg"])
    y_all, s_all, g_all = [], [], []
    fields = FRAME0_FIELDS if cfg.model_type == "frame0_probe" else HISTORY_FIELDS
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        sl = {k: data[k][start:end] for k in fields}
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


def save_checkpoint(path, params, opt_state, step, seed, mode, cfg, nominal_gain, archive_sha, sampler):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=params,
        archive_sha256=archive_sha,
        input_schema="rgb-unit-motion-cm-v1",
        sampler_state_json=json.dumps(sampler.bit_generator.state),
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


def train_one(mode, model_type, seed, train_data, steps, batch_size, nominal_gain, ckpt_dir, archive_sha):
    """Fit a fixed number of updates using fitting rows alone."""
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
    fields = FRAME0_FIELDS if model_type == "frame0_probe" else HISTORY_FIELDS
    train_data = {k: train_data[k] for k in (*fields, "yaw_deg", "xy_scale", "gripper_sign")}
    n_train = len(train_data["yaw_deg"])
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(n_train, size=batch_size)
        batch = {k: jnp.asarray(v[idx]) for k, v in train_data.items()}
        params, opt_state, loss, metrics = update_step(params, opt_state, batch)
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite {mode}/{model_type}/{seed} loss at update {step}")
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
    if not all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(params)):
        raise FloatingPointError(f"Nonfinite final weights for {mode}/{model_type}/{seed}")
    save_checkpoint(ckpt_path, params, opt_state, steps, seed, mode, cfg, nominal_gain, archive_sha, rng)
    return params, cfg, dict(seed=seed, seconds=round(time.monotonic() - started, 3),
                             checkpoint=str(ckpt_path), curve=curve)


def audit_model(params, cfg, dev_data, dev_rows, donor_perm):
    y_ret, s_ret, g_ret = predict_in_batches(params, cfg, dev_data)
    cleared_data = dict(dev_data, valid=np.zeros_like(dev_data["valid"]))
    y_clr, s_clr, g_clr = predict_in_batches(params, cfg, cleared_data)

    # Scoring the donor's retained prediction is identical to rerunning the reader
    # on that tape. All members of each group share the identical reset observation.
    if cfg.model_type == "frame0_probe":
        for a, b in zip((y_ret, s_ret, g_ret), (y_clr, s_clr, g_clr), strict=True):
            if not np.allclose(a, b, atol=1e-5, rtol=0) or not np.allclose(a, a[donor_perm], atol=1e-5, rtol=0):
                raise ValueError("Frame-0 prediction changed within a matched reset")
    diagnostics = {}
    if cfg.model_type == "factored_equiv":
        indices = [i for i, row in enumerate(dev_rows) if row["regime"] == "continuous_angle_ood"][:4]
        if indices:
            sample = {k: dev_data[k][indices] for k in ("img_before", "img_after", "valid")}
            vectors = np.asarray(visual_motion_vectors(params, sample))
            diagnostics["sample_cont_ood"] = [
                dict(embodiment=dev_rows[i]["embodiment"], true_yaw=dev_rows[i]["yaw_deg"], pred_yaw=float(y_ret[i]),
                     v_hat_first_chunks=vectors[j, :3, :2].tolist(), valid=sample["valid"][j, :3].tolist(),
                     issued_xy_first_chunks=dev_data["issued_mean"][i, :3, :2].tolist())
                for j, i in enumerate(indices)
            ]
    return dict(
        diagnostics=diagnostics, retained=score_predictions(dev_rows, y_ret, s_ret, g_ret),
        cleared=score_predictions(dev_rows, y_clr, s_clr, g_clr),
        swapped_donor=score_predictions(dev_rows, y_ret[donor_perm], s_ret[donor_perm], g_ret[donor_perm]),
    )


def signal_summary(data):
    """Measure observed motions in archive units, excluding unexecuted chunks."""
    valid = data["valid"]
    eef_mm = 1000 * np.linalg.norm(data["delta_eef"][valid], axis=-1)
    jaw_mm = 1000 * np.abs(data["delta_jaw"][..., 0][valid])
    before = data["img_before"][valid].astype(np.int16)
    diff = np.abs(data["img_after"][valid] - before)
    counts, frequencies = np.unique(data["chunk_steps"].sum(axis=1), return_counts=True)
    return dict(
        controls_per_tape={str(n): int(f) for n, f in zip(counts, frequencies, strict=True)},
        valid_chunks=int(valid.sum()), padded_chunks=int((~valid).sum()),
        mean_eef_motion_mm=float(eef_mm.mean()), mean_abs_jaw_motion_mm=float(jaw_mm.mean()),
        mean_abs_rgb_difference_0_255=float(diff.mean()),
    )


def analytic_audit(train_data, train_rows, dev_data, dev_rows, donor_perm):
    nom = np.array([r["embodiment"] == "nom" for r in train_rows])
    gain = fit_nominal_gain(issued_exposure(train_data)[nom], train_data["delta_eef"][nom],
                            train_data["valid"][nom])
    issued = issued_exposure(dev_data)
    deef, jaw, valid = (dev_data[k] for k in ("delta_eef", "delta_jaw", "valid"))
    yaw, scale, grip = analytic_proprio_sysid(issued, deef, jaw, valid, gain)
    cleared = score_predictions(dev_rows, *analytic_proprio_sysid(issued, deef, jaw, np.zeros_like(valid), gain))

    def report(y):
        return dict(retained=score_predictions(dev_rows, y, scale, grip), cleared=cleared,
                    swapped_donor=score_predictions(dev_rows, y[donor_perm], scale[donor_perm], grip[donor_perm]))

    nom_valid = train_data["valid"][nom]
    motion = image_motion(train_data["img_before"][nom], train_data["img_after"][nom], nom_valid)
    try:
        projection = fit_visual_projection(motion, train_data["delta_eef"][nom], nom_valid)
    except ValueError as err:
        visual = dict(available=False, reason=str(err))
    else:
        motion = image_motion(dev_data["img_before"], dev_data["img_after"], valid)
        visual = dict(available=True, projection=projection.tolist(),
                      **report(analytic_visual_yaw(issued, motion, deef[..., 2], valid, projection)))

    # These are the constant predictors for the fitting loss under its empirical factor frequencies.
    angles = np.deg2rad(train_data["yaw_deg"])
    prior = (np.degrees(np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())),
             np.exp(np.log(train_data["xy_scale"]).mean()), 1 if train_data["gripper_sign"].mean() >= 0 else -1)
    prior_scores = score_predictions(dev_rows, *(np.full(len(dev_rows), x) for x in prior))
    return dict(nominal_gain=gain, constant_nominal=cleared, training_prior=prior_scores,
                analytic_proprio_sysid=dict(nominal_gain=gain, **report(yaw)), analytic_visual_sysid=visual)


def offline_gate(report, threshold):
    """Apply the compute stop once, at the fixed final step of all active equivariant fits."""
    if threshold is None:
        return None, {"status": "unrequested"}
    active = report["modes"].get("active_probe", {})
    result = active.get("models", {}).get("factored_equiv", {})
    yaw = result.get("retained", {}).get("continuous_angle_ood", {}).get("yaw_err_deg", {}).get("mean")
    passed = yaw is not None and np.isfinite(yaw) and yaw < threshold
    return bool(passed), dict(threshold_deg=threshold, active_probe_cont_angle_yaw_err_deg=yaw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    parser.add_argument("--gate-yaw-deg", type=float, default=None)
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch size must be positive")
    if args.gate_yaw_deg is not None and not 0 < args.gate_yaw_deg <= 180:
        parser.error("gate angle must be in (0, 180]")
    raw_archive = args.archive.read_bytes()
    archive = json.loads(raw_archive)
    validate_archive(archive)
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    if len(set(seeds)) != len(seeds) or not seeds:
        parser.error("Supply distinct fitting seeds")
    if archive["panel"] == "full" and seeds != SEEDS:
        parser.error("Full panel requires fitting seeds 41,42,43")
    if args.gate_yaw_deg is not None and "active_probe" not in archive["modes"]:
        parser.error("The yaw gate requires active probes")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"
    ckpt_dir.mkdir()
    packages = {p: metadata.version(p) for p in ("jax", "jaxlib", "flax", "optax", "numpy")}
    devices = jax.devices()
    requested_backend = os.environ.get("JAX_PLATFORMS")
    if requested_backend == "cuda" and (len(devices) != 1 or devices[0].platform != "gpu"):
        raise RuntimeError(f"Expected one JAX CUDA device in the allocated runtime, got {devices}")
    if requested_backend == "cpu" and any(d.platform != "cpu" for d in devices):
        raise RuntimeError(f"CPU qualification exposed an accelerator: {devices}")
    runtime = dict(
        python=sys.executable, packages=packages, backend=jax.default_backend(),
        devices=[str(d) for d in devices],
        package_locations={p: str(metadata.distribution(p).locate_file("")) for p in packages},
    )
    report = dict(
        complete=False, panel=archive["panel"], archive=str(args.archive), archive_sha256=archive_sha,
        matched_support_cap=SUPPORT_BUDGET, runtime=runtime, steps=args.steps,
        batch_size=args.batch_size, seeds=list(seeds), modes={}, offline_gate_passed=None,
    )
    for mode in archive["modes"]:
        train_data, train_rows = load_mode_split(args.archive.parent, archive["trials"], mode, "train")
        dev_data, dev_rows = load_mode_split(args.archive.parent, archive["trials"], mode, "dev")
        donor_perm = make_donor_permutation(dev_rows, np.random.default_rng(20261119))
        result = analytic_audit(train_data, train_rows, dev_data, dev_rows, donor_perm)
        models = {}
        report["modes"][mode] = dict(**result, models=models, train_trials=len(train_rows), dev_trials=len(dev_rows),
                                     train_signal=signal_summary(train_data), dev_signal=signal_summary(dev_data),
                                     donor_indices=[dev_rows[i]["index"] for i in donor_perm])
        for model_type in MODEL_NAMES:
            per_seed = []
            for seed in seeds:
                params, cfg, fit = train_one(mode, model_type, seed, train_data, args.steps, args.batch_size,
                                             result["nominal_gain"], ckpt_dir, archive_sha)
                train_predictions = predict_in_batches(params, cfg, train_data)
                fit["training"] = score_predictions(train_rows, *train_predictions)
                fit["fitting"] = score_predictions(train_rows, *predict_in_batches(params, cfg, train_data))
                fit.update(audit_model(params, cfg, dev_data, dev_rows, donor_perm))
                per_seed.append(fit)
                print(json.dumps(dict(mode=mode, model=model_type, seed=seed, overall=fit["retained"]["overall"])),
                      flush=True)
            models[model_type] = {k: aggregate_seeds([s[k] for s in per_seed])
                                  for k in ("retained", "cleared", "swapped_donor")}
            models[model_type]["seeds"] = per_seed
    report["offline_gate_passed"], report["offline_gate_details"] = offline_gate(report, args.gate_yaw_deg)
    report["complete"] = True
    write_json(args.output / "offline-audit.json", report)
    print(json.dumps(dict(offline_gate_passed=report["offline_gate_passed"])), flush=True)
    if report["offline_gate_passed"] is False:
        raise SystemExit(4)


if __name__ == "__main__":
    main()
