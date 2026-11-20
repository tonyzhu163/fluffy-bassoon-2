"""Train 3-seed cross-morphology readers on 7-DoF Panda/Sawyer and run Direction 3 audit on held-out 6-DoF UR5e."""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fluffy_bassoon.xemb.embodiment import EVAL_REGIMES, circular_yaw_error_deg
from fluffy_bassoon.xemb.joint_audit import evaluate_cross_morphology_joint_jacobian
from fluffy_bassoon.xemb.readers import (
    FRAME0_FIELDS,
    HISTORY_FIELDS,
    MODEL_NAMES,
    ReaderConfig,
    analytic_proprio_sysid,
    analytic_visual_yaw,
    fit_nominal_gain,
    fit_visual_projection,
    image_motion,
    init_reader,
    predict_embodiment,
    reader_loss,
)
from collect_ur5e import DEV_ARM, SUPPORT_BUDGET, TRAIN_ARMS, issued_exposure, read_support, validate_archive
from ur5e_motor import write_json


SEEDS = (41, 42, 43)


def load_mode_split(archive_dir: Path, trials: list, mode: str, split_name: str):
    subset = [r for r in trials if r["split"] == split_name]
    tapes = [read_support(archive_dir, row, mode) for row in subset]
    data = {k: np.stack([t[k] for t in tapes]) for k in tapes[0]}
    for label in ("yaw_deg", "xy_scale", "gripper_sign"):
        data[label] = np.array([r[label] for r in subset], dtype=np.float32)
    return data, subset


def score_predictions(rows: list, yaw_hat, scale_hat, grip_hat):
    yaw_errs = np.array(
        [circular_yaw_error_deg(yh, r["yaw_deg"]) for yh, r in zip(yaw_hat, rows, strict=True)],
        dtype=np.float64,
    )
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
        group = tuple(row[k] for k in ("arm", "suite", "task_id", "init_state_id", "seed"))
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
    return params, cfg, dict(seed=seed, seconds=round(time.monotonic() - started, 3), checkpoint=str(ckpt_path), curve=curve)


def audit_model(params, cfg, dev_data, dev_rows, donor_perm):
    y_ret, s_ret, g_ret = predict_in_batches(params, cfg, dev_data)
    cleared_data = dict(dev_data, valid=np.zeros_like(dev_data["valid"]))
    y_clr, s_clr, g_clr = predict_in_batches(params, cfg, cleared_data)

    if cfg.model_type == "frame0_probe":
        for a, b in zip((y_ret, s_ret, g_ret), (y_clr, s_clr, g_clr), strict=True):
            if not np.allclose(a, b, atol=1e-5, rtol=0) or not np.allclose(a, a[donor_perm], atol=1e-5, rtol=0):
                raise ValueError("Frame-0 prediction changed within a matched reset")
    return dict(
        retained=score_predictions(dev_rows, y_ret, s_ret, g_ret),
        cleared=score_predictions(dev_rows, y_clr, s_clr, g_clr),
        swapped_donor=score_predictions(dev_rows, y_ret[donor_perm], s_ret[donor_perm], g_ret[donor_perm]),
    )


def analytic_audit(train_data, train_rows, dev_data, dev_rows, donor_perm, *, clean_visual_projection=None):
    nom = np.array([r["embodiment"] == "nom" for r in train_rows])
    gain = fit_nominal_gain(issued_exposure(train_data)[nom], train_data["delta_eef"][nom], train_data["valid"][nom])
    issued = issued_exposure(dev_data)
    deef, jaw, valid = (dev_data[k] for k in ("delta_eef", "delta_jaw", "valid"))
    yaw, scale, grip = analytic_proprio_sysid(issued, deef, jaw, valid, gain)
    cleared = score_predictions(dev_rows, *analytic_proprio_sysid(issued, deef, jaw, np.zeros_like(valid), gain))

    def report(y):
        return dict(
            retained=score_predictions(dev_rows, y, scale, grip),
            cleared=cleared,
            swapped_donor=score_predictions(dev_rows, y[donor_perm], scale[donor_perm], grip[donor_perm]),
        )

    # Analytic visual baseline (mode-matched projection).
    nom_valid = train_data["valid"][nom]
    train_motion = image_motion(train_data["img_before"][nom], train_data["img_after"][nom], nom_valid)
    dev_motion = image_motion(dev_data["img_before"], dev_data["img_after"], valid)
    try:
        projection = fit_visual_projection(train_motion, train_data["delta_eef"][nom], nom_valid)
    except ValueError as err:
        visual = dict(available=False, reason=str(err))
    else:
        visual = dict(
            available=True,
            projection=projection.tolist(),
            **report(analytic_visual_yaw(issued, dev_motion, deef[..., 2], valid, projection)),
        )

    # Strongest analytic visual control: clean active_probe projection applied to passive_attempt1.
    visual_clean = None
    if clean_visual_projection is not None:
        proj_c = np.asarray(clean_visual_projection, dtype=np.float64)
        visual_clean = dict(
            available=True,
            projection=proj_c.tolist(),
            **report(analytic_visual_yaw(issued, dev_motion, deef[..., 2], valid, proj_c)),
        )

    # Cross-morphology joint-Jacobian audit (7-DoF Panda nominal -> held-out 6-DoF UR5e).
    panda_nom = np.array([r["arm"] == "Panda" and r["embodiment"] == "nom" for r in train_rows])
    j_audit = evaluate_cross_morphology_joint_jacobian(
        train_data["delta_joint"][panda_nom],
        train_data["delta_eef"][panda_nom],
        train_data["valid"][panda_nom],
        issued,
        dev_data["delta_joint"],
        deef,
        valid,
    )
    joint_report = dict(
        source_arm="Panda",
        source_dof=j_audit["source_dof"],
        target_arm=DEV_ARM,
        target_dof=j_audit["target_dof"],
        applicable=j_audit["applicable"],
        reason=j_audit["reason"],
        zero_padded=report(np.asarray(j_audit["zero_padded_yaw_deg"], dtype=np.float32)),
        self_calibrated_local=report(np.asarray(j_audit["self_calibrated_local_yaw_deg"], dtype=np.float32)),
    )

    return dict(
        nominal_gain=float(gain),
        analytic_proprio_sysid=report(yaw),
        analytic_visual_sysid=visual,
        analytic_visual_sysid_clean_proj=visual_clean,
        analytic_joint_jacobian=joint_report,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    args = parser.parse_args()

    raw_archive = args.archive.read_bytes()
    archive_sha = hashlib.sha256(raw_archive).hexdigest()
    archive = json.loads(raw_archive)
    validate_archive(archive)
    archive_dir = args.archive.parent
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    train_arms = sorted({r["arm"] for r in archive["trials"] if r["split"] == "train"})
    dev_arms = sorted({r["arm"] for r in archive["trials"] if r["split"] == "dev"})
    if set(train_arms) != set(TRAIN_ARMS) or dev_arms != [DEV_ARM]:
        raise ValueError(f"Expected train_arms={TRAIN_ARMS}, dev_arms={[DEV_ARM]}, got {train_arms} / {dev_arms}")

    args.output.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output / "checkpoints"

    modes_out = {}
    clean_active_proj = None
    for mode in archive["modes"]:
        train_data, train_rows = load_mode_split(archive_dir, archive["trials"], mode, "train")
        dev_data, dev_rows = load_mode_split(archive_dir, archive["trials"], mode, "dev")
        donor_perm = make_donor_permutation(dev_rows, np.random.default_rng(20261120))

        an = analytic_audit(
            train_data,
            train_rows,
            dev_data,
            dev_rows,
            donor_perm,
            clean_visual_projection=(clean_active_proj if mode == "passive_attempt1" else None),
        )
        if mode == "active_probe" and an["analytic_visual_sysid"].get("available"):
            clean_active_proj = an["analytic_visual_sysid"]["projection"]

        models_out = {}
        for model_type in MODEL_NAMES:
            seed_retained, seed_cleared, seed_swapped, curves = [], [], [], []
            for seed in seeds:
                params, cfg, info = train_one(
                    mode,
                    model_type,
                    seed,
                    train_data,
                    args.steps,
                    args.batch_size,
                    an["nominal_gain"],
                    ckpt_dir,
                    archive_sha,
                )
                aud = audit_model(params, cfg, dev_data, dev_rows, donor_perm)
                seed_retained.append(aud["retained"])
                seed_cleared.append(aud["cleared"])
                seed_swapped.append(aud["swapped_donor"])
                curves.append(info)
            models_out[model_type] = dict(
                retained=aggregate_seeds(seed_retained),
                cleared=aggregate_seeds(seed_cleared),
                swapped_donor=aggregate_seeds(seed_swapped),
                training=curves,
            )
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        model=model_type,
                        overall_yaw=round(models_out[model_type]["retained"]["overall"]["yaw_err_deg"]["mean"], 2),
                        overall_all=round(models_out[model_type]["retained"]["overall"]["all_correct"]["mean"], 3),
                        comp_yaw=round(models_out[model_type]["retained"]["compositional_ood"]["yaw_err_deg"]["mean"], 2),
                        cont_yaw=round(models_out[model_type]["retained"]["continuous_angle_ood"]["yaw_err_deg"]["mean"], 2),
                    )
                ),
                flush=True,
            )

        modes_out[mode] = dict(**an, models=models_out)

    summary = dict(
        complete=True,
        panel=archive["panel"],
        train_arms=train_arms,
        dev_arm=DEV_ARM,
        archive=str(args.archive),
        archive_sha256=archive_sha,
        matched_support_cap=SUPPORT_BUDGET,
        steps=args.steps,
        batch_size=args.batch_size,
        seeds=list(seeds),
        runtime=dict(
            packages={p: metadata.version(p) for p in ("jax", "jaxlib", "flax", "optax", "numpy")},
            devices=[str(d) for d in jax.devices()],
            backend=jax.default_backend(),
        ),
        modes=modes_out,
    )
    write_json(args.output / "offline-audit.json", summary)
    print(json.dumps(dict(complete=True, modes=list(modes_out.keys()), seeds=list(seeds)), indent=2), flush=True)


if __name__ == "__main__":
    main()
