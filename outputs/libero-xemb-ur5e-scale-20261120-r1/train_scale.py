"""Train 3-seed appearance-invariant (luminance_flow) & RGB ablation readers on 1,344 7-DoF Panda/Sawyer tapes."""

import argparse
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

BASE_DIR = Path(__file__).resolve().parents[1] / "libero-xemb-ur5e-sysid-20261120-r1"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fluffy_bassoon.xemb.readers import FRAME0_FIELDS, HISTORY_FIELDS, MODEL_NAMES, ReaderConfig, init_reader, reader_loss
from collect_scale import validate_scale_archive
from collect_ur5e import DEV_ARM, SUPPORT_BUDGET, TRAIN_ARMS
from train_ur5e_readers import (
    aggregate_seeds,
    analytic_audit,
    audit_model,
    load_mode_split,
    make_donor_permutation,
)
from ur5e_motor import write_json


def save_checkpoint(path, params, opt_state, step, seed, mode, cfg, nominal_gain, archive_sha, sampler):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        params=params,
        archive_sha256=archive_sha,
        input_schema=f"{cfg.visual_mode}-unit-motion-cm-v2",
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


def train_one(
    mode,
    model_type,
    seed,
    train_data,
    steps,
    batch_size,
    nominal_gain,
    ckpt_dir,
    archive_sha,
    *,
    visual_mode: str,
    cnn_width: int,
    mlp_width: int,
    tag_prefix: str = "",
):
    max_chunks = int(train_data["valid"].shape[1])
    cfg = ReaderConfig(
        model_type=model_type,
        max_chunks=max(max_chunks, 32),
        cnn_width=cnn_width,
        mlp_width=mlp_width,
        visual_mode=visual_mode,
    )
    params = init_reader(jax.random.key(seed), cfg)
    schedule = optax.cosine_decay_schedule(init_value=1e-3, decay_steps=steps, alpha=0.05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=1e-4))
    opt_state = optimizer.init(params)

    @jax.jit
    def update_step(p, os_state, batch):
        (loss, metrics), grads = jax.value_and_grad(lambda pp: reader_loss(pp, cfg, batch), has_aux=True)(p)
        updates, os_state = optimizer.update(grads, os_state, p)
        return optax.apply_updates(p, updates), os_state, loss, metrics

    rng = np.random.default_rng(seed)
    fields = FRAME0_FIELDS if model_type == "frame0_probe" else HISTORY_FIELDS
    train_sub = {k: train_data[k] for k in (*fields, "yaw_deg", "xy_scale", "gripper_sign")}
    n_train = len(train_sub["yaw_deg"])
    curve = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        idx = rng.integers(n_train, size=batch_size)
        batch = {k: jnp.asarray(v[idx]) for k, v in train_sub.items()}
        params, opt_state, loss, metrics = update_step(params, opt_state, batch)
        if step == 1 or step % max(1, steps // 5) == 0 or step == steps:
            if not np.isfinite(float(loss)):
                raise FloatingPointError(f"Nonfinite {mode}/{model_type}/{seed} loss at step {step}")
            curve.append(
                dict(
                    step=step,
                    loss=float(loss),
                    yaw_loss=float(metrics["yaw_loss"]),
                    scale_loss=float(metrics["scale_loss"]),
                    grip_loss=float(metrics["grip_loss"]),
                )
            )

    ckpt_name = f"{tag_prefix}{mode}-{model_type}-seed{seed}.msgpack"
    ckpt_path = ckpt_dir / ckpt_name
    save_checkpoint(ckpt_path, params, opt_state, steps, seed, mode, cfg, nominal_gain, archive_sha, rng)
    return params, cfg, dict(seed=seed, seconds=round(time.monotonic() - started, 3), checkpoint=str(ckpt_path), curve=curve)


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
        # Train all 4 architectures with appearance-invariant luminance_flow + raw-RGB ablation of factored_equiv.
        specs = [
            (m, "luminance_flow", "") for m in MODEL_NAMES
        ] + [
            ("factored_equiv", "rgb", "rgb-"),
        ]
        for model_type, vis_mode, prefix in specs:
            out_key = f"{model_type}_rgb" if prefix else model_type
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
                    visual_mode=vis_mode,
                    cnn_width=args.cnn_width,
                    mlp_width=args.mlp_width,
                    tag_prefix=prefix,
                )
                aud = audit_model(params, cfg, dev_data, dev_rows, donor_perm)
                seed_retained.append(aud["retained"])
                seed_cleared.append(aud["cleared"])
                seed_swapped.append(aud["swapped_donor"])
                curves.append(info)
            models_out[out_key] = dict(
                visual_mode=vis_mode,
                retained=aggregate_seeds(seed_retained),
                cleared=aggregate_seeds(seed_cleared),
                swapped_donor=aggregate_seeds(seed_swapped),
                training=curves,
            )
            print(
                json.dumps(
                    dict(
                        mode=mode,
                        model=out_key,
                        overall_yaw=round(models_out[out_key]["retained"]["overall"]["yaw_err_deg"]["mean"], 2),
                        overall_all=round(models_out[out_key]["retained"]["overall"]["all_correct"]["mean"], 3),
                        fam_yaw=round(models_out[out_key]["retained"]["familiar"]["yaw_err_deg"]["mean"], 2),
                        comp_yaw=round(models_out[out_key]["retained"]["compositional_ood"]["yaw_err_deg"]["mean"], 2),
                        cont_yaw=round(models_out[out_key]["retained"]["continuous_angle_ood"]["yaw_err_deg"]["mean"], 2),
                    )
                ),
                flush=True,
            )

        modes_out[mode] = dict(**an, models=models_out)

    summary = dict(
        complete=True,
        panel=archive["panel"],
        train_arms=list(TRAIN_ARMS),
        dev_arm=DEV_ARM,
        train_trials=sum(1 for r in archive["trials"] if r["split"] == "train"),
        dev_trials=sum(1 for r in archive["trials"] if r["split"] == "dev"),
        archive=str(args.archive),
        archive_sha256=archive_sha,
        matched_support_cap=SUPPORT_BUDGET,
        steps=args.steps,
        batch_size=args.batch_size,
        cnn_width=args.cnn_width,
        mlp_width=args.mlp_width,
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
