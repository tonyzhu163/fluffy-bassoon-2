"""Offline check: which inner loop recovers a command interface outside yaw/gain/sign?

Recorded 15-control support traces are relabeled to sampled 2x2 planar interfaces
(see fluffy_bassoon.xemb.wider_family). No simulator or VLA rollout is used.
Fixed solvers run on frozen SymMeta readers; Perceivers are trained on
continuously sampled interfaces. Every estimate is scored as the planar command
map the frozen policy would be left with after correction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

from fluffy_bassoon.xemb.embodiment import circular_yaw_error_deg
from fluffy_bassoon.xemb.equiv_v2 import (
    ReaderV2Config,
    _apply_perceiver,
    _apply_res_cnn,
    _init_perceiver,
    _init_res_cnn,
    _vis_inputs_v2,
    apply_reader_v2,
    extract_visual_vectors_v2,
    init_reader_v2,
    reader_loss_v2,
)
from fluffy_bassoon.xemb.wider_family import (
    FAMILIES,
    command_spread,
    delay_commands,
    operator_error,
    procrustes_yaw_deg,
    relabel,
    ridge_linear_map,
    rot2_batch,
    sample_interfaces,
)

SEEDS = (41, 42, 43)
SUPPORT_CHUNKS = 3
HORIZONS = (3, 6, 12, 24)
RIDGES = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0)
DRAWS = 4
CHUNK_KEYS = ("img_before", "img_after", "dis_flow", "issued_mean", "delta_eef", "delta_jaw", "valid")
LABEL_KEYS = ("yaw_deg", "xy_scale", "gripper_sign")

SYMMETA = ReaderV2Config(model_type="factored_equiv_v2", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, mlp_width=128, use_dz_parallax=True)
PERCEIVER = ReaderV2Config(model_type="perceiver_icba", visual_mode="dis_lumaflow", max_chunks=24, cnn_width=96, perceiver_latents=8, perceiver_width=192, perceiver_layers=3, perceiver_heads=4)

# name, output, architecture, training interfaces, peak learning rate, warmup updates, updates
TRAINED = (
    ("symmeta_cont", "factors", SYMMETA, ("family",), 1.5e-3, 0, 2500),
    ("perceiver_cont", "factors", PERCEIVER, ("family",), 1.5e-3, 0, 2500),
    ("perceiver_cont_warm", "factors", PERCEIVER, ("family",), 3e-4, 200, 2500),
    ("perceiver_cont_warm_4x", "factors", PERCEIVER, ("family",), 3e-4, 200, 10000),
    ("perceiver_map_family", "map", PERCEIVER, ("family",), 3e-4, 200, 10000),
    ("perceiver_map_wide", "map", PERCEIVER, FAMILIES, 3e-4, 200, 10000),
)


def take_chunks(batch: dict, k: int) -> dict:
    return {key: (val[:, :k] if key in CHUNK_KEYS else val) for key, val in batch.items()}


def tile(batch: dict, reps: int) -> dict:
    return {key: np.concatenate([val] * reps, axis=0) for key, val in batch.items()}


def sample_mixed(rng, families, n: int) -> dict:
    counts = rng.multinomial(n, np.full(len(families), 1.0 / len(families)))
    parts = [sample_interfaces(rng, fam, int(c)) for fam, c in zip(families, counts, strict=True) if c]
    return {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0]}


# ---------------------------------------------------------------- data

def load_recorded(diverse_dir: Path, reuse_support_dir: Path, campaign_dir: Path):
    """Stage-3 Panda+Sawyer source traces and the locked UR5e test supports."""
    sys.path.insert(0, str(campaign_dir))
    from eval_stage3_scaling import load_ur5e_official_test_batch

    parts = []
    for shard in sorted(diverse_dir.glob("shard-*")):
        with np.load(shard / "diverse.npz") as z:
            parts.append({k: np.asarray(z[k]) for k in z.files})
    train = {k: np.concatenate([p[k] for p in parts], axis=0) for k in (*CHUNK_KEYS, *LABEL_KEYS)}
    test, meta = load_ur5e_official_test_batch(reuse_support_dir)
    test = {k: test[k] for k in (*CHUNK_KEYS, *LABEL_KEYS)}
    return train, test, [m["regime"] for m in meta]


def synthetic_traces(rng, n: int, k: int, blob: float, cam_deg: float):
    """Code-path check only: a blob moves by the commanded world motion under a fixed camera map."""
    from fluffy_bassoon.xemb.baselines import compute_dis_flow_maps

    yaw = rng.uniform(-180, 180, n)
    gain = np.exp(rng.uniform(np.log(0.65), np.log(1.5), n))
    grip = rng.choice([-1, 1], n)
    issued = np.zeros((n, k, 7), dtype=np.float32)
    base = rng.uniform(-0.6, 0.6, (n, 1, 3))
    issued[..., :3] = np.clip(base + 0.25 * rng.standard_normal((n, k, 3)), -1, 1)
    issued[..., 6] = rng.choice([-1.0, 1.0], (n, 1))
    world = 0.04 * np.einsum("bij,bkj->bki", gain[:, None, None] * rot2_batch(yaw), issued[..., :2])
    world += 0.002 * rng.standard_normal(world.shape)
    delta = np.zeros((n, k, 3), dtype=np.float32)
    delta[..., :2] = np.einsum("bij,bkj->bki", rot2_batch(-yaw), world)
    delta[..., 2] = 0.04 * gain[:, None] * issued[..., 2]
    jaw = np.zeros((n, k, 2), dtype=np.float32)
    jaw[..., 0] = 0.01 * grip[:, None] * issued[..., 6]

    cam = 220.0 * rot2_batch(cam_deg)
    pix = np.einsum("ij,bkj->bki", cam, world) + 90.0 * delta[..., 2:3] * np.array([0.3, -1.0])
    pos = rng.uniform(22, 42, (n, 1, 2)) + np.concatenate((np.zeros((n, 1, 2)), np.cumsum(pix, axis=1)), axis=1)
    yy, xx = np.mgrid[0:64, 0:64]
    texture = rng.uniform(40, 90, (n, 1, 64, 64))
    frames = texture + 160.0 * np.exp(-((xx - pos[..., 0, None, None]) ** 2 + (yy - pos[..., 1, None, None]) ** 2) / (2 * blob**2))
    frames = np.repeat(np.clip(frames, 0, 255).astype(np.uint8)[..., None], 3, axis=-1)
    batch = dict(
        img_before=frames[:, :-1],
        img_after=frames[:, 1:],
        issued_mean=issued,
        delta_eef=delta,
        delta_jaw=jaw,
        valid=np.ones((n, k), dtype=bool),
        yaw_deg=yaw.astype(np.float32),
        xy_scale=gain.astype(np.float32),
        gripper_sign=grip.astype(np.int32),
    )
    batch["dis_flow"] = compute_dis_flow_maps(batch["img_before"], batch["img_after"], batch["valid"])
    return batch


# ---------------------------------------------------------------- learned models

def init_model(key, output: str, cfg: ReaderV2Config):
    import jax

    if output == "factors":
        return init_reader_v2(key, cfg)
    k1, k2 = jax.random.split(key)
    return dict(
        cnn=_init_res_cnn(k1, 9, cfg.cnn_width),
        perceiver=_init_perceiver(k2, cfg.cnn_width + 12, cfg.max_chunks, cfg.perceiver_latents, cfg.perceiver_width, cfg.perceiver_layers, 5),
    )


def apply_map(params, cfg: ReaderV2Config, batch):
    """Perceiver whose output is the 2x2 planar interface itself, plus a gripper logit."""
    import jax.numpy as jnp

    valid = jnp.asarray(batch["valid"], dtype=bool)
    images, b, k = _vis_inputs_v2(batch, cfg.visual_mode)
    vis = _apply_res_cnn(params["cnn"], images).reshape(b, k, -1)

    def sig(field, gain=1.0):
        return jnp.where(valid[..., None], gain * jnp.asarray(batch[field], dtype=jnp.float32), 0.0)

    tok = jnp.concatenate((vis, sig("issued_mean"), sig("delta_eef", 100.0), sig("delta_jaw", 100.0)), axis=-1)
    raw = _apply_perceiver(params["perceiver"], tok, valid, cfg.perceiver_heads)
    return jnp.eye(2)[None] + raw[:, :4].reshape(b, 2, 2), raw[:, 4]


def model_loss(params, output: str, cfg: ReaderV2Config, batch):
    import jax
    import jax.numpy as jnp

    if output == "factors":
        return reader_loss_v2(params, cfg, batch)[0]
    op, grip_logit = apply_map(params, cfg, batch)
    grip_t = jnp.asarray(batch["gripper_sign"] > 0, dtype=jnp.float32)
    grip_loss = jnp.mean(jax.nn.softplus(grip_logit) - grip_t * grip_logit)
    return jnp.mean(jnp.sum(jnp.square(op - batch["op"]), axis=(-2, -1))) + grip_loss


def train_model(train: dict, output: str, cfg: ReaderV2Config, families, lr: float, warmup: int, steps: int, seed: int, batch_size: int):
    import jax
    import optax

    params = init_model(jax.random.key(seed), output, cfg)
    if warmup:
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, end_value=0.05 * lr)
    else:
        sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(p, st, b):
        loss, grads = jax.value_and_grad(lambda pp: model_loss(pp, output, cfg, b))(p)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    rng = np.random.default_rng(seed * 10007 + 31)
    n = len(train["valid"])
    loss = np.nan
    for _ in range(steps):
        sel = rng.choice(n, size=batch_size, replace=True)
        batch = relabel({k: v[sel] for k, v in train.items()}, sample_mixed(rng, families, batch_size))
        params, state, loss = step(params, state, batch)
    return params, float(loss)


def predict_ops(params, output: str, cfg: ReaderV2Config, batch: dict):
    """Estimated planar interface [N, 2, 2] and, for factor outputs, the yaw estimate."""
    import jax

    fn = jax.jit(lambda b: apply_map(params, cfg, b)[0] if output == "map" else apply_reader_v2(params, cfg, b)[:3])
    ops, yaws = [], []
    n = len(batch["valid"])
    for start in range(0, n, 96):
        part = {k: v[start:start + 96] for k, v in batch.items() if k in CHUNK_KEYS}
        out = fn(part)
        if output == "map":
            ops.append(np.asarray(out))
            continue
        cos_h, sin_h, log_scale = map(np.asarray, out)
        yaw = np.degrees(np.arctan2(sin_h, cos_h))
        ops.append(np.exp(np.clip(log_scale, np.log(0.25), np.log(4.0)))[:, None, None] * rot2_batch(yaw))
        yaws.append(yaw)
    return np.concatenate(ops), (np.concatenate(yaws) if yaws else None)


def measure(params, cfg: ReaderV2Config, batch: dict):
    """Command-blind visual vectors and confidences of a SymMeta reader."""
    import jax

    fn = jax.jit(lambda b: extract_visual_vectors_v2(params, cfg, b))
    vs, ws = [], []
    for start in range(0, len(batch["valid"]), 96):
        v, w = fn({k: val[start:start + 96] for k, val in batch.items() if k in CHUNK_KEYS})
        vs.append(np.asarray(v, dtype=np.float64))
        ws.append(np.asarray(w, dtype=np.float64))
    return np.concatenate(vs), np.concatenate(ws)


# ---------------------------------------------------------------- fixed solvers

def fit_metres_per_command(train: dict) -> float:
    valid = train["valid"].astype(np.float64)
    cmd = np.linalg.norm(train["issued_mean"][..., :2], axis=-1) * train["xy_scale"][:, None]
    moved = np.linalg.norm(train["delta_eef"][..., :2], axis=-1)
    return float(np.sum(valid * cmd * moved) / np.sum(valid * cmd * cmd))


def solver_ops(batch: dict, v, w, paper_op, metres_per_command: float, ridge: float) -> dict:
    """Interfaces estimated from one reader's vectors by each fixed solver."""
    valid = batch["valid"].astype(np.float64)
    cmd = batch["issued_mean"][..., :2].astype(np.float64)
    moved = batch["delta_eef"][..., :2].astype(np.float64) / metres_per_command
    wv = w * valid
    eye = np.eye(2)[None]

    def energy(x):
        return np.sum(valid * np.sum(x * x, axis=-1), axis=-1)

    gain = np.sqrt(energy(moved) / np.maximum(energy(cmd), 1e-12))
    out = {}
    for tag, lam in (("ls", 0.0), ("ridge", ridge)):
        # Proprioception gives the in-frame 2x2 map; vision gives the frame rotation.
        shape = ridge_linear_map(cmd, moved, valid, gain[:, None, None] * eye, lam)
        out[f"proprio_only_{tag}"] = shape
        out[f"factored_{tag}"] = rot2_batch(procrustes_yaw_deg(moved, v, wv)) @ shape
        # One unconstrained 2x2 map from commands to visual vectors, rescaled to proprioceptive travel.
        prior = paper_op / np.sqrt(np.abs(np.linalg.det(paper_op)))[:, None, None]
        vis = ridge_linear_map(cmd, v, wv, prior * np.sqrt(energy(v) / np.maximum(energy(cmd), 1e-12))[:, None, None], lam)
        pred = np.einsum("bij,bkj->bki", vis, cmd)
        out[f"visual_{tag}"] = vis * np.sqrt(energy(moved) / np.maximum(energy(pred), 1e-12))[:, None, None]
    return out


def summarize(true_op, est_op) -> dict:
    err = operator_error(true_op, est_op)
    return dict(mean=float(np.mean(err)), median=float(np.median(err)), q75=float(np.percentile(err, 75)))


# ---------------------------------------------------------------- conditions

def has_yaw(condition: str) -> bool:
    return condition == "family" or condition.startswith(("recorded", "delay"))


def build_conditions(test: dict, k: int) -> dict:
    """Test supports at k chunks: as recorded, relabeled to each family, and with a one-chunk delay."""
    base = take_chunks(test, k)
    n = len(base["valid"])
    recorded = dict(base)
    recorded["op"] = (base["xy_scale"].astype(np.float64)[:, None, None] * rot2_batch(base["yaw_deg"])).astype(np.float32)
    conds = {"recorded": recorded}
    for i, fam in enumerate(FAMILIES):
        rng = np.random.default_rng(20261123 + i)
        conds[fam] = relabel(tile(base, DRAWS), sample_interfaces(rng, fam, n * DRAWS))
    if test["valid"].shape[1] > k and test["valid"][:, k].all():
        delayed = take_chunks(delay_commands(take_chunks(test, k + 1), 1), k)
        delayed["op"] = recorded["op"]
        conds["delay_1_chunk"] = delayed
    else:
        # Supports end at k chunks: a delayed trace keeps k-1 motions, so pair it with a k-1 control.
        short = dict(recorded)
        short["valid"] = recorded["valid"].copy()
        short["valid"][:, k - 1:] = False
        conds[f"recorded_{k - 1}_chunks"] = short
        delayed = delay_commands(base, 1)
        delayed["op"] = recorded["op"]
        conds[f"delay_1_chunk_{k - 1}_chunks"] = delayed
    return conds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--diverse-dir", type=Path)
    parser.add_argument("--reuse-support-dir", type=Path)
    parser.add_argument("--campaign-dir", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--synthetic", action="store_true", help="code-path check on generated blobs")
    parser.add_argument("--step-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    args.output.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        rng = np.random.default_rng(7)
        train_full = synthetic_traces(rng, 384, SUPPORT_CHUNKS, blob=3.0, cam_deg=20.0)
        test = synthetic_traces(rng, 96, 6, blob=4.0, cam_deg=20.0)
        regimes = ["continuous_angle_ood"] * 96
    else:
        train_full, test, regimes = load_recorded(args.diverse_dir, args.reuse_support_dir, args.campaign_dir)
    train = take_chunks(train_full, SUPPORT_CHUNKS)
    ood = np.array([r != "familiar" for r in regimes])
    metres = fit_metres_per_command(train)
    horizons = [k for k in HORIZONS if k <= test["valid"].shape[1] and test["valid"][:, k - 1].all()]
    print(f"[data] train={len(train['valid'])} test={len(test['valid'])} metres_per_command={metres:.4f} horizons={horizons}", flush=True)

    conds = {k: build_conditions(test, k) for k in horizons}
    rows = []
    data_stats = {}
    for k in horizons:
        for name, cond in conds[k].items():
            valid = cond["valid"].astype(np.float64)
            cmd = cond["issued_mean"][..., :2]
            spread = command_spread(cmd, valid)
            none = summarize(cond["op"], np.broadcast_to(np.eye(2), cond["op"].shape))
            data_stats[f"{name}@{k}"] = dict(
                n=len(valid),
                command_spread_quartiles=[float(q) for q in np.percentile(spread, (25, 50, 75))],
                commands_outside_unit_box=float(np.mean((np.abs(cmd) > 1.0).any(axis=-1)[cond["valid"]])),
                no_adaptation=none,
            )

    def record(model, seed, k, name, cond, est_op, yaw=None, extra=None):
        row = dict(model=model, seed=seed, chunks=k, condition=name, **summarize(cond["op"], est_op))
        if yaw is not None:
            errs = np.array([circular_yaw_error_deg(float(a), float(b)) for a, b in zip(yaw, cond["yaw_deg"], strict=True)])
            row["yaw_mae_deg"] = float(errs.mean())
            if name == "recorded":
                row["yaw_mae_ood_deg"] = float(errs[ood].mean())
        rows.append({**row, **(extra or {})})

    for seed in seeds:
        trained = {}
        for name, output, cfg, families, lr, warmup, steps in TRAINED:
            t0 = time.monotonic()
            n_steps = max(20, int(steps * args.step_scale))
            params, loss = train_model(train, output, cfg, families, lr, min(warmup, n_steps // 4), n_steps, seed, args.batch_size)
            trained[name] = (params, output, cfg)
            print(f"[train] {name} seed={seed} steps={n_steps} loss={loss:.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
            cond_k = conds[SUPPORT_CHUNKS]
            for cname, cond in cond_k.items():
                est, yaw = predict_ops(params, output, cfg, cond)
                record(name, seed, SUPPORT_CHUNKS, cname, cond, est, yaw if has_yaw(cname) else None)

        readers = {"symmeta_cont": trained["symmeta_cont"][0::2]}
        if args.checkpoint_dir:
            sys.path.insert(0, str(args.campaign_dir))
            from eval_stage2a_live_ur5e import load_v2_checkpoint

            p, c, _ = load_v2_checkpoint(args.checkpoint_dir / f"equiv_v2_dis_dz-passive_attempt1-seed{seed}.msgpack")
            readers["symmeta_paper"] = (p, c)
            p, c, _ = load_v2_checkpoint(args.checkpoint_dir / f"perceiver_icba_dis-passive_attempt1-seed{seed}.msgpack")
            for cname, cond in conds[SUPPORT_CHUNKS].items():
                est, yaw = predict_ops(p, "factors", c, cond)
                record("perceiver_paper", seed, SUPPORT_CHUNKS, cname, cond, est, yaw if has_yaw(cname) else None)

        for rname, (params, cfg) in readers.items():
            # Ridge strength is chosen on the source arms, pooled over all families.
            src = relabel(train, sample_mixed(np.random.default_rng(seed), FAMILIES, len(train["valid"])))
            v_src, w_src = measure(params, cfg, src)
            paper_src, _ = predict_ops(params, "factors", cfg, src)
            scores = {}
            for lam in RIDGES[1:]:
                ops = solver_ops(src, v_src, w_src, paper_src, metres, lam)
                scores[lam] = {key: float(np.mean(operator_error(src["op"], ops[f"{key}_ridge"]))) for key in ("factored", "visual")}
            best = {key: min(scores, key=lambda lam: scores[lam][key]) for key in ("factored", "visual")}
            print(f"[ridge] {rname} seed={seed} chosen={best}", flush=True)

            for k in horizons:
                v_test, w_test = measure(params, cfg, take_chunks(test, k))
                for cname, cond in conds[k].items():
                    reps = len(cond["valid"]) // len(v_test)
                    v, w = np.concatenate([v_test] * reps), np.concatenate([w_test] * reps)
                    paper_op, yaw = predict_ops(params, "factors", cfg, cond)
                    if k != SUPPORT_CHUNKS or rname != "symmeta_cont":
                        record(f"{rname}+procrustes", seed, k, cname, cond, paper_op, yaw if has_yaw(cname) else None)
                    ops = solver_ops(cond, v, w, paper_op, metres, 0.0)
                    for key in ("factored", "visual"):
                        ops[f"{key}_ridge"] = solver_ops(cond, v, w, paper_op, metres, best[key])[f"{key}_ridge"]
                        record(f"{rname}+{key}_ridge", seed, k, cname, cond, ops[f"{key}_ridge"], extra=dict(ridge=best[key]))
                    for key in ("factored_ls", "visual_ls", "proprio_only_ls"):
                        record(f"{rname}+{key}", seed, k, cname, cond, ops[key])
                    frame = procrustes_yaw_deg(cond["delta_eef"][..., :2].astype(np.float64), v, w * cond["valid"])
                    rows[-3]["yaw_mae_deg"] = float(np.mean([circular_yaw_error_deg(float(a), float(b)) for a, b in zip(frame, cond["yaw_deg"], strict=True)]))
        (args.output / "rows.partial.json").write_text(json.dumps(rows, indent=1))

    table = {}
    for row in rows:
        table.setdefault(f"{row['model']}@{row['chunks']}", {}).setdefault(row["condition"], []).append(row)
    summary = {}
    for model, by_cond in table.items():
        summary[model] = {}
        for cname, rs in by_cond.items():
            entry = dict(
                operator_error_mean=round(float(np.mean([r["mean"] for r in rs])), 4),
                operator_error_seed_std=round(float(np.std([r["mean"] for r in rs])), 4),
                operator_error_median=round(float(np.mean([r["median"] for r in rs])), 4),
                per_seed=[round(r["mean"], 4) for r in rs],
            )
            for key in ("yaw_mae_deg", "yaw_mae_ood_deg"):
                if key in rs[0]:
                    entry[key] = [round(r[key], 2) for r in rs]
            summary[model][cname] = entry
    result = dict(
        synthetic=bool(args.synthetic),
        seeds=list(seeds),
        support_chunks=SUPPORT_CHUNKS,
        draws_per_trace=DRAWS,
        metres_per_command=metres,
        step_scale=args.step_scale,
        data=data_stats,
        summary=summary,
        rows=rows,
    )
    (args.output / "wider-family-summary.json").write_text(json.dumps(result, indent=1))
    (args.output / "rows.partial.json").unlink()

    names = [c for c in conds[SUPPORT_CHUNKS]]
    print("\noperator error, mean over seeds (0 = exact; 'none' = no adaptation)")
    print(f"{'model':44s} " + " ".join(f"{c[:9]:>9s}" for c in names))
    print(f"{'none@3':44s} " + " ".join(f"{data_stats[f'{c}@{SUPPORT_CHUNKS}']['no_adaptation']['mean']:9.3f}" for c in names))
    for model in sorted(summary):
        cells = [summary[model].get(c, {}).get("operator_error_mean") for c in conds[int(model.split("@")[1])]]
        print(f"{model:44s} " + " ".join(f"{x:9.3f}" if x is not None else f"{'-':>9s}" for x in cells))


if __name__ == "__main__":
    main()
