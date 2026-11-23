"""Offline check: where does a closed-form inner loop differ from a tuned Perceiver?

Same traces, relabeling and score as the wider-family run. Four comparisons at
15 controls on the locked UR5e supports: the number of training interfaces, a
correction on an unshifted interface, shorter supports and gains outside the
trained range, and a one-chunk command delay. No simulator or VLA rollout is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "symmeta-wider-family-20261123-r1"))

from eval_wider_family import (  # noqa: E402
    PERCEIVER,
    SEEDS,
    SUPPORT_CHUNKS,
    SYMMETA,
    fit_metres_per_command,
    init_model,
    load_recorded,
    model_loss,
    predict_ops,
    sample_mixed,
    summarize,
    synthetic_traces,
    take_chunks,
    tile,
)

from fluffy_bassoon.xemb.embodiment import TRAIN_EMBODIMENTS, circular_yaw_error_deg  # noqa: E402
from fluffy_bassoon.xemb.equiv_v2 import ReaderV2Config, _apply_scale_and_grip, extract_visual_vectors_v2, init_reader_v2  # noqa: E402
from fluffy_bassoon.xemb.wider_family import (  # noqa: E402
    FAMILIES,
    delay_commands,
    delay_keep_motion,
    lagged,
    operator_error,
    relabel,
    rot2_batch,
    sample_interfaces,
    soft_shape,
)

DRAWS = 4
GATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)
SOFT_KEYS = ("img_before", "img_after", "dis_flow", "issued_mean", "delta_eef", "delta_jaw", "valid", "cmd_valid")

# name, output, architecture, training interfaces, share of delayed traces, peak rate, warmup, updates.
# Interfaces: "paper7" is the paper's seven, an integer is that many fixed in-family draws,
# a tuple of families is a fresh draw for every trace and update.
TRAINED = (
    ("symmeta_7", "factors", SYMMETA, "paper7", 0.0, 1.5e-3, 0, 2500),
    ("symmeta_16", "factors", SYMMETA, 16, 0.0, 1.5e-3, 0, 2500),
    ("symmeta_64", "factors", SYMMETA, 64, 0.0, 1.5e-3, 0, 2500),
    ("symmeta_cont", "factors", SYMMETA, ("family",), 0.0, 1.5e-3, 0, 2500),
    ("perceiver_7", "factors", PERCEIVER, "paper7", 0.0, 3e-4, 200, 2500),
    ("perceiver_16", "factors", PERCEIVER, 16, 0.0, 3e-4, 200, 2500),
    ("perceiver_64", "factors", PERCEIVER, 64, 0.0, 3e-4, 200, 2500),
    ("perceiver_cont", "factors", PERCEIVER, ("family",), 0.0, 3e-4, 200, 2500),
    ("perceiver_delay", "factors", PERCEIVER, ("family",), 0.5, 3e-4, 200, 2500),
    ("perceiver_map_wide_delay", "map", PERCEIVER, FAMILIES, 0.5, 3e-4, 200, 10000),
    ("soft_7", "soft", SYMMETA, "paper7", 0.0, 1.5e-3, 200, 2500),
    ("soft_wide_delay", "soft", SYMMETA, FAMILIES, 0.5, 1.5e-3, 200, 2500),
)


def interface_sampler(spec):
    if isinstance(spec, tuple):
        return lambda rng, n: sample_mixed(rng, spec, n)
    if spec == "paper7":
        gain = np.array([e.xy_scale for e in TRAIN_EMBODIMENTS], dtype=np.float64)
        pool = dict(
            yaw_deg=np.array([e.yaw_deg for e in TRAIN_EMBODIMENTS], dtype=np.float64),
            shape=gain[:, None, None] * np.eye(2),
            z_gain=gain,
            gripper_sign=np.array([e.gripper_sign for e in TRAIN_EMBODIMENTS]),
        )
    else:
        pool = sample_interfaces(np.random.default_rng(20261123 + spec), "family", spec)

    def draw(rng, n):
        idx = rng.integers(len(pool["yaw_deg"]), size=n)
        return {key: val[idx] for key, val in pool.items()}

    return draw


# ---------------------------------------------------------------- closed-form reader with a soft prior

def init_soft(key, cfg: ReaderV2Config):
    import jax.numpy as jnp

    return dict(
        reader=init_reader_v2(key, cfg),
        log_ridge=jnp.asarray(np.log(0.3), dtype=jnp.float32),
        lag=jnp.asarray([np.log(5.0), np.log(5.0), 0.0], dtype=jnp.float32),
        log_kappa=jnp.asarray(0.0, dtype=jnp.float32),
    )


def apply_soft(params, cfg: ReaderV2Config, batch, metres: float):
    """Planar interface from command-blind visual vectors, proprioceptive travel and closed-form solves.

    Frame rotation: weighted Procrustes from travel to visual vectors, which no command timing enters.
    In-frame map: soft_shape at command lags of 0 and 1 chunk, mixed by how well each lag is explained.
    Learned beyond the SymMeta reader: the ridge, three lag-evidence scalars and a yaw concentration scale.
    """
    import jax
    import jax.numpy as jnp

    valid = jnp.asarray(batch["valid"], dtype=bool)
    cmd_valid = jnp.asarray(batch["cmd_valid"], dtype=bool)

    def sig(field):
        return jnp.where(valid[..., None], jnp.asarray(batch[field], dtype=jnp.float32), 0.0)

    v, w = extract_visual_vectors_v2(params["reader"], cfg, batch)
    issued = sig("issued_mean")
    cmd = jnp.where(cmd_valid[..., None], issued[..., :2], 0.0)
    moved = sig("delta_eef")[..., :2] / metres

    wv = jnp.where(valid, w, 0.0)
    dot = jnp.sum(wv * (moved[..., 0] * v[..., 0] + moved[..., 1] * v[..., 1]), axis=-1)
    cross = jnp.sum(wv * (moved[..., 0] * v[..., 1] - moved[..., 1] * v[..., 0]), axis=-1)
    length = jnp.sqrt(dot * dot + cross * cross + 1e-12)
    cos, sin = dot / length, cross / length
    rot = jnp.stack((jnp.stack((cos, -sin), axis=-1), jnp.stack((sin, cos), axis=-1)), axis=-2)

    ridge = jnp.exp(jnp.clip(params["log_ridge"], np.log(1e-3), np.log(1e3)))
    shapes, logits, shares = [], [], []
    for lag in (0, 1):
        pair = valid & lagged(cmd_valid, lag, jnp)
        shape, resid, turn, share = soft_shape(lagged(cmd, lag, jnp), moved, pair, ridge, jnp)
        # A delay needs two pairs: one pair is always fitted exactly.
        enough = jnp.sum(pair, axis=-1) >= 1 + lag
        score = -jnp.exp(params["lag"][0]) * resid - jnp.exp(params["lag"][1]) * turn + lag * params["lag"][2]
        shapes.append(shape)
        shares.append(share)
        logits.append(jnp.where(enough, score, -1e9))
    mix = jax.nn.softmax(jnp.stack(logits, axis=-1), axis=-1)
    shape = mix[:, 0, None, None] * shapes[0] + mix[:, 1, None, None] * shapes[1]
    _, grip_logit = _apply_scale_and_grip(params["reader"], issued, 100.0 * sig("delta_eef"), 100.0 * sig("delta_jaw"), valid)
    info = dict(cos=cos, sin=sin, kappa=jnp.exp(params["log_kappa"]) * length, lag1=mix[:, 1], share=mix[:, 0] * shares[0] + mix[:, 1] * shares[1])
    return rot @ shape, grip_logit, info


def soft_loss(params, cfg: ReaderV2Config, batch, metres: float):
    import jax
    import jax.numpy as jnp

    op, grip_logit, info = apply_soft(params, cfg, batch, metres)
    grip_t = jnp.asarray(batch["gripper_sign"] > 0, dtype=jnp.float32)
    grip_loss = jnp.mean(jax.nn.softplus(grip_logit) - grip_t * grip_logit)
    rad = jnp.deg2rad(jnp.asarray(batch["yaw_deg"], dtype=jnp.float32))
    # Von Mises likelihood of the frame yaw; it trains the concentration only, so kappa reports yaw precision.
    align = jax.lax.stop_gradient(info["cos"] * jnp.cos(rad) + info["sin"] * jnp.sin(rad))
    nll = jnp.mean(info["kappa"] * (1.0 - align) + jnp.log(jax.scipy.special.i0e(info["kappa"])))
    return jnp.mean(jnp.sum(jnp.square(op - batch["op"]), axis=(-2, -1))) + grip_loss + 0.1 * nll


def predict_soft(params, cfg: ReaderV2Config, batch: dict, metres: float):
    import jax

    fn = jax.jit(lambda b: apply_soft(params, cfg, b, metres))
    ops, infos = [], []
    for start in range(0, len(batch["valid"]), 96):
        op, _, info = fn({k: batch[k][start:start + 96] for k in SOFT_KEYS})
        ops.append(np.asarray(op))
        infos.append({k: np.asarray(val) for k, val in info.items()})
    info = {k: np.concatenate([i[k] for i in infos]) for k in infos[0]}
    return np.concatenate(ops), np.degrees(np.arctan2(info["sin"], info["cos"])), info


def train_model(train: dict, output: str, cfg, spec, delay: float, lr: float, warmup: int, steps: int, seed: int, batch_size: int, metres: float):
    import jax
    import optax

    key = jax.random.key(seed)
    params = init_soft(key, cfg) if output == "soft" else init_model(key, output, cfg)
    if warmup:
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, end_value=0.05 * lr)
    else:
        sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    def loss_fn(p, b):
        return soft_loss(p, cfg, b, metres) if output == "soft" else model_loss(p, output, cfg, b)

    @jax.jit
    def step(p, st, b):
        loss, grads = jax.value_and_grad(loss_fn)(p, b)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    draw = interface_sampler(spec)
    rng = np.random.default_rng(seed * 10007 + 31)
    n = len(train["valid"])
    loss = np.nan
    for _ in range(steps):
        sel = rng.choice(n, size=batch_size, replace=True)
        batch = relabel({k: v[sel] for k, v in train.items()}, draw(rng, batch_size))
        batch["cmd_valid"] = batch["valid"].copy()
        if delay:
            batch = delay_keep_motion(batch, 1, rows=rng.random(batch_size) < delay)
        if output != "soft":
            del batch["cmd_valid"]
        params, state, loss = step(params, state, batch)
    return params, float(loss)


def predict(params, output: str, cfg, batch: dict, metres: float):
    if output == "soft":
        return predict_soft(params, cfg, batch, metres)
    est, yaw = predict_ops(params, output, cfg, batch)
    return est, yaw, {}


# ---------------------------------------------------------------- conditions

def keep_chunks(batch: dict, k: int) -> dict:
    out = dict(batch)
    for key in ("valid", "cmd_valid"):
        out[key] = batch[key].copy()
        out[key][:, k:] = False
    return out


def has_yaw(condition: str) -> bool:
    return condition.startswith(("recorded", "family", "nominal", "gain", "delay"))


def build_conditions(test: dict) -> dict:
    base = take_chunks(test, SUPPORT_CHUNKS)
    n = len(base["valid"])
    recorded = dict(base)
    recorded["op"] = (base["xy_scale"].astype(np.float64)[:, None, None] * rot2_batch(base["yaw_deg"])).astype(np.float32)
    conds = {"recorded": recorded}
    rng = np.random.default_rng(20261123)
    conds["family"] = relabel(tile(base, DRAWS), sample_interfaces(rng, "family", n * DRAWS))
    conds["nominal"] = relabel(base, dict(yaw_deg=np.zeros(n), shape=np.broadcast_to(np.eye(2), (n, 2, 2)).copy(), z_gain=np.ones(n), gripper_sign=np.ones(n, dtype=np.int64)))
    for name, span in (("gain_low", (0.3, 0.5)), ("gain_high", (2.0, 3.0))):
        conds[name] = relabel(tile(base, DRAWS), sample_interfaces(rng, "family", n * DRAWS, gain_range=span))
    for i, fam in enumerate(FAMILIES[1:], start=1):
        conds[fam] = relabel(tile(base, DRAWS), sample_interfaces(np.random.default_rng(20261123 + i), fam, n * DRAWS))
    for cond in conds.values():
        cond["cmd_valid"] = cond["valid"].copy()
    for name in ("recorded", "family"):
        for k in (2, 1):
            conds[f"{name}_{k}_chunks"] = keep_chunks(conds[name], k)
    # Two views of one delayed trace: every motion kept, or (as in the wider-family run) the last chunk hidden.
    conds["delay_3_motions"] = delay_keep_motion(recorded, 1)
    hidden = delay_commands(recorded, 1)
    hidden["cmd_valid"] = hidden["valid"].copy()
    conds["delay_2_motions"] = hidden
    return conds


def rank_corr(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)).astype(np.float64), np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def gated(est, gate: float):
    """Leave the interface alone when the estimated correction is small."""
    eye = np.broadcast_to(np.eye(2), est.shape)
    small = np.linalg.norm(est - eye, axis=(-2, -1)) / np.sqrt(2.0) < gate
    return np.where(small[:, None, None], eye, est), small


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
    parser.add_argument("--models", default="", help="comma-separated subset of trained models")
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    args.output.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        rng = np.random.default_rng(7)
        train_full = synthetic_traces(rng, 384, SUPPORT_CHUNKS, blob=3.0, cam_deg=20.0)
        test = synthetic_traces(rng, 96, SUPPORT_CHUNKS, blob=4.0, cam_deg=20.0)
        regimes = ["continuous_angle_ood"] * 96
    else:
        train_full, test, regimes = load_recorded(args.diverse_dir, args.reuse_support_dir, args.campaign_dir)
    train = take_chunks(train_full, SUPPORT_CHUNKS)
    ood = np.array([r != "familiar" for r in regimes])
    metres = fit_metres_per_command(train)
    conds = build_conditions(test)
    print(f"[data] train={len(train['valid'])} test={len(test['valid'])} metres_per_command={metres:.4f} conditions={list(conds)}", flush=True)

    # Source traces for choosing each reader's gate: half unshifted, half in-family.
    n_src = len(train["valid"])
    src_rng = np.random.default_rng(20261124)
    src = {
        "nominal": relabel(train, dict(yaw_deg=np.zeros(n_src), shape=np.broadcast_to(np.eye(2), (n_src, 2, 2)).copy(), z_gain=np.ones(n_src), gripper_sign=np.ones(n_src, dtype=np.int64))),
        "family": relabel(train, sample_interfaces(src_rng, "family", n_src)),
    }
    for part in src.values():
        part["cmd_valid"] = part["valid"].copy()

    data_stats = {
        name: dict(n=len(cond["valid"]), no_adaptation=summarize(cond["op"], np.broadcast_to(np.eye(2), cond["op"].shape)))
        for name, cond in conds.items()
    }
    rows, diagnostics = [], []

    def record(model, seed, name, cond, est, yaw, extra=None):
        row = dict(model=model, seed=seed, condition=name, **summarize(cond["op"], est))
        if yaw is not None and has_yaw(name):
            errs = np.array([circular_yaw_error_deg(float(a), float(b)) for a, b in zip(yaw, cond["yaw_deg"], strict=True)])
            row["yaw_mae_deg"] = float(errs.mean())
            if len(errs) == len(ood):
                row["yaw_mae_ood_deg"] = float(errs[ood].mean())
        rows.append({**row, **(extra or {})})

    def evaluate(name, seed, params, output, cfg):
        src_est = {key: predict(params, output, cfg, part, metres)[0] for key, part in src.items()}
        scores = {gate: float(np.mean([np.mean(operator_error(src[key]["op"], gated(src_est[key], gate)[0])) for key in src])) for gate in GATES}
        gate = min(scores, key=scores.get)
        for cname, cond in conds.items():
            est, yaw, info = predict(params, output, cfg, cond, metres)
            record(name, seed, cname, cond, est, yaw)
            if cname in ("nominal", "family", "recorded"):
                kept, small = gated(est, gate)
                record(f"{name}+gate", seed, cname, cond, kept, None, dict(gate=gate, left_alone=float(small.mean())))
            if info:
                err = operator_error(cond["op"], est)
                diag = dict(model=name, seed=seed, condition=cname, lag1_chosen=float(np.mean(info["lag1"] > 0.5)), prior_share_mean=float(info["share"].mean()), prior_share_vs_error=rank_corr(info["share"], err))
                if has_yaw(cname):
                    yaw_err = np.array([circular_yaw_error_deg(float(a), float(b)) for a, b in zip(yaw, cond["yaw_deg"], strict=True)])
                    order = np.argsort(info["kappa"])
                    half = len(order) // 2
                    diag.update(kappa_vs_yaw_error=rank_corr(info["kappa"], yaw_err), yaw_mae_low_kappa_half=float(yaw_err[order[:half]].mean()), yaw_mae_high_kappa_half=float(yaw_err[order[half:]].mean()))
                diagnostics.append(diag)

    chosen = [m for m in TRAINED if not args.models or m[0] in args.models.split(",")]
    for seed in seeds:
        for name, output, cfg, spec, delay, lr, warmup, steps in chosen:
            t0 = time.monotonic()
            n_steps = max(20, int(steps * args.step_scale))
            params, loss = train_model(train, output, cfg, spec, delay, lr, min(warmup, n_steps // 4), n_steps, seed, args.batch_size, metres)
            learned = dict(ridge=float(np.exp(params["log_ridge"])), lag=[float(x) for x in params["lag"]]) if output == "soft" else {}
            print(f"[train] {name} seed={seed} steps={n_steps} loss={loss:.4f} ({time.monotonic() - t0:.0f}s) {learned}", flush=True)
            evaluate(name, seed, params, output, cfg)
        if args.checkpoint_dir:
            sys.path.insert(0, str(args.campaign_dir))
            from eval_stage2a_live_ur5e import load_v2_checkpoint

            for name, stem in (("symmeta_paper", "equiv_v2_dis_dz"), ("perceiver_paper", "perceiver_icba_dis")):
                p, c, _ = load_v2_checkpoint(args.checkpoint_dir / f"{stem}-passive_attempt1-seed{seed}.msgpack")
                evaluate(name, seed, p, "factors", c)
        (args.output / "rows.partial.json").write_text(json.dumps(rows, indent=1))

    summary = {}
    for row in rows:
        summary.setdefault(row["model"], {}).setdefault(row["condition"], []).append(row)
    for model, by_cond in summary.items():
        for cname, rs in by_cond.items():
            entry = dict(
                operator_error_mean=round(float(np.mean([r["mean"] for r in rs])), 4),
                operator_error_seed_std=round(float(np.std([r["mean"] for r in rs])), 4),
                operator_error_median=round(float(np.mean([r["median"] for r in rs])), 4),
                per_seed=[round(r["mean"], 4) for r in rs],
            )
            for key in ("yaw_mae_deg", "yaw_mae_ood_deg", "gate", "left_alone"):
                if key in rs[0]:
                    entry[key] = [round(r[key], 3) for r in rs]
            by_cond[cname] = entry
    result = dict(
        synthetic=bool(args.synthetic),
        seeds=list(seeds),
        support_chunks=SUPPORT_CHUNKS,
        draws_per_trace=DRAWS,
        metres_per_command=metres,
        step_scale=args.step_scale,
        data=data_stats,
        summary=summary,
        soft_diagnostics=diagnostics,
        rows=rows,
    )
    (args.output / "soft-inner-summary.json").write_text(json.dumps(result, indent=1))
    (args.output / "rows.partial.json").unlink()

    print("\noperator error, mean over seeds (0 = exact)")
    print(f"{'model':30s} " + " ".join(f"{c[:8]:>8s}" for c in conds))
    print(f"{'none':30s} " + " ".join(f"{data_stats[c]['no_adaptation']['mean']:8.3f}" for c in conds))
    for model in summary:
        cells = [summary[model].get(c, {}).get("operator_error_mean") for c in conds]
        print(f"{model:30s} " + " ".join(f"{x:8.3f}" if x is not None else f"{'-':>8s}" for x in cells))


if __name__ == "__main__":
    main()
