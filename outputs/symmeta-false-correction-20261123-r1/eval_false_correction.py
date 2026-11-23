"""Offline check: how much does each reader change an interface that needs no correction?

Readers are scored on the locked UR5e supports relabeled to the nominal interface, next to the
recorded and in-family interfaces, with and without a rule that leaves the interface alone when
the estimated correction is small. Same traces, relabeling and operator error as the wider-family
run. No simulator or VLA rollout is used.
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

from fluffy_bassoon.xemb.embodiment import TRAIN_EMBODIMENTS  # noqa: E402
from fluffy_bassoon.xemb.wider_family import operator_error, relabel, rot2_batch, sample_interfaces  # noqa: E402

DRAWS = 4
GATES = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5)

# name, architecture, training interfaces ("paper7" or a fresh in-family draw per trace), peak rate, warmup, updates
TRAINED = (
    ("symmeta_7", SYMMETA, "paper7", 1.5e-3, 0, 2500),
    ("symmeta_cont", SYMMETA, "family", 1.5e-3, 0, 2500),
    ("perceiver_cont", PERCEIVER, "family", 3e-4, 200, 2500),
)


def nominal(n: int) -> dict:
    return dict(yaw_deg=np.zeros(n), shape=np.broadcast_to(np.eye(2), (n, 2, 2)).copy(), z_gain=np.ones(n), gripper_sign=np.ones(n, dtype=np.int64))


def interface_sampler(spec: str):
    if spec == "family":
        return lambda rng, n: sample_mixed(rng, ("family",), n)
    gain = np.array([e.xy_scale for e in TRAIN_EMBODIMENTS], dtype=np.float64)
    pool = dict(
        yaw_deg=np.array([e.yaw_deg for e in TRAIN_EMBODIMENTS], dtype=np.float64),
        shape=gain[:, None, None] * np.eye(2),
        z_gain=gain,
        gripper_sign=np.array([e.gripper_sign for e in TRAIN_EMBODIMENTS]),
    )

    def draw(rng, n):
        idx = rng.integers(len(gain), size=n)
        return {key: val[idx] for key, val in pool.items()}

    return draw


def train_model(train: dict, cfg, spec: str, lr: float, warmup: int, steps: int, seed: int, batch_size: int):
    import jax
    import optax

    params = init_model(jax.random.key(seed), "factors", cfg)
    if warmup:
        sched = optax.warmup_cosine_decay_schedule(0.0, lr, warmup, steps, end_value=0.05 * lr)
    else:
        sched = optax.cosine_decay_schedule(lr, steps, alpha=0.05)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4))
    state = opt.init(params)

    @jax.jit
    def step(p, st, b):
        loss, grads = jax.value_and_grad(lambda q: model_loss(q, "factors", cfg, b))(p)
        updates, st = opt.update(grads, st, p)
        return optax.apply_updates(p, updates), st, loss

    draw = interface_sampler(spec)
    rng = np.random.default_rng(seed * 10007 + 31)
    loss = np.nan
    for _ in range(steps):
        sel = rng.choice(len(train["valid"]), size=batch_size, replace=True)
        params, state, loss = step(params, state, relabel({k: v[sel] for k, v in train.items()}, draw(rng, batch_size)))
    return params, float(loss)


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
    args = parser.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))
    args.output.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        rng = np.random.default_rng(7)
        train_full = synthetic_traces(rng, 384, SUPPORT_CHUNKS, blob=3.0, cam_deg=20.0)
        test = synthetic_traces(rng, 96, SUPPORT_CHUNKS, blob=4.0, cam_deg=20.0)
    else:
        train_full, test, _ = load_recorded(args.diverse_dir, args.reuse_support_dir, args.campaign_dir)
    train = take_chunks(train_full, SUPPORT_CHUNKS)
    base = take_chunks(test, SUPPORT_CHUNKS)
    n, n_src = len(base["valid"]), len(train["valid"])
    recorded = dict(base)
    recorded["op"] = (base["xy_scale"].astype(np.float64)[:, None, None] * rot2_batch(base["yaw_deg"])).astype(np.float32)
    conds = {
        "nominal": relabel(base, nominal(n)),
        "family": relabel(tile(base, DRAWS), sample_interfaces(np.random.default_rng(20261123), "family", n * DRAWS)),
        "recorded": recorded,
    }
    # Source arms, half unshifted and half in-family, choose each reader's gate.
    src = {
        "nominal": relabel(train, nominal(n_src)),
        "family": relabel(train, sample_interfaces(np.random.default_rng(20261124), "family", n_src)),
    }
    no_adaptation = {name: summarize(c["op"], np.broadcast_to(np.eye(2), c["op"].shape)) for name, c in conds.items()}
    print(f"[data] train={n_src} test={n}", flush=True)
    rows = []

    def evaluate(model, seed, params, cfg):
        src_est = {key: predict_ops(params, "factors", cfg, part)[0] for key, part in src.items()}
        scores = {g: float(np.mean([np.mean(operator_error(src[k]["op"], gated(src_est[k], g)[0])) for k in src])) for g in GATES}
        gate = min(scores, key=scores.get)
        for name, cond in conds.items():
            est, yaw = predict_ops(params, "factors", cfg, cond)
            kept, small = gated(est, gate)
            row = dict(model=model, seed=seed, condition=name, gate=gate, left_alone=float(small.mean()),
                       always=summarize(cond["op"], est), gated=summarize(cond["op"], kept))
            if name == "nominal":
                turn = np.abs((np.asarray(yaw) + 180.0) % 360.0 - 180.0)
                row.update(yaw_abs_mean_deg=float(turn.mean()), yaw_abs_p90_deg=float(np.percentile(turn, 90)),
                           yaw_over_15_deg=float(np.mean(turn > 15.0)))
            rows.append(row)

    for seed in seeds:
        for model, cfg, spec, lr, warmup, steps in TRAINED:
            t0 = time.monotonic()
            n_steps = max(20, int(steps * args.step_scale))
            params, loss = train_model(train, cfg, spec, lr, min(warmup, n_steps // 4), n_steps, seed, args.batch_size)
            print(f"[train] {model} seed={seed} steps={n_steps} loss={loss:.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
            evaluate(model, seed, params, cfg)
        if args.checkpoint_dir:
            sys.path.insert(0, str(args.campaign_dir))
            from eval_stage2a_live_ur5e import load_v2_checkpoint

            for model, stem in (("symmeta_paper", "equiv_v2_dis_dz"), ("perceiver_paper", "perceiver_icba_dis")):
                p, c, _ = load_v2_checkpoint(args.checkpoint_dir / f"{stem}-passive_attempt1-seed{seed}.msgpack")
                evaluate(model, seed, p, c)
        (args.output / "rows.partial.json").write_text(json.dumps(rows, indent=1))

    summary = {}
    for row in rows:
        summary.setdefault(row["model"], {}).setdefault(row["condition"], []).append(row)
    for by_cond in summary.values():
        for name, rs in by_cond.items():
            entry = dict(
                always_mean=round(float(np.mean([r["always"]["mean"] for r in rs])), 4),
                gated_mean=round(float(np.mean([r["gated"]["mean"] for r in rs])), 4),
                always_per_seed=[round(r["always"]["mean"], 4) for r in rs],
                gated_per_seed=[round(r["gated"]["mean"], 4) for r in rs],
                gate=[r["gate"] for r in rs],
                left_alone=[round(r["left_alone"], 3) for r in rs],
            )
            for key in ("yaw_abs_mean_deg", "yaw_abs_p90_deg", "yaw_over_15_deg"):
                if key in rs[0]:
                    entry[key] = [round(r[key], 3) for r in rs]
            by_cond[name] = entry
    result = dict(synthetic=bool(args.synthetic), seeds=list(seeds), support_chunks=SUPPORT_CHUNKS, draws_per_trace=DRAWS,
                  step_scale=args.step_scale, gates=list(GATES), no_adaptation=no_adaptation, summary=summary, rows=rows)
    (args.output / "false-correction-summary.json").write_text(json.dumps(result, indent=1))
    (args.output / "rows.partial.json").unlink()

    print("\noperator error, mean over seeds: always correct / with the gate (0 = exact)")
    print(f"{'model':18s} " + " ".join(f"{c:>16s}" for c in conds))
    print(f"{'none':18s} " + " ".join(f"{no_adaptation[c]['mean']:16.3f}" for c in conds))
    for model, by_cond in summary.items():
        print(f"{model:18s} " + " ".join(f"{by_cond[c]['always_mean']:8.3f}/{by_cond[c]['gated_mean']:7.3f}" for c in conds))


if __name__ == "__main__":
    main()
