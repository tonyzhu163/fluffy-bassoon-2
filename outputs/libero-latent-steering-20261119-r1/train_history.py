"""Stage-2 trainer for the decoupled causal history belief encoder."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fluffy_bassoon.libero.history_encoder import (
    HistoryEncoderConfig,
    apply_history_encoder,
    counterfactual_margin_loss,
    decode_steering_latent,
    history_belief_loss,
    init_history_encoder,
    pack_control_tokens,
)


SEEDS = (41, 42, 43)


def load_archive(archive_path, cases, cfg):
    """Load collected support/prefix/query episodes grouped by partition."""
    archive_dir = archive_path.parent
    payload = json.loads(archive_path.read_text())
    if tuple(payload["cases"]) != tuple(cases):
        raise ValueError(f"Archive cases {payload['cases']} do not match adapter cases {cases}")
    by_split = {}
    for split_name in ("train", "dev"):
        entries = [row for row in payload["trials"] if row["split"] == split_name]
        if not entries:
            raise ValueError(f"No trials found for split {split_name!r} in {archive_path}")
        s_tokens, s_valid, p_tokens, p_valid = [], [], [], []
        q_reset, q_prefix, cond_ids, keys = [], [], [], []
        for row in entries:
            with np.load(archive_dir / row["support_path"]) as sup:
                st, sv = pack_control_tokens(
                    sup["issued"],
                    sup["positions"],
                    sup["initial_position"],
                    sup["gripper"],
                    sup["rewards"],
                    sup["waits"],
                    max_controls=cfg.max_support_controls,
                    controls_per_token=cfg.controls_per_token,
                    episode_boundary_at_start=True,
                )
            with np.load(archive_dir / row["prefix_path"]) as pfx:
                pt, pv = pack_control_tokens(
                    pfx["issued"],
                    pfx["positions"],
                    pfx["initial_position"],
                    pfx["gripper"],
                    pfx["rewards"],
                    pfx["waits"],
                    max_controls=cfg.max_prefix_controls,
                    controls_per_token=cfg.controls_per_token,
                    episode_boundary_at_start=True,
                )
            qr = np.load(archive_dir / row["query_reset_embedding"]).astype(np.float32)
            qp = np.load(archive_dir / row["query_prefix_embedding"]).astype(np.float32)
            s_tokens.append(st)
            s_valid.append(sv)
            p_tokens.append(pt)
            p_valid.append(pv)
            q_reset.append(qr)
            q_prefix.append(qp)
            cond_ids.append(int(row["condition_id"]))
            keys.append((row["suite"], int(row["task_id"]), row["variant"]))
        by_split[split_name] = dict(
            support_tokens=np.stack(s_tokens),
            support_valid=np.stack(s_valid),
            prefix_tokens=np.stack(p_tokens),
            prefix_valid=np.stack(p_valid),
            query_reset=np.stack(q_reset),
            query_prefix=np.stack(q_prefix),
            condition_id=np.asarray(cond_ids, dtype=np.int32),
            keys=keys,
        )
    return by_split


def sample_minibatch(split, batch_size, rng):
    """Sample balanced condition rows for reset and prefix views plus paired contrasts."""
    idx = rng.integers(len(split["condition_id"]), size=batch_size)
    empty_pfx_t = np.zeros_like(split["prefix_tokens"][idx])
    empty_pfx_v = np.zeros_like(split["prefix_valid"][idx])
    empty_sup_t = np.zeros_like(split["support_tokens"][idx])
    empty_sup_v = np.zeros_like(split["support_valid"][idx])

    reset_batch = dict(
        support_tokens=jnp.asarray(split["support_tokens"][idx]),
        support_valid=jnp.asarray(split["support_valid"][idx]),
        prefix_tokens=jnp.asarray(empty_pfx_t),
        prefix_valid=jnp.asarray(empty_pfx_v),
        query_features=jnp.asarray(split["query_reset"][idx]),
        condition_id=jnp.asarray(split["condition_id"][idx]),
    )
    prefix_batch = dict(
        support_tokens=jnp.asarray(empty_sup_t),
        support_valid=jnp.asarray(empty_sup_v),
        prefix_tokens=jnp.asarray(split["prefix_tokens"][idx]),
        prefix_valid=jnp.asarray(split["prefix_valid"][idx]),
        query_features=jnp.asarray(split["query_prefix"][idx]),
        condition_id=jnp.asarray(split["condition_id"][idx]),
    )
    # Pair consecutive halves for counterfactual margin loss on contrasting conditions.
    half = batch_size // 2
    idx_a = rng.integers(len(split["condition_id"]), size=half)
    idx_b = rng.integers(len(split["condition_id"]), size=half)
    pair_mask = split["condition_id"][idx_a] != split["condition_id"][idx_b]
    cf_a = dict(
        support_tokens=jnp.asarray(split["support_tokens"][idx_a]),
        support_valid=jnp.asarray(split["support_valid"][idx_a]),
        prefix_tokens=jnp.asarray(empty_pfx_t[:half]),
        prefix_valid=jnp.asarray(empty_pfx_v[:half]),
        query_features=jnp.asarray(split["query_reset"][idx_a]),
        condition_id=jnp.asarray(split["condition_id"][idx_a]),
    )
    cf_b = dict(
        support_tokens=jnp.asarray(split["support_tokens"][idx_b]),
        support_valid=jnp.asarray(split["support_valid"][idx_b]),
        prefix_tokens=jnp.asarray(empty_pfx_t[:half]),
        prefix_valid=jnp.asarray(empty_pfx_v[:half]),
        query_features=jnp.asarray(split["query_reset"][idx_b]),
        condition_id=jnp.asarray(split["condition_id"][idx_b]),
    )
    return reset_batch, prefix_batch, (cf_a, cf_b, jnp.asarray(pair_mask))


def evaluate_views(params, cfg, split, condition_table, rng):
    """Score retained, cleared, current_prefix, and swapped_donor offline accuracy and latent MSE."""
    n = len(split["condition_id"])
    cond_ids = jnp.asarray(split["condition_id"], dtype=jnp.int32)
    target_z = condition_table[cond_ids]
    empty_pfx_t = jnp.zeros_like(split["prefix_tokens"])
    empty_pfx_v = jnp.zeros_like(split["prefix_valid"])

    donor_idx = np.arange(n)
    for i in range(n):
        candidates = np.flatnonzero(split["condition_id"] != split["condition_id"][i])
        if len(candidates):
            donor_idx[i] = int(candidates[rng.integers(len(candidates))])

    views = {
        "retained": (
            split["support_tokens"],
            split["support_valid"],
            empty_pfx_t,
            empty_pfx_v,
            split["query_reset"],
            True,
            False,
        ),
        "cleared": (
            split["support_tokens"],
            split["support_valid"],
            empty_pfx_t,
            empty_pfx_v,
            split["query_reset"],
            False,
            False,
        ),
        "current_prefix": (
            split["support_tokens"],
            split["support_valid"],
            split["prefix_tokens"],
            split["prefix_valid"],
            split["query_prefix"],
            False,
            True,
        ),
        "swapped_donor": (
            split["support_tokens"][donor_idx],
            split["support_valid"][donor_idx],
            empty_pfx_t,
            empty_pfx_v,
            split["query_reset"],
            True,
            False,
        ),
    }
    report = {}
    for name, (st, sv, pt, pv, qf, ret_s, ret_p) in views.items():
        logits, delta = apply_history_encoder(
            params,
            cfg,
            jnp.asarray(st),
            jnp.asarray(sv),
            jnp.asarray(pt),
            jnp.asarray(pv),
            jnp.asarray(qf),
            retain_support=ret_s,
            retain_prefix=ret_p,
        )
        preds = jnp.argmax(logits, axis=-1)
        soft_z = decode_steering_latent(logits, delta, condition_table, mode="soft")
        report[name] = dict(
            accuracy=float(jnp.mean(preds == cond_ids)),
            latent_mse=float(jnp.mean(jnp.square(soft_z - target_z))),
        )
    return report


def save_checkpoint(path, params, opt_state, step, seed, cfg, cases, adapter_sha, recipe):
    state = dict(
        params=params,
        optimizer=opt_state,
        step=step,
        seed=seed,
        encoder_config=asdict(cfg),
        cases=list(cases),
        adapter_sha256=adapter_sha,
        recipe=recipe,
    )
    temp = path.with_suffix(".tmp")
    temp.write_bytes(serialization.to_bytes(state))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--adapter-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--seeds", type=str, default="41,42,43")
    args = parser.parse_args()

    adapter_bytes = args.adapter_checkpoint.read_bytes()
    adapter_sha = hashlib.sha256(adapter_bytes).hexdigest()
    adapter_state = serialization.msgpack_restore(adapter_bytes)
    cases = tuple(adapter_state["cases"])
    condition_table = jax.lax.stop_gradient(jnp.asarray(adapter_state["params"]["condition_table"], dtype=jnp.float32))
    latent_dim = int(condition_table.shape[-1])

    cfg = HistoryEncoderConfig(num_conditions=len(cases), latent_dim=latent_dim)
    splits = load_archive(args.archive, cases, cfg)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    args.output.mkdir(parents=True, exist_ok=False)
    recipe = dict(
        archive=str(args.archive),
        archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
        adapter_sha256=adapter_sha,
        cases=list(cases),
        steps=args.steps,
        batch=args.batch,
        counterfactual_weight=args.counterfactual_weight,
        learning_rate=3e-4,
        seeds=list(seeds),
    )
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(recipe["learning_rate"], weight_decay=1e-4))

    def total_loss(params, reset_batch, prefix_batch, cf_tuple):
        loss_reset, metrics = history_belief_loss(
            params, cfg, reset_batch, condition_table, cleared_weight=0.5, prefix_weight=0.0
        )
        loss_pfx, pfx_metrics = history_belief_loss(
            params, cfg, prefix_batch, condition_table, cleared_weight=0.0, prefix_weight=1.0
        )
        loss = (loss_reset + 0.5 * loss_pfx) / 1.5
        if args.counterfactual_weight > 0.0:
            cf_a, cf_b, pair_mask = cf_tuple
            la, _ = apply_history_encoder(
                params,
                cfg,
                cf_a["support_tokens"],
                cf_a["support_valid"],
                cf_a["prefix_tokens"],
                cf_a["prefix_valid"],
                cf_a["query_features"],
                retain_support=True,
                retain_prefix=False,
            )
            lb, _ = apply_history_encoder(
                params,
                cfg,
                cf_b["support_tokens"],
                cf_b["support_valid"],
                cf_b["prefix_tokens"],
                cf_b["prefix_valid"],
                cf_b["query_features"],
                retain_support=True,
                retain_prefix=False,
            )
            ta = jax.nn.one_hot(cf_a["condition_id"], cfg.num_conditions, dtype=jnp.float32)
            tb = jax.nn.one_hot(cf_b["condition_id"], cfg.num_conditions, dtype=jnp.float32)
            cf_loss = counterfactual_margin_loss(la, lb, ta, tb, pair_mask=pair_mask)
            loss = loss + args.counterfactual_weight * cf_loss
        else:
            cf_loss = jnp.asarray(0.0, dtype=jnp.float32)
        metrics = dict(metrics, prefix_acc=pfx_metrics["prefix_acc"], cf_loss=cf_loss)
        return loss, metrics

    @jax.jit
    def update(params, opt_state, reset_batch, prefix_batch, cf_tuple):
        (loss, metrics), grad = jax.value_and_grad(total_loss, has_aux=True)(
            params, reset_batch, prefix_batch, cf_tuple
        )
        changes, opt_state = optimizer.update(grad, opt_state, params)
        return optax.apply_updates(params, changes), opt_state, loss, metrics

    all_summaries = {}
    for seed in seeds:
        started = time.monotonic()
        rng = np.random.default_rng(seed)
        params = init_history_encoder(jax.random.key(seed), cfg)
        opt_state = optimizer.init(params)
        for step in range(1, args.steps + 1):
            reset_b, prefix_b, cf_tuple = sample_minibatch(splits["train"], args.batch, rng)
            params, opt_state, loss, metrics = update(params, opt_state, reset_b, prefix_b, cf_tuple)
            if step == 1 or step % 100 == 0 or step == args.steps:
                print(
                    json.dumps(
                        dict(
                            seed=seed,
                            step=step,
                            loss=float(loss),
                            retained_acc=float(metrics["retained_acc"]),
                            cleared_acc=float(metrics["cleared_acc"]),
                            prefix_acc=float(metrics["prefix_acc"]),
                            cf_loss=float(metrics["cf_loss"]),
                        )
                    ),
                    flush=True,
                )
        ckpt_path = args.output / f"seed-{seed}.msgpack"
        save_checkpoint(ckpt_path, params, opt_state, args.steps, seed, cfg, cases, adapter_sha, recipe)
        dev_eval = evaluate_views(params, cfg, splits["dev"], condition_table, np.random.default_rng(seed + 1000))
        all_summaries[str(seed)] = dict(
            checkpoint=str(ckpt_path),
            seconds=round(time.monotonic() - started, 3),
            dev_offline=dev_eval,
        )

    (args.output / "summary.json").write_text(json.dumps(dict(recipe=recipe, seeds=all_summaries), indent=2) + "\n")
    print(json.dumps(all_summaries, indent=2), flush=True)


if __name__ == "__main__":
    main()
