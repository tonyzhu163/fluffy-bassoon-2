"""Stage-1 trainer for the multi-perturbation continuous-latent velocity adapter."""

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

from data import DemonstrationSampler
from fluffy_bassoon.libero.latent_adapter import (
    AdapterConfig,
    apply_latent_adapter,
    init_latent_adapter,
    latent_norm_penalty,
    lookup_condition_latent,
)
from motor import FLOW_DIM, HORIZON, LatentMotor


def masked_loss(prediction, target, mask):
    error = jnp.square(prediction - target).mean(axis=-1)
    return jnp.sum(jnp.where(mask, error, 0.0)) / jnp.maximum(mask.sum(), 1)


def objective(params, features, target, condition_ids, mask, prototype_reg):
    latent = lookup_condition_latent(params, condition_ids)
    predicted = apply_latent_adapter(params, features, latent)
    flow_loss = masked_loss(predicted, target, mask)
    return flow_loss + prototype_reg * latent_norm_penalty(params), flow_loss


def validate(motor, params, dataset, batch_size, n_batches):
    rng = np.random.default_rng(961821)
    num_conditions = len(dataset.cases)
    losses = np.zeros((3, num_conditions), dtype=np.float64)
    counts = np.zeros(num_conditions, dtype=np.int64)
    for index in range(n_batches):
        batch = dataset.sample(batch_size, rng, partition="validation")
        features, target = motor.training_features(batch, jax.random.fold_in(jax.random.key(961821), index))
        cond_ids = jnp.asarray(batch["condition_ids"], dtype=jnp.int32)
        wrong_ids = (cond_ids + 1) % num_conditions
        mask = jnp.asarray(batch["action_mask"])
        arms = (
            jnp.zeros_like(target),
            apply_latent_adapter(params, features, lookup_condition_latent(params, cond_ids)),
            apply_latent_adapter(params, features, lookup_condition_latent(params, wrong_ids)),
        )
        for label in range(num_conditions):
            label_mask = mask & (cond_ids[:, None] == label)
            if not bool(jnp.any(label_mask)):
                continue
            counts[label] += 1
            for arm_idx, values in enumerate(arms):
                losses[arm_idx, label] += float(masked_loss(values, target, label_mask))
    safe_counts = np.maximum(counts, 1)
    mean_by_cond = (losses / safe_counts[None]).tolist()
    return {
        "cases": list(dataset.cases),
        "frozen": mean_by_cond[0],
        "correct_z": mean_by_cond[1],
        "wrong_z": mean_by_cond[2],
    }


def save(path, params, optimizer_state, rng, step, cfg, cases, recipe):
    state = dict(
        params=params,
        optimizer=optimizer_state,
        rng=json.dumps(rng.bit_generator.state),
        step=step,
        config=asdict(cfg),
        cases=list(cases),
        recipe=recipe,
    )
    temp = path.with_suffix(".tmp")
    temp.write_bytes(serialization.to_bytes(state))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--no-film", action="store_true")
    parser.add_argument("--prototype-reg", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=19112026)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=256)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    index_data = json.loads(args.index.read_text())
    cases = tuple(index_data["cases"])
    if len(cases) < 2:
        parser.error("Require at least two conditions in the training index")
    if min(args.steps, args.validation_batches, args.checkpoint_every) < 1:
        parser.error("Require positive steps, validation-batches, and checkpoint-every")
    if args.batch < len(cases) or args.batch % len(cases) != 0:
        parser.error(f"Batch size ({args.batch}) must be a positive multiple of {len(cases)} conditions")

    args.output.mkdir()
    recipe = dict(
        index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(),
        cases=list(cases),
        seed=args.seed,
        batch=args.batch,
        steps=args.steps,
        learning_rate=3e-4,
        prototype_reg=args.prototype_reg,
        latent_dim=args.latent_dim,
        width=args.width,
        use_film=not args.no_film,
        validation_batches=args.validation_batches,
    )
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")

    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(recipe["learning_rate"], weight_decay=1e-4))

    @jax.jit
    def update(params, state, features, target, condition_ids, mask):
        (loss, flow_loss), gradient = jax.value_and_grad(objective, has_aux=True)(
            params, features, target, condition_ids, mask, args.prototype_reg
        )
        changes, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, changes), state, loss, flow_loss, optax.global_norm(gradient)

    motor = LatentMotor()
    with DemonstrationSampler(args.index) as dataset, (args.output / "samples.jsonl").open("w") as samples:
        preview = dataset.sample(args.batch, np.random.default_rng(1921), partition="train")
        features, _ = motor.training_features(preview, jax.random.fold_in(key, 0))
        cfg = AdapterConfig(
            feature_dim=int(features.shape[-1]),
            action_dim=7,
            latent_dim=args.latent_dim,
            width=args.width,
            num_conditions=len(cases),
            use_film=not args.no_film,
        )
        params = init_latent_adapter(jax.random.fold_in(key, 7), cfg)
        z_preview = lookup_condition_latent(params, preview["condition_ids"])
        residual = np.asarray(apply_latent_adapter(params, features, z_preview))
        if not np.all(residual == 0.0):
            raise ValueError("Untrained velocity adapter must emit exact zero residual")
        opt_state = optimizer.init(params)
        motor.params = params
        motor.adapter_config = cfg
        motor.cases = cases

        noise = np.random.default_rng(9317).standard_normal((HORIZON, FLOW_DIM)).astype(np.float32)
        z0 = motor.condition_latent(int(preview["condition_ids"][0]))
        original = motor.proposal(preview["raw"][0], preview["prompt"][0], noise)
        zero_adapter = motor.proposal(preview["raw"][0], preview["prompt"][0], noise, latent=z0, active=True)
        sham = motor.proposal(preview["raw"][0], preview["prompt"][0], noise, latent=z0, active=False)
        if not np.array_equal(sham, zero_adapter):
            raise ValueError("Untrained adapter and active=False shared-solver control must agree exactly")
        parity = float(np.max(np.abs(original - zero_adapter)))
        print(
            json.dumps(
                dict(
                    native_vs_shared_solver_max_command_error=parity,
                    shared_solver_zero_adapter_exact=True,
                    adapter_parameters=sum(int(p.size) for p in jax.tree.leaves(params)),
                    cases=list(cases),
                )
            ),
            flush=True,
        )

        start = 0
        if args.resume:
            template = dict(
                params=params,
                optimizer=opt_state,
                rng="",
                step=0,
                config=asdict(cfg),
                cases=list(cases),
                recipe=recipe,
            )
            state = serialization.from_bytes(template, args.resume.read_bytes())
            if state["recipe"] != recipe or state["cases"] != list(cases) or state["step"] >= args.steps:
                raise ValueError("Resume requires matching unfinished recipe and cases")
            params, opt_state, start = state["params"], state["optimizer"], state["step"]
            rng.bit_generator.state = json.loads(state["rng"])

        start_time = time.monotonic()
        for step in range(start + 1, args.steps + 1):
            batch = dataset.sample(args.batch, rng, partition="train")
            features, target = motor.training_features(batch, jax.random.fold_in(key, step))
            params, opt_state, loss, flow_loss, grad = update(
                params,
                opt_state,
                features,
                target,
                jnp.asarray(batch["condition_ids"], dtype=jnp.int32),
                jnp.asarray(batch["action_mask"]),
            )
            loss_f, flow_f, grad_f = float(loss), float(flow_loss), float(grad)
            if not np.isfinite([loss_f, flow_f, grad_f]).all() or (step == 1 and grad_f <= 0.0):
                raise ValueError(f"Invalid loss/gradient at update {step}: {loss_f}, {grad_f}")
            samples.write(json.dumps(dict(step=step, sample_ids=batch["sample_ids"]), allow_nan=False) + "\n")
            if step == start + 1 or step % 16 == 0:
                print(
                    json.dumps(
                        dict(
                            step=step,
                            loss=loss_f,
                            flow_loss=flow_f,
                            gradient_l2=grad_f,
                            elapsed_s=time.monotonic() - start_time,
                        )
                    ),
                    flush=True,
                )
            if step == start + 1 or step % args.checkpoint_every == 0 or step == args.steps:
                save(args.output / "checkpoint.msgpack", params, opt_state, rng, step, cfg, cases, recipe)
                samples.flush()
                validation = validate(motor, params, dataset, args.batch, args.validation_batches)
                result = dict(step=step, validation=validation, elapsed_s=time.monotonic() - start_time)
                with (args.output / "validation.jsonl").open("a") as report:
                    report.write(json.dumps(result, allow_nan=False) + "\n")
                print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
