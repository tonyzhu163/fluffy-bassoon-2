"""Train the timing-conditioned velocity correction on aligned oracle chunks."""

import argparse
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
from motor import FLOW_DIM, HORIZON, Motor, adapter_init, correction


def masked_loss(prediction, target, mask):
    error = jnp.square(prediction - target).mean(axis=-1)
    return jnp.sum(jnp.where(mask, error, 0.)) / jnp.maximum(mask.sum(), 1)


def objective(params, features, target, timing, mask):
    return masked_loss(correction(params, features, timing), target, mask)


def validate(motor, params, dataset, batch_size, n_batches):
    # Reuse these draws at every checkpoint without consuming training randomness.
    rng = np.random.default_rng(961821)
    losses = np.zeros((3, 2), dtype=np.float64)
    for index in range(n_batches):
        batch = dataset.sample(batch_size, rng, partition="validation")
        features, target = motor.training_features(batch, jax.random.fold_in(jax.random.key(961821), index))
        timing = jnp.asarray(batch["timing"])
        mask = jnp.asarray(batch["action_mask"])
        predicted = [jnp.zeros_like(target), correction(params, features, timing), correction(params, features, 1 - timing)]
        for arm, values in enumerate(predicted):
            for label in (0, 1):
                losses[arm, label] += float(masked_loss(values, target, mask & (timing[:, None] == label)))
    return dict(zip(("frozen", "correct", "wrong"), (losses / n_batches).tolist(), strict=True))


def save(path, params, optimizer_state, rng, step, recipe):
    state = dict(params=params, optimizer=optimizer_state, rng=json.dumps(rng.bit_generator.state), step=step, recipe=recipe)
    temp = path.with_suffix(".tmp")
    temp.write_bytes(serialization.to_bytes(state))
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seed", type=int, default=19112026)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=128)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.steps, args.validation_batches, args.checkpoint_every) < 1 or args.batch < 2 or args.batch % 2:
        parser.error("Require an even balanced batch and positive update/validation intervals")
    args.output.mkdir()
    recipe = dict(index_sha256=hashlib.sha256(args.index.read_bytes()).hexdigest(), seed=args.seed, batch=args.batch,
                  steps=args.steps, learning_rate=3e-4, validation_batches=args.validation_batches)
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    key = jax.random.key(args.seed)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(recipe["learning_rate"], weight_decay=1e-4))

    @jax.jit
    def update(params, state, features, target, timing, mask):
        loss, gradient = jax.value_and_grad(objective)(params, features, target, timing, mask)
        changes, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, changes), state, loss, optax.global_norm(gradient)

    motor = Motor()
    with DemonstrationSampler(args.index) as dataset, (args.output / "samples.jsonl").open("w") as samples:
        preview = dataset.sample(args.batch, np.random.default_rng(1921), partition="train")
        features, _ = motor.training_features(preview, jax.random.fold_in(key, 0))
        params = adapter_init(jax.random.fold_in(key, 7), features.shape[-1] + 1)
        opt_state = optimizer.init(params)
        motor.params = params
        noise = np.random.default_rng(9317).standard_normal((HORIZON, FLOW_DIM)).astype(np.float32)
        original = motor.proposal(preview["raw"][0], preview["prompt"][0], noise)
        zero_adapter = motor.proposal(preview["raw"][0], preview["prompt"][0], noise, int(preview["timing"][0]))
        parity = float(np.max(np.abs(original - zero_adapter)))
        if parity > 1e-4:
            raise ValueError(f"Zero adapter changed decoded commands by {parity}")
        print(json.dumps(dict(zero_adapter_max_command_error=parity,
                              adapter_parameters=sum(int(p.size) for p in jax.tree.leaves(params)))), flush=True)
        start = 0
        if args.resume:
            template = dict(params=params, optimizer=opt_state, rng="", step=0, recipe=recipe)
            state = serialization.from_bytes(template, args.resume.read_bytes())
            if state["recipe"] != recipe or state["step"] >= args.steps:
                raise ValueError("Resume requires the same unfinished fit recipe")
            params, opt_state, start = state["params"], state["optimizer"], state["step"]
            rng.bit_generator.state = json.loads(state["rng"])

        start_time = time.monotonic()
        for step in range(start + 1, args.steps + 1):
            batch = dataset.sample(args.batch, rng, partition="train")
            features, target = motor.training_features(batch, jax.random.fold_in(key, step))
            params, opt_state, loss, grad = update(params, opt_state, features, target,
                                                  jnp.asarray(batch["timing"]), jnp.asarray(batch["action_mask"]))
            loss, grad = float(loss), float(grad)
            if not np.isfinite([loss, grad]).all() or (step == 1 and grad <= 0.):
                raise ValueError(f"Invalid loss/gradient at update {step}: {loss}, {grad}")
            samples.write(json.dumps(dict(step=step, sample_ids=batch["sample_ids"]), allow_nan=False) + "\n")
            if step == start + 1 or step % 16 == 0:
                print(json.dumps(dict(step=step, loss=loss, gradient_l2=grad, elapsed_s=time.monotonic() - start_time)), flush=True)
            if step == start + 1 or step % args.checkpoint_every == 0 or step == args.steps:
                save(args.output / "checkpoint.msgpack", params, opt_state, rng, step, recipe)
                samples.flush()
                validation = validate(motor, params, dataset, args.batch, args.validation_batches)
                result = dict(step=step, validation=validation, elapsed_s=time.monotonic() - start_time)
                with (args.output / "validation.jsonl").open("a") as report:
                    report.write(json.dumps(result, allow_nan=False) + "\n")
                print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
