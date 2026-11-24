"""Direct per-example OpenPI losses with one Adam update per matched global batch."""
import hashlib
import json
import os
import time

import flax.nnx as nnx
import flax.serialization as ser
import jax
import jax.numpy as jnp
import numpy as np
import optax

from flow_head import CONDITIONS, full_params, initial_params, install, live_loss, token_loss

SEED, BATCH, EPOCHS, LR = 20261124, 32, 8, .001
DRAWS = {"train": 2, "heldout": 2}


def native_functions(model):
    # Pass frozen weights as dynamic inputs to XLA. Construct every NNX module inside
    # its grad trace so installing a condition cannot mutate another condition's weights.
    graph, frozen = nnx.split(model)

    def scalar_loss(p, state, key, obs, action, history):
        m = nnx.merge(graph, state)
        return live_loss(m, key, obs, action, p, history, train=True).mean()

    names = CONDITIONS[1:]
    template = initial_params(model.action_out_proj.in_features, "joint")

    def stack_conditions(params):
        expanded = [full_params(params[c], template) for c in names]
        return jax.tree.map(lambda *xs: jnp.stack(xs), *expanded)

    # vmap differentiates genuine singleton Pi0.compute_loss calls. Only heads have
    # a mapped axis; observations, model state and RNG (including image aug) do not.
    row_grad = jax.vmap(jax.value_and_grad(scalar_loss), in_axes=(0, None, None, None, None, None))

    @jax.jit
    def accumulate(state, key, obs, action, history, params, totals, losses):
        val, grad = row_grad(stack_conditions(params), state, key, obs, action, history)
        sums = {c: {k: totals[c][k] + grad[k][i] for k in params[c]} for i, c in enumerate(names)}
        return sums, {c: losses[c] + val[i] for i, c in enumerate(names)}

    def score(state, key, obs, action, history, params, *, train):
        _, noise_key, _ = jax.random.split(key, 3)
        target = jax.random.normal(noise_key, action.shape) - action

        def measure(p):
            m = nnx.merge(graph, state)
            layer = install(m.action_out_proj, p, history)
            velocity = []

            def record(h):
                v = layer(h)
                velocity.append(v)
                return v

            m.action_out_proj = record
            native = m.compute_loss(key, obs, action, train=train)
            physical = token_loss(velocity[0][:, :5, :7], target[:, :5, :7]).mean()
            return jnp.stack([native.mean(), physical])

        fitted = jax.vmap(measure)(stack_conditions(params))
        return {"original": measure({}), **{c: fitted[i] for i, c in enumerate(names)}}

    return frozen, accumulate, jax.jit(score, static_argnames=("train",))


def train_heads(model, engine, samples, out, prior, contract):
    frozen, accumulate, _ = engine
    params = {c: initial_params(model.action_out_proj.in_features, c) for c in CONDITIONS[1:]}
    tx = optax.adam(LR)
    states = {c: tx.init(p) for c, p in params.items()}

    @jax.jit
    def update(p, s, grad, count):
        mean = jax.tree.map(lambda g: g / count, grad)
        change, s = tx.update(mean, s, p)
        return optax.apply_updates(p, change), s

    checkpoint = out / "optimizer.msgpack"
    logs = []
    start_step = 0
    if prior and (prior / checkpoint.name).exists():
        data = ser.msgpack_restore((prior / checkpoint.name).read_bytes())
        if data["contract"] != contract:
            raise ValueError("Recovery checkpoint differs from this experiment")
        params, states = ser.from_state_dict((params, states), data["optimizer"])
        logs = json.loads(data["logs"])
        start_step = int(data["step"])
        if start_step != len(logs):
            raise ValueError("Inconsistent checkpoint position")
    n = len(samples)
    per_epoch = (n + BATCH - 1) // BATCH
    zeros = jax.tree.map(jnp.zeros_like, params)
    zero_losses = {c: jnp.array(0., jnp.float32) for c in params}
    for epoch in range(EPOCHS):
        order = np.random.default_rng(np.random.SeedSequence([SEED, epoch, 193])).permutation(n)
        order_hash = hashlib.sha256(order.astype("<i8").tobytes()).hexdigest()
        for batch, off in enumerate(range(0, n, BATCH)):
            step = epoch * per_epoch + batch + 1
            if step <= start_step:
                continue
            tic = time.monotonic()
            ids = order[off:off+BATCH]
            grads, sums = zeros, zero_losses
            for i in ids:
                key, obs, action, history, _ = samples[i]
                grads, sums = accumulate(frozen, key, obs, action, history, params, grads, sums)
            for c in params:
                params[c], states[c] = update(params[c], states[c], grads[c], len(ids))
            jax.block_until_ready((params, states))
            losses = {c: float(v) / len(ids) for c, v in sums.items()}
            if not all(np.isfinite(x) for x in losses.values()):
                raise FloatingPointError(losses)
            seconds = time.monotonic() - tic
            logs.append(dict(step=step, epoch=epoch+1, batch=batch+1, rows=len(ids), seconds=seconds,
                             order_sha256=order_hash, training_mse=losses, rows_per_second=len(ids)/seconds))
            print(json.dumps(logs[-1]), flush=True)
            payload = dict(optimizer=ser.to_state_dict((params, states)), step=step, contract=contract,
                           logs=json.dumps(logs, allow_nan=False))
            tmp = checkpoint.with_suffix(".part")
            tmp.write_bytes(ser.msgpack_serialize(jax.device_get(payload)))
            tmp.replace(checkpoint)
    if not checkpoint.exists() and prior:
        os.link(prior / checkpoint.name, checkpoint)
    epochs = []
    for epoch in range(1, EPOCHS+1):
        rows = [r for r in logs if r["epoch"] == epoch]
        epochs.append(dict(epoch=epoch, batches=len(rows), seconds=sum(r["seconds"] for r in rows),
                           order_sha256=rows[0]["order_sha256"],
                           training_mse={c: sum(r["rows"]*r["training_mse"][c] for r in rows)/n for c in params}))
    return {"original": {}, **params}, epochs


def direct_scores(engine, params, samples, split):
    frozen, _, score = engine
    values = {c: [] for c in params}
    trajectory = []
    for i, (key, obs, action, history, demo) in enumerate(samples):
        result = jax.device_get(score(frozen, key, obs, action, history, params, train=split == "train"))
        for c, loss in result.items():
            values[c].append(loss)
        trajectory.append(demo)
        if (i+1) % BATCH == 0:
            print(f"direct scores {split} {i+1}/{len(samples)}", flush=True)
    trajectory = np.asarray(trajectory)
    demos = sorted(set(trajectory.tolist()))
    by_condition = {}
    for c, pairs in values.items():
        all_dims, physical = np.asarray(pairs).T
        by_condition[c] = {"native_mse_32": float(all_dims.mean()), "first_five_physical_mse_7": float(physical.mean()),
                           "trajectory_mse_32": {s: float(all_dims[trajectory == s].mean()) for s in demos}}
    ix = np.random.default_rng(SEED).integers(len(demos), size=(2000, len(demos)))
    contrasts = {}
    for c in CONDITIONS[:-1]:
        delta = np.array([by_condition["joint"]["trajectory_mse_32"][s] - by_condition[c]["trajectory_mse_32"][s]
                          for s in demos])
        contrasts[c] = {"trajectory_mean_joint_minus_control": float(delta.mean()),
                        "bootstrap_trajectory_ci95": np.quantile(delta[ix].mean(1), [.025,.975]).tolist()
                            if len(demos) > 1 else None}
    return {"conditions": by_condition, "paired_contrasts": contrasts, "trajectory_count": len(demos)}
