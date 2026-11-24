"""Linux CPU numerical checks against the independently restored stock output layer."""
import argparse
import json
import os
import resource
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from cotrain_policy import CHECKPOINT, array_sha, base_identity
from flow_head import cached_forward, effective_kernel, initial_params, token_loss


def checkpoint_projection():
    # Orbax OCDBT tensors use flattened dot paths and Zarr metadata.
    def read(name):
        spec = {"driver": "zarr", "kvstore": {"driver": "ocdbt", "base": f"file://{CHECKPOINT}/params/"},
                "path": f"params.action_out_proj.{name}.value"}
        return jnp.asarray(ts.open(spec, open=True).result().read().result(), dtype=jnp.bfloat16)

    w, b = read("kernel"), read("bias")
    layer = nnx.Linear(*w.shape, rngs=nnx.Rngs(0))
    layer.kernel.value, layer.bias.value = w, b
    return layer


def check():
    if jax.default_backend() != "cpu":
        raise RuntimeError("CPU qualification only")
    start = time.monotonic()
    layer = checkpoint_projection()
    key = jax.random.key(341)
    hidden = jax.random.normal(key, (2, 10, layer.in_features)).astype(jnp.bfloat16)
    target = jax.random.normal(jax.random.fold_in(key, 1), (2, 10, 32))
    history = jax.random.normal(jax.random.fold_in(key, 2), (2, 52))
    predict = jax.jit(cached_forward(layer))
    params = initial_params(layer.in_features, "joint")
    stock = layer(hidden)
    np.testing.assert_array_equal(predict(params, hidden, history), stock)
    loss = lambda p: token_loss(predict(p, hidden, history), target).mean()
    first = jax.grad(loss)(params)
    changed = jax.tree.map(lambda p, g: p - 0.01 * g, params, first)
    second = jax.grad(loss)(changed)
    norms = lambda grads: {k: float(jnp.linalg.norm(v)) for k, v in grads.items()}
    if not all(v > 0 and np.isfinite(v) for v in norms(second).values()):
        raise AssertionError("Every enabled factor should receive gradient after zero initialization opens")
    actual = nnx.clone(layer)
    actual.kernel.value = effective_kernel(layer.kernel.value, changed)
    zero = jnp.zeros_like(history)
    np.testing.assert_array_equal(predict(changed, hidden, zero), actual(hidden))
    if np.array_equal(np.asarray(stock), np.asarray(actual(hidden))):
        raise AssertionError("Real output kernel did not affect native projection")
    if np.array_equal(np.asarray(predict(changed, hidden, zero)), np.asarray(predict(changed, hidden, history))):
        raise AssertionError("History did not affect the fitted field")
    return {"base": base_identity(verify_files=False), "backend": jax.default_backend(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "kernel_shape": list(layer.kernel.value.shape),
            "hidden_shape": list(hidden.shape), "velocity_shape": list(stock.shape),
            "base_kernel": array_sha(layer.kernel.value), "changed_kernel": array_sha(actual.kernel.value),
            "stock_hidden_fixture_velocity": array_sha(stock), "zero_init_bitwise": True,
            "merged_stock_layer_zero_history_bitwise": True, "initial_grad_norm": norms(first),
            "updated_grad_norm": norms(second), "jax": jax.__version__, "wall_s": time.monotonic() - start,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("out")
    args = p.parse_args()
    result = check()
    with open(args.out, "x") as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result))
