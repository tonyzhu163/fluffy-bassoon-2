"""Reference checks on one observation using real OpenPI training and sampling paths."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from cotrain_policy import array_sha, load_stock, transformed
from flow_head import bind, cached_forward, capture, full_params, initial_params, live_loss, live_sample, token_loss


def check(stock, item, history, *, gradient=True):
    layer = stock._model.action_out_proj
    obs, actions = transformed(stock, item)
    rng = jax.random.key(2026)
    history = jnp.asarray(history)[None]
    get = bind(stock._model, capture, static_argnames=("train",))
    hidden, target, stock_loss = get(rng, obs, actions, train=True)
    predict = cached_forward(layer)
    template = initial_params(layer.in_features, "joint")
    # Keep the point off both zero-output initializations for the gradient test.
    key1, key2 = jax.random.split(rng)
    perturbed = dict(template, right=1e-3 * jax.random.normal(key1, template["right"].shape),
                     history_out=1e-3 * jax.random.normal(key2, template["history_out"].shape))
    cached_stock = token_loss(predict({}, hidden, history), target)
    np.testing.assert_allclose(cached_stock, stock_loss, rtol=3e-6, atol=3e-6)
    report = dict(hidden_shape=list(hidden.shape), hidden_dtype=str(hidden.dtype),
                  target_shape=list(target.shape), native_loss=float(stock_loss.mean()),
                  cache_stock_max_abs=float(jnp.max(jnp.abs(cached_stock - stock_loss))))
    if gradient:
        direct = bind(stock._model, live_loss, static_argnames=("train",))

        def cached_loss(p):
            return token_loss(predict(p, hidden, history), target).mean()

        def native_loss(p):
            return direct(rng, obs, actions, p, history, train=True).mean()

        cached_val, cached_grad = jax.jit(jax.value_and_grad(cached_loss))(perturbed)
        live_val, live_grad = jax.value_and_grad(native_loss)(perturbed)
        np.testing.assert_allclose(cached_val, live_val, rtol=3e-6, atol=3e-6)
        for k in cached_grad:
            # BF16 is the checkpoint's projection/input precision. Gradients can differ
            # by final reduction rounding between independently compiled graphs.
            std = float(jnp.linalg.norm(live_grad[k]))
            diff = float(jnp.linalg.norm(live_grad[k] - cached_grad[k]))
            if not std > 0 or not diff <= float(jnp.finfo(hidden.dtype).eps) * std:
                raise AssertionError(f"Cached vs direct gradient {k}: difference={diff}, norm={std}")
        report["gradient"] = {k: {"live_l2": float(jnp.linalg.norm(live_grad[k])),
                                  "difference_l2": float(jnp.linalg.norm(live_grad[k] - cached_grad[k]))}
                              for k in cached_grad}
    sampler = bind(stock._model, live_sample)
    noise = np.asarray(jax.random.normal(rng, actions.shape), np.float32)
    inference_item = {k: v for k, v in item.items() if k != "actions"}
    ordinary = stock.infer(inference_item, noise=noise[0])["actions"]
    def decoded(params):
        result = sampler(rng, obs, noise, params, history)
        return stock._output_transform({"actions": np.asarray(result[0]), "state": np.asarray(obs.state[0])})["actions"]
    noop = decoded(full_params({}, template))
    np.testing.assert_array_equal(noop, ordinary)
    fitted = decoded(perturbed)
    if np.array_equal(fitted, ordinary):
        raise AssertionError("Nonzero projection and history injection failed to change the sampler")
    report.update(stock_action_sha=array_sha(ordinary), noop_action_sha=array_sha(noop),
                  changed_action_sha=array_sha(fitted), live_steps=10)
    return report


if __name__ == "__main__":
    import argparse
    import resource
    from openpi.policies.libero_policy import make_libero_example

    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    args = p.parse_args()
    if jax.default_backend() != "cpu":
        raise RuntimeError("Standalone native qualification runs on CPU")
    start = time.monotonic()
    np.random.seed(2026)
    stock = load_stock()
    print("Stock policy restored", flush=True)
    item = make_libero_example()
    item["actions"] = np.random.uniform(-1, 1, (10, 7)).astype(np.float32)
    result = check(stock, item, np.ones(52, np.float32))
    result.update(device=str(jax.devices()[0]), wall_s=time.monotonic() - start,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
