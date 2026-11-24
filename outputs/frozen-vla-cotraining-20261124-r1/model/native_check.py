"""Reference checks on one observation using real OpenPI training and sampling paths."""
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from cotrain_policy import NativeSampler, array_sha, load_stock, transformed
from flow_head import bind, cached_forward, capture, full_params, initial_params, live_loss, token_loss


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
    cached_stock = jax.jit(lambda h, z, y: token_loss(predict({}, h, z), y))(hidden, history, target)
    report = dict(hidden_shape=list(hidden.shape), hidden_dtype=str(hidden.dtype),
                  target_shape=list(target.shape), native_loss=float(stock_loss.mean()),
                  cache_stock_max_abs=float(jnp.max(jnp.abs(cached_stock - stock_loss))))
    if gradient:
        direct = bind(stock._model, live_loss, static_argnames=("train",))

        def native_loss(p):
            return direct(rng, obs, actions, p, history, train=True).mean()

        cached_val, cached_grad = jax.value_and_grad(lambda p: jax.jit(
            lambda weights, h, hist, y: token_loss(predict(weights, h, hist), y).mean()
        )(p, hidden, history, target))(perturbed)
        live_val, live_grad = jax.value_and_grad(native_loss)(perturbed)
        report.update(direct_perturbed_loss=float(live_val), cached_perturbed_loss=float(cached_val),
                      fit_engine="direct-native-only")
        report["gradient"] = {}
        for k in cached_grad:
            norm = float(jnp.linalg.norm(live_grad[k]))
            diff = float(jnp.linalg.norm(live_grad[k] - cached_grad[k]))
            if not np.isfinite(norm) or not norm > 0:
                raise AssertionError(f"Invalid native qualification gradient {k}: {norm}")
            report["gradient"][k] = {"live_l2": norm, "difference_l2": diff, "relative_difference": diff / norm,
                                     "within_original_bf16_bound": diff <= float(jnp.finfo(hidden.dtype).eps) * norm}
    sampler = NativeSampler(stock, {"original": full_params({}, template), "joint": perturbed})
    noise = np.asarray(jax.random.normal(rng, actions.shape), np.float32)[0]
    inference_item = {k: v for k, v in item.items() if k != "actions"}
    parity = sampler.parity(inference_item, noise)
    changed = sampler.infer(inference_item, noise, "joint", np.asarray(history[0]))["actions"]
    if not np.isfinite(changed).all() or array_sha(changed) == parity["stock_sha256"]:
        raise AssertionError("Nonzero native intervention failed to change decoded actions")
    report.update(parity=parity, changed_action_sha=array_sha(changed), live_steps=10)
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
