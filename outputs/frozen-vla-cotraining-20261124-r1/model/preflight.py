"""Bounded Linux CPU exercise of the recovery optimizer, input and output interfaces."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from openpi.models.pi0 import Pi0
from openpi.policies import policy_config
from openpi.training import config

import direct_fit
import source_data
from cotrain_policy import CHECKPOINT, NativeSampler, array_sha, load_stock
from flow_head import full_params, initial_params
from qualify import checkpoint_projection
from train import json_write, raw_item, samples_for_split


class SmallTower(nnx.Module):
    def __call__(self, tokens, **kwargs):
        return tuple(tokens), ()


class CPUFixture(nnx.Module):
    """Stock Pi0 loss, image augmentation and Euler loop, with inexpensive token embeddings."""
    compute_loss = Pi0.compute_loss
    sample_actions = Pi0.sample_actions

    def __init__(self):
        self.action_horizon, self.action_dim = 10, 32
        self.action_out_proj = checkpoint_projection()
        self.PaliGemma = nnx.Dict(llm=SmallTower())

    def embed_prefix(self, obs):
        token = jnp.mean(obs.images["base_0_rgb"], axis=(1, 2, 3))[:, None, None]
        return token, jnp.ones(token.shape[:2], bool), jnp.array([False])

    def embed_suffix(self, obs, actions, time):
        x = jnp.tile(actions, (1,1,32)) + time[:,None,None]
        return x.astype(jnp.bfloat16), jnp.ones(x.shape[:2], bool), jnp.arange(10) == 0, None


def run(args):
    if jax.default_backend() != "cpu":
        raise RuntimeError("Linux CPU preflight only")
    if args.stock:
        stock = load_stock()
    else:
        fixture = CPUFixture()
        cfg = config.get_config("pi05_libero")
        with patch("openpi.models.model.restore_params", return_value={}), patch.object(type(cfg.model), "load", return_value=fixture):
            stock = policy_config.create_trained_policy(cfg, CHECKPOINT)
    ds = source_data.SourceDataset(args.data, "train")
    row = next(row for row in ds if np.any(row["history"]))
    # One real example, its two native flow keys and a final short global batch.
    samples = samples_for_split(stock, [row], "train", args.out)
    engine = direct_fit.native_functions(stock._model)
    with patch.object(direct_fit, "EPOCHS", 1):
        params, logs = direct_fit.train_heads(stock._model, engine, samples, args.out, None, "cpu-smoke")
        with tempfile.TemporaryDirectory(dir=args.out) as recover:
            restored, replay = direct_fit.train_heads(stock._model, engine, samples, Path(recover), args.out, "cpu-smoke")
    for p, q in zip(jax.tree.leaves(params), jax.tree.leaves(restored), strict=True):
        np.testing.assert_array_equal(p, q)
    assert logs == replay
    # The three zero-output initialized conditions execute the same native graph.
    if len(set(logs[0]["training_mse"].values())) != 1:
        raise AssertionError("Unmatched initial per-condition losses")
    scores = direct_fit.direct_scores(engine, params, samples, "train")
    template = initial_params(1024, "joint")
    motor = NativeSampler(stock, {c: full_params(p, template) for c,p in params.items()})
    item = raw_item(row)
    item.pop("actions")
    noise = np.random.default_rng(2026).normal(size=(10,32)).astype(np.float32)
    parity = motor.parity(item, noise)
    zero = np.zeros_like(row["history"])
    stock_action = stock.infer(item, noise=noise)["actions"]
    np.testing.assert_array_equal(motor.infer(item, noise, "adapter_only", zero)["actions"], stock_action)
    actions = {c: array_sha(motor.infer(item, noise, c, row["history"])["actions"]) for c in params}
    if len(set(actions.values())) != len(actions):
        raise AssertionError("A fitted CPU fixture condition failed to change the decoded chunk")
    json_write(args.out / "preflight.json", dict(stock_towers=args.stock, demo=row["trajectory_id"], t=int(row["t"]),
        keys=[np.asarray(jax.random.key_data(x[0])).tolist() for x in samples],
        action_shape=list(samples[0][2].shape), image_shapes={k:list(v.shape) for k,v in samples[0][1].images.items()},
        state_shape=list(samples[0][1].state.shape), history_shape=list(samples[0][3].shape),
        direct_epochs=logs, optimizer_recovered=True, direct_scores=scores, parity=parity, decoded_sha=actions))
    print("Passed direct optimizer, recovery, score and independent native sampling control", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--stock", action="store_true")
    run(p.parse_args())
