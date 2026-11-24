"""Cache native supervised denoiser features, fit three controlled heads, score held-out source demos."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

import source_data
from ur5e_motor import to_openpi_observation
from cotrain_policy import SCHEMA, LockedPolicy, base_identity, file_sha, kernel_receipt, load_stock, transformed
from flow_head import CONDITIONS, bind, cached_forward, capture, initial_params, token_loss
import native_check

SEED, BATCH, EPOCHS, LR = 20261124, 32, 8, 1e-3
DRAWS = {"train": 2, "heldout": 2}
BLOCK = 32


def json_write(path, value):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def raw_item(row):
    actions = np.asarray(row["actions"])
    history = np.asarray(row["history"])
    for x, shape in ((actions, (10, 7)), (history, (52,))):
        if x.shape != shape or x.dtype != np.float32 or not np.isfinite(x).all():
            raise ValueError(f"Invalid source row at {row['trajectory_id']}/{row['t']}")
    return dict(to_openpi_observation(row["raw"], row["prompt"], canonicalize_quat=False), actions=actions)


def schedule(ds, split):
    if not len(ds):
        raise ValueError(f"Empty source split {split}")
    # An independent reproducible key per draw and split, stable across resume.
    return [(int(i), draw, (int(i) * 2 + draw) * 2 + int(split == "heldout"))
            for i in range(len(ds)) for draw in range(DRAWS[split])]


def cache_split(stock, extract, ds, split, cache_dir, prior, contract_hash, out):
    sched = schedule(ds, split)
    blocks = []
    start = time.monotonic()
    reused = 0
    for off in range(0, len(sched), BLOCK):
        block_name = f"{split}-{off:06d}.npz"
        previous = prior / "cache" / block_name if prior else None
        dest = cache_dir / block_name
        if previous and previous.exists():
            os.link(previous, dest)
            reused += 1
        if not dest.exists():
            entries = []
            for i, draw, fold in sched[off:off + BLOCK]:
                row = ds[i]
                obs, action = transformed(stock, raw_item(row))
                if action.shape != (1, 10, 32) or not np.isfinite(np.asarray(action)).all():
                    raise ValueError("Invalid stock-normalized expert actions")
                key = jax.random.fold_in(jax.random.key(SEED), fold)
                h, y, native = extract(key, obs, action, train=(split == "train"))
                # Float32 losslessly represents every captured BF16 value. Avoid NumPy's
                # unportable bfloat16 .npy encoding; cast back to original dtype on load.
                entries.append((np.asarray(h[0], np.float32), np.asarray(y[0]), row["history"],
                                np.asarray(native[0]), str(row["trajectory_id"]), int(row["t"])))
                if off == 0 and len(entries) == 1:
                    print(f"first {split} feature {h.shape} {h.dtype}, {time.monotonic()-start:.2f}s", flush=True)
            fields = dict(zip(("hidden", "target", "history", "native", "trajectory", "t"), map(np.stack, zip(*entries))))
            with dest.with_suffix(".part").open("wb") as f:
                np.savez(f, **fields, contract=contract_hash, keys=sched[off:off + BLOCK], dtype=str(h.dtype))
            dest.with_suffix(".part").replace(dest)
        with np.load(dest, allow_pickle=False) as f:
            if str(f["contract"]) != contract_hash or not np.array_equal(f["keys"], sched[off:off + BLOCK]):
                raise ValueError(f"Cache contract mismatch: {dest}")
            blocks.append({k: f[k] for k in ("hidden", "target", "history", "native", "trajectory", "t", "keys")})
            dtype = str(f["dtype"])
        print(f"{split} {min(off+BLOCK,len(sched))}/{len(sched)} rows, bank {dest.name}", flush=True)
    data = {k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]}
    for k in ("hidden", "target", "history", "native"):
        if not np.isfinite(data[k]).all():
            raise ValueError(f"Nonfinite {split}/{k}")
    data["hidden"] = data["hidden"].astype(jnp.dtype(dtype))
    elapsed = time.monotonic() - start
    json_write(out / f"{split}-cache.json", {"rows": len(sched), "source_dataset_rows": len(ds),
               "examples_before_draws": len(sched) // DRAWS[split], "wall_s": elapsed, "reused_blocks": reused,
               "execution_batch": 1, "hidden_dtype": dtype, "native_augmented": split == "train",
               "block_sha256": {p.name: file_sha(p) for p in sorted(cache_dir.glob(f"{split}-*.npz"))}})
    return data


def batch_data(data, ids):
    return tuple(jnp.asarray(data[k][ids]) for k in ("hidden", "history", "target"))


def train_heads(stock, train, out, prior, contract_hash):
    layer = stock._model.action_out_proj
    predict = cached_forward(layer)
    tx = optax.adam(LR)
    params = {c: initial_params(layer.in_features, c) for c in CONDITIONS[1:]}
    state = {c: tx.init(p) for c, p in params.items()}

    @jax.jit
    def step(p, state, h, history, y):
        def loss(weights):
            return token_loss(predict(weights, h, history), y).mean()
        loss_value, grad = jax.value_and_grad(loss)(p)
        updates, state = tx.update(grad, state, p)
        return optax.apply_updates(p, updates), state, loss_value

    checkpoint = out / "optimizer.msgpack"
    epoch0 = 0
    previous = prior / checkpoint.name if prior else None
    if previous and previous.exists():
        template = dict(params=params, state=state, epoch=0, contract=contract_hash)
        saved = serialization.msgpack_restore(previous.read_bytes())
        logs = json.loads(saved.pop("logs_json"))
        restored = serialization.from_state_dict(template, saved)
        params, state, epoch0 = restored["params"], restored["state"], restored["epoch"]
        if restored["contract"] != contract_hash:
            raise ValueError("Optimizer checkpoint contract mismatch")
    else:
        logs = []
    for epoch in range(epoch0, EPOCHS):
        ids = np.random.default_rng(np.random.SeedSequence([SEED, epoch, 193])).permutation(len(train["target"]))
        start = time.monotonic()
        losses = {c: 0. for c in params}
        for off in range(0, len(ids), BATCH):
            ix = ids[off:off + BATCH]
            args = batch_data(train, ix)
            for c in params:
                params[c], state[c], value = step(params[c], state[c], *args)
                losses[c] += float(value) * len(ix)
        row = dict(epoch=epoch+1, batches=int(np.ceil(len(ids)/BATCH)), seconds=time.monotonic()-start,
                   order_sha256=hashlib.sha256(ids.astype("<i8").tobytes()).hexdigest(),
                   training_mse={c: v / len(ids) for c, v in losses.items()})
        logs.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)
        payload = dict(params=params, state=state, epoch=epoch+1, contract=contract_hash, logs_json=json.dumps(logs))
        tmp = checkpoint.with_suffix(".part")
        tmp.write_bytes(serialization.to_bytes(payload))
        tmp.replace(checkpoint)
    # A fresh recovery attempt may start after the last complete epoch.
    if not checkpoint.exists() and previous:
        os.link(previous, checkpoint)
    return {"original": {}, **params}, logs


def scores(layer, params, data):
    predict = cached_forward(layer)

    @jax.jit
    def score(p, h, hist, y):
        v = predict(p, h, hist)
        return token_loss(v, y), token_loss(v[:, :5, :7], y[:, :5, :7])

    by_condition = {}
    trajectory = data["trajectory"]
    demos = sorted(set(trajectory.tolist()))
    for c, p in params.items():
        all_dims, physical = [], []
        for off in range(0, len(trajectory), BATCH):
            ids = slice(off, off+BATCH)
            mse, mse7 = score(p, *batch_data(data, ids))
            if c == "original":
                np.testing.assert_allclose(np.asarray(mse), data["native"][ids], rtol=3e-6, atol=3e-6)
            all_dims.extend(np.asarray(mse).mean(axis=1))
            physical.extend(np.asarray(mse7).mean(axis=1))
        all_dims, physical = np.array(all_dims), np.array(physical)
        by_condition[c] = {"native_mse_32": float(all_dims.mean()), "first_five_physical_mse_7": float(physical.mean()),
                           "trajectory_mse_32": {s: float(all_dims[trajectory == s].mean()) for s in demos}}
    contrasts = {}
    ix = np.random.default_rng(SEED).integers(len(demos), size=(2000, len(demos)))
    for c in CONDITIONS[:-1]:
        delta = np.array([by_condition["joint"]["trajectory_mse_32"][s] - by_condition[c]["trajectory_mse_32"][s]
                          for s in demos])
        contrasts[c] = {"trajectory_mean_joint_minus_control": float(delta.mean()),
                        "bootstrap_trajectory_ci95": np.quantile(delta[ix].mean(1), [.025,.975]).tolist()
                            if len(demos) > 1 else None}
    return {"conditions": by_condition, "paired_contrasts": contrasts, "trajectory_count": len(demos)}


def run(args):
    if len(jax.devices()) != 1 or jax.default_backend() != "gpu":
        raise RuntimeError("Fit requires one allocated GPU")
    start = time.monotonic()
    out = args.out
    manifest_hash = file_sha(args.data / "manifest.json")
    if manifest_hash != args.manifest:
        raise ValueError("Source manifest mismatch")
    manifest = json.loads((args.data / "manifest.json").read_text())
    tapes = {}
    for split in DRAWS:
        part = manifest["splits"][split]
        if file_sha(args.data / part["file"]) != part["sha256"]:
            raise ValueError(f"Changed staged {split} HDF5")
        tapes[split] = {t["sequence_sha256"] for t in part["trajectories"]}
    if tapes["train"] & tapes["heldout"]:
        raise ValueError("The two source splits share an action/state tape")
    ds = {s: source_data.SourceDataset(args.data, split=s) for s in DRAWS}
    src = Path(__file__).resolve().parent
    spec = {"schema": SCHEMA, "data_manifest_sha256": manifest_hash,
            "source_code": file_sha(source_data.__file__),
            "model_code": {p.name: file_sha(p) for p in src.glob("*.py")},
            "conversion_code": file_sha(__import__("ur5e_motor").__file__),
            "base": base_identity(verify_files=False), "seed": SEED, "draws": DRAWS,
            "batch": BATCH, "epochs": EPOCHS, "optimizer": "Adam", "lr": LR}
    contract_hash = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    json_write(out / "config.json", spec)
    if args.prior and json.loads((args.prior / "config.json").read_text()) != spec:
        raise ValueError("Read-only recovery input has a different contract")
    stock = load_stock()
    model = stock._model
    first = next(row for row in ds["train"] if np.any(row["history"]))
    parity = native_check.check(stock, raw_item(first), first["history"])
    parity.update(trajectory_id=str(first["trajectory_id"]), t=int(first["t"]))
    json_write(out / "native-parity.json", parity)
    extract = bind(model, capture, static_argnames=("train",))
    cache_dir = out / "cache"
    cache_dir.mkdir()
    data = {s: cache_split(stock, extract, ds[s], s, cache_dir, args.prior, contract_hash, out) for s in DRAWS}
    if set(data["train"]["trajectory"]) & set(data["heldout"]["trajectory"]):
        raise ValueError("Overlapping train and held-out source trajectories")
    params, logs = train_heads(stock, data["train"], out, args.prior, contract_hash)
    layer = model.action_out_proj
    metrics = {s: scores(layer, params, d) for s, d in data.items()}
    heads = out / "heads.npz"
    with heads.with_suffix(".part").open("wb") as f:
        np.savez(f, **{f"{c}/{k}": np.asarray(v) for c, p in params.items() for k, v in p.items()})
    heads.with_suffix(".part").replace(heads)
    receipts = {c: kernel_receipt(layer, p) for c, p in params.items()}
    base_kernel = receipts["original"]["action_out_proj/kernel"]
    for c in CONDITIONS:
        if (receipts[c]["action_out_proj/kernel"] != base_kernel) != (c in ("policy_only", "joint")):
            raise AssertionError(f"Effective kernel intervention failed: {c}")
    report = dict(schema=SCHEMA, contract=contract_hash, base=spec["base"], code_revision=args.pin, data_manifest=manifest_hash,
                  parameters=receipts, heads_sha256=file_sha(heads), metrics=metrics, epochs=logs,
                  resources={"wall_s": time.monotonic()-start, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                             "device": str(jax.devices()[0]), "device_kind": jax.devices()[0].device_kind,
                             "jax_memory_stats": jax.devices()[0].memory_stats()})
    # Validate the exported bundle before publishing the completion file.
    check_dir = out / ".verify"
    check_dir.mkdir()
    os.link(heads, check_dir / heads.name)
    json_write(check_dir / "fit.json", report)
    motor = LockedPolicy(stock, check_dir)
    item = raw_item(first)
    item.pop("actions")
    noise = np.random.default_rng(SEED).normal(size=(10,32)).astype(np.float32)
    parity = motor.parity(item, noise=noise)
    for c in CONDITIONS:
        actions = motor.infer(item, noise=noise, condition=c, history=first["history"])["actions"]
        if actions.shape != (10,7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid reloaded deployment output: {c}")
    report["reloaded_parity"] = parity
    (check_dir / "fit.json").unlink()
    (check_dir / heads.name).unlink()
    check_dir.rmdir()
    json_write(out / "fit.json", report)
    print(f"Ready {out/'fit.json'} {file_sha(out/'fit.json')}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pin", required=True)
    p.add_argument("--prior", type=Path)
    run(p.parse_args())
