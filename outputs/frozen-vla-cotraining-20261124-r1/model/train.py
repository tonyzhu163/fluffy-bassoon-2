"""Fit and score three controlled heads with the differentiable native Pi0.compute_loss."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import jax
import jax.numpy as jnp
import numpy as np

import source_data
from ur5e_motor import to_openpi_observation
from cotrain_policy import SCHEMA, LockedPolicy, base_identity, file_sha, kernel_receipt, load_stock, transformed
from flow_head import CONDITIONS
from direct_fit import SEED, BATCH, EPOCHS, LR, DRAWS, direct_scores, native_functions, train_heads
import native_check


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


def samples_for_split(stock, ds, split, out):
    # Only deterministic public input transforms are reused. Every model call runs
    # the native stochastic preprocessing, noise/time sampling and denoising tower.
    inputs = []
    identities = []
    for row in ds:
        item = raw_item(row)
        for name in ("observation/image", "observation/wrist_image"):
            if item[name].shape != (224,224,3) or item[name].dtype != np.uint8:
                raise ValueError(f"Invalid public camera {name}")
        obs, actions = transformed(stock, item)
        if (actions.shape != (1,10,32) or obs.state.shape != (1,32)
                or obs.tokenized_prompt.shape != (1,200) or obs.tokenized_prompt_mask.shape != (1,200)
                or not np.isfinite(np.asarray(actions)).all()):
            raise ValueError(f"Invalid stock transform shapes for {row['trajectory_id']}/{row['t']}")
        for img in obs.images.values():
            if img.shape != (1,224,224,3) or img.dtype != jnp.float32:
                raise ValueError("Stock camera transform changed")
        inputs.append((obs, actions, jnp.asarray(row["history"])[None], str(row["trajectory_id"])))
        identities.append([str(row["trajectory_id"]), int(row["t"])])
    order = schedule(ds, split)
    json_write(out / f"{split}-inputs.json", dict(identities=identities, draws=order,
               native_train=split == "train", execution_batch=1, rows=len(order)))
    return [(jax.random.fold_in(jax.random.key(SEED), fold), *inputs[i]) for i, _, fold in order]


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
    spec = {"schema": SCHEMA, "engine": "direct-differentiable-Pi0.compute_loss", "data_manifest_sha256": manifest_hash,
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
    data = {s: samples_for_split(stock, ds[s], s, out) for s in DRAWS}
    engine = native_functions(model)
    params, logs = train_heads(model, engine, data["train"], out, args.prior, contract_hash)
    layer = model.action_out_proj
    metrics = {s: direct_scores(engine, params, samples, s) for s, samples in data.items()}
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
