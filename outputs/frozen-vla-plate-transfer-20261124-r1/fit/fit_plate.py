"""Two source-only native Pi0.compute_loss fits sharing a frozen stock tree and executable."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import resource
import time

import flax.nnx as nnx
import flax.serialization as serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from plate_policy import FITS, SCHEMA, PlatePolicy, head_digest, read_json, state_digest, stock_digest, write_json
from cotrain_policy import array_sha, base_identity, file_sha, load_stock
from direct_fit import BATCH, DRAWS, EPOCHS, LR, SEED
from flow_head import initial_params, install, live_loss, token_loss
from source_data import SourceDataset
from train import raw_item, samples_for_split

BODIES = ("Panda", "Sawyer")


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def source(root, entry, split, layouts, counts):
    directory = root / entry["root"]
    manifest = directory / "manifest.json"
    require(file_sha(manifest) == entry["manifest_sha256"], f"{manifest}: file digest")
    part = read_json(manifest)["splits"][split]
    require(file_sha(directory / part["file"]) == part["sha256"], f"{directory}: {split} HDF5 digest")
    ds = SourceDataset(directory, split)
    rows, counts_by_layout = [], defaultdict(Counter)
    for traj in part["trajectories"]:
        seed, body = traj["layout_seed"], traj["body"]
        require(seed in layouts and layouts[seed] == traj["scene_hash"], f"{traj['id']}: physical layout")
        require(body in counts and traj["prompt"].strip().lower().rstrip(".") == "push the plate to the front of the stove",
                f"{traj['id']}: arm or task")
        for t, phase in zip(traj["t"], traj["phase_slots"], strict=True):
            rows.append(dict(layout=seed, phase=phase, body=body, trajectory_id=traj["id"], t=t, tape=traj["sequence_sha256"]))
            counts_by_layout[seed][body] += 1
    require(len(rows) == len(ds) == part["examples"] and set(counts_by_layout) == set(layouts)
            and all(dict(v) == counts for v in counts_by_layout.values()), "source composition")
    require(len({(r["trajectory_id"], r["t"]) for r in rows}) == len(rows), "duplicate source window")
    return ds, rows


def read_data(root, expected):
    require(file_sha(root / "seal.json") == expected, "source seal SHA-256")
    seal = read_json(root / "seal.json")
    layouts = {s: {r["layout_seed"]: r["scene_hash"] for r in seal["layouts"] if r["split"] == s}
               for s in ("train", "heldout")}
    require(len(layouts["train"]) == 16 and len(layouts["heldout"]) == 4
            and len({r["layout_seed"] for r in seal["layouts"]}) == 20
            and len({r["scene_hash"] for r in seal["layouts"]}) == 20
            and len({r["movable_xy_hash"] for r in seal["layouts"]}) == 20, "20 disjoint physical layouts")
    datasets, meta = {}, {}
    for c, counts in (("panda", {"Panda": 12}), ("mixed", {"Panda": 6, "Sawyer": 6})):
        datasets[c], meta[c] = source(root, seal["variants"][c], "train", layouts["train"], counts)
    for body in BODIES:
        datasets[body], meta[body] = source(root, seal["diagnostics"][body], "heldout", layouts["heldout"], {body: 8})
    train_tapes = {r["tape"] for c in FITS for r in meta[c]}
    require(train_tapes.isdisjoint(r["tape"] for b in BODIES for r in meta[b]), "held-out tape overlaps training")
    for left, right in zip(meta["panda"], meta["mixed"], strict=True):
        require((left["layout"], left["phase"]) == (right["layout"], right["phase"]), "unpaired layout or phase")
        if right["body"] == "Panda":
            require(left == right, "shared Panda window identity differs")
    return datasets, meta


def engine(model):
    graph, frozen = nnx.split(model)

    def direct(p, state, key, obs, action, history):
        return live_loss(nnx.merge(graph, state), key, obs, action, p, history, train=True).mean()

    @jax.jit
    def accumulate(state, key, obs, action, history, params, grads, total):
        loss, grad = jax.value_and_grad(direct)(params, state, key, obs, action, history)
        return jax.tree.map(jnp.add, grads, grad), total + loss

    @jax.jit
    def score(state, key, obs, action, history, params):
        _, noise_key, _ = jax.random.split(key, 3)
        target = jax.random.normal(noise_key, action.shape) - action
        result = {}
        for c, p in params.items():
            m = nnx.merge(graph, state)
            layer = install(m.action_out_proj, p, history)
            velocity = []

            def record(hidden):
                v = layer(hidden)
                velocity.append(v)
                return v

            m.action_out_proj = record
            loss = m.compute_loss(key, obs, action, train=False)
            issued = token_loss(velocity[0][:, :5, :7], target[:, :5, :7]).mean()
            result[c] = jnp.stack([loss.mean(), issued])
        return result

    return frozen, accumulate, score


def optimize(native, params, samples, out, contract, prior=None):
    state, accumulate, _ = native
    tx = optax.adam(LR)
    states = {c: tx.init(p) for c, p in params.items()}
    log = []
    if prior:
        previous = serialization.msgpack_restore((prior / "optimizer.msgpack").read_bytes())
        require(previous["contract"] == contract, "checkpoint contract")
        params, states = serialization.from_state_dict((params, states), previous["optimizer"])
        log = json.loads(previous["progress"])
        require(len(log) == previous["step"], "checkpoint cursor")
        (out / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
        (out / "optimizer.msgpack").write_bytes((prior / "optimizer.msgpack").read_bytes())

    @jax.jit
    def update(p, moment, grad, count):
        change, moment = tx.update(jax.tree.map(lambda g: g / count, grad), moment, p)
        return optax.apply_updates(p, change), moment

    n = len(samples["panda"])
    require(n == len(samples["mixed"]), "paired draw counts")
    zeros = jax.tree.map(jnp.zeros_like, params)
    for epoch in range(EPOCHS):
        perm = np.random.default_rng(np.random.SeedSequence([SEED, epoch, 193])).permutation(n)
        order_sha = hashlib.sha256(perm.astype("<i8").tobytes()).hexdigest()
        for batch, off in enumerate(range(0, n, BATCH)):
            step = epoch * ((n + BATCH - 1) // BATCH) + batch + 1
            if step <= len(log):
                continue
            start = time.monotonic()
            ids = perm[off:off+BATCH]
            losses = {}
            for c in FITS:
                grad, loss = zeros[c], jnp.float32(0)
                for i in ids:
                    key, obs, action, history, _ = samples[c][i]
                    grad, loss = accumulate(state, key, obs, action, history, params[c], grad, loss)
                params[c], states[c] = update(params[c], states[c], grad, len(ids))
                losses[c] = float(loss) / len(ids)
            jax.block_until_ready((params, states))
            require(np.isfinite(list(losses.values())).all(), "nonfinite direct loss")
            record = dict(step=step, epoch=epoch+1, batch=batch+1, rows=len(ids), order_sha256=order_sha,
                          seconds=time.monotonic()-start, train_mse=losses)
            log.append(record)
            with (out / "progress.jsonl").open("a") as f:
                print(json.dumps(record), file=f)
            print(json.dumps(record), flush=True)
            payload = dict(contract=contract, step=step, progress=json.dumps(log),
                           optimizer=serialization.to_state_dict((params, states)))
            tmp = out / "optimizer.part"
            tmp.write_bytes(serialization.msgpack_serialize(jax.device_get(payload)))
            tmp.replace(out / "optimizer.msgpack")
    return params, log


def evaluate(native, params, samples, rows):
    state, _, score = native
    groups = defaultdict(lambda: defaultdict(list))
    for i, (key, obs, action, history, _) in enumerate(samples):
        meta = rows[i // DRAWS["heldout"]]
        values = jax.device_get(score(state, key, obs, action, history, {"stock": {}, **params}))
        for c, val in values.items():
            require(np.isfinite(val).all(), "nonfinite source evaluation")
            groups[meta["body"], c][meta["layout"]].append(val)
    result = {}
    for (body, label), layouts in groups.items():
        per_layout = {i: np.mean(arr, axis=0, dtype=np.float64).tolist() for i, arr in layouts.items()}
        mean = np.mean(list(per_layout.values()), axis=0)
        result.setdefault(body, {})[label] = dict(native_mse_32=float(mean[0]), issued_mse_7=float(mean[1]),
                                                  per_layout_native_and_issued=per_layout)
    return result


def source_sample(motor, row, noise):
    item = raw_item(row)
    item.pop("actions")
    control = motor.parity(item, noise=noise)
    control.update(trajectory_id=row["trajectory_id"], t=row["t"], nonempty_history={})
    for c in ("stock", *FITS):
        result, route = motor.infer_with_receipt(item, noise=noise, condition=c, history=row["history"])
        a = result["actions"]
        require(a.shape == (10, 7) and np.isfinite(a).all(), f"{c} native output")
        control["nonempty_history"][c] = dict(actions_sha256=array_sha(a), route=route)
    return control


def run(args):
    require(jax.default_backend() == "gpu" and len(jax.devices()) == 1, "one allocated GPU")
    start = time.monotonic()
    out = args.out
    datasets, meta = read_data(args.data, args.seal)
    held_rows = [r for b in BODIES for r in meta[b]]
    config = dict(schema=SCHEMA, data_seal_sha256=args.seal, code_revision=args.pin, base=base_identity(verify_files=False),
                  seed=SEED, draws=DRAWS, batch=BATCH, epochs=EPOCHS, optimizer="Adam", lr=LR,
                  engine="direct-differentiable-Pi0.compute_loss", trainable_shapes={"history_in":[52,16],"history_out":[16,320]})
    contract = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    write_json(out / "config.json", config)
    stock = load_stock()
    frozen_sha = stock_digest(stock)
    model = stock._model
    params = {c: initial_params(model.action_out_proj.in_features, "adapter_only") for c in FITS}
    init_sha = head_digest(params["panda"])
    require(init_sha == head_digest(params["mixed"]), "paired initialization")
    valid = {b: next(row for row in datasets[b] if np.any(row["history"])) for b in BODIES}
    noise = np.random.default_rng(SEED).normal(size=(10,32)).astype(np.float32)
    initial_motor = PlatePolicy(stock, params)
    before = {b: source_sample(initial_motor, row, noise) for b, row in valid.items()}
    write_json(out / "native-before.json", before)
    del initial_motor

    train = {}
    for c in FITS:
        directory = out / c
        directory.mkdir()
        write_json(directory / "windows.json", meta[c])
        train[c] = samples_for_split(stock, datasets[c], "train", directory)
    write_json(out / "heldout-windows.json", held_rows)
    held = samples_for_split(stock, [row for b in BODIES for row in datasets[b]], "heldout", out)
    native = engine(model)
    pre = evaluate(native, params, held, held_rows)
    write_json(out / "source-before.json", pre)
    params, progress = optimize(native, params, train, out, contract, args.prior)
    after = evaluate(native, params, held, held_rows)
    write_json(out / "source-after.json", after)
    family = {**config, "resident_stock": frozen_sha, "initial_parameters": init_sha, "updates_per_fit": len(progress),
              "exports": {}, "source_metrics": {name: file_sha(out / f"source-{name}.json") for name in ("before", "after")}}
    for c in FITS:
        directory = out / c
        head = directory / "head.npz"
        with head.open("xb") as f:
            np.savez(f, **jax.device_get(params[c]))
        digest = head_digest(params[c])
        require(all(digest[k] != init_sha[k] for k in digest), f"{c} adapter failed to train both weights")
        report = dict(schema=SCHEMA, condition=c, base=config["base"], resident_stock=frozen_sha, code_revision=args.pin,
                      data_seal_sha256=args.seal, head_sha256=file_sha(head), parameters=digest, contract=contract,
                      windows_sha256=file_sha(directory / "windows.json"), draws_sha256=file_sha(directory / "train-inputs.json"),
                      heldout={b: after[b][c] for b in BODIES}, updates=len(progress))
        write_json(directory / "fit.json", report)
        family["exports"][c] = file_sha(directory / "fit.json")
    motor = PlatePolicy.restore(stock, out, family)
    post = {b: source_sample(motor, row, noise) for b, row in valid.items()}
    write_json(out / "native-after.json", post)
    deployed_stock = state_digest(motor.state.filter(nnx.Param))
    require(stock_digest(stock) == frozen_sha == deployed_stock, "published or deployed stock changed during fit")
    family["deployed_stock"] = deployed_stock
    for b in BODIES:
        require(before[b]["stock_sha256"] == post[b]["stock_sha256"] and pre[b]["stock"] == after[b]["stock"],
                f"{b}: stock outputs changed")
    family.update(native_before_sha256=file_sha(out / "native-before.json"), native_after_sha256=file_sha(out / "native-after.json"),
                  optimizer_sha256=file_sha(out / "optimizer.msgpack"), progress_sha256=file_sha(out / "progress.jsonl"),
                  heldout_draws_sha256=file_sha(out / "heldout-inputs.json"), heldout_windows_sha256=file_sha(out / "heldout-windows.json"),
                  resources=dict(device=jax.devices()[0].device_kind, seconds=time.monotonic()-start,
                                 max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    write_json(out / "fit.json", family)
    print("Sealed", out / "fit.json", file_sha(out / "fit.json"), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seal", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--prior", type=Path)
    run(parser.parse_args())
