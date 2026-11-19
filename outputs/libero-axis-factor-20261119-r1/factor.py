"""Fit support-only calibration MLPs and score the corrected first motor command.

For simulator inference, import predict_support and pass shared-41.npz plus
float support [B,2,4]. Rows are X- then Y-dominant probes. Columns are issued
X, issued Y, measured displacement X, measured displacement Y. Multiply the
proposal's XY coordinates by the returned [B,2] signs. Inference needs NumPy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


PRIOR = Path(__file__).resolve().parents[1] / "libero-axis-calibration-20261119-r1"
NAMES = ("nom", "x", "y", "xy")
SIGNS = np.asarray(((1, 1), (-1, 1), (1, -1), (-1, -1)), dtype=np.float32)
SEEDS = (41, 42, 43)
STEPS, SOURCE_BATCH, EMPTY_SOURCES = 400, 40, 4
WIDTH, LR, WEIGHT_DECAY = 32, 0.001, 0.01
CONDITIONS = ("matched", "cleared", "opposite")
FORMAT = "axis-shared-relu-v1"


def features(support, arm):
    if arm == "joint":
        return support.reshape(*support.shape[:-2], 8)
    if arm != "shared":
        raise ValueError(f"Unknown model {arm!r}")
    x_probe, y_probe = support[..., 0, :], support[..., 1, :]
    x = np.stack((x_probe[..., 0], x_probe[..., 2], y_probe[..., 0], y_probe[..., 2]), axis=-1)
    y = np.stack((y_probe[..., 1], y_probe[..., 3], x_probe[..., 1], x_probe[..., 3]), axis=-1)
    return np.stack((x, y), axis=-2)


def sign(values):
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite calibration scores or motor commands")
    return np.where(values >= 0, 1, -1).astype(np.int8)


def predict_support(checkpoint: Path, support: np.ndarray) -> np.ndarray:
    """Return signed int8 calibration [B,2] from the portable shared-41.npz."""
    support = np.asarray(support, dtype=np.float32)
    if support.ndim != 3 or support.shape[1:] != (2, 4) or not np.isfinite(support).all():
        raise ValueError("Expected finite support [B,2,4]")
    with np.load(checkpoint, allow_pickle=False) as state:
        if state["format"].item() != FORMAT:
            raise ValueError(f"Unsupported model format in {checkpoint}")
        value = (features(support, "shared") - state["mean"]) / state["scale"]
        for layer in (0, 1, 2):
            # Torch Linear stores weights [out, in]. Hidden layers use ReLU.
            value = value @ state[f"weight_{layer}"].T + state[f"bias_{layer}"]
            if layer < 2:
                value = np.maximum(value, 0)
    return sign(value.squeeze(-1))


def support_grid(raw, conventions):
    """Return [direction, convention, trial, probe, field] from nominal tapes."""
    reflected = raw[None] * np.tile(SIGNS, (1, 2))[:, None, None, :]
    gains = np.concatenate((conventions, np.ones_like(conventions)), axis=-1)
    return reflected[:, None] * gains[None, :, None, None, :]


def fit_inputs(raw, arm):
    import torch

    values = features(support_grid(raw, SIGNS[:3]), arm)
    rows = values.reshape(-1, values.shape[-1]).astype(np.float64)
    mean = rows.mean(0).astype(np.float32)
    std = rows.std(0).astype(np.float32)
    scale = np.where(std > 1e-8, std, np.float32(1.0))
    inputs = torch.from_numpy((values - mean) / scale)
    # Empty history is zero in raw support coordinates, before standardization.
    empty = torch.from_numpy(-mean / scale)
    return inputs, empty, mean, scale


def fit(panel, arm, seed, prepared, output):
    import torch
    from torch import nn

    inputs, empty, mean, scale = prepared
    torch.manual_seed(seed)
    sampler = torch.Generator().manual_seed(seed)
    model = nn.Sequential(nn.Linear(len(mean), WIDTH), nn.ReLU(), nn.Linear(WIDTH, WIDTH),
                          nn.ReLU(), nn.Linear(WIDTH, 2 if arm == "joint" else 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    target = torch.from_numpy((SIGNS[:3] > 0).astype(np.float32)).expand(SOURCE_BATCH, -1, -1)
    lane = torch.arange(SOURCE_BATCH)
    draw_hash = hashlib.sha256()
    for step in range(1, STEPS + 1):
        ids = torch.randint(len(panel["keys"]), (SOURCE_BATCH,), generator=sampler)
        draw_hash.update(ids.numpy().tobytes())
        direction = (lane + step) % len(NAMES)
        # Each source lane sees all three conventions at the same reflection.
        # Each reflection has 10 lanes, one of them cleared, at every update.
        batch = inputs[direction, :, ids]
        batch[:EMPTY_SOURCES] = empty
        logits = model(batch)
        if arm == "shared":
            logits = logits.squeeze(-1)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"{arm} seed {seed} update {step}/{STEPS} loss {loss.item():.6f}", flush=True)

    details = dict(arm=arm, seed=seed, step=step, width=WIDTH, lr=LR, weight_decay=WEIGHT_DECAY,
                   source_batch=SOURCE_BATCH, empty_sources=EMPTY_SOURCES, fitting_keys=panel["keys"],
                   fitting_conventions=NAMES[:3], directions=NAMES, final_loss=loss.item(),
                   parameters=sum(p.numel() for p in model.parameters()), draw_sha256=draw_hash.hexdigest())
    torch.save(dict(**details, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    torch_rng=torch.get_rng_state(), sampler_rng=sampler.get_state(),
                    mean=torch.from_numpy(mean), scale=torch.from_numpy(scale)), output / f"{arm}-{seed}.pt")
    if (arm, seed) == ("shared", 41):
        weights = {}
        for i, layer in enumerate((model[0], model[2], model[4])):
            weights[f"weight_{i}"] = layer.weight.detach().numpy()
            weights[f"bias_{i}"] = layer.bias.detach().numpy()
        np.savez(output / "shared-41.npz", format=FORMAT, mean=mean, scale=scale, **weights)
    return model.eval(), details


def counts(correct):
    rows = correct.reshape(-1, 2)
    return dict(n=len(rows), joint=int(rows.all(-1).sum()), x=int(rows[:, 0].sum()), y=int(rows[:, 1].sum()))


def evaluate(panel, models, norms, arrays, output):
    import torch

    raw, proposal = panel["support"], panel["query"]
    matched = support_grid(raw, SIGNS)
    labels = np.broadcast_to(SIGNS[None, :, None, :], matched.shape[:3] + (2,)).astype(np.int8)
    target = proposal * labels
    arrays.update(calibration_target=labels, command_target=target, command_target_sign=sign(target))
    pooled, detail = {}, {}
    for condition, support in zip(CONDITIONS, (matched, np.zeros_like(matched), support_grid(raw, -SIGNS))):
        scores = dict(true_sign_oracle=labels, sign_correlation=(support[..., :2] * support[..., 2:]).sum(-2))
        for arm in ("joint", "shared"):
            mean, scale = norms[arm]
            inputs = torch.from_numpy((features(support, arm) - mean) / scale)
            for seed in SEEDS:
                method = f"{arm}-{seed}"
                logits = models[method](inputs).numpy().reshape(labels.shape)
                scores[method] = logits
                if method == "shared-41":
                    flat = support.reshape(-1, 2, 4)
                    portable = predict_support(output / "shared-41.npz", flat)
                    if not np.array_equal(portable.reshape(labels.shape), sign(logits)):
                        raise ValueError(f"Portable signs differ from Torch under {condition} support")

        for method, score in scores.items():
            estimate = sign(score)
            command = proposal * estimate
            correct = sign(command) == sign(target)
            prefix = f"{method}/{condition}/"
            arrays[prefix + "score"] = score
            arrays[prefix + "calibration_prediction"] = estimate
            arrays[prefix + "command_prediction"] = command
            pooled.setdefault(method, {})[condition] = {
                name: counts(correct[:, c]) for c, name in enumerate(NAMES)}
            detail.setdefault(method, {})[condition] = {
                direction: {name: counts(correct[d, c]) for c, name in enumerate(NAMES)}
                for d, direction in enumerate(NAMES)}
    return pooled, detail


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(PRIOR))
    from panel import load_panel

    prior = json.loads((PRIOR / "cpu01-readout.json").read_text())["data"]
    panels = {}
    for split in ("fitting", "development"):
        panel = load_panel(args.episodes, split)
        if panel["audit"] != prior[split]:
            raise ValueError(f"{split} panel differs from the original calibration run")
        panels[split] = panel
    fitting = panels["fitting"]
    args.output.mkdir(parents=True, exist_ok=False)
    models, fits, norms = {}, {}, {}
    for arm in ("joint", "shared"):
        prepared = fit_inputs(fitting["support"], arm)
        norms[arm] = prepared[2:]
        for seed in SEEDS:
            method = f"{arm}-{seed}"
            models[method], fits[method] = fit(fitting, arm, seed, prepared, args.output)
    for seed in SEEDS:
        if fits[f"joint-{seed}"]["draw_sha256"] != fits[f"shared-{seed}"]["draw_sha256"]:
            raise ValueError(f"Unpaired source batches for seed {seed}")

    arrays = dict(convention=np.asarray(NAMES), direction=np.asarray(NAMES), signs=SIGNS,
                  axis=np.asarray(["x", "y"]))
    for split, panel in panels.items():
        arrays[f"{split}/trial"] = np.asarray(panel["keys"])
        arrays[f"{split}/nominal_support"] = panel["support"]
        arrays[f"{split}/nominal_proposal"] = panel["query"]
    with torch.inference_mode():
        pooled, detail = evaluate(panels["development"], models, norms, arrays, args.output)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    (args.output / "counts.json").write_text(json.dumps(detail, allow_nan=False) + "\n")
    summary = dict(fitting=fitting["audit"], development=panels["development"]["audit"],
                   episodes=str(args.episodes.resolve()), seeds=SEEDS, steps=STEPS, optimizer="AdamW",
                   conventions=dict(zip(NAMES, SIGNS.astype(int).tolist())), held_out="xy",
                   fit_target="Calibration signs, per-axis BCE with logits; zero logits give +1.",
                   normalizers={arm: dict(mean=m.tolist(), scale=s.tolist()) for arm, (m, s) in norms.items()},
                   normalizer_population="All matched fitting sources x three conventions x four directions; "
                                         "population std, pooled axes for shared. Scale 1 for std <= 1e-8.",
                   fits=fits, counts=pooled, counts_by_direction="counts.json",
                   prediction_axes=["direction", "convention", "development/trial", "axis"],
                   predictions="predictions.npz", portable="shared-41.npz",
                   portable_verified="Signs equal Torch on every development support condition.",
                   scoring="Compare signs of nominal proposal times estimated signs with the target "
                           "command signs. Counts pool four directions per source. Zero correlation gives +1.")
    (args.output / "summary.json").write_text(json.dumps(summary, allow_nan=False) + "\n")
    print(f"Saved six fits, development predictions and counts in {args.output}", flush=True)


if __name__ == "__main__":
    main()
