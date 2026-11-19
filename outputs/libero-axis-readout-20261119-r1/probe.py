"""CPU-only, fixed ridge probes of the original balanced calibration readers.

Run with --episodes <collector>/episodes --fits <original>/cpu01/fit
--output <fresh-directory>. predictions.npz contains one array per split,
method, context and quantity, indexed by direction, convention, trial, axis.
The one-dimensional label arrays give the order of those indices.
"""

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

RUN = Path(__file__).resolve().parent
sys.path[:0] = [str(RUN.parent / "libero-axis-calibration-20261119-r1"),
                str(RUN.parents[1] / "src")]
import reader  # Original Reader, panel selection, convention and direction transforms.


NAMES, SIGNS = reader.NAMES, reader.SIGNS
SEEDS = (41, 42, 43)
WIDTH, RIDGE = 48, 1.0
ATOL, RTOL = 1e-6, 1e-5


def sign(values):
    return torch.where(values >= 0, 1, -1)


def contexts(raw, conventions, direction):
    return reader.relabel(raw * direction.repeat(2), conventions)


def activations(model, probes, query):
    """Return both post-final-norm states from the same causal Reader sequence."""
    conventions, trials = probes.shape[:2]
    history = model.support_token((probes.flatten(0, 1) - model.support_mean) / model.support_scale)
    current = model.query_token((query.repeat(conventions, 1) - model.query_mean) / model.query_scale)
    tokens = torch.cat((history, current[:, None]), dim=1).transpose(0, 1)
    valid = torch.ones(tokens.shape[:2], dtype=torch.bool)
    states = model.core.sequence(tokens, reset=torch.zeros_like(valid), valid=valid)
    return {token: state.reshape(conventions, trials, WIDTH)
            for token, state in zip(("support", "query"), states[-2:])}


def load_models(fitting, seed, folder):
    keys, raw, query, _ = fitting
    path = folder / f"seed-{seed}.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    expected = dict(seed=seed, step=reader.STEPS, balanced=True, core=reader.CORE,
                    fitting_conventions=NAMES[:3], fitting_keys=keys, source_batch=reader.SOURCE_BATCH,
                    empty_sources=reader.EMPTY_SOURCES, lr=reader.LR)
    for field, value in expected.items():
        if ckpt[field] != value:
            raise ValueError(f"{path}: incompatible checkpoint field {field}")

    torch.manual_seed(seed)
    support = reader.relabel(raw, SIGNS[:3])
    fit_inputs = torch.cat([support * direction.repeat(2) for direction in SIGNS])
    random = reader.Reader(fit_inputs, query).requires_grad_(False).eval()
    trained = copy.deepcopy(random)
    trained.load_state_dict(ckpt["model"], strict=True)
    for name, value in trained.state_dict().items():
        if not torch.equal(value, ckpt["model"][name]) or not value.isfinite().all():
            raise ValueError(f"{path}: invalid loaded state {name}")
    normalizers = {}
    for name in ("support_mean", "support_scale", "query_mean", "query_scale"):
        value = getattr(trained, name)
        if not torch.equal(value, getattr(random, name)):
            raise ValueError(f"{path}: reconstructed fitting normalizer differs: {name}")
        normalizers[name] = value.tolist()
    initial = dict(random.named_parameters())
    changed = [name for name, value in trained.named_parameters()
               if not name.startswith("head.") and not torch.equal(value, initial[name])]
    if not changed:
        raise ValueError(f"{path}: checkpoint encoder equals initialization")
    audit = dict(expected, path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                 strict_state_tensors_verified=len(ckpt["model"]), changed_encoder_parameters=changed,
                 shared_input_normalizers=normalizers, final_loss=ckpt["final_loss"])
    return {"trained": trained, "random": random}, audit


def fit_probes(model, fitting):
    keys, raw, query, _ = fitting
    # Build only the three fitting conventions. The outer index is direction.
    features = [activations(model, contexts(raw, SIGNS[:3], d), query) for d in SIGNS]
    labels = SIGNS[None, :3, None, :].expand(4, 3, len(keys), 2).reshape(-1, 2).double()
    probes, stats = {}, {}
    for token in ("support", "query"):
        x = torch.stack([entry[token] for entry in features]).reshape(-1, WIDTH).double()
        mean, std = x.mean(0), x.std(0, unbiased=False)
        scale = torch.where(std > 0, std, 1.0)
        z = (x - mean) / scale
        intercept = labels.mean(0)
        # Centering gives an unpenalized intercept. The Gram matrix is a sum.
        weights = torch.linalg.solve(z.T @ z + RIDGE * torch.eye(WIDTH, dtype=z.dtype),
                                     z.T @ (labels - intercept))
        fitted = z @ weights + intercept
        if not torch.isfinite(fitted).all():
            raise ValueError(f"Nonfinite {token} ridge fit")
        probes[token] = (mean, scale, weights, intercept)
        stats[token] = dict(rows=len(x), width=WIDTH, lambda_sum=RIDGE,
                            positive_labels=(labels > 0).sum(0).tolist(),
                            feature_mean=mean.tolist(), feature_std=std.tolist(), scale=scale.tolist(),
                            weights=weights.tolist(), intercept=intercept.tolist(),
                            fit_mse_by_axis=(fitted - labels).square().mean(0).tolist())
    return probes, stats


def accuracy(correct):
    rows = correct.reshape(-1, 2)
    return dict(rows=len(rows), both=rows.all(-1).double().mean().item(),
                x=rows[:, 0].double().mean().item(), y=rows[:, 1].double().mean().item())


def metrics(correct, keys):
    result = {}
    for group, subset in (("original_direction", correct[:1]), ("all_directions", correct)):
        result[group] = dict(
            overall=accuracy(subset),
            by_convention={name: accuracy(subset[:, i]) for i, name in enumerate(NAMES)},
            by_trial={key: accuracy(subset[:, :, j]) for j, key in enumerate(keys)},
        )
    return result


def evaluate(split, panel, models, probes, arrays):
    keys, raw, query, _ = panel
    nominal = query.sign().long()
    target = SIGNS[:, None] * nominal
    arrays[f"{split}/trial"] = np.asarray(keys)
    arrays[f"{split}/nominal_query"] = query.numpy()
    arrays[f"{split}/command_target"] = target.to(torch.int8).numpy()
    report = {}
    for condition, context in (("matched", SIGNS), ("opposite", -SIGNS)):
        collected = {}
        for direction in SIGNS:
            support = contexts(raw, context, direction)
            correlation = (support[..., :2] * support[..., 2:]).sum(2)
            collected.setdefault("sign_correlation", []).append(correlation)
            for method, model in models.items():
                states = activations(model, support, query)
                if method.startswith("trained-"):
                    logits = model.head(states["query"].reshape(-1, WIDTH)).reshape_as(target)
                    collected.setdefault(method + "-head", []).append(logits)
                for token, state in states.items():
                    mean, scale, weights, intercept = probes[method][token]
                    score = ((state.double() - mean) / scale) @ weights + intercept
                    collected.setdefault(f"{method}-{token}", []).append(score)

        for method, entries in collected.items():
            score = torch.stack(entries)
            if not score.isfinite().all():
                raise ValueError(f"Nonfinite score for {split}/{method}/{condition}")
            if method.endswith("-head"):
                command = sign(score)
                calibration = command * nominal
                quantity = "command_logits"
            else:
                calibration = sign(score)
                command = calibration * nominal
                quantity = "calibration_score"
            prefix = f"{split}/{method}/{condition}/"
            arrays[prefix + quantity] = score.numpy()
            arrays[prefix + "calibration_prediction"] = calibration.to(torch.int8).numpy()
            arrays[prefix + "command_prediction"] = command.to(torch.int8).numpy()
            report.setdefault(method, {})[condition] = metrics(command == target, keys)
    return report


def verify_original(folder, keys, arrays):
    """Compare by identity and require complete, unique reference coverage."""
    index = {key: j for j, key in enumerate(keys)}
    methods = {f"seed-{s}": f"trained-{s}-head" for s in SEEDS}
    seen, errors = set(), {}
    with (folder / "predictions.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            method, condition = row["method"], row["support"]
            if method not in methods or condition not in ("matched", "opposite"):
                continue
            identity = (method, condition, row["direction"], row["convention"], row["key"])
            d, c, j = NAMES.index(row["direction"]), NAMES.index(row["convention"]), index[row["key"]]
            prefix = f"development/{methods[method]}/{condition}/"
            logits = arrays[prefix + "command_logits"][d, c, j]
            prediction = arrays[prefix + "command_prediction"][d, c, j]
            target = arrays["development/command_target"][c, j]
            reference = np.asarray(row["logits"], dtype=np.float64)
            if (identity in seen or not np.array_equal(target, row["target"])
                    or not np.array_equal(prediction, row["prediction"]) or reference.shape != (2,)
                    or not np.allclose(logits, reference, atol=ATOL, rtol=RTOL)):
                raise ValueError(f"Original reader replay mismatch: {identity}")
            seen.add(identity)
            difference = float(np.abs(logits - reference).max())
            errors[condition] = max(errors.get(condition, 0.0), difference)
    expected = len(methods) * 2 * len(NAMES)**2 * len(keys)
    if len(seen) != expected:
        raise ValueError(f"Incomplete original reader replay: {len(seen)} of {expected}")
    return dict(rows=len(seen), signs_and_targets_exact=True, max_abs_logit_error=errors,
                atol=ATOL, rtol=RTOL, source=str(folder / "predictions.jsonl"))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("episodes", "fits", "output"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    folder = args.fits / "balanced"
    prior = json.loads((args.fits / "summary.json").read_text())
    fitting = reader.read(args.episodes, "fitting")
    if len(fitting[0]) != 16 or prior["fitting"] != fitting[3]:
        raise ValueError("Expected the original 16 fitting trials and audit")
    models, probes, checkpoints, fit_stats = {}, {}, {}, {}
    for seed in SEEDS:
        pair, checkpoints[seed] = load_models(fitting, seed, folder)
        for arm, model in pair.items():
            method = f"{arm}-{seed}"
            models[method] = model
            probes[method], fit_stats[method] = fit_probes(model, fitting)

    development = reader.read(args.episodes, "development")
    if prior["development"] != development[3] or set(fitting[0]).intersection(development[0]):
        raise ValueError("Expected original, disjoint eligible development trials")
    arrays = dict(direction=np.asarray(NAMES), convention=np.asarray(NAMES),
                  signs=SIGNS.numpy(), axis=np.asarray(["x", "y"]))
    panels = {"fitting": fitting, "development": development}
    results = {split: evaluate(split, panel, models, probes, arrays) for split, panel in panels.items()}
    replay = verify_original(folder, development[0], arrays)
    trial_cases = {}
    for split, (keys, _, _, _) in panels.items():
        rows = json.loads((args.episodes / split / "progress.json").read_text())
        trial_cases[split] = {r["trial"]: dict(support=r["support"]["case"],
                                               query=r["queries"]["default"]["case"])
                              for r in rows if r["trial"] in keys}
    summary = dict(
        split_audits={split: panel[3] for split, panel in panels.items()}, trial_cases=trial_cases,
        training=dict(split="fitting", keys=fitting[0], conventions=NAMES[:3], directions=NAMES,
                      context="matched", target="calibration signs", lambda_sum=RIDGE,
                      intercept="unpenalized", width=WIDTH, standardization="fit population std",
                      zero_std_scale=1, seed_selection="fixed", device="cpu"),
        checkpoints=checkpoints, ridge_fits=fit_stats, original_head_replay=replay,
        metrics=results, scoring="Both axes, X and Y compare corrected command signs to the fixed "
        "matched target. This also equals calibration-sign accuracy. Zero scores predict +1.",
        prediction_axes=["direction", "convention", "trial", "axis"],
        target_axes=["convention", "trial", "axis"],
    )
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output / "predictions.npz", **arrays)
    (args.output / "summary.json").write_text(json.dumps(summary, allow_nan=False) + "\n")
    print(f"Verified {replay['rows']} original head rows. Saved results in {args.output}.", flush=True)


if __name__ == "__main__":
    main()
