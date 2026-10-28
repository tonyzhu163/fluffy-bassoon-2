"""Linear probes of private trial settings from exported per-attempt core state.

Reads one condition archive written by `evaluate --export-states` and fits a
closed-form ridge classifier per state entry and attempt. The first half of the
trials fits the probe, the second half scores it. No model is constructed, so
this runs anywhere.
"""

import argparse
from pathlib import Path

import numpy as np

from fluffy_bassoon.evaluation_stages import sha256, write_json


def ridge_probe(train_x, train_y, test_x, test_y, *, ridge=1.):
    """Standardize on the fit split, regress one-hot labels, score argmax accuracy."""
    classes = np.unique(train_y)
    mean, scale = train_x.mean(0), train_x.std(0)
    scale[scale < 1e-8] = 1.
    fit, held = (train_x - mean) / scale, (test_x - mean) / scale
    targets = (train_y[:, None] == classes[None]).astype(np.float64)
    # Dual form: features usually outnumber fitting trials.
    weights = fit.T @ np.linalg.solve(fit @ fit.T + ridge * np.eye(len(fit)), targets - targets.mean(0))
    predicted = classes[(held @ weights + targets.mean(0)).argmax(1)]
    majority = classes[(train_y[:, None] == classes[None]).sum(0).argmax()]
    return dict(accuracy=float((predicted == test_y).mean()), majority_baseline=float((test_y == majority).mean()),
                fit_classes=len(classes))


def state_features(archive):
    features = {key[len("state_"):]: archive[key] for key in archive if key.startswith("state_")}
    if "alpha" in features and "beta" in features:
        # The CMM readout normalizes alpha by beta before its RMS norm.
        features["alpha_over_beta"] = features["alpha"] / np.maximum(features["beta"], np.finfo(np.float32).tiny)
    return features


def probe_archive(archive, *, ridge=1.):
    features = state_features(archive)
    labels = {key[len("hidden_"):]: archive[key] for key in archive if key.startswith("hidden_")}
    if not features or not labels:
        raise ValueError("Archive lacks exported states or probe targets; evaluate with --export-states")
    trials, attempts = next(iter(features.values())).shape[:2]
    if trials < 4:
        raise ValueError("Require at least four trials to fit and score a probe")
    half = trials // 2
    results = {}
    for label, values in labels.items():
        rows = {}
        for entry, state in features.items():
            if not state[0, 0].size:
                continue  # An empty store, such as zero long-term tokens.
            flat = state.reshape(trials, attempts, -1).astype(np.float64)
            curve = []
            for attempt in range(attempts):
                target = values[:, attempt] if values.ndim == 2 else values
                curve.append(ridge_probe(flat[:half, attempt], target[:half], flat[half:, attempt],
                                         target[half:], ridge=ridge))
            rows[entry] = dict(accuracy_by_attempt=[c["accuracy"] for c in curve],
                               majority_baseline_by_attempt=[c["majority_baseline"] for c in curve])
        results[label] = rows
    return dict(trials=int(trials), fit_trials=int(half), attempts=int(attempts), ridge=ridge, probes=results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True, help="Condition .npz from evaluate --export-states")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ridge", type=float, default=1.)
    args = parser.parse_args(argv)
    if not args.ridge > 0:
        raise ValueError("Ridge strength must be positive")
    with np.load(args.archive, allow_pickle=False) as archive:
        result = probe_archive(dict(archive), ridge=args.ridge)
    result["archive"] = dict(path=str(args.archive), sha256=sha256(args.archive))
    write_json(args.output, result)
    return result


if __name__ == "__main__":
    main()
