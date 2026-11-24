"""Fit a three-way selector to paired source-arm forks at a shared decision prefix.

The two regression targets are signed differences in native success relative to
continuing. The factorized correction head sees only motion; its cue head sees
only progress. Both generic heads see the concatenated vector. Each uses 16
fixed tanh random features and fits 16 slopes plus an unpenalized intercept
(34 trainable parameters in either selector). The common projection bank and
offset are seeded independently of the data and stored in the model.

The caller verifies that each outcome triplet shares its prefix. Case metadata
enforces the source-arm split and is never a predictor. Rows from other arms
are skipped before reading features or labels. A query needs only motion and progress.
"""

import numpy as np


INTERVENTIONS = ("correct", "cue")
SOURCE_ARMS = ("Panda", "UR5e")
CASE_FIELDS = {"arm", "suite", "task_id", "init_state_id", "seed"}
WIDTH = 16
PROJECTION_SEED = 20261124


def _vector(values, name):
    array = np.asarray(values)
    if array.ndim != 1 or not array.size or array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a nonempty one-dimensional numeric vector")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a non-finite value")
    return array


def _training_arrays(rows):
    observations, targets = [], []
    sizes = None
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {i} must be a dictionary")
        case = row.get("case")
        if not isinstance(case, dict) or not CASE_FIELDS.issubset(case):
            raise ValueError(f"row {i} is missing source case metadata")
        if case["arm"] not in SOURCE_ARMS:
            continue
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, dict) or set(outcomes) != {"continue", *INTERVENTIONS}:
            raise ValueError(f"row {i} needs all three paired outcomes")
        if not all(isinstance(value, (bool, np.bool_)) for value in outcomes.values()):
            raise ValueError(f"row {i} outcomes must be boolean success values")
        motion = _vector(row.get("motion"), "motion")
        progress = _vector(row.get("progress"), "progress")
        row_sizes = (motion.size, progress.size)
        if sizes is not None and row_sizes != sizes:
            raise ValueError(f"inconsistent motion/progress dimensions in row {i}")
        sizes = row_sizes
        observations.append(np.concatenate((motion, progress)))
        baseline = int(outcomes["continue"])
        targets.append([int(outcomes[key]) - baseline for key in INTERVENTIONS])
    if not observations:
        raise ValueError("fit needs at least one paired source row")
    return np.stack(observations), np.asarray(targets), sizes


def _basis(x, projection, offset, kind, action, motion_size):
    if kind == "factorized":
        columns = slice(0, motion_size) if action == "correct" else slice(motion_size, None)
        x, projection = x[..., columns], projection[columns]
    elif kind != "generic":
        raise ValueError("kind must be 'factorized' or 'generic'")
    return np.tanh((x @ projection) / np.sqrt(x.shape[-1]) + offset)


def fit(rows, kind):
    """Return a JSON-compatible selector trained on the provided source forks."""
    if kind not in ("factorized", "generic"):
        raise ValueError("kind must be 'factorized' or 'generic'")
    inputs, gains, sizes = _training_arrays(rows)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        center = inputs.mean(axis=0)
        scale = inputs.std(axis=0)
        # Training-constant columns carry no information about their effects.
        varying = np.any(inputs != inputs[0], axis=0)
        inverse_scale = np.divide(1.0, scale, where=varying, out=np.zeros_like(scale))
        x = (inputs - center) * inverse_scale
    rng = np.random.default_rng(PROJECTION_SEED)
    projection = rng.standard_normal((x.shape[1], WIDTH))
    offset = rng.standard_normal(WIDTH)
    heads = {}
    for column, action in enumerate(INTERVENTIONS):
        basis = _basis(x, projection, offset, kind, action, sizes[0])
        mean_basis = basis.mean(axis=0)
        centered = basis - mean_basis
        target = gains[:, column]
        mean_target = target.mean()
        # Minimize sum of squared errors + ||slopes||². Centering frees the intercept.
        slopes = np.linalg.solve(
            centered.T @ centered + np.eye(WIDTH),
            centered.T @ (target - mean_target),
        )
        heads[action] = {
            "slopes": slopes.tolist(),
            "intercept": float(mean_target - mean_basis @ slopes),
        }
    return {
        "kind": kind, "motion_size": sizes[0], "progress_size": sizes[1],
        "center": center.tolist(), "inverse_scale": inverse_scale.tolist(),
        "projection": projection.tolist(), "offset": offset.tolist(), "heads": heads,
        "trainable_parameters": 2 * (WIDTH + 1),
    }


def choose(model, motion, progress):
    """Choose a unique, strictly positive gain winner. Every top-score tie continues."""
    motion = _vector(motion, "motion")
    progress = _vector(progress, "progress")
    if (motion.size, progress.size) != (model["motion_size"], model["progress_size"]):
        raise ValueError("motion/progress dimensions differ from training")
    x = (np.concatenate((motion, progress)) - model["center"]) * model["inverse_scale"]
    projection = np.asarray(model["projection"])
    scores = []
    for action in INTERVENTIONS:
        basis = _basis(x, projection, model["offset"], model["kind"], action, motion.size)
        head = model["heads"][action]
        scores.append(float(basis @ head["slopes"] + head["intercept"]))
    if not np.isfinite(scores).all():
        raise ValueError("non-finite predicted gain")
    correct, cue = scores
    if correct > max(0, cue):
        return "correct"
    if cue > max(0, correct):
        return "cue"
    return "continue"
