"""Small frozen NumPy readers for the public Goal 2 command and wrist/jaw tape."""

import json
from operator import index
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


SEEDS = (41, 42, 43)
CURRENT_SEED = 41
PROBE = 10
CLOSE = 12
STEPS = 250
RATE = 0.025
L2 = 0.01
ARCHIVE = "readers.npz"
FEATURES = (
    "command_weighted_x_step_first_five",
    "width_before_first_close",
    "width_after_first_close",
    "width_after_first_five_consecutive_closes",
    "width_after_first_twelve_consecutive_closes",
    "max_single_step_widening_while_closing",
    "max_widening_from_running_minimum_while_closing",
    "max_width_after_eight_or_more_consecutive_closes",
    "max_wrist_lift_from_start_of_a_closing_streak",
)
SCALE_FLOOR = np.array([0.00025] + [0.001] * 8)  # metres


def _tape(commands, states):
    c, s = (np.asarray(a, dtype=np.float64) for a in (commands, states))
    if c.ndim != 2 or c.shape[1] != 7 or s.ndim != 2 or s.shape[1] != 9:
        raise ValueError("Expected public commands (N, 7) and states (N+1, 9)")
    if len(s) != len(c) + 1 and (len(c) or len(s)):
        raise ValueError("Expected the state before and after each completed control")
    if not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError("Non-finite public tape")
    return c, s


def _features(c, s):
    """Each feature uses only completed controls and their observed responses."""
    values = np.full(len(FEATURES), np.nan)
    k = min(5, len(c))
    pulse = c[:k, 0]
    if np.abs(pulse).sum() > 0:
        values[0] = pulse @ np.diff(s[:k + 1, 0]) / np.abs(pulse).sum()

    closing = c[PROBE:, 6] > 0
    edges = np.diff(np.r_[0, closing, 0])
    starts = PROBE + np.flatnonzero(edges == 1)
    stops = PROBE + np.flatnonzero(edges == -1)
    width = np.abs(s[:, 7] - s[:, 8])
    for start, stop in zip(starts, stops):
        # Command[start] acts on state[start]. State[stop] is the last response.
        w = width[start:stop + 1]
        if start == starts[0]:
            values[1:3] = w[:2]
        if len(w) > 5 and np.isnan(values[3]):
            values[3] = w[5]
        if len(w) > CLOSE and np.isnan(values[4]):
            values[4] = w[CLOSE]
        values[5] = np.fmax(values[5], max(0.0, np.diff(w).max()))
        values[6] = np.fmax(values[6], (w - np.minimum.accumulate(w)).max())
        if len(w) > 8:
            values[7] = np.fmax(values[7], w[8:].max())
        lift = s[start:stop + 1, 2].max() - s[start, 2]
        values[8] = np.fmax(values[8], lift)
    return values


def _normalizer(x, floor):
    observed = np.isfinite(x)
    count = observed.sum(axis=0)
    mean = np.nansum(x, axis=0) / np.maximum(count, 1)
    variance = np.nansum((x - mean) ** 2, axis=0) / np.maximum(count, 1)
    return np.stack((mean, np.maximum(np.sqrt(variance), floor), count))


def _normalize(x, norm):
    mean, scale, count = norm
    observed = np.isfinite(x) & (count > 0)
    return np.where(observed, np.clip((x - mean) / scale, -6, 6), 0.0)


def _fit_net(x, y, hidden, seed):
    """Full-batch Adam on summed binary cross-entropies per tape plus L2."""
    rng = np.random.default_rng(seed)
    n, inputs = x.shape
    outputs = y.shape[1]
    w = rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden))
    b = np.zeros(hidden)
    v = rng.normal(0, 1 / np.sqrt(hidden), (hidden, outputs))
    d = np.zeros(outputs)
    params = (w, b, v, d)
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    for step in range(1, STEPS + 1):
        h = np.tanh(x @ w + b)
        logits = h @ v + d
        residual = (np.exp(-np.logaddexp(0, -logits)) - y) / n
        back = (residual @ v.T) * (1 - h * h)
        gradients = (x.T @ back, back.sum(axis=0), h.T @ residual, residual.sum(axis=0))
        for j, (p, grad) in enumerate(zip(params, gradients)):
            grad = grad + L2 * p
            first[j] = 0.9 * first[j] + 0.1 * grad
            second[j] = 0.999 * second[j] + 0.001 * grad ** 2
            m = first[j] / (1 - 0.9 ** step)
            var = second[j] / (1 - 0.999 ** step)
            p -= RATE * m / (np.sqrt(var) + 1e-8)
    return dict(zip(("w", "b", "v", "d"), params))


def _save_net(arrays, name, x, y, hidden, seed):
    for key, value in _fit_net(x, y, hidden, seed).items():
        arrays[f"{name}_{key}"] = value


def fit(samples, outdir):
    """Fit on the supplied source supports, freeze all readers, and return the model."""
    full, labels, current, grip = [], [], [], []
    for sample in samples:
        wire, case = sample["wire"], sample["condition"]
        if (wire, case) not in ((1, "BASE"), (-1, "BASE"), (1, "D17")):
            raise ValueError("Expected a registered fitting cell")
        c, s = _tape(sample["commands"], sample["states"])
        full.append(_features(c, s))
        labels.append((wire == 1, case == "D17"))
        boundary = sample["choice_after"]
        if boundary is not None:
            t = index(boundary)
            if not 0 <= t <= len(c):
                raise ValueError("choice_after outside recorded tape")
            prefix = _features(c[:t], s[:t + 1])
            if np.isnan(prefix[4]):
                raise ValueError("choice_after precedes the public closure latch")
            current.append(prefix[1:])
            grip.append(case == "D17")
    if not full:
        raise ValueError("Need at least one fitting support")

    full, labels = np.stack(full), np.asarray(labels, dtype=np.float64)
    arrays = {"full_norm": _normalizer(full, SCALE_FLOOR)}
    x = _normalize(full, arrays["full_norm"])
    for seed in SEEDS:
        _save_net(arrays, f"x_{seed}", x[:, :1], labels[:, :1], 2, seed)
        _save_net(arrays, f"grip_{seed}", x[:, 1:], labels[:, 1:], 3, seed)
        _save_net(arrays, f"joint_{seed}", x, labels, 3, seed)
    if current:
        p = np.stack(current)
        arrays["current_norm"] = _normalizer(p, SCALE_FLOOR[1:])
        _save_net(arrays, "current", _normalize(p, arrays["current_norm"]),
                  np.asarray(grip, dtype=np.float64)[:, None], 3, CURRENT_SEED)
    spec = dict(
        version=1, features=FEATURES, seeds=SEEDS, current_seed=CURRENT_SEED,
        fit_supports=len(full), eligible_current_prefixes=len(current),
        positive_labels=["wire +1", "D17"], probe_controls=PROBE, close_controls=CLOSE,
        widths=dict(factored_x=2, factored_grip=3, joint=3, current=3),
        parameters=dict(factored=38, joint=38, current=31 if current else 0),
        activation="tanh", output="binary logits, positive when strictly above zero",
        objective="sum of head BCEs averaged over tapes + L2/2 times sum of all squared parameters",
        optimizer="full-batch Adam", updates=STEPS, learning_rate=RATE, l2=L2,
        betas=[0.9, 0.999], epsilon=1e-8, initialization="normal sd=1/sqrt(fan_in), zero biases",
        normalization="observed-only mean and population sd, clip at +/-6, zero masked features",
        scale_floor_metres=SCALE_FLOOR.tolist(),
        current="analytic first-pulse sign; eligible public prefixes only; no eligible closure selects grip 0",
    )
    arrays["training_spec"] = np.array(json.dumps(spec, sort_keys=True))
    folder = Path(outdir)
    folder.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=folder, suffix=".npz", delete=False) as stream:
        path = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
        except BaseException:
            path.unlink(missing_ok=True)
            raise
    path.replace(folder / ARCHIVE)
    return Model(arrays)


class Model:
    def __init__(self, arrays):
        self._arrays = arrays

    def _logits(self, name, x):
        a = self._arrays
        h = np.tanh(x @ a[name + "_w"] + a[name + "_b"])
        return h @ a[name + "_v"] + a[name + "_d"]

    def predict(self, commands, states, kind, seed):
        """Return decoder sign and grasp flag from the supplied public prefix."""
        raw = _features(*_tape(commands, states))
        if kind == "current":
            sign = -1 if raw[0] < 0 else 1
            if np.isnan(raw[4]) or "current_norm" not in self._arrays:
                return sign, 0
            x = _normalize(raw[1:], self._arrays["current_norm"])
            return sign, int(self._logits("current", x)[0] > 0)
        if seed not in SEEDS:
            raise ValueError(f"Unknown reset-reader seed {seed!r}")
        x = _normalize(raw, self._arrays["full_norm"])
        if kind == "factored":
            wire = self._logits(f"x_{seed}", x[:1])[0]
            grip = self._logits(f"grip_{seed}", x[1:])[0]
        elif kind == "joint":
            wire, grip = self._logits(f"joint_{seed}", x)
        else:
            raise ValueError(f"Unknown reader kind {kind!r}")
        return (1 if wire > 0 else -1), int(grip > 0)


def load(outdir):
    """Load a frozen archive for inference using NumPy alone."""
    with np.load(Path(outdir) / ARCHIVE, allow_pickle=False) as archive:
        arrays = dict(archive)
    if json.loads(arrays["training_spec"].item())["version"] != 1:
        raise ValueError("Unsupported frozen reader archive")
    return Model(arrays)
