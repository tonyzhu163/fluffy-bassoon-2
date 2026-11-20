"""Source-only bounded action residual and one signed native-outcome gate."""

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tempfile

import numpy as np


ARMS = ("Panda", "Sawyer")
FORK = 60
HORIZON = 300
CHUNK = 5
AXES = 7
CAMERA_FEATURES = 4 * 4 * 3
RAW_START = 2 * CAMERA_FEATURES + 8
RAW = slice(RAW_START, RAW_START + CHUNK * AXES)
DIM = RAW.stop + 1
RIDGE = 1.0
GATE_THRESHOLD = 0.10
CAPS = np.array([0.50, 0.50, 0.50, 0.25, 0.25, 0.25, 2.00])
SCALE_FLOOR = np.repeat([0.04, 0.02, 0.05, 5.0], [2 * CAMERA_FEATURES, 8, CHUNK * AXES, 1])
SPEC = dict(schema="source-correction-gate/v1", ridge=RIDGE, gate_threshold=GATE_THRESHOLD,
            fork=FORK, horizon=HORIZON, action_bounds=[-1, 1], residual_caps=CAPS.tolist(),
            features=["rgb:4x4x3:mean/255", "wrist:4x4x3:mean/255", "state:8",
                      "raw:5x7:time-major", "elapsed:1"], scale_floor=SCALE_FLOOR.tolist(),
            weighting="equal eligible arms, layouts within arms, rows within layouts per head",
            gate_target="teacher_success - raw_success", residual_target="issued_teacher - frozen_raw")


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Expected integer {name}")
    return int(value)


def _boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Expected boolean {name}")
    return bool(value)


def _raw(raw_chunk):
    raw = np.asarray(raw_chunk)
    if (raw.dtype != np.float32 or raw.shape != (CHUNK, AXES)
            or not np.isfinite(raw).all() or (np.abs(raw) > 1).any()):
        raise ValueError("Expected a finite, bounded R2 float32[5,7] proposal")
    return raw


def features(obs, raw_chunk, step):
    """Compute identical public features for collection and deployment."""
    step = _integer(step, "elapsed step")
    if not 0 <= step < HORIZON or set(obs) != {"rgb", "wrist", "state", "prompt"}:
        raise ValueError("Expected an actionable step and the four R2 public observation fields")
    state = np.asarray(obs["state"])
    if state.shape != (8,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("Expected R2 float32[8] public state")
    x = np.empty(DIM, np.float32)
    for i, name in enumerate(("rgb", "wrist")):
        rgb = np.asarray(obs[name])
        if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
            raise ValueError(f"Expected R2 uint8[256,256,3] {name}")
        cells = rgb.reshape(4, 64, 4, 64, 3).mean(axis=(1, 3), dtype=np.float32)
        x[i * CAMERA_FEATURES:(i + 1) * CAMERA_FEATURES] = cells.ravel() / 255.0
    x[2 * CAMERA_FEATURES:RAW_START] = state
    x[RAW] = _raw(raw_chunk).ravel()
    x[-1] = step
    return x


@dataclass
class _Head:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    bias: np.ndarray

    def predict(self, x):
        return ((x - self.mean) / self.scale) @ self.coef + self.bias


def _correct(raw, prediction):
    # Exact per-axis caps also hold after conversion to executable float32.
    raw64 = raw.astype(np.float64)
    action = np.clip(raw64 + np.clip(prediction, -CAPS, CAPS), -1, 1).astype(np.float32)
    rounded_past_cap = np.abs(action.astype(np.float64) - raw64) > CAPS
    return np.where(rounded_past_cap, np.nextafter(action, raw), action)


@dataclass
class Policy:
    gate: _Head
    residual: tuple

    def score(self, obs, raw_chunk, step=FORK):
        if step != FORK:
            raise ValueError("Score the gate once at the step-60 fork")
        return float(self.gate.predict(features(obs, raw_chunk, step))[0])

    def apply(self, obs, raw_chunk, step, enabled):
        raw = _raw(raw_chunk)
        if not _boolean(enabled, "gate decision"):
            return raw
        if _integer(step, "elapsed step") < FORK:
            raise ValueError("Correction begins at the step-60 fork")
        x = features(obs, raw, step)
        return _correct(raw, np.stack([head.predict(x) for head in self.residual]))


def _matrix(value, shape, name):
    a = np.asarray(value)
    if a.shape != shape or a.dtype != np.float32 or not np.isfinite(a).all():
        raise ValueError(f"Expected finite float32 {name} of shape {shape}")
    return a


def _check_x(x):
    if (np.any(x[..., :2 * CAMERA_FEATURES] < 0) or np.any(x[..., :2 * CAMERA_FEATURES] > 1)
            or np.any(np.abs(x[..., RAW]) > 1)):
        raise ValueError("Out-of-range camera feature or frozen proposal in training input")


def _digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(f"{a.dtype.str}/{a.shape}/".encode())
        h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class _Pair:
    arm: str
    layout: int
    fork: object
    x: np.ndarray
    actions: np.ndarray
    lengths: np.ndarray
    record: dict

    @property
    def delta(self):
        return self.record["outcome_difference"]


def _pair(row):
    if row["split"] != "train" or row["arm"] not in ARMS:
        raise ValueError("Fit requires source training pairs")
    arm = str(row["arm"])
    layout = _integer(row["layout_id"], "layout_id")
    raw_win = _boolean(row["raw_success"], "raw_success")
    teacher_win = _boolean(row["teacher_success"], "teacher_success")
    fork = row["fork_x"]
    reached = fork is not None
    if "decision_reached" in row and _boolean(row["decision_reached"], "decision_reached") != reached:
        raise ValueError("decision_reached disagrees with fork_x")
    if reached:
        fork = _matrix(fork, (DIM,), "fork_x")
        _check_x(fork)
        if fork[-1] != FORK:
            raise ValueError("Gate features must be measured at step 60")
    elif raw_win != teacher_win:
        raise ValueError("Outcomes disagree before the fork")

    lengths = np.asarray(row["valid_lengths"])
    n = lengths.size
    if lengths.ndim != 1 or (n and lengths.dtype.kind not in "iu"):
        raise ValueError("Expected integer valid_lengths vector")
    lengths = lengths.astype(np.int64)
    if n > (HORIZON - FORK) // CHUNK or (n and (
            not reached or np.any(lengths[:-1] != CHUNK) or not 1 <= lengths[-1] <= CHUNK)):
        raise ValueError("Invalid sequence of issued teacher chunks")
    if reached and teacher_win and not n:
        raise ValueError("Successful teacher is missing its actionable suffix")
    if n:
        x = _matrix(row["suffix_x"], (n, DIM), "suffix_x")
        actions = _matrix(row["teacher_actions"], (n, CHUNK, AXES), "teacher_actions")
        _check_x(x)
        valid = np.arange(CHUNK) < lengths[:, None]
        if (not np.array_equal(x[0], fork)
                or not np.array_equal(x[:, -1], FORK + CHUNK * np.arange(n))
                or np.any(np.abs(actions) > 1) or np.any(actions[~valid] != 0)):
            raise ValueError("Unmatched fork, wrong suffix cadence or invalid teacher actions")
    else:
        if np.size(row["suffix_x"]) or np.size(row["teacher_actions"]):
            raise ValueError("Unexpected suffix data without issued teacher controls")
        x = np.empty((0, DIM), np.float32)
        actions = np.empty((0, CHUNK, AXES), np.float32)
    if "base_chunks" in row:
        base = np.asarray(row["base_chunks"])
        if base.size or n:
            base = _matrix(base, (n, CHUNK, AXES), "base_chunks")
            if not np.array_equal(base, x[:, RAW].reshape(base.shape)):
                raise ValueError("Stored raw chunks differ from feature proposals")

    record = dict(arm=arm, layout_id=layout, decision_reached=reached, raw_success=raw_win,
                  teacher_success=teacher_win, outcome_difference=int(teacher_win) - int(raw_win),
                  suffix_chunks=n, suffix_controls=int(lengths.sum()), valid_lengths=lengths.tolist(),
                  residual_rows_by_offset=[int((lengths > k).sum()) if teacher_win else 0
                                           for k in range(CHUNK)],
                  feature_target_sha256=_digest(fork if reached else x[:0], x, actions, lengths))
    if row.get("layout_seed") is not None:
        record["layout_seed"] = _integer(row["layout_seed"], "layout_seed")
    return _Pair(arm, layout, fork, x, actions, lengths, record)


def _fit_head(groups, width):
    """Each group is one eligible layout with its own possibly unequal number of rows."""
    counts = Counter(pair.arm for pair, _, _ in groups)
    if groups:
        x = np.concatenate([x for _, x, _ in groups]).astype(np.float64)
        y = np.concatenate([y for _, _, y in groups]).astype(np.float64)
        weights = np.concatenate([np.full(len(xx), 1 / (len(counts) * counts[p.arm] * len(xx)))
                                  for p, xx, _ in groups])
        mean = weights @ x
        centered = x - mean
        scale = np.maximum(np.sqrt(weights @ (centered ** 2)), SCALE_FLOOR)
        z = centered / scale
        bias = weights @ y
        ztw = z.T * weights
        coef = np.linalg.solve(ztw @ z + RIDGE * np.eye(DIM), ztw @ (y - bias))
    else:
        mean, scale = np.zeros(DIM), np.ones(DIM)
        coef, bias = np.zeros((DIM, width)), np.zeros(width)
    head = _Head(mean, scale, coef, bias)
    info = dict(eligible_layouts={arm: counts[arm] for arm in ARMS},
                arm_weight={arm: 1 / len(counts) if counts[arm] else 0 for arm in ARMS},
                rows=sum(len(x) for _, x, _ in groups), mean=mean.tolist(), scale=scale.tolist(),
                bias=bias.tolist())
    return head, info


def _fit(pairs):
    groups = [(p, p.fork[None, :], np.array([[p.delta]])) for p in pairs if p.fork is not None]
    gate, gate_info = _fit_head(groups, 1)
    for p in pairs:
        score = None if p.fork is None else float(gate.predict(p.fork)[0])
        p.record.update(gate_score=score, gate_on=None if score is None else score > GATE_THRESHOLD)
        p.record["gate_layout_weight"] = (
            gate_info["arm_weight"][p.arm] / gate_info["eligible_layouts"][p.arm]
            if p.fork is not None else 0.0)
        p.record["residual_layout_weights"] = []
    residual, residual_info = [], []
    for offset in range(CHUNK):
        groups = []
        for p in pairs:
            valid = p.lengths > offset
            if p.record["teacher_success"] and valid.any():
                x = p.x[valid]
                base = x[:, RAW].reshape(-1, CHUNK, AXES)[:, offset]
                delta = p.actions[valid, offset].astype(np.float64) - base
                groups.append((p, x, delta))
        head, info = _fit_head(groups, AXES)
        for p in pairs:
            weight = (info["arm_weight"][p.arm] / info["eligible_layouts"][p.arm]
                      if p.record["residual_rows_by_offset"][offset] else 0.0)
            p.record["residual_layout_weights"].append(weight)
        errors = {arm: [] for arm in ARMS}
        for p, x, delta in groups:
            raw = x[:, RAW].reshape(-1, CHUNK, AXES)[:, offset]
            corrected = _correct(raw, head.predict(x))
            remainder = (corrected.astype(np.float64) - raw) - delta
            errors[p.arm].append([np.mean(delta ** 2), np.mean(remainder ** 2)])
        info["by_arm_layout_mean_mse"] = {
            arm: dict(zip(("raw", "corrected"), np.mean(values, axis=0).tolist())) if values else None
            for arm, values in errors.items()}
        residual_info.append(info)
        residual.append(head)
    return Policy(gate, tuple(residual)), dict(gate=gate_info, residual_by_offset=residual_info)


def _arm_counts(records):
    rows = [r for r in records if r["decision_reached"]]
    return dict(pairs=len(records), early_terminal=len(records) - len(rows), forks=len(rows),
                paired_outcomes={f"raw{int(b)}_teacher{int(t)}": sum(
                    r["raw_success"] == b and r["teacher_success"] == t for r in rows)
                    for b in (False, True) for t in (False, True)},
                gate_labels={str(v): sum(r["outcome_difference"] == v for r in rows) for v in (-1, 0, 1)},
                gate_on=sum(r["gate_on"] for r in rows), gate_off=sum(not r["gate_on"] for r in rows),
                residual_layouts=sum(r["teacher_success"] for r in rows),
                residual_chunks=sum(r["suffix_chunks"] for r in rows if r["teacher_success"]),
                residual_controls=sum(r["suffix_controls"] for r in rows if r["teacher_success"]))


def _json(obj):
    return json.dumps(obj, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)


def fit(pairs, out_dir):
    """Fit signed gate and bounded residual from paired, manifest-selected training rows."""
    pairs = sorted((_pair(row) for row in pairs), key=lambda p: (p.arm, p.layout))
    if not pairs or len({(p.arm, p.layout) for p in pairs}) != len(pairs):
        raise ValueError("Expected a nonempty training panel with unique arm/layout pairs")
    records = [p.record for p in pairs]
    fit_id = hashlib.sha256(_json([SPEC, records]).encode()).hexdigest()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".fit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        report = out / "fit.json"
        if report.exists():
            if json.loads(report.read_text())["fit_id"] != fit_id:
                raise ValueError("A different training fit already exists in this directory")
            return load(out)
        policy, heads = _fit(pairs)
        stream = io.BytesIO()
        arrays = {}
        for name in ("mean", "scale", "coef", "bias"):
            arrays["gate_" + name] = getattr(policy.gate, name)
            arrays["residual_" + name] = np.stack([getattr(h, name) for h in policy.residual])
        np.savez_compressed(stream, fit_id=fit_id, spec=_json(SPEC), **arrays)
        blob = stream.getvalue()
        summary = dict(fit_id=fit_id, split="train", spec=SPEC, training_exposure=records, heads=heads,
                       per_arm={arm: _arm_counts([r for r in records if r["arm"] == arm]) for arm in ARMS},
                       model_sha256=hashlib.sha256(blob).hexdigest())
        _write(out / "model.npz", blob)
        _write(report, (json.dumps(summary, indent=2, allow_nan=False) + "\n").encode())
        return policy


def load(out_dir):
    out = Path(out_dir)
    summary = json.loads((out / "fit.json").read_text())
    blob = (out / "model.npz").read_bytes()
    if hashlib.sha256(blob).hexdigest() != summary["model_sha256"]:
        raise ValueError("Checkpoint bytes differ from the fit report")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        if (data["spec"].item() != _json(SPEC) or summary["spec"] != SPEC
                or data["fit_id"].item() != summary["fit_id"]):
            raise ValueError("Checkpoint specification or training exposure ID differs")
        names = ("mean", "scale", "coef", "bias")
        gate = _Head(*(data["gate_" + name] for name in names))
        values = [data["residual_" + name] for name in names]
        residual = tuple(_Head(*(v[k] for v in values)) for k in range(CHUNK))
    return Policy(gate, residual)


def selftest():
    """Small Linux CPU test with closed-form gate scores and masked residual means."""
    if platform.system() != "Linux":
        raise RuntimeError("Run the numerical selftest on Linux")
    rgb = np.zeros((256, 256, 3), np.uint8)
    wrist = rgb.copy()
    rgb[:64, 64:128, 2] = 255
    wrist[-64:, :64, 1] = 255
    raw = np.zeros((CHUNK, AXES), np.float32)
    raw[:, 0], raw[:, 2] = -0.125, 0.875
    raw[:, 1] = np.arange(CHUNK) / 16 - 0.25
    raw[:, 3] = -0.0

    def obs(signal):
        state = np.array([signal] + [0] * 7, np.float32)
        return dict(rgb=rgb, wrist=wrist, state=state, prompt="public task text")

    x = features(obs(1), raw, FORK)
    np.testing.assert_array_equal(np.flatnonzero(x[:2 * CAMERA_FEATURES]), [5, 85])
    np.testing.assert_array_equal(x[RAW], raw.ravel())
    np.testing.assert_array_equal(x, features(dict(obs(1), prompt="different"), raw, FORK))

    def pair(arm, layout, signal, raw_win, teacher_win, lengths=(), delta=0):
        xx = [features(obs(signal), raw, FORK + CHUNK * i) for i in range(len(lengths))]
        actions = np.zeros((len(xx), CHUNK, AXES), np.float32)
        for i, length in enumerate(lengths):
            actions[i, :length] = raw[:length]
            actions[i, :length, 0] += delta
            actions[i, :length, 1] += np.arange(length) / 8
        return dict(split="train", arm=arm, layout_id=layout, raw_success=raw_win,
                    teacher_success=teacher_win, fork_x=features(obs(signal), raw, FORK),
                    suffix_x=np.array(xx), teacher_actions=actions, valid_lengths=lengths,
                    base_chunks=np.repeat(raw[None], len(xx), axis=0))

    # Unequal numbers of layouts and unequal-duration teacher trajectories.
    pairs = [pair("Panda", 0, 1, False, True, [5, 2], 0.75),
             pair("Panda", 1, -1, True, False, [5], -0.5),
             pair("Panda", 2, 0, False, False), pair("Panda", 3, 0, False, False),
             pair("Sawyer", 0, 1, False, True, [5, 5, 1], 0.5),
             pair("Sawyer", 1, -1, True, False),
             pair("Sawyer", 2, 0, True, True, [1], -0.25)]
    early = dict(pair("Sawyer", 3, 0, True, True), fork_x=None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        policy = fit([*pairs, early], root / "fit")
        restored = load(root / "fit")
        report = json.loads((root / "fit/fit.json").read_text())
        assert report["per_arm"]["Panda"]["gate_labels"] == {"-1": 1, "0": 2, "1": 1}
        assert report["per_arm"]["Sawyer"]["early_terminal"] == 1
        np.testing.assert_allclose(policy.gate.scale[2 * CAMERA_FEATURES], np.sqrt(7 / 12))
        for signal in (-1, 0, 1):
            public = obs(signal)
            score = restored.score(public, raw, FORK)
            np.testing.assert_allclose(score, signal / (1 + RIDGE), atol=1e-12)
            on = score > GATE_THRESHOLD
            issued = restored.apply(public, raw, FORK, on)
            expected = policy.apply(public, raw, FORK, True) if on else raw
            assert issued.tobytes() == expected.tobytes()
        assert policy.apply(None, raw, 0, False) is raw
        np.testing.assert_allclose([h.bias[0] for h in restored.residual], [0.4375] + [0.625] * 4)
        np.testing.assert_allclose([h.bias[1] for h in restored.residual], np.arange(CHUNK) / 8)
        issued = restored.apply(obs(1), raw, FORK, True)
        assert issued.dtype == np.float32 and np.any(issued != raw)
        assert np.all(np.abs(issued.astype(np.float64) - raw) <= CAPS)
        saturated = _correct(raw, np.full_like(raw, 10))
        np.testing.assert_array_equal(saturated, np.clip(raw + CAPS, -1, 1))
        null = fit([pairs[1], early], root / "null")
        assert null.score(obs(1), raw) == -1
        np.testing.assert_array_equal(null.apply(obs(1), raw, FORK, True), raw)
        empty_fork = fit([early], root / "no-forks")
        assert empty_fork.score(obs(1), raw) == 0
    print(json.dumps(dict(status="passed", platform=platform.system(), numpy=np.__version__,
                          features=DIM, signed_gate_scores=[-0.5, 0.0, 0.5],
                          residual_offset_bias=[0.4375] + [0.625] * 4)))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("command", choices=("selftest",))
    cli.parse_args()
    selftest()
