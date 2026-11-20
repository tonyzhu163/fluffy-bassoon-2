"""Estimate forward command yaw and gain from archived public tapes, on CPU."""

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np


BASE = "1967f2b31fd4f64c0c9771fb04df2a268c1e2e41"
TAPES_SHA256 = "c1207da02bd988de7fb51cab00a894ed29f1a31ca7d7306bfe9c9c7fb05c3ebf"
ARMS = ("Panda", "Sawyer")
SOURCE = {"nominal": (0, 1.0), "yaw_p90": (90, 1.0), "yaw_m90": (-90, 1.0), "gain065": (0, 0.65)}
HELD = {"yaw_p45": (45, 1.0), "yaw_p90_gain065": (90, 0.65)}
LAGS = (0, 1, 2, 3)
PENALTY = 0.01


def features(command_xy, motion_xy):
    # A public command at t precedes the displacement recorded in row t + lag.
    u = command_xy[:, 0] + 1j * command_xy[:, 1]
    d = motion_xy[:, 0] + 1j * motion_xy[:, 1]
    result = []
    for lag in LAGS:
        command = u[:len(u) - lag]
        command = command - command.mean()
        motion = d[lag:] - d[lag:].mean()
        energy = np.vdot(command, command).real
        if energy == 0:
            raise ValueError(f"constant planar command at lag {lag}")
        response = np.vdot(command, motion) / energy
        result.extend((response.real, response.imag))
    return result


def check_features():
    # Two public pulses, followed two steps later by +90 degree motion and constant drift.
    command = np.zeros((16, 2))
    command[2, 0] = command[7, 1] = 1
    motion = np.zeros_like(command)
    motion[2:] = 0.005 * np.column_stack((-command[:-2, 1], command[:-2, 0]))
    measured = np.reshape(features(command + [0.2, -0.3], motion + [0.01, -0.02]), (-1, 2))
    np.testing.assert_allclose(measured[LAGS.index(2)], [0, 0.005], rtol=0, atol=1e-14)
    if LAGS[np.argmax(np.linalg.norm(measured, axis=1))] != 2:
        raise ValueError("the delayed synthetic response peaked at the wrong lag")


def load_tapes(path):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TAPES_SHA256:
        raise ValueError(f"unexpected SHA-256 for {path}: {digest}")
    expected = {
        (split, arm, setting, role, index)
        for split, settings, size in (("train", SOURCE, 12), ("eval", HELD, 8))
        for arm, setting, role, index in product(ARMS, settings, ("support", "query"), range(size))
    }
    blocks, held_features = {}, {}
    source_features, targets = [], []
    query_count = transition_count = 0
    for line in raw.splitlines():
        row = json.loads(line)
        split, arm, setting, role, index = key = tuple(
            row[field] for field in ("split", "family", "setting", "role", "index")
        )
        if key not in expected:
            raise ValueError(f"duplicate or unregistered tape {key}")
        expected.remove(key)
        command = np.asarray(row["commands"], dtype=np.float64)
        if command.shape != (16, 7) or len(row["motion"]) != 16:
            raise ValueError(f"unexpected control count or dimensions at {key}")
        transition_count += len(command)
        block = (split, role, index)
        identity = (row["layout_seed"], row["scene_hash"])
        previous, reference_command = blocks.setdefault(block, (identity, command))
        if previous != identity or not np.array_equal(command, reference_command):
            raise ValueError(f"mismatched donor or public probe at {key}")

        if split == "eval" and role == "query":
            query_count += 1
            continue
        motion = np.asarray(row["motion"], dtype=np.float64)
        if motion.shape != (16, 3) or not np.isfinite(motion).all() or not np.isfinite(command).all():
            raise ValueError(f"invalid public tape at {key}")
        x = features(command[:, :2], motion[:, :2])
        if split == "train":
            yaw_deg, gain = SOURCE[setting]
            target = gain * np.exp(1j * np.deg2rad(yaw_deg))
            source_features.append(x)
            targets.append((target.real, target.imag))
        else:
            held_features[arm, setting, index] = x

    seeds = {block: identity[0] for block, (identity, _) in blocks.items()}
    scenes = {identity[1] for identity, _ in blocks.values()}
    counts = dict(tapes=len(targets) + len(held_features) + query_count,
                  source_tapes=len(targets), held_support_tapes=len(held_features),
                  held_query_tapes=query_count, transitions=transition_count,
                  distinct_layouts=len(set(seeds.values())))
    required = dict(tapes=256, source_tapes=192, held_support_tapes=32,
                    held_query_tapes=32, transitions=4096, distinct_layouts=40)
    if expected or len(scenes) != 40 or counts != required:
        raise ValueError(f"incomplete registered panel: {counts}")
    support_seeds = [seeds["eval", "support", index] for index in range(8)]
    return np.asarray(source_features), np.asarray(targets), held_features, support_seeds, counts


def fit_readers(x, y, support):
    intercept = y.mean(axis=0)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    z = (x - mean) / scale
    yield "source_mean", {"intercept_re_im": intercept.tolist()}, np.tile(intercept, (len(support), 1))
    for name, lags in (("lag_0", (0,)), ("lags_0_to_3", LAGS)):
        width = 2 * len(lags)
        design = z[:, :width]
        gram = design.T @ design + len(x) * PENALTY * np.eye(width)
        weights = np.linalg.solve(gram, design.T @ (y - intercept))
        fit = dict(lags=lags, lambda_mean_objective=PENALTY, intercept_re_im=intercept.tolist(),
                   feature_mean=mean[:width].tolist(), feature_scale=scale[:width].tolist(),
                   weights=weights.tolist())
        yield name, fit, intercept + ((support[:, :width] - mean[:width]) / scale[:width]) @ weights


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def polar(real, imag):
    gain = float(np.hypot(real, imag))
    yaw = wrap_deg(np.rad2deg(np.arctan2(imag, real))) if gain > 0 else None
    return yaw, gain


def score(estimate, target):
    yaw, gain = estimate
    true_yaw, true_gain = HELD[target]
    signed_error = abs_error = None
    nearest = []
    if yaw is not None:
        signed_error = wrap_deg(yaw - true_yaw)
        abs_error = abs(signed_error)
        distances = {angle: abs(wrap_deg(yaw - angle)) for angle, _ in HELD.values()}
        best = min(distances.values())
        nearest = [angle for angle, distance in distances.items() if distance <= best + 1e-12]
    return dict(yaw_deg=yaw, gain=gain, signed_circular_error_deg=signed_error,
                abs_circular_error_deg=abs_error, abs_gain_error=abs(gain - true_gain),
                nearest_yaws_deg=nearest, nearest_credit=1 / len(nearest) if true_yaw in nearest else 0)


def distribution(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"n": 0}
    return dict(n=len(values), mean=float(np.mean(values)), median=float(np.median(values)),
                min=float(min(values)), max=float(max(values)))


def summarize(scores):
    summary = {key: distribution(row[key] for row in scores)
               for key in ("gain", "abs_circular_error_deg", "abs_gain_error")}
    angles = [row["yaw_deg"] for row in scores if row["yaw_deg"] is not None]
    vector = np.mean(np.exp(1j * np.deg2rad(angles))) if angles else 0j
    summary.update(yaw_count=len(angles), yaw_mean_resultant_length=abs(vector) if angles else None,
                   yaw_circular_mean_deg=wrap_deg(np.angle(vector, deg=True)) if abs(vector) > 1e-15 else None,
                   nearest_accuracy=float(np.mean([row["nearest_credit"] for row in scores])))
    return summary


def evaluate(predictions, support_seeds):
    cells, pairs = [], []
    first, second = HELD
    for arm in ARMS:
        for target, other in ((first, second), (second, first)):
            rows = [
                dict(index=i, layout_seed=seed, true=score(predictions[arm, target, i], target),
                     swap=score(predictions[arm, other, i], target))
                for i, seed in enumerate(support_seeds)
            ]
            cells.append(dict(arm=arm, target=target, swap_donor=other, blocks=rows,
                              summary={k: summarize([row[k] for row in rows]) for k in ("true", "swap")}))
        # One signed second-minus-first comparison per arm/block, shared by both target cells.
        rows = []
        for i, seed in enumerate(support_seeds):
            a, _ = predictions[arm, first, i]
            b, _ = predictions[arm, second, i]
            gap = wrap_deg(b - a) if a is not None and b is not None else None
            rows.append(dict(index=i, layout_seed=seed, signed_gap_deg=gap,
                             separation_deg=abs(gap) if gap is not None else None))
        pairs.append(dict(arm=arm, from_setting=first, to_setting=second,
                          ideal_signed_gap_deg=wrap_deg(HELD[second][0] - HELD[first][0]), blocks=rows,
                          separation_deg=distribution(row["separation_deg"] for row in rows)))
    return dict(cells=cells, paired_angles=pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=Path, required=True, help="registered native02 tapes.jsonl")
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run the offline reader on Linux CPU")
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")
    check_features()
    x, y, held, seeds, counts = load_tapes(args.tapes)
    support = np.asarray(list(held.values()))
    report = dict(design_base=BASE, factor_py_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tapes_sha256=TAPES_SHA256, numpy=np.__version__, synthetic_feature_check="passed",
                  counts=counts, held_truth_yaw_deg_gain=HELD, readers={})
    readers = list(fit_readers(x, y, support))
    for name, fit, values in readers:
        predictions = {key: polar(*value) for key, value in zip(held, values, strict=True)}
        report["readers"][name] = dict(fit=fit, **evaluate(predictions, seeds))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.mkdir(parents=True, exist_ok=False)
    path = args.out / "report.json"
    path.write_text(payload, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
