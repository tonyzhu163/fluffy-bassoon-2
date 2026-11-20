"""Estimate forward XY command yaw and gain from archived H1 tapes on Linux CPU."""

import argparse
import hashlib
import json
import sys
from itertools import product
from pathlib import Path
from typing import NamedTuple

import numpy as np


BASE = "1967f2b31fd4f64c0c9771fb04df2a268c1e2e41"
TAPE_SHA256 = "c1207da02bd988de7fb51cab00a894ed29f1a31ca7d7306bfe9c9c7fb05c3ebf"
FAMILIES = ("Panda", "Sawyer")
SOURCE = ("nominal", "yaw_p90", "yaw_m90", "gain065")
HELD = ("yaw_p45", "yaw_p90_gain065")
TRUTH = {"yaw_p45": (45.0, 1.0), "yaw_p90_gain065": (90.0, 0.65)}
ROLES = ("support", "query")
STEPS, START = 16, 3
BOOT_SEED, BOOT_DRAWS = 20261120, 10000


class Tape(NamedTuple):
    command: np.ndarray  # public x + iy, before the hidden map
    motion: np.ndarray  # measured world dx + i dy, metres


def matrix(row, name, width):
    a = np.asarray(row[name])
    if a.shape != (STEPS, width) or a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"invalid {name}, expected {STEPS} x {width} finite numbers")
    return a.astype(np.float64)


def load_tapes(path):
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != TAPE_SHA256:
        raise ValueError(f"H1 tape SHA-256 mismatch: {digest}")

    expected = {
        (family, setting, role, i)
        for settings, count in ((SOURCE, 12), (HELD, 8))
        for family, setting, role, i in product(FAMILIES, settings, ROLES, range(count))
    }
    fingerprints = ("initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash")
    tapes, blocks = {}, {}
    for line_no, line in enumerate(data.splitlines(), 1):
        try:
            row = json.loads(line)
            family, setting, role, i = key = tuple(row[k] for k in ("family", "setting", "role", "index"))
            if type(i) is not int or key not in expected or key in tapes:
                raise ValueError(f"unexpected or duplicate tape {key}")
            train = setting in SOURCE
            split = "train" if train else "eval"
            seed = (960000 if train else 970000) + (100 if role == "query" else 0) + i
            pulse = train and role == "support" and i == 0
            assignment = (split, seed, seed + 1000000, seed + 2000000, "signed_pulses" if pulse else "mixed")
            if tuple(row[k] for k in ("split", "layout_seed", "layout_id", "probe_seed", "probe_kind")) != assignment:
                raise ValueError(f"incorrect split, layout or probe for {key}")
            cmd, mov = matrix(row, "commands", 7), matrix(row, "motion", 3)
            hashes = tuple(row[k] for k in fingerprints)
            if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
                   for h in hashes):
                raise ValueError(f"invalid layout fingerprints for {key}")
            ref_cmd, ref_hashes = blocks.setdefault(seed, (cmd, hashes))
            if not np.array_equal(cmd, ref_cmd) or hashes != ref_hashes:
                raise ValueError(f"unmatched public commands or layout for {key}")
            tapes[key] = Tape(cmd[:, 0] + 1j * cmd[:, 1], mov[:, 0] + 1j * mov[:, 1])
        except (ValueError, TypeError, KeyError) as err:
            raise ValueError(f"{path}:{line_no}: {err}") from err
    if tapes.keys() != expected:
        raise ValueError(f"incomplete H1 panel, missing {len(expected - tapes.keys())} tapes")
    for j, field in enumerate(fingerprints):
        if len({hashes[j] for _, hashes in blocks.values()}) != len(blocks):
            raise ValueError(f"distinct layout seeds share {field}")
    return tapes, {"path": str(path), "sha256": digest, "bytes": len(data),
                   "tapes": len(tapes), "placement_blocks": len(blocks)}


def lag_matrix(command, taps):
    # Construct lags within one tape. Every column targets times START .. T-1.
    return np.column_stack([command[START - lag:len(command) - lag] for lag in range(taps)])


def fit_fir(nominal, taps):
    x = np.vstack([lag_matrix(t.command, taps) for t in nominal])
    y = np.concatenate([t.motion[START:] for t in nominal])
    coef, _, rank, _ = np.linalg.lstsq(np.vstack((x.real, x.imag)), np.r_[y.real, y.imag], rcond=None)
    if rank != taps:
        raise ValueError("nominal source commands do not identify the requested FIR")
    return coef


def response(command, fir):
    return lag_matrix(command, len(fir)) @ fir


def fit_multiplier(predicted, measured):
    energy = np.vdot(predicted, predicted).real
    if energy == 0:
        raise ValueError("zero predicted XY energy, multiplier is unidentifiable")
    return np.vdot(predicted, measured) / energy


def source_common(source, fir):
    return fit_multiplier(np.concatenate([response(t.command, fir) for t in source]),
                          np.concatenate([t.motion[START:] for t in source]))


def xy_mse(residual):
    return float(np.mean(residual.real**2 + residual.imag**2) / 2)


def wrap(angle):
    return (angle + 180.0) % 360.0 - 180.0


def yaw(multiplier):
    if multiplier == 0:
        raise ValueError("zero support gain, forward yaw is undefined")
    return float(wrap(np.angle(multiplier, deg=True)))


def nearest(angle):
    d45, d90 = (abs(wrap(angle - ref)) for ref in (45, 90))
    return None if d45 == d90 else (45 if d45 < d90 else 90)


def score_case(family, setting, index, query, fir, correct, swapped, common, scale):
    baseline = response(query.command, fir)
    mappings = {"correct": correct, "swapped": swapped, "nominal": 1.0, "common": common}
    losses = {k: xy_mse(c * baseline - query.motion[START:]) for k, c in mappings.items()}
    mse = {k: v / scale**2 for k, v in losses.items()}
    # Truth first enters here, after every prediction has been determined.
    target_yaw, target_gain = TRUTH[setting]
    angle, other_angle = yaw(correct), yaw(swapped)
    error = wrap(angle - target_yaw)
    separation = wrap(angle - other_angle)
    oriented = separation if setting == HELD[1] else wrap(-separation)
    label = nearest(angle)
    return {
        "family": family, "setting": setting, "index": index, "query_rows": len(baseline),
        "true_yaw_deg": target_yaw, "true_gain": target_gain,
        "correct_complex": [float(correct.real), float(correct.imag)],
        "swapped_complex": [float(swapped.real), float(swapped.imag)],
        "forward_yaw_deg": angle, "gain": abs(correct), "swapped_yaw_deg": other_angle,
        "swapped_gain": abs(swapped), "yaw_error_deg": error, "abs_yaw_error_deg": abs(error),
        "gain_error": abs(correct) - target_gain, "abs_gain_error": abs(abs(correct) - target_gain),
        "nearest_angle_deg": label, "classified_correctly": label == target_yaw,
        "correct_minus_swapped_yaw_deg": separation, "abs_donor_change_deg": abs(separation),
        "yaw90_minus_yaw45_deg": oriented, "fraction_of_45deg_separation": oriented / 45,
        "correct_minus_swapped_abs_yaw_error_deg": abs(error) - abs(wrap(other_angle - target_yaw)),
        "query_mse_m2": losses, "query_mse": mse,
        "control_minus_correct_mse": {k: v - mse["correct"] for k, v in mse.items() if k != "correct"},
        "prediction_change_mse": xy_mse((correct - swapped) * baseline) / scale**2,
    }


def means(rows, field):
    return {k: float(np.mean([r[field][k] for r in rows])) for k in rows[0][field]}


def summarize(rows, draws):
    groups = [[r for r in rows if r["index"] == i] for i in sorted({r["index"] for r in rows})]
    layouts = []
    for group in groups:
        layouts.append({
            "index": group[0]["index"], "cases": len(group),
            "control_minus_correct_mse": means(group, "control_minus_correct_mse"),
            "yaw90_minus_yaw45_deg": float(np.mean([r["yaw90_minus_yaw45_deg"] for r in group])),
            "abs_donor_change_deg": float(np.mean([r["abs_donor_change_deg"] for r in group])),
        })

    def interval(values):
        samples = np.asarray(values)[draws].mean(axis=1)
        return np.percentile(samples, [2.5, 97.5]).tolist()

    fields = ("gain", "yaw_error_deg", "abs_yaw_error_deg", "gain_error", "abs_gain_error",
              "correct_minus_swapped_yaw_deg", "abs_donor_change_deg",
              "yaw90_minus_yaw45_deg", "fraction_of_45deg_separation",
              "correct_minus_swapped_abs_yaw_error_deg", "prediction_change_mse")
    n_correct = sum(r["classified_correctly"] for r in rows)
    mean_direction = np.mean(np.exp(1j * np.deg2rad([r["forward_yaw_deg"] for r in rows])))
    return {
        "support_query_cases": len(rows), "paired_layout_indices": len(layouts),
        "distinct_family_index_donor_pairs": len({(r["family"], r["index"]) for r in rows}),
        "query_rows": sum(r["query_rows"] for r in rows), "nearest_angle_correct": n_correct,
        "nearest_angle_total": len(rows), "nearest_angle_ties": sum(r["nearest_angle_deg"] is None for r in rows),
        "nearest_angle_accuracy": n_correct / len(rows),
        "circular_mean_forward_yaw_deg": yaw(mean_direction) if abs(mean_direction) > np.finfo(float).eps else None,
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in fields},
        "median_abs_donor_change_deg": float(np.median([r["abs_donor_change_deg"] for r in rows])),
        "query_mse": means(rows, "query_mse"), "query_mse_m2": means(rows, "query_mse_m2"),
        "control_minus_correct_mse": means(rows, "control_minus_correct_mse"),
        "control_minus_correct_mse_ci95": {
            k: interval([g["control_minus_correct_mse"][k] for g in layouts])
            for k in rows[0]["control_minus_correct_mse"]
        },
        "yaw90_minus_yaw45_deg_ci95": interval([g["yaw90_minus_yaw45_deg"] for g in layouts]),
        "by_layout": layouts,
    }


def evaluate(tapes, nominal, source, taps, scale, draws):
    fir = fit_fir(nominal, taps)
    common = source_common(source, fir)
    cases = []
    for family, i in product(FAMILIES, range(8)):
        estimates = []
        for setting in HELD:
            support = tapes[family, setting, "support", i]
            estimates.append(fit_multiplier(response(support.command, fir), support.motion[START:]))
        for j, setting in enumerate(HELD):
            query = tapes[family, setting, "query", i]
            cases.append(score_case(family, setting, i, query, fir, estimates[j], estimates[1 - j], common, scale))

    slices = []
    for family, setting in product((None, *FAMILIES), (None, *HELD)):
        subset = [r for r in cases if (family is None or r["family"] == family)
                  and (setting is None or r["setting"] == setting)]
        slices.append({"family": family, "setting": setting, **summarize(subset, draws)})
    return {"lags": list(range(taps)), "fir_m_per_command": fir.tolist(),
            "fixed_common_complex": [float(common.real), float(common.imag)], "cases": cases, "slices": slices}


def self_test():
    # Generate motion independently of the estimator's lag construction.
    rng = np.random.default_rng(147)
    commands = [rng.normal(size=STEPS) + 1j * rng.normal(size=STEPS) for _ in range(3)]
    for fir in (np.array([0.01]), np.array([0.002, 0.011, -0.003, 0.004])):
        def synthetic(command, gain=1.0, deg=0.0):
            rot = gain * np.exp(1j * np.deg2rad(deg))
            motion = rot * np.convolve(command, fir)[:STEPS]
            motion[:START] = 100 + 200j  # A scored or fitted warm-up row must break exact recovery.
            return Tape(command, motion)

        recovered = fit_fir([synthetic(cmd) for cmd in commands[:2]], len(fir))
        np.testing.assert_allclose(recovered, fir, rtol=0, atol=1e-14)
        c45 = np.exp(1j * np.pi / 4)
        c90 = 0.65j
        donors = [synthetic(commands[1], 1, 45), synthetic(commands[1], 0.65, 90)]
        common = source_common(donors, recovered)
        np.testing.assert_allclose(common, (c45 + c90) / 2, rtol=0, atol=1e-12)
        for t, truth in zip(donors + [synthetic(commands[1], 0.7, -115)],
                            [c45, c90, 0.7 * np.exp(-115j * np.pi / 180)]):
            fitted = fit_multiplier(response(t.command, recovered), t.motion[START:])
            np.testing.assert_allclose(fitted, truth, rtol=0, atol=1e-12)
            q = synthetic(commands[2], abs(truth), np.angle(truth, deg=True))
            np.testing.assert_allclose(fitted * response(q.command, recovered), q.motion[START:], rtol=0, atol=1e-14)
        for j, (setting, gain, deg) in enumerate(zip(HELD, (1.0, 0.65), (45.0, 90.0))):
            c = (c45, c90)
            row = score_case("synthetic", setting, 0, synthetic(commands[2], gain, deg),
                             recovered, c[j], c[1 - j], common, 0.02)
            np.testing.assert_allclose(row["query_mse"]["correct"], 0, rtol=0, atol=1e-24)
            np.testing.assert_allclose(row["correct_minus_swapped_yaw_deg"], (-45, 45)[j], atol=1e-12)
            np.testing.assert_allclose(row["yaw90_minus_yaw45_deg"], 45, atol=1e-12)
            if not (row["classified_correctly"] and row["query_rows"] == 13
                    and min(row["control_minus_correct_mse"].values()) > 0):
                raise AssertionError("synthetic classification, query pairing or row alignment failed")
    if wrap(179 - (-179)) != -2 or nearest(67.5) is not None or nearest(-179) != 90:
        raise AssertionError("circular distances or tie handling failed")
    print("Synthetic rotation self-test passed on Linux CPU.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=Path, help="exact H1 native02 collection/tapes.jsonl")
    parser.add_argument("--out", type=Path, help="fresh result directory")
    parser.add_argument("--self-test", action="store_true", help="synthetic numerical test only")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run the numerical self-test and evidence analysis on Linux CPU")
    if args.self_test:
        if args.tapes is not None or args.out is not None:
            parser.error("--self-test takes neither --tapes nor --out")
        self_test()
        return
    if args.tapes is None or args.out is None:
        parser.error("analysis requires --tapes PATH --out FRESH_DIRECTORY")
    if args.out.exists():
        parser.error(f"output directory already exists: {args.out}")

    tapes, receipt = load_tapes(args.tapes)
    source = [tapes[f, s, "support", i] for f, s, i in product(FAMILIES, SOURCE, range(12))]
    nominal = [tapes[f, "nominal", "support", i] for f, i in product(FAMILIES, range(12))]
    source_motion = np.concatenate([t.motion[START:] for t in source])
    scale = float(np.sqrt(xy_mse(source_motion)))
    if not scale > 0:
        raise ValueError("source XY motion scale is zero")
    draws = np.random.default_rng(BOOT_SEED).integers(8, size=(BOOT_DRAWS, 8))
    report = {
        "base": BASE, "estimator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input": receipt, "python": sys.version, "numpy": np.__version__, "argv": sys.argv[1:],
        "nominal_fit_supports": len(nominal), "nominal_fit_rows": sum(len(t.motion[START:]) for t in nominal),
        "common_fit_supports": len(source), "common_fit_rows": len(source_motion),
        "source_xy_rms_m": scale, "first_scored_step_zero_based": START,
        "bootstrap": {"unit": "paired layout index", "seed": BOOT_SEED, "draws": BOOT_DRAWS},
        "models": {name: evaluate(tapes, nominal, source, n, scale, draws)
                   for name, n in (("four_tap", 4), ("one_tap", 1))},
    }
    contents = json.dumps(report, indent=2, allow_nan=False) + "\n"
    args.out.mkdir(parents=True, exist_ok=False)
    tmp = args.out / "report.json.tmp"
    tmp.write_text(contents, encoding="utf-8")
    target = args.out / "report.json"
    tmp.replace(target)
    print(target)


if __name__ == "__main__":
    main()
