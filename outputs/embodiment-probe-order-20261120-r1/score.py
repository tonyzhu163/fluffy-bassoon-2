"""Score the frozen Panda/Sawyer probe-order comparison on Linux CPU."""

import argparse
import hashlib
import importlib.util
import json
from itertools import product
from pathlib import Path
import re
import sys
from typing import NamedTuple

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = "4bd494d188bfdcb757c554bb335b5c19d3bc97e6"
PRIOR = HERE.parent / "embodiment-tape-observability-20261120-r1"
H1 = HERE.parent / "embodiment-command-frame-20261120-r1"
REPORTS = PRIOR / "evidence" / "cpu01"
FAMILIES = ("Panda", "Sawyer")
YAWS = (45, 90)
GAINS = (0.65, 1.0)
SCHEDULES = ("blocked", "shuffled")
INDICES = range(16)
READERS = ("physical", "lag_ridge")
YAW_HALF, GAIN_HALF = 22.5, 0.175
BOOT_SEED, BOOT_DRAWS = 20261120, 10_000
PLACEMENT = ("initial_state_sha256", "scene_hash", "movable_xy_hash", "mobile_joint_xy_hash")
SETTLED = ("initial_state_fingerprint", "post_settle_scene_fingerprint")
REPORT_HASHES = {
    "estimate": "b6acf89b4e5f245391e886d5a0ebb01351d24687737cd508b2171417fe501c59",
    "factor": "8dcd9a2fb1eb3e005c82744e21fc4bd5f9b13aebf7092e9bbe2783ffc321152c",
}


def pinned_module(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"source differs from {BASE}: {path}")
    spec = importlib.util.spec_from_file_location(f"probe_order_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


estimate = pinned_module(PRIOR / "estimate.py", "4ee95d7ebe62415699d0da755ff613a01fbf46f943f8c29604df4bfbdfd88eef")
factor = pinned_module(PRIOR / "factor.py", "79a084f75928219ec4c995af8c6ed4aecbe5471193e244f75b2efa739f2d7dcd")
h1 = pinned_module(H1 / "collect.py", "018e71cc0b5819c1e294d91e087ae0e3e64512d8e29fe4c122e09298c49fe537")


class Frozen(NamedTuple):
    fir: np.ndarray
    common: complex
    xy_rms: float
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    intercept: np.ndarray


def load_frozen():
    reports = {}
    for name, digest in REPORT_HASHES.items():
        path = REPORTS / name / "report.json"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"frozen report differs from {BASE}: {path}")
        reports[name] = json.loads(raw)
    response = reports["estimate"]["models"]["four_tap"]
    fit = reports["factor"]["readers"]["lags_0_to_3"]["fit"]
    if (response["lags"] != list(range(4)) or fit["lags"] != list(factor.LAGS)
            or reports["estimate"]["first_scored_step_zero_based"] != estimate.START):
        raise ValueError("frozen functions and report windows disagree")
    return Frozen(np.array(response["fir_m_per_command"]), complex(*response["fixed_common_complex"]),
                  reports["estimate"]["source_xy_rms_m"], np.array(fit["feature_mean"]),
                  np.array(fit["feature_scale"]), np.array(fit["weights"]), np.array(fit["intercept_re_im"]))


def layout_seed(index, schedule):
    return (991300 if schedule == "query" else 991200) + index


def permutation(index):
    return np.random.default_rng(2991200 + index).permutation(16)


def expected_commands(index, schedule):
    if schedule == "query":
        return h1.probe(2991300 + index, False)
    directions = np.array([[0.39, 0], [-0.39, 0], [0, 0.39], [0, -0.39]])
    commands = np.zeros((16, 7))
    commands[:, :2] = np.repeat(directions, 4, axis=0)
    commands[:, 6] = -1
    return commands if schedule == "blocked" else commands[permutation(index)]


def fingerprints(row, fields):
    result = tuple(row[k] for k in fields)
    if not all(isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h) for h in result):
        raise ValueError(f"invalid SHA-256 in {fields}")
    return result


def check_bounds(commands):
    if np.any(np.abs(commands) > 1):
        raise ValueError("action outside native [-1, 1] bounds")


def reset_receipt(row, placement, family, seed):
    reset = row["reset"]
    physics = reset["environment_reproducibility"]
    if (reset["model_identity"]["arm"] != family or physics["layout_seed"] != seed
            or ("layout_id" in row and physics["source_index"] != row["layout_id"])
            or physics["generated_row_fingerprint"] != placement[0] or physics["scene_hash"] != placement[1]):
        raise ValueError("native reset does not identify the recorded family and donor layout")
    settled = fingerprints(physics, SETTLED)
    wrist = np.asarray(row["wrist_world_m"])
    if wrist.shape != (17, 3) or wrist.dtype.kind not in "fiu" or not np.isfinite(wrist).all():
        raise ValueError("invalid native wrist observations")
    opening = np.asarray(row["opening_state"])
    if (opening.shape != (8,) or opening.dtype.kind not in "fiu" or not np.isfinite(opening).all()
            or not np.array_equal(opening[:3], wrist[0])):
        raise ValueError("invalid public reset opening")
    check_bounds(estimate.matrix(row, "issued_actions", 7))
    if row["worker_step_index"] != list(range(1, 17)):
        raise ValueError("expected native control indices 1 through 16")
    for field in ("rewards", "success", "terminated", "truncated"):
        if not isinstance(row[field], list) or len(row[field]) != 16:
            raise ValueError(f"expected 16 native {field} entries")
    if any(row["terminated"][:-1]) or any(row["truncated"][:-1]):
        raise ValueError("native episode ended before control 16")
    return settled + (tuple(opening),)


def load_panel(path):
    expected = set(product(FAMILIES, INDICES, YAWS, GAINS, (*SCHEDULES, "query")))
    tapes, placements, resets, public = {}, {}, {}, {}
    raw = path.read_bytes()
    for line_no, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            family, i, yaw, gain, schedule = key = tuple(
                row[k] for k in ("family", "index", "yaw_deg", "gain", "schedule")
            )
            if type(i) is not int or isinstance(gain, bool) or key not in expected or key in tapes:
                raise ValueError(f"unknown or duplicate cell {key}")
            seed = layout_seed(i, schedule)
            if row["layout_seed"] != seed or row["role"] != ("query" if schedule == "query" else "support"):
                raise ValueError(f"wrong seed or role for {key}")
            expected_probe_seed = None if schedule == "blocked" else seed + 2000000
            if "probe_seed" in row and row["probe_seed"] != expected_probe_seed:
                raise ValueError(f"wrong probe seed for {key}")
            if (schedule == "shuffled" and "blocked_row_indices" in row
                    and row["blocked_row_indices"] != permutation(i).tolist()):
                raise ValueError(f"wrong permutation receipt for {key}")
            cmd, motion = estimate.matrix(row, "commands", 7), estimate.matrix(row, "motion", 3)
            check_bounds(cmd)
            target = expected_commands(i, schedule)
            # The registered .39 may be serialized directly or after conversion to native float32.
            if not (np.array_equal(cmd, target) or np.array_equal(cmd, target.astype(np.float32))):
                raise ValueError(f"commands differ from the registered {schedule} probe")
            if not np.array_equal(public.setdefault((i, schedule), cmd), cmd):
                raise ValueError(f"arms/settings do not share the same {schedule} commands")
            placement = fingerprints(row, PLACEMENT)
            if placements.setdefault(seed, placement) != placement:
                raise ValueError(f"unmatched donor layout at seed {seed}")
            opening = reset_receipt(row, placement, family, seed)
            if resets.setdefault((family, seed), opening) != opening:
                raise ValueError(f"unmatched settled reset at {(family, seed)}")
            # Store only the two public XY signals. Metadata is exclusively a pairing key.
            tapes[key] = (cmd[:, :2], motion[:, :2])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if expected != tapes.keys():
        missing = sorted(expected - tapes.keys())
        raise ValueError(f"missing {len(missing)} registered tapes, first missing: {missing[:1]}")
    for j, field in enumerate(PLACEMENT):
        if len({p[j] for p in placements.values()}) != 32:
            raise ValueError(f"expected 32 distinct donor layouts for {field}")
    for i in INDICES:
        if not np.array_equal(public[i, "shuffled"], public[i, "blocked"][permutation(i)]):
            raise ValueError(f"shuffled schedule does not permute the exact blocked rows at index {i}")
    counts = {s: sum(key[-1] == s for key in tapes) for s in (*SCHEDULES, "query")}
    return tapes, dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest(),
                       tapes=len(tapes), actions=16 * len(tapes), tapes_by_schedule=counts,
                       unique_query_rows_scored=counts["query"] * (16 - estimate.START),
                       layout_seeds=len(placements), family_layout_resets=len(resets))


def as_complex(xy):
    return xy[:, 0] + 1j * xy[:, 1]


def read_support(command_xy, motion_xy, frozen):
    predicted = estimate.response(as_complex(command_xy), frozen.fir)
    physical = estimate.fit_multiplier(predicted, as_complex(motion_xy)[estimate.START:])
    x = np.asarray(factor.features(command_xy, motion_xy))
    ridge = frozen.intercept + ((x - frozen.mean) / frozen.scale) @ frozen.weights
    return dict(physical=complex(physical), lag_ridge=complex(*ridge))


def query_metrics(multipliers, baseline, measured, frozen, yaw, gain):
    mse = {key: estimate.xy_mse(value * baseline - measured) / frozen.xy_rms**2
           for key, value in multipliers.items()}
    matched = multipliers["matching"]
    angle = estimate.yaw(matched)
    yaw_donor = estimate.yaw(multipliers["yaw_swap"])
    gain_donor = abs(multipliers["gain_swap"])
    yaw_gap = estimate.wrap(angle - yaw_donor if yaw == 90 else yaw_donor - angle)
    gain_gap = abs(matched) - gain_donor if gain == 1 else gain_donor - abs(matched)
    yaw_error = estimate.wrap(angle - yaw)
    gain_error = abs(matched) - gain
    yaw_close, gain_close = abs(yaw_error) < YAW_HALF, abs(gain_error) < GAIN_HALF
    return {
        **{f"{k}_mse": v for k, v in mse.items()},
        **{f"{k}_minus_matching_mse": v - mse["matching"] for k, v in mse.items() if k != "matching"},
        "yaw_error_deg": yaw_error, "gain_error": gain_error,
        "abs_yaw_error_deg": abs(yaw_error), "abs_gain_error": abs(gain_error),
        "yaw90_minus_yaw45_deg": yaw_gap, "gain1_minus_gain065": gain_gap,
        "abs_yaw_gap_error_deg": abs(estimate.wrap(yaw_gap - 45)), "abs_gain_gap_error": abs(gain_gap - 0.35),
        "yaw_below_half_spacing": yaw_close, "gain_below_half_spacing": gain_close,
        "both_below_half_spacing": yaw_close and gain_close,
    }


def score_reader(reader, tapes, estimates, frozen):
    cases = []
    for family, i, yaw, gain in product(FAMILIES, INDICES, YAWS, GAINS):
        other_yaw = 45 if yaw == 90 else 90
        other_gain = 0.65 if gain == 1 else 1.0
        command, motion = tapes[family, i, yaw, gain, "query"]
        baseline = estimate.response(as_complex(command), frozen.fir)
        measured = as_complex(motion)[estimate.START:]
        schedules = {}
        for schedule in SCHEDULES:
            maps = {
                "matching": estimates[family, i, yaw, gain, schedule][reader],
                "yaw_swap": estimates[family, i, other_yaw, gain, schedule][reader],
                "gain_swap": estimates[family, i, yaw, other_gain, schedule][reader],
                "common": frozen.common, "nominal": 1 + 0j,
                "ideal": gain * np.exp(1j * np.deg2rad(yaw)),
            }
            metrics = query_metrics(maps, baseline, measured, frozen, yaw, gain)
            schedules[schedule] = dict(multipliers_re_im={k: [v.real, v.imag] for k, v in maps.items()},
                                       estimated_yaw_deg=estimate.yaw(maps["matching"]),
                                       estimated_gain=abs(maps["matching"]), metrics=metrics)
        blocked, shuffled = (schedules[s]["metrics"] for s in SCHEDULES)
        delta = {k: float(blocked[k]) - float(shuffled[k]) for k in blocked}
        cases.append(dict(family=family, index=i, yaw_deg=yaw, gain=gain,
                          support_layout_seed=layout_seed(i, "blocked"), query_layout_seed=layout_seed(i, "query"),
                          yaw_swap_donor_yaw_deg=other_yaw, gain_swap_donor_gain=other_gain,
                          query_rows=len(measured), **schedules, blocked_minus_shuffled=delta))
    return cases


def bootstrap_counts():
    draws = np.random.default_rng(BOOT_SEED).integers(16, size=(BOOT_DRAWS, 16))
    return np.column_stack([(draws == i).sum(axis=1) for i in INDICES])


def summarize(rows, counts):
    # One row per index/cell, retaining both schedules and all six predictions together.
    metrics = list(rows[0]["blocked"]["metrics"])
    groups = [[r for r in rows if r["index"] == i] for i in INDICES]
    layout_means = []
    for group in groups:
        values = []
        for row in group:
            conditions = [row[s]["metrics"] for s in SCHEDULES] + [row["blocked_minus_shuffled"]]
            values.append([[c[k] for k in metrics] for c in conditions])
        layout_means.append(np.mean(values, axis=0))
    matrix = np.asarray(layout_means)
    means = matrix.mean(axis=0)
    boot = (counts @ matrix.reshape(16, -1) / 16).reshape(-1, *means.shape)
    low, high = np.percentile(boot, [2.5, 97.5], axis=0)
    summary = dict(layout_indices=16, cells_per_index=len(groups[0]), unique_queries=len(rows))
    for j, schedule in enumerate((*SCHEDULES, "blocked_minus_shuffled")):
        summary[schedule] = {k: dict(mean=float(means[j, n]), ci95=[float(low[j, n]), float(high[j, n])])
                             for n, k in enumerate(metrics)}
    return summary


def slices(rows, counts):
    result = []
    for family, yaw, gain in product((None, *FAMILIES), (None, *YAWS), (None, *GAINS)):
        selected = [r for r in rows if (family is None or r["family"] == family)
                    and (yaw is None or r["yaw_deg"] == yaw) and (gain is None or r["gain"] == gain)]
        result.append(dict(family=family, yaw_deg=yaw, gain=gain, **summarize(selected, counts)))
    return result


def self_test():
    # Generate observations with real X/Y equations and per-axis convolution,
    # independently of the scorer's complex arithmetic and lag matrix.
    def observed(command, yaw, gain, fir):
        angle = np.deg2rad(yaw)
        x, y = (np.convolve(axis, fir)[:16] for axis in command.T)
        return gain * np.column_stack((np.cos(angle) * x - np.sin(angle) * y,
                                       np.sin(angle) * x + np.cos(angle) * y))

    def truth(yaw, gain):
        return [gain * np.cos(np.deg2rad(yaw)), gain * np.sin(np.deg2rad(yaw))]

    frozen = Frozen(np.array([0.004, 0.003, -0.001, 0.0005]), 0.3 + 0.2j, 0.002,
                    np.arange(8) / 128, np.arange(1, 9) / 16, np.zeros((8, 2)), np.array([0.2, -0.1]))
    # Make real and imaginary outputs depend on different lag components.
    frozen.weights[0, 1], frozen.weights[5, 0] = -1.5, 2.0
    factor.check_features()
    check_bounds(np.array([-1, 1]))
    for sign in (-1, 1):
        try:
            check_bounds(np.array([sign * np.nextafter(1.0, 2.0)]))
        except ValueError:
            continue
        raise AssertionError("action outside the inclusive native bound was accepted")

    tapes, estimates = {}, {}
    for arm, i, yaw, gain, schedule in product(FAMILIES, INDICES, YAWS, GAINS, (*SCHEDULES, "query")):
        key = (arm, i, yaw, gain, schedule)
        cmd = expected_commands(i, schedule)[:, :2].astype(float)
        # Give each support block a recognizable gain and give shuffled a yaw offset.
        extra = 0 if schedule == "query" else (i + 1 + 16 * (arm == "Sawyer")) / 1000
        angle = yaw + 10 * (schedule == "shuffled")
        motion = observed(cmd, angle, gain + extra, frozen.fir)
        if schedule == "query":
            motion[:estimate.START] = 100  # Fails if the three warm-up labels enter any query score.
        else:
            readings = read_support(cmd, motion, frozen)
            estimates[key] = readings
            c = readings["physical"]
            np.testing.assert_allclose([c.real, c.imag], truth(angle, gain + extra), atol=1e-13, rtol=0)
            x = factor.features(cmd, motion)
            ridge_expected = [0.2 + 2 * (x[5] - 5 / 128) / (6 / 16),
                              -0.1 - 1.5 * (x[0] - 0 / 128) / (1 / 16)]
            ridge = readings["lag_ridge"]
            np.testing.assert_allclose([ridge.real, ridge.imag], ridge_expected, atol=1e-13, rtol=0)
        tapes[key] = (cmd, motion)

    rows = score_reader("physical", tapes, estimates, frozen)
    for row in rows:
        arm, i, yaw, gain = (row[k] for k in ("family", "index", "yaw_deg", "gain"))
        extra = (i + 1 + 16 * (arm == "Sawyer")) / 1000
        cmd, motion = tapes[arm, i, yaw, gain, "query"]
        for schedule in SCHEDULES:
            angle = yaw + 10 * (schedule == "shuffled")
            swap_angle = (135 - yaw) + 10 * (schedule == "shuffled")
            swap_gain = 1.0 if gain == 0.65 else 0.65
            metrics = row[schedule]["metrics"]
            for name, deg, amplitude in (("matching", angle, gain + extra), ("yaw_swap", swap_angle, gain + extra),
                                         ("gain_swap", angle, swap_gain + extra), ("ideal", yaw, gain)):
                predicted = observed(cmd, deg, amplitude, frozen.fir)
                expected_mse = np.mean((predicted[3:] - motion[3:])**2) / frozen.xy_rms**2
                np.testing.assert_allclose(metrics[f"{name}_mse"], expected_mse, atol=1e-14, rtol=1e-12)
                np.testing.assert_allclose(row[schedule]["multipliers_re_im"][name], truth(deg, amplitude),
                                           atol=1e-13, rtol=0)
            np.testing.assert_allclose([metrics["yaw90_minus_yaw45_deg"], metrics["gain1_minus_gain065"],
                                        metrics["abs_yaw_error_deg"], metrics["abs_gain_error"]],
                                       [45, 0.35, angle - yaw, extra], atol=1e-12, rtol=0)
    # A free complex multiplier minimizes the synthetic support residual even if
    # there is unmodelled motion. The true-map diagnostic on a query has no such bound.
    cmd, motion = tapes["Panda", 1, 90, 0.65, "blocked"]
    noisy = motion.copy()
    noisy[6:10, 0] += 0.002
    best = read_support(cmd, noisy, frozen)["physical"]
    z = estimate.response(as_complex(cmd), frozen.fir)
    d = as_complex(noisy)[3:]
    offset = -0.13 + 0.27j
    cost_gap = estimate.xy_mse((best + offset) * z - d) - estimate.xy_mse(best * z - d)
    np.testing.assert_allclose(cost_gap, abs(offset)**2 * estimate.xy_mse(z), rtol=1e-12, atol=0)

    summary = summarize(rows, bootstrap_counts())
    delta = summary["blocked_minus_shuffled"]
    np.testing.assert_allclose(delta["abs_yaw_error_deg"]["ci95"], [-10, -10], atol=1e-12)
    if delta["matching_mse"]["ci95"][1] >= 0 or delta["ideal_mse"]["ci95"] != [0, 0]:
        raise AssertionError("paired schedule contrast lost its orientation or its shared query")
    print("Synthetic Linux CPU self-test passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tapes", type=Path, help="completed JSONL (default: collection/tapes.jsonl beside script)")
    parser.add_argument("--out", type=Path, help="fresh output directory (default: score beside script)")
    parser.add_argument("--self-test", action="store_true", help="exercise synthetic signals only")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("run the self-test and scoring on Linux CPU")
    if args.self_test:
        if args.tapes is not None or args.out is not None:
            parser.error("--self-test takes neither --tapes nor --out")
        self_test()
        return
    out = args.out if args.out is not None else HERE / "score"
    if out.exists():
        parser.error(f"output must be fresh: {out}")
    frozen = load_frozen()
    tapes, receipt = load_panel(args.tapes if args.tapes is not None else HERE / "collection" / "tapes.jsonl")
    estimates = {key: read_support(*signals, frozen) for key, signals in tapes.items() if key[-1] in SCHEDULES}
    counts = bootstrap_counts()
    readers = {}
    for name in READERS:
        cases = score_reader(name, tapes, estimates, frozen)
        readers[name] = dict(cases=cases, summary=slices(cases, counts))
    physical = {(s["family"], s["yaw_deg"], s["gain"]): s for s in readers["physical"]["summary"]}
    by_arm = {a: physical[a, None, None]["blocked_minus_shuffled"]["matching_mse"]["mean"] for a in FAMILIES}
    pooled = physical[None, None, None]["blocked_minus_shuffled"]["matching_mse"]
    report = dict(design_base=BASE, scorer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  input=receipt, source_reports_sha256=REPORT_HASHES,
                  source_xy_rms_m=frozen.xy_rms, first_query_row=estimate.START, last_query_row=15,
                  bootstrap=dict(unit="shared layout index", indices=16, draws=BOOT_DRAWS, seed=BOOT_SEED),
                  half_spacings=dict(yaw_deg=YAW_HALF, gain=GAIN_HALF),
                  physical_probe_order=dict(blocked_minus_shuffled_matching_mse_by_arm=by_arm, pooled=pooled,
                                            prefer_blocked=all(v < 0 for v in by_arm.values()) and pooled["ci95"][1] < 0),
                  readers=readers, python=sys.version, numpy=np.__version__, argv=sys.argv[1:])
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "report.json.tmp"
    tmp.write_text(payload, encoding="utf-8")
    target = out / "report.json"
    tmp.replace(target)
    print(target)


if __name__ == "__main__":
    main()
