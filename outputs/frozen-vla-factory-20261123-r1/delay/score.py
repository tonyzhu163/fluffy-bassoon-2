"""Score fixed-budget timing adapters using NumPy and the committed evidence."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1]
SOURCE = OUTPUTS / "libero-xemb-friction-dev-20261119-r1" / "cpu01"
CAMERA = OUTPUTS / "symmeta-camera-command-20261123-r1" / "evidence"
CROSSARM = OUTPUTS / "symmeta-fk-crossarm-20261123-r1" / "evidence"
TAPS, MAX_LAG, BUDGET = 5, 16, 64
BOOTSTRAP_SEED = 20261123


def load_sources():
    """Only commands and wrist increments from the archived nominal tapes."""
    tapes = {}
    for p in sorted(SOURCE.glob("*-ordinary.npz")):
        suite, task, mode, _ = p.stem.split("-")
        with np.load(p) as data:
            u = data["issued"][:, :3].astype(float)
            y = np.diff(data["robot0_eef_pos"].astype(float), axis=0)
        assert u.shape == y.shape and len(y) >= 80
        tapes[(f"{suite}/{task}", mode)] = (u, y)
    assert len(tapes) == 16
    return tapes


def taps(u, t):
    return np.stack([u[t-k] for k in range(TAPS)], axis=-1)


def fit_kernel(tapes, excluded_task=None):
    x, y = [], []
    t = np.arange(TAPS-1, BUDGET)
    for (task, _), (u, motion) in tapes.items():
        if task != excluded_task:
            x.append(taps(u, t).reshape(-1, TAPS))
            y.append(motion[t].ravel())
    h, _, rank, _ = np.linalg.lstsq(np.concatenate(x), np.concatenate(y), rcond=None)
    assert rank == TAPS and h.sum() > 0
    return h


def gain_ls(x, y):
    energy = np.sum(x*x)
    assert energy > 0
    return max(0., float(np.sum(x*y) / energy))


def centered_cosine(x, yc):
    xc = x - x.mean(axis=0)
    return np.sum(xc*yc) / (np.linalg.norm(xc)*np.linalg.norm(yc))


def source_peak_offset(tapes, held_out):
    sample = np.arange(MAX_LAG + TAPS - 1, BUDGET)
    peaks = []
    for (task, _), (u, y) in tapes.items():
        if task == held_out:
            continue
        yc = y[sample] - y[sample].mean(axis=0)
        peaks.append(np.argmax([centered_cosine(u[sample-d], yc) for d in range(TAPS)]))
    return int(np.rint(np.median(peaks)))


def aligned_fit(u, support_y, h, peak_offset):
    """Select lag by residual or correlation. No target label arguments."""
    end = len(support_y)
    sample = np.arange(MAX_LAG + TAPS - 1, end)
    y = support_y[sample]
    yc = y - y.mean(axis=0)
    joint, unit, corr, calibrated_corr = [], [], [], []
    for lag in range(MAX_LAG+1):
        est = taps(u, sample-lag) @ h
        gain = gain_ls(est, y)
        joint.append((np.sum((y-gain*est)**2), lag, gain))
        unit.append((np.sum((y-est)**2), lag, 1.0))
        instant = u[sample-lag] * h.sum()
        # Same centered vector cosine as the repo timing rule, with matched rows.
        cosine = centered_cosine(instant, yc)
        corr.append((-cosine, lag, gain_ls(instant, y)))
        calibrated_corr.append((-centered_cosine(u[sample-lag-peak_offset], yc), lag, gain))
    return {"joint_fir": min(joint)[1:], "zero_fir": joint[0][1:],
            "unit_fir": min(unit)[1:], "xcorr_instant": min(corr)[1:],
            "xcorr_calibrated_fir": min(calibrated_corr)[1:]}


def summarize(rows, keys, metrics):
    """Equal-task means and paired task bootstrap, never resampling individual controls."""
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        group = tuple(r[k] for k in keys)
        grouped[group][r["task"]].append([r[m] for m in metrics])
    out = []
    for group, tasks in sorted(grouped.items()):
        means = np.array([np.mean(tasks[t], axis=0) for t in sorted(tasks)])
        n = len(means)
        indices = np.random.default_rng(BOOTSTRAP_SEED).integers(0, n, size=(10000, n))
        ci = np.percentile(means[indices].mean(axis=1), [2.5, 97.5], axis=0)
        estimates = {m: [round(float(x), 7) for x in (means[:,i].mean(), *ci[:,i])]
                     for i,m in enumerate(metrics)}
        out.append(dict(zip(keys, group), tasks=n, rows=sum(map(len, tasks.values())), **estimates))
    return out


def paired_changes(rows, group_keys, measures, reference, candidates):
    """Pair models before resampling. Positive values favor the reference."""
    cases = defaultdict(dict)
    for r in rows:
        cases[r["case"]][r["method"]] = r
    changes = []
    for models in cases.values():
        ref = models[reference]
        for alt in candidates:
            changes.append({**ref, "baseline": alt,
                            **{m: models[alt][m] - ref[m] for m in measures}})
    return summarize(changes, [*group_keys, "baseline"], measures)


def score_native(tapes, folder):
    receipt = json.loads((folder / "replay-receipt.json").read_text())
    assert len(receipt["cases"]) == 96
    tasks = sorted({t for t,_ in tapes})
    kernels = {t:fit_kernel(tapes, t) for t in tasks}
    offsets = {t:source_peak_offset(tapes, t) for t in tasks}
    records = []
    timing_truth = {"BASE": 0, "BASE_fifo5": 5, "S20": 14}
    seen = set()
    with np.load(folder / "wrist.npz") as states:
        assert len(states.files) == len(receipt["cases"])
        for r in receipt["cases"]:
            case, task, mode = r["key"], r["task"], r["mode"]
            identity = (task, mode, r["condition"], r["gain"])
            assert identity not in seen
            seen.add(identity)
            u = tapes[(task, mode)][0]
            y = np.diff(states[case].astype(float), axis=0)
            assert y.shape == (80,3) and np.isfinite(y).all()
            h = kernels[task]
            pred = aligned_fit(u[:BUDGET], y[:BUDGET], h, offsets[task])
            lag_true = timing_truth[r["condition"]]
            query = np.arange(BUDGET, len(y))
            for method, (lag, gain) in pred.items():
                if method == "xcorr_instant":
                    response = u[query-lag] * h.sum()
                else:
                    response = taps(u, query-lag) @ h
                rms = np.sqrt(np.mean(np.sum((y[query]-gain*response)**2, axis=-1)))
                records.append(dict(case=case, task=task, mode=mode, condition=r["condition"],
                                    target_gain=r["gain"], method=method, lag_hat=lag, gain_hat=gain,
                                    lag_exact=float(lag==lag_true), lag_mae=abs(lag-lag_true),
                                    gain_mae=abs(gain-r["gain"]), continuation_rms_mm=float(1000*rms)))
    err = ["lag_mae","gain_mae","continuation_rms_mm"]
    return dict(scores=summarize(records, ["condition","method"], ["lag_exact", *err]),
                improvements=paired_changes(records, ["condition"], err, "joint_fir",
                                            ["zero_fir","unit_fir","xcorr_instant","xcorr_calibrated_fir"]),
                joint_by_gain=summarize([r for r in records if r["method"]=="joint_fir"],
                                        ["condition","target_gain"], ["gain_hat","lag_exact","gain_mae"]),
                kernels={k:v.tolist() for k,v in sorted(kernels.items())}, source_peak_offsets=offsets)


def load_json_gz(path):
    with gzip.open(path) as stream:
        return json.load(stream)


def complex_xy(values):
    a = np.asarray(values, dtype=float)[:, :2]
    assert a.shape == (3, 2) and np.isfinite(a).all()
    return a[:,0] + 1j*a[:,1]


def fit_similarity(x, y):
    """Complex least squares is an isotropic gain and proper in-plane rotation."""
    den = np.vdot(x,x).real
    assert den > 0
    op = np.vdot(x,y) / den
    return float(np.sum(np.abs(x*op-y)**2)), op


def score_coarse(tapes):
    # These eight source tasks and all eight official target tasks are disjoint.
    source_tasks = {t for t,_ in tapes}
    chunk_response = 5 * fit_kernel(tapes).sum()
    world = load_json_gz(CAMERA / "world.json.gz")
    labels = {r["id"]:(r["true_scale"],r["true_yaw_deg"])
              for r in load_json_gz(CAMERA / "roll.json.gz")["rows"] if r["roll_deg"]==0}
    panels = {"UR5e/pi05":[(r, r["issued_mean"], r["delta_world_m"]) for r in world["rows"]]}
    for path in sorted(CROSSARM.glob("*-cases.json.gz")):
        d = load_json_gz(path)
        panels[d["arm"]+"/"+d["policy"]] = [(r,r["issued_xy"],r["world_delta_xy_m"])
                                              for r in d["rows"] if r["fk_available"]]
    rows = []
    for panel, supports in panels.items():
        for row, issued, motion in supports:
            task = f'{row["suite"]}/{row["task_id"]}'
            assert task not in source_tasks
            x, y = complex_xy(issued) * chunk_response, complex_xy(motion)
            for shift in (0,1):
                # Both labels give two valid endpoints at t=1,2; no padding cue.
                observed = y[1-shift:3-shift]
                candidates = [fit_similarity(x[1-d:3-d], observed) for d in (0,1)]
                best = int(np.argmin([sse for sse,_ in candidates]))
                # Reader has committed its decisions before scoring any native label.
                scale,yaw = labels[f'{task}/{row["embodiment"]}']
                for method,lag in (("joint_lag_similarity",best),("zero_lag_similarity",0)):
                    op = candidates[lag][1]
                    err = abs((np.rad2deg(np.angle(op))-yaw+180) % 360-180)
                    rows.append(dict(case=(panel,task,row["index"],shift), task=task,
                                     shift=shift, panel=panel, method=method,
                                     lag_exact=float(lag==shift), lag_mae=abs(lag-shift),
                                     gain_mae=abs(abs(op)-scale), yaw_mae_deg=err))
    assert sum(len(s) for s in panels.values()) == 360
    errors = ["lag_mae","gain_mae","yaw_mae_deg"]
    return dict(panda_calibration_metres_per_mean_command=float(chunk_response),
                scores=summarize(rows, ["shift","method"], ["lag_exact", *errors]),
                panel_scores=summarize(rows, ["panel","shift","method"], ["lag_exact",*errors]),
                improvements=paired_changes(rows, ["shift"], errors,
                                            "joint_lag_similarity", ["zero_lag_similarity"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE)
    parser.add_argument("--output", type=Path, default=HERE/"metrics.json")
    args = parser.parse_args()
    tapes = load_sources()
    output = dict(baseline="757a44cd4374833cad790852df8b87ddf412e638",
                  bootstrap_seed=BOOTSTRAP_SEED, interval_order=["mean","p2.5","p97.5"],
                  full_rate=score_native(tapes, args.input), chunk_rate=score_coarse(tapes))
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
