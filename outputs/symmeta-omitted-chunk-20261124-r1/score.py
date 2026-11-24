"""Open evaluation labels after verifying an independently addressable frozen prediction seal."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import judge as j

BOOTSTRAPS = 2000
BOOT_SEED = 20261124
METHODS = ("adaptive", "tuned", "constant", "midpoint", "geom", "hard_yaw", "oracle")
COMPARISONS = ("tuned", "constant", "midpoint", "geom", "oracle")
HARD_DELTA = "hard_minus_constant_concordance"
PRIMARY = ("mae_adaptive_minus_tuned", "mae_adaptive_minus_constant", HARD_DELTA)


def observations(data, label, pred, gate):
    y = label["true_yaw"]
    e = {name: np.abs(j.wrap(pred[name] - y)) for name in METHODS if name != "oracle"}
    e["oracle"] = np.minimum(e["tuned"], e["geom"])
    # Stable equivalent of ||(g/ghat)R(y−yhat)−I||_F / sqrt(2).
    ratio = label["true_gain"] / pred["tuned_gain"]
    op = {k: np.sqrt((ratio - 1)**2 + 4 * ratio * np.sin(np.deg2rad(v) / 2)**2)
          for k, v in e.items()}
    cols = {}
    for metric, errors in (("mae", e), ("operator", op)):
        for name in METHODS:
            cols[f"{metric}_{name}"] = errors[name]
        for name in COMPARISONS:
            cols[f"{metric}_adaptive_minus_{name}"] = errors["adaptive"] - errors[name]
    unique = data["full_valid"].all(axis=1) & (e["geom"] != e["tuned"])
    win_g = e["geom"] < e["tuned"]
    hard = np.where(unique, (pred["hard"] == win_g).astype(float), np.nan)
    constant = np.where(unique, (gate["hard_default"] == win_g).astype(float), np.nan)
    cols["hard_concordance"] = hard
    cols["constant_concordance"] = constant
    cols[HARD_DELTA] = hard - constant
    cols["gripper_accuracy"] = (pred["tuned_sign"] == label["true_sign"]).astype(float)
    return cols


def average(cols, mass):
    # Conditional concordance is the ratio of weighted correct and uniquely adjudicable rows.
    result = {}
    for k, x in cols.items():
        ok = np.isfinite(x)
        denom = mass[ok].sum()
        result[k] = float(mass[ok] @ x[ok] / denom) if denom > 0. else np.nan
    return result


def json_means(x):
    return {k: v if np.isfinite(v) else None for k, v in x.items()}


def interval(point, draws):
    out = {}
    for name, value in point.items():
        a = np.array([rep[name] for rep in draws])
        ok = np.isfinite(a)
        out[name] = dict(mean=value if np.isfinite(value) else None, draws=int(ok.sum()),
                         ci95=np.quantile(a[ok], [.025, .975]).tolist() if ok.any() else None)
    return out


def signflip(cols, mass, tasks):
    # Under symmetric independent task signs, flip the paired numerator per task.
    # For concordance, each flip leaves its observed eligible mass unchanged.
    out = {}
    for key in PRIMARY:
        x = np.where(np.isfinite(cols[key]), cols[key], 0.)
        by_task = np.bincount(tasks, weights=x * mass)
        signs = np.array(list(itertools.product((-1., 1.), repeat=len(by_task))))
        ref = np.abs(signs @ by_task)
        observed = abs(by_task.sum())
        p = np.mean(ref >= observed - 64 * np.finfo(float).eps * max(1., observed))
        out[key] = dict(two_sided_exact_p=float(p), assignments=len(ref),
                        exchangeable_unit="task under a symmetric paired null",
                        observed_eligible_rows=int(np.isfinite(cols[key]).sum())) if np.isfinite(cols[key]).any() else None
    return out


def coverage(data, d, pred, lab=None):
    _, ec = np.unique(data["seed"], return_counts=True)
    r = dict(rows=len(d), tasks=len(set(data["task"])), arms=sorted(set(data["arm"])),
             physical_traces=len(set(data["physical_trace_id"])), case_ids=len(set(data["case_id"])),
             seed_counts=dict(zip(map(str, sorted(set(data["seed"]))), map(int, ec), strict=True)),
             valid_fold_decoders_by_slot=data["loo_valid"].sum(axis=0).tolist(),
             valid_full_decoders=data["full_valid"].sum(axis=0).tolist(),
             valid_visual_command_chunks=int((j.unit(data["flow"])[1] & j.unit(data["cmd"])[1] & data["valid"]).sum()),
             scoreable=int(np.isfinite(d).sum()), equal_scores=int(np.sum(d == 0.)),
             zero_score_or_invalid=int((~np.isfinite(d) | (d == 0.)).sum()),
             invalid_score_with_constant_fallback=int((~np.isfinite(d) & pred["blend_ok"]).sum()),
             undefined_arc_with_tuned_fallback=int((~pred["blend_ok"]).sum()),
             equal_full_endpoints=int((j.endpoints(data)[1][pred["blend_ok"]] == 0.).sum()),
             mixing_weight_min_mean_max=[float(f(pred["weight"])) for f in (np.min, np.mean, np.max)])
    both = data["loo_valid"] & data["full_valid"][:, None, :]
    changed = both & (j.wrap(data["loo_yaw"].astype(float) - data["full_yaw"][:, None, :]) != 0.)
    r.update(identified_full_vs_loo_by_slot=both.sum(axis=0).tolist(),
             changed_full_vs_loo_by_slot=changed.sum(axis=0).tolist(),
             all_three_omissions_leave_yaw_identical=(both.all(axis=1) & ~changed.any(axis=1)).sum(axis=0).tolist(),
             zero_flow_chunks=int((~j.unit(data["flow"])[1]).sum()),
             zero_cmd_chunks=int((~j.unit(data["cmd"])[1]).sum()))
    if lab is not None:
        err = np.abs(j.wrap(data["full_yaw"] - lab["true_yaw"][:, None]))
        good = data["full_valid"].all(axis=1)
        r["endpoint_error_ties"] = int((good & (err[:, 0] == err[:, 1])).sum())
        r["hard_eligible"] = int((good & (err[:, 0] != err[:, 1])).sum())
    return r


def verify(out, seal_sha, runtime):
    j.require(j.digest(out / "seal.json") == seal_sha, "Pre-label seal digest mismatch")
    seal = json.loads((out / "seal.json").read_bytes())
    j.require(runtime["code_sha256"] == seal["runtime"]["code_sha256"], "Frozen scoring code changed")
    for k, sha in seal["outputs"].items():
        j.require(j.digest(out / k) == sha, f"Sealed {k} changed")
    parts = {r: j.load_arrays(out / f"compact-{r}.npz") for r in j.ROLES}
    p = j.load_arrays(out / "predictions.npz")
    j.assert_disjoint(parts)
    terms = j.fit_terms(parts["fit"])
    matrix = j.projection(terms, np.ones(len(terms[0])))
    np.testing.assert_array_equal(matrix, seal["matrix"])
    _, dcal = j.residuals(parts["cal"], matrix)
    errors, d = j.residuals(parts["eval"], matrix)
    _, _, mass = j.equal_units(parts["cal"])
    gate = j.calibrate(j.cal_cache(parts["cal"]), dcal, mass)
    j.require(gate == seal["gate"], "Mismatch reproducing calibration")
    for k, v in {**j.decide(parts["eval"], d, gate), "fold_error": errors, "advantage": d,
                 **{k: parts["eval"][k] for k in (*j.ID, "seed")}}.items():
        np.testing.assert_array_equal(p[k], v)
    return seal, parts, p, terms, dcal


def statistic(parts, p, terms, dcal, label, gate, draws=BOOTSTRAPS):
    cal, ev = parts["cal"], parts["eval"]
    cf, ct, cu = j.equal_units(cal)
    ef, et, eu = j.equal_units(ev)
    cc = j.cal_cache(cal)
    obs = observations(ev, label, p, gate)
    point = average(obs, eu)
    rng = np.random.default_rng(BOOT_SEED)
    joint, conditional = [], []
    singular = zero_scale = zero_slope = 0
    for _ in range(draws):
        nf, nc, ne = len(terms[0]), len(cf), len(ef)
        fc, gc, ec = [np.bincount(rng.integers(n, size=n), minlength=n) for n in (nf, nc, ne)]
        matrix = j.projection(terms, fc)
        singular += int(not np.any(matrix))
        _, dc = j.residuals(cal, matrix)
        gate_rep = j.calibrate(cc, dc, cu * gc[ct])
        zero_scale += int(gate_rep["s"] == 0.)
        zero_slope += int(gate_rep["b"] == 0.)
        _, de = j.residuals(ev, matrix)
        pred = j.decide(ev, de, gate_rep)
        emass = eu * ec[et]
        joint.append(average(observations(ev, label, pred, gate_rep), emass))
        conditional.append(average(obs, emass))
    joint_ci = interval(point, joint)
    def group_report(field, values):
        return {str(k): json_means(average(obs, eu * (field == k))) for k in values}
    verdict = all(joint_ci[k]["draws"] == draws and (joint_ci[k]["ci95"][0] > 0.
                  if k == HARD_DELTA else joint_ci[k]["ci95"][1] < 0.) for k in PRIMARY)
    d = p["advantage"]
    return dict(reps=draws, seed=BOOT_SEED, joint_refit=joint_ci, test_only=interval(point, conditional),
                singular_refit_draws=singular, zero_scale_draws=zero_scale, zero_slope_draws=zero_slope,
                sign_flip=signflip(obs, eu, et), registered_advancement_supported=bool(verdict),
                per_task=group_report(ev["task"], ef), per_seed=group_report(ev["seed"], j.SEEDS),
                per_arm=group_report(ev["arm"], sorted(set(ev["arm"]))),
                fit=dict(task_ids=sorted(set(parts["fit"]["task"])), physical_traces=len(set(parts["fit"]["physical_trace_id"])),
                         case_ids=len(parts["fit"]["case_id"]), valid_chunks=int(terms[2].sum()),
                         total_chunks=int(terms[2].size)),
                cal=coverage(cal, dcal, j.decide(cal, dcal, gate), cal),
                eval=coverage(ev, d, p, label))


def run(frozen, sha, label_path, out, runtime):
    seal, parts, p, terms, dcal = verify(frozen, sha, runtime)
    out.mkdir(parents=True, exist_ok=False)
    # Opening evaluation labels is downstream of seal verification and exact prediction replay.
    label_sha = j.digest(label_path)
    j.require(label_sha == seal["expected_eval_label_sha256"], "Evaluation label file differs from source seal")
    label = j.labels(label_path, parts["eval"])
    report = statistic(parts, p, terms, dcal, label, seal["gate"])
    report.update(runtime=runtime, seal_sha256=sha, labels_sha256=label_sha, gate=seal["gate"],
                  probes=seal["probes"], registered_tasks=seal["registered_tasks"])
    j.write_json(out / "metrics.json", report)
    for k in PRIMARY:
        print(k, report["joint_refit"][k])
    print("Registered advancement supported:", report["registered_advancement_supported"])
    print("Metrics SHA-256", j.digest(out / "metrics.json"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pin", required=True)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--seal-sha", required=True)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="fresh result directory")
    args = parser.parse_args()
    run(args.frozen, args.seal_sha, args.labels, args.out, j.execution(args.pin))


if __name__ == "__main__":
    main()
