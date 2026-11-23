"""Source-only model selection and independent native-arm forward-response scoring."""
import argparse
from collections import Counter
from hashlib import sha256
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from motion import correction_matrix, fit_source, fit_target, map_action

HERE = Path(__file__).resolve().parent
SCALE = "libero-xemb-ur5e-scale-20261120-r1"
CAMPAIGN = "xemb-icml-campaign-20261121-r1"
ARCHIVE_SHA = "37f062958cb63b962ceefb760f0592fdb120da966369ee308b509e2d82449076"
PANELS = {
    "UR5e": (SCALE, "gpu09test", "test_live", "5dd5fa914bbbfe25b9cf4ca5fa504c8a472e997b"),
    "IIWA": (CAMPAIGN, "gpu09iiwa", "live", "490f25c5716aa067199c5b65e32caf6cd15421a4"),
    "Kinova3": (CAMPAIGN, "gpu06kinova3", "live", "465f33dc657ceab341dbceaf6b98355b94810059"),
}
STRENGTHS = [1e-5, 1e-4, 1e-3, 1e-2, .1, 1., 10.]
METHODS = ("source", "planar", "target_ridge", "jacobian")


def check(value, reason):
    if not value:
        raise ValueError(str(reason))


def dump(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def get_trace(path, **identity):
    raw = path.read_bytes()
    with np.load(io.BytesIO(raw), allow_pickle=False) as tape:
        valid = tape["valid"]
        check(valid.dtype == bool and valid[:3].all() and not valid[3:].any(), path)
        commands = tape["issued_mean"][valid, :3].astype(float)
        response = tape["delta_eef"][valid].astype(float)
    check(commands.shape == response.shape == (3, 3) and np.isfinite([commands, response]).all(), path)
    check(abs(commands).max() <= 1, path)
    previous = np.vstack([np.zeros(3), commands[:-1]])
    return dict(inputs=np.column_stack([commands, previous]), response=response, **identity,
                support=dict(path=str(path), sha256=sha256(raw).hexdigest(), **identity))


def source_traces(root, splits):
    directory = root / SCALE / "gpu01"
    launch = (directory / "launch.txt").read_text().split()[0]
    check(launch == "d597ee0f1e99937bbeb2705132d709b514a4b5c2", launch)
    archive = directory / "archive/archive.json"
    raw = archive.read_bytes()
    check(sha256(raw).hexdigest() == ARCHIVE_SHA, archive)
    expected = {(a, s, t, i) for a in ("Panda", "Sawyer") for s, ids in splits["train"].items()
                for t in ids for i in range(4)}
    records = []
    for r in json.loads(raw)["trials"]:
        if r["split"] != "train" or r["embodiment"] != "nom":
            continue
        ident = tuple(r[k] for k in ("arm", "suite", "task_id", "init_state_id"))
        expected.remove(ident)
        check((r["yaw_deg"], r["xy_scale"], r["gripper_sign"], r["passive_controls"]) == (0, 1, 1, 15), ident)
        arm, suite, task, start = ident
        records.append(get_trace(archive.parent / r["passive_path"], arm=arm, task=f"{suite}/{task}",
                                 start=start, seed=r["seed"], collector_pin=launch))
    check(len(records) == 192 and not expected, "Wrong nominal source panel")
    return records


def target_traces(root, splits):
    suites = sorted(splits["test"])
    tasks = [(s, t) for s in suites for t in splits["test"][s]]
    check(len(tasks) == len(set(tasks)) == 8, tasks)
    records = {}
    for arm, (campaign, job, subdir, pin) in PANELS.items():
        directory = root / campaign / job
        check((directory / "launch.txt").read_text().split()[0] == pin, directory)
        records[arm] = []
        for i, (suite, task) in enumerate(tasks):
            shard = directory / subdir / f"shard-{i}"
            meta = json.loads((shard / "support-meta.json").read_bytes())[0]
            path = shard / "support/00-fam_nom-passive.npz"
            check(meta["embodiment"] == "fam_nom" and Path(meta["passive_npz"]) == path, shard)
            # Read only identifiers from the old index to resolve tasks. No query responses enter fits.
            if arm != "UR5e":
                check_row = json.loads((shard / "live-shard.json").read_bytes())["rows"][0]
                check(tuple(check_row[k] for k in ("index", "suite", "task_id", "embodiment")) ==
                      (12 * i, suite, task, "fam_nom"), shard)
            records[arm].append(get_trace(path, arm=arm, task=f"{suite}/{task}", start=11,
                controls=15, seed=31800000 + 10000 * suites.index(suite) + 100 * task + 11,
                physics=meta["physics"], collector_pin=pin))
    return records


def query_panel(rows):
    return np.stack([r["inputs"] for r in rows]), np.stack([r["response"] for r in rows])


def errors(coef, inputs, response):
    return 1000 * np.linalg.norm(inputs @ coef - response, axis=-1).mean(axis=-1)


def tune(source):
    # Each validation support adapts its own pseudo-target. Score all other tasks on that arm.
    costs = {m: np.zeros(len(STRENGTHS)) for m in METHODS[1:]}
    for target_arm in ("Panda", "Sawyer"):
        prior, _ = fit_source([r for r in source if r["arm"] != target_arm])
        target = [r for r in source if r["arm"] == target_arm]
        inputs, response = query_panel(target)
        for support in target:
            keep = np.array([r["task"] != support["task"] for r in target])
            for method in costs:
                for k, coef in enumerate(STRENGTHS):
                    fit = fit_target(method, prior, support, coef)
                    costs[method][k] += errors(fit, inputs[keep], response[keep]).mean() / len(source)
    return ({m: STRENGTHS[np.argmin(vals)] for m, vals in costs.items()},
            {m: vals.tolist() for m, vals in costs.items()})


def summary(grid):
    # Delete one of the eight unique tasks in both roles; pairs share tasks.
    n = len(grid)
    replicas = []
    for i in range(n):
        keep = np.arange(n) != i
        replicas.append(np.nanmean(grid[np.ix_(keep, keep)]))
    se = np.sqrt((n - 1) / n * np.sum((replicas - np.mean(replicas))**2))
    mean = np.nanmean(grid)
    return dict(mean_mm=mean, delete_one_task_se_mm=se, approximate_95_mm=[mean - 2.365 * se, mean + 2.365 * se])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    splits = json.loads(args.splits.read_bytes())
    source = source_traces(args.data_root, splits)
    strength, cv = tune(source)
    prior, noise = fit_source(source)
    check(noise > 0 and np.isfinite(noise), "Degenerate inverse calibration")
    source_record = dict(prior=prior.tolist(), residual_power=noise, choice=strength, grid=STRENGTHS,
        cv_mm=cv, archive_sha256=ARCHIVE_SHA, split_sha256=sha256(args.splits.read_bytes()).hexdigest(),
        implementation={p.name:sha256(p.read_bytes()).hexdigest() for p in (HERE/"experiment.py", HERE/"motion.py")},
        evaluation_code_git_head=subprocess.check_output(["git","-C",str(HERE),"rev-parse","HEAD"], text=True).strip(),
        numpy=np.__version__, counts=dict(Counter(r["arm"] for r in source)), sources=[r["support"] for r in source])
    dump(out / "source-fit.json", source_record)
    targets = target_traces(args.data_root, splits)
    source_tasks = set(r["task"] for r in source)
    check(all(r["task"] not in source_tasks for rs in targets.values() for r in rs), "Task overlap")
    results, matrices, raw_scores = {}, {}, []
    for arm, rows in targets.items():
        inputs, response = query_panel(rows)
        grids = {m:np.full((8,8), np.nan) for m in (*METHODS, "wrong_arm_jacobian")}
        for i, support in enumerate(rows):
            key = f"{arm}/{support['task']}"
            matrices[key] = dict(support=support["support"], source_fit_sha256=sha256((out/"source-fit.json").read_bytes()).hexdigest())
            for method in METHODS:
                fit = fit_target(method, prior, support, strength.get(method))
                grid = grids[method]
                grid[i, :] = errors(fit, inputs, response)
                grid[i, i] = np.nan
                if method in ("jacobian","planar","source"):
                    matrix = correction_matrix(prior, fit, noise)
                    check(np.isfinite(matrix).all(), key)
                    if method == "source":
                        np.testing.assert_array_equal(matrix, np.eye(3))
                    if method == "jacobian":
                        matrices[key]["matrix"] = matrix.tolist()
                    elif method == "planar":
                        matrices[key]["planar_matrix"] = matrix.tolist()
            wrong = []
            for other, other_rows in targets.items():
                if arm == other:
                    continue
                check(other_rows[i]["task"] == support["task"], key)
                wrong.append(errors(fit_target("jacobian", prior, other_rows[i], strength["jacobian"]), inputs, response))
            grids["wrong_arm_jacobian"][i, :] = np.mean(wrong, axis=0)
            grids["wrong_arm_jacobian"][i, i] = np.nan
        results[arm] = dict(errors={m:summary(g) for m,g in grids.items()},
             jacobian_minus={m:summary(grids["jacobian"] - g) for m,g in grids.items() if m != "jacobian"})
        for i,s in enumerate(rows):
            for j,q in enumerate(rows):
                if i != j:
                    raw_scores.append(dict(arm=arm,support=s["task"],query=q["task"],errors_mm={m:g[i,j] for m,g in grids.items()}))
    fixture = np.array([[0.2,0.3,-0.4,0.3,-0.2,0.1,1]], dtype=np.float32)
    custom = np.array([[0,2,0],[-2,0,0],[0,0,4]])
    want = fixture.copy()
    want[0, :3] = [-0.6, 0.4, -1]
    np.testing.assert_allclose(map_action(fixture, custom), want)
    np.testing.assert_array_equal(map_action(fixture, np.eye(3)), fixture)
    dump(out/"matrices.json", matrices)
    dump(out/"paired-scores.json", raw_scores)
    dump(out/"summary.json", results)
    for arm,row in results.items():
        print(arm, {m:round(s["mean_mm"], 3) for m,s in row["errors"].items()}, flush=True)


if __name__ == "__main__":
    main()
