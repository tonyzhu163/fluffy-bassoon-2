"""Post-score CPU audit of the two native panel32 attempts and their paired starts."""

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

PIN = "200647539605980ceb81bb10d32cc003a6f736dd"
RUN = Path("outputs/reground-low-budget-20261123-r1")
SEEDS = (0, 3)
JOBS = {0: "job-oba1tdxissci", 3: "job-sc23pialxxia"}
BUDGETS = (15, 100, 1000)
ARMS = ("base", "b15", "b100", "b1000", "b10000")
PAIRS = [(f"b{b}", ctl) for b in BUDGETS for ctl in ("base", "b10000")] + [
    ("b100", "b15"), ("b1000", "b15"), ("b1000", "b100"), ("b10000", "base")
]
DRAWS = 10000
BOOT_SEED = 20261123


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        table = csv.DictWriter(f, fieldnames=list(rows[0]))
        table.writeheader()
        table.writerows(rows)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def run_panel_readers(seeds, scripts, out):
    for seed in seeds:
        for budget in BUDGETS:
            dest = out / f"panel-s{seed}-b{budget}.json"
            result = subprocess.run(
                [sys.executable, "-B", str(scripts / "panel.py"), "--seed", str(seed),
                 "--budget", str(budget), "--panel", "panel32"],
                stdout=subprocess.PIPE, text=True, check=True,
            )
            # Some versions of native dependencies print import banners before the JSON.
            payload = json.loads(result.stdout.splitlines()[-1])
            summary = payload["summary"]
            assert (summary["seed"], summary["budget"], summary["panel"], summary["episodes"]) == (
                seed, budget, "panel32", 32
            )
            dest.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")


def collect_seed(seed, work, out):
    from common import SOURCE_STORE, ref
    from panel import read_rows
    from runner import Inputs, panel_jobs
    from reground.eval.jobs import load_traces
    from reground.utils import noise_seed
    import torch

    attempt = work / "attempts" / f"panel-s{seed}-01"
    launch = ref.read(attempt / "attempt.json")
    exit_receipt = ref.read(attempt / "exit.json")
    probe = ref.read(attempt / "gpu.json")
    assert launch["code"] == PIN and launch["host"] == JOBS[seed] + "-master-0"
    assert launch["seed"] == seed and launch["budgets"] == list(BUDGETS)
    assert launch["phase"] == "fit-panel32" and exit_receipt["exit_code"] == 0
    assert exit_receipt["gpus"] == len(probe["devices"]) == 1
    assert probe["hostname"] == launch["host"] and probe["status"] == "passed"

    panel_rows, stores = {}, {}
    for budget in BUDGETS:
        dest = out / f"panel-s{seed}-b{budget}.json"
        payload = json.loads(dest.read_text())
        arm = f"b{budget}"
        panel_rows[arm] = {(r["task"], r["start"], r["rep"]): r for r in payload["rows"]}
        eval_dir = work / "production" / f"seed{seed}/b{budget}/eval"
        store, = (p.parent for p in eval_dir.glob("*/panel32") if p.is_dir())
        stores[arm] = store

    arms, comparisons = [], []
    for job in panel_jobs(Inputs(seed, BUDGETS[0]), "panel32"):
        shard = job["shard"]
        name = f"shard{shard:03d}"
        keys = [tuple(e) for e in job["episodes"]]
        openings = {}
        for arm in ARMS:
            if arm in stores:
                method, root = "reground", stores[arm]
                path = root / "panel32" / name
                assert ref.read(path / "manifest.json")["host"] == launch["host"]
            else:
                method = "base" if arm == "base" else "reground"
                root = SOURCE_STORE
                path = root / "shards" / f"seed{seed}/rot_z/90" / method / name
            checkpoint = root / "batches" / f"seed{seed}/rot_z/90" / method / name / "batch0000.pt"
            saved = torch.load(checkpoint, weights_only=False, map_location="cpu")
            assert saved["upstream"] == ref.PIN and saved["policy"] == ref.POLICY
            assert [tuple(e[:3]) for e in saved["episodes"]] == keys
            traces = load_traces(path)
            rows = read_rows(path)
            for lane, key in enumerate(keys):
                task, start, rep = key
                raw = rows[key]
                tr = traces[key]
                noise = int(raw["noise_seed"])
                scene = int(saved["episodes"][lane][3])
                assert noise == noise_seed(start, rep)
                r = dict(seed=seed, task=task, start=start, rep=rep, shard=shard, lane=lane,
                         arm=arm, scene_seed=scene, noise_seed=noise)
                for col in ("success", "unsafe", "steps", "speed_viol", "contact_steps", "fallen", "displaced"):
                    r[col] = int(raw[col])
                assert r["unsafe"] == int(any(r[c] for c in ("speed_viol", "contact_steps", "fallen", "displaced")))
                if arm in stores:
                    sources = [(panel_rows[arm][key], "low")]
                else:
                    prefix = "base" if arm == "base" else "reground"
                    sources = [(reader[key], prefix) for reader in panel_rows.values()]
                for record, prefix in sources:
                    assert all(record[f"{prefix}_{m}"] == r[m] for m in ("success", "unsafe", "steps"))
                arms.append(r)
                openings[arm, key] = scene, noise, tr["X"][0], tr["A_vla"][:10]

        for left, right in PAIRS:
            for key in keys:
                sc_a, nz_a, x_a, a = openings[left, key]
                sc_b, nz_b, x_b, b = openings[right, key]
                n = min(len(a), len(b))
                assert n > 0
                robot = np.array_equal(x_a, x_b)
                proposal = np.array_equal(a[:n], b[:n])
                if left in stores and right in ("base", "b10000"):
                    field = "base_opening_exact" if right == "base" else "reground_opening_exact"
                    assert panel_rows[left][key][field] == (robot and proposal)
                comparisons.append(dict(seed=seed, task=key[0], start=key[1], rep=key[2], left=left,
                                        right=right, scene_seed_exact=sc_a == sc_b, noise_exact=nz_a == nz_b,
                                        initial_robot_exact=robot, proposal_exact=proposal, prefix_steps=n,
                                        max_abs_x=float(np.max(np.abs(x_a - x_b))),
                                        max_abs_proposal=float(np.max(np.abs(a[:n] - b[:n])))))

    elapsed = exit_receipt["elapsed_gpu_seconds"]
    start_utc = datetime.fromtimestamp(launch["start"], timezone.utc).isoformat()
    ledger = dict(job=JOBS[seed], seed=seed, attempt=attempt.name, start_utc=start_utc, pin=PIN,
                  exit_code=exit_receipt["exit_code"], gpus=1, device=probe["devices"][0]["cuda_name"],
                  reserved_seconds=launch["wall_seconds"], elapsed_gpu_seconds=elapsed,
                  elapsed_gpu_hours=elapsed / 3600,
                  eval_contracts={arm: path.name for arm, path in stores.items()})
    return arms, comparisons, ledger


def interval_summary(values, stratified, pooled):
    # Axis zero is the task-start cluster, with the seed repeats and every arm/difference aligned.
    means = values.mean(axis=1)
    result = dict(n=int(values.size), sum=int(values.sum()), mean=float(values.mean()))
    for name, ix in (("stratified_start_ci95", stratified), ("pooled_start_ci95", pooled)):
        result[name] = np.percentile(means[ix].mean(axis=1), [2.5, 97.5]).tolist()
    return result


def analyze(records, checks, registered, out):
    cells = {(r["task"], r["start"], r["rep"], r["seed"], r["arm"]): r for r in records}
    starts = sorted(map(tuple, registered))
    tasks = sorted({task for task, _, _ in starts})
    assert len(set(starts)) == len({(task, start) for task, start, _ in starts}) == 32
    assert len(tasks) == 8 and all(sum(k[0] == t for k in starts) == 4 for t in tasks)
    assert set(cells) == {(*k, s, a) for k in starts for s in SEEDS for a in ARMS}
    assert len(records) == len(cells)
    checks_by_key = {(r["task"], r["start"], r["rep"], r["seed"], r["left"], r["right"]): r for r in checks}
    assert len(checks_by_key) == len(checks) == len(starts) * len(SEEDS) * len(PAIRS)
    rng = np.random.default_rng(BOOT_SEED)
    strata = []
    for task in tasks:
        ids = [i for i, key in enumerate(starts) if key[0] == task]
        strata.append(rng.choice(ids, size=(DRAWS, len(ids)), replace=True))
    stratified = np.concatenate(strata, axis=1)
    pooled = rng.integers(len(starts), size=(DRAWS, len(starts)))
    summary, task_counts = {}, []
    match_fields = ("scene_seed_exact", "noise_exact", "initial_robot_exact", "proposal_exact")
    for scope, seeds in (("both", SEEDS), ("seed0", (0,)), ("seed3", (3,))):
        absolute, paired = {}, {}
        for metric in ("success", "unsafe"):
            scores = {a: np.asarray([[cells[(*k, s, a)][metric] for s in seeds] for k in starts]) for a in ARMS}
            absolute[metric] = {a: interval_summary(v, stratified, pooled) for a, v in scores.items()}
            paired[metric] = {}
            for left, right in PAIRS:
                delta = scores[left] - scores[right]
                result = interval_summary(delta, stratified, pooled)
                result.update(left_only=int((delta == 1).sum()), right_only=int((delta == -1).sum()),
                              ties=int((delta == 0).sum()))
                selected = [[checks_by_key[(*k, s, left, right)] for s in seeds] for k in starts]
                exact = np.asarray([[all(r[c] for c in match_fields) for r in rr] for rr in selected])
                changed = int((~exact).sum())
                fixed = int(delta[exact].sum())
                result["opening"] = dict(
                    mismatches={c: sum(not r[c] for rr in selected for r in rr) for c in match_fields},
                    exact=int(exact.sum()), inexact=changed,
                    left_events_exact=int(scores[left][exact].sum()), right_events_exact=int(scores[right][exact].sum()),
                    exact_mean_delta=(fixed / int(exact.sum())) if exact.any() else None,
                    arbitrary_inexact_bounds=[(fixed - changed) / delta.size, (fixed + changed) / delta.size],
                )
                paired[metric][f"{left}_minus_{right}"] = result
                if len(seeds) == 1:
                    for task in tasks:
                        indices = [i for i, key in enumerate(starts) if key[0] == task]
                        v = delta[indices]
                        task_counts.append(dict(seed=seeds[0], task=task, metric=metric, left=left, right=right,
                                                n=int(v.size), left_events=int(scores[left][indices].sum()),
                                                right_events=int(scores[right][indices].sum()),
                                                left_only=int((v == 1).sum()), right_only=int((v == -1).sum()),
                                                ties=int((v == 0).sum())))
        summary[scope] = dict(absolute=absolute, paired=paired)
    write_csv(out / "paired-by-task-seed.csv", task_counts)
    write_csv(out / "openings.csv", checks)
    write_csv(out / "assignments.csv", records)
    return dict(bootstrap=dict(draws=DRAWS, rng="numpy.default_rng", rng_seed=BOOT_SEED, distinct_starts=32,
                               stratified="Four starts with replacement per fixed task; retain every seed and arm.",
                               pooled="32 starts with replacement from all tasks; retain every seed and arm.",
                               limits="Percentile 2.5 and 97.5; fixed observed tasks and fitted seeds for primary intervals."),
                estimates=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output
    work = Path(os.environ["REGROUND_RUNTIME_ROOT"]).parent / RUN.name
    checkout = work / f"src-{PIN}"
    scripts = checkout / RUN
    assert subprocess.check_output(["git", "-C", checkout, "rev-parse", "HEAD"], text=True).strip() == PIN
    assert Path(out).resolve().is_relative_to(work) and len(os.sched_getaffinity(0)) <= 2
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
    sys.path.insert(0, str(scripts))
    from common import configure, ref

    configure()
    out.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        receipt = ref.read(work / "attempts" / f"panel-s{seed}-01" / "exit.json")
        assert receipt["exit_code"] == 0
    run_panel_readers(SEEDS, scripts, out)
    records, checks, ledger = [], [], []
    for seed in SEEDS:
        cases, comparisons, job = collect_seed(seed, work, out)
        records.extend(cases)
        checks.extend(comparisons)
        ledger.append(job)
    # The handoff fixes these shard IDs independently of the low-budget outcomes.
    shards = {0, 3, 5, 7, 9, 11, 14, 16}
    registered = [episode for job in ref.units("reground", [0]) if job["shard"] in shards
                  for episode in job["episodes"][:4]]
    summary = analyze(records, checks, registered, out)
    write_json(out / "summary.json", summary)
    write_json(out / "jobs.json", ledger)


if __name__ == "__main__":
    main()
