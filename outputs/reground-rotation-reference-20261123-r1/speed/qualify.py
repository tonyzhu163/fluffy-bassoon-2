"""Small CPU-only native babble fixtures on Baige. Never use the registered output store."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import time

import reference as ref


def fixture(work, arm):
    root = ref.ROOT / "speed"
    if work.resolve().parent != root or not work.name.startswith("qualification."):
        raise ValueError("Use a fresh qualification directory under the run's speed/")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA for CPU qualification")
    ref.STORE = work / arm
    ref.NATIVE = ref.STORE / "runs"
    ref.configure()


def plan():
    grid = ref.units("babble", ref.SEEDS)
    # One native whole babbling episode per shard. Exercise all eight released tasks.
    return [dict(next(j for j in grid if j["shard"] == i and
                      j["cell"] == ref.CELLS[i % 2] and j["seed"] == i % 5), n=1)
            for i in range(8)]


def evidence(job):
    path = ref.output(job)
    return dict(unit=ref.label(job), npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                meta=ref.read(path.with_suffix(".json")), **ref.check_pool_shard(path, job))


def serial(work):
    fixture(work, "serial")
    jobs = plan()
    attempt = ref.STORE / "attempts/first"
    attempt.mkdir(parents=True)
    rows = []
    start = time.monotonic()
    for job in jobs:
        t = time.monotonic()
        with ref.claim(job) as mine:
            ref.require(mine, "Unexpected claim in private fixture")
            ref.perform(job, None, attempt / "partial" / ref.label(job), ref.output(job))
        rows.append(dict(seconds=time.monotonic() - t, **evidence(job)))
        print(rows[-1]["unit"], rows[-1]["seconds"], flush=True)
    ref.write(work / "serial.json", dict(wall_s=time.monotonic() - start, rows=rows))


def isolated_plan():
    # Use a short-lived planner so pandas threads never enter the forking parent.
    return json.loads(subprocess.check_output(
        [sys.executable, "-B", "-c", "import json, qualify; print(json.dumps(qualify.plan()))"],
        cwd=Path(__file__).resolve().parent))


def parallel(work, arm="parallel"):
    from babble import run

    fixture(work, arm)
    jobs = isolated_plan()
    attempt = ref.STORE / "attempts/first"
    attempt.mkdir(parents=True)
    result = run(jobs, attempt, 8)
    ref.require(result["verified"] == 8, "Incomplete private fixture")
    rows = [evidence(job) for job in jobs]
    baseline = ref.read(work / "serial.json")
    for control, actual in zip(baseline["rows"], rows, strict=True):
        for k in ("unit", "npz_sha256", "episodes", "transitions"):
            ref.require(control[k] == actual[k], f"{k} differs at {actual['unit']}")
        ref.require(scientific(control) == scientific(actual), "Native sidecars differ")
    ref.write(work / f"{arm}.json", dict(**result, preload=ref.read(attempt / "preload.json"), rows=rows))


def scientific(row):
    return {k: v for k, v in row["meta"].items() if k != "wall_s"}


def legacy_one(job, attempt):
    with ref.claim(job) as mine:
        ref.require(mine, ref.label(job))
        ref.perform(job, None, attempt / "partial" / ref.label(job), ref.output(job))


def four(work):
    # Four independent legacy lanes. Each invokes the registered per-shard spawn.
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from itertools import repeat
    fixture(work, "four")
    jobs = plan()
    attempt = ref.STORE / "attempts/first"
    attempt.mkdir(parents=True)
    t = time.monotonic()
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn"),
                             initializer=fixture, initargs=(work, "four")) as lanes:
        list(lanes.map(legacy_one, jobs, repeat(attempt)))
    wall = time.monotonic() - t
    control = ref.read(work / "serial.json")["rows"]
    for j, c in zip(jobs, control, strict=True):
        actual = evidence(j)
        ref.require(c["npz_sha256"] == actual["npz_sha256"] and scientific(c) == scientific(actual),
                    ref.label(j))
    ref.write(work / "four.json", dict(wall_s=wall, compared=len(control)))


def resume(work):
    from babble import run
    fixture(work, "resume")
    first, second = isolated_plan()[:2]
    attempt = ref.STORE / "attempts/locked"
    attempt.mkdir(parents=True)
    dest = ref.output(first)
    dest.parent.mkdir(parents=True)
    src = work / "parallel" / dest.relative_to(ref.STORE)
    for suffix in (".npz", ".json"):
        shutil.copy2(src.with_suffix(suffix), dest.with_suffix(suffix))
    stamp = dest.stat().st_mtime_ns
    missing = ref.output(second)
    missing.parent.mkdir(parents=True)
    missing.with_suffix(".json").write_text('{"incomplete":true}')
    # Simulate a prior torn publication, a stale private stage, and an unrelated live claim.
    (attempt / "partial" / ref.label(second)).mkdir(parents=True)
    script = ("import json, sys, reference as r; from pathlib import Path; "
              "r.STORE = Path(sys.argv[1]); job=json.loads(sys.argv[2]); "
              "ctx = r.claim(job); print(ctx.__enter__(), flush=True); sys.stdin.read(); "
              "ctx.__exit__(None,None,None)")
    with subprocess.Popen([sys.executable, "-B", "-c", script, str(ref.STORE), json.dumps(second)],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True) as legacy:
        try:
            ref.require(legacy.stdout.readline().strip() == "True", "Legacy claim failed")
            locked = run([first, second], attempt, 8)
            ref.require(locked["outstanding"] == [ref.label(second)] and not missing.exists(), locked)
        finally:
            legacy.stdin.close()
    fresh = ref.STORE / "attempts/fresh"
    fresh.mkdir()
    result = run([first, second], fresh, 8)
    actual = evidence(second)
    control = ref.read(work / "serial.json")["rows"][1]
    ref.require(actual["npz_sha256"] == control["npz_sha256"] and scientific(actual) == scientific(control)
                and dest.stat().st_mtime_ns == stamp and result["verified"] == 2, result)
    ref.write(work / "resume.json", dict(while_claimed=locked, after_release=result,
                                         previous_npz_untouched=True, recovered_npz_sha256=actual["npz_sha256"]))


def budget(work):
    from types import SimpleNamespace
    fixture(work, "budget")
    os.environ.update(REFERENCE_PHASE="babble", REFERENCE_CODE_PIN="budget-fixture",
                      REFERENCE_STARTED=str(int(time.time())))
    attempts = ref.STORE / "attempts"
    legacy, fresh = attempts / "legacy", attempts / "fresh"
    legacy.mkdir(parents=True)
    fresh.mkdir()
    def reserve(attempt, gpus, wall):
        ref.reserve(SimpleNamespace(attempt=attempt, gpus=gpus, wall=wall, cap=1, seeds=list(ref.SEEDS)))
    reserve(legacy, 4, 800)
    try:
        reserve(fresh, 1, 600)
    except ValueError as exc:
        ref.require("3200+600" in str(exc), str(exc))
    else:
        raise AssertionError("Overlapping reservations exceeded the fixture cap")
    ref.write(legacy / "exit.json", {"allocated_gpu_seconds": 2800})
    reserve(fresh, 1, 600)
    receipt = ref.read(fresh / "reservation.json")
    ref.require(receipt["prior_gpu_seconds"] == 2800 and receipt["max_gpu_seconds"] == 600, receipt)
    ref.write(work / "budget.json", dict(rejected_unfinished_gpu_seconds=3800,
                                          reconciled_prior=2800, one_card_reservation=600, cap=3600))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("serial", "parallel", "four", "private", "resume", "budget"))
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    {"serial": serial, "parallel": parallel, "four": four, "resume": resume, "budget": budget,
     "private": lambda work: parallel(work, "private")}[args.action](args.work)
