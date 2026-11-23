"""Run one whole, proposal-discordant native shard in the separate common-opening store."""
import argparse
import csv
from functools import partial
import os
from pathlib import Path
import socket
import signal
import time
from unittest.mock import patch

import numpy as np

import reference as ref
import symmeta_adapter as stream


def original_inputs(job):
    """Select solely on registered proposals. Bind every row to the original state and nonce."""
    from reground.eval.jobs import load_traces
    from reground.policies.vla import chunk_seed
    from reground.utils import noise_seed
    from opening import identical

    keys = list(map(tuple, job["episodes"]))
    paths = {m: ref.output(dict(job, method=m)) for m in ("base", "reground")}
    paths["symmeta"] = stream.STORE / "shards" / ref.label(dict(job, method="symmeta"))
    traces = {}
    for arm, path in paths.items():
        if arm == "reground" and not path.exists():
            continue
        with (path / "results.csv").open() as f:
            rows = list(csv.DictReader(f))
        ref.require([(r["task"], int(r["init_id"]), int(r["rep"])) for r in rows] == keys
                    and all(r["split"] == "test" and int(r["adapt_seed"]) == job["seed"]
                            and r["method"] == arm and r["policy"] == "pi05" and r["shift"] == "rot_z"
                            and float(r["level"]) == 90 and int(r["noise_seed"]) == noise_seed(i, rep)
                            for r, (_, i, rep) in zip(rows, keys)), f"Original assignments differ: {path}")
        traces[arm] = load_traces(path)
        ref.require(set(traces[arm]) == set(keys), f"Original trace keys differ: {path}")
    costs = ref.read(paths["symmeta"] / "calibration.json")
    result, different = {}, []
    for key, cost in zip(keys, costs, strict=True):
        nonce = noise_seed(*key[1:])
        ref.require((cost["task"], cost["init_id"], cost["rep"]) == key and cost["adapt_seed"] == job["seed"]
                    and cost["scene_seed"] == 0 and cost["noise_seed"] == nonce
                    and cost["queries"][0]["start_control"] == 0
                    and cost["queries"][0]["chunk_seed"] == chunk_seed(nonce, 0), "Original nonce differs")
        starts = [tr[key]["X"][0] for tr in traces.values()]
        ref.require(all(identical(x, starts[0]) for x in starts), f"Diagnose original reset: {key}")
        proposals = [tr[key]["A_vla"][:10] for tr in traces.values()]
        if any(not identical(a[:min(len(a), len(b))], b[:min(len(a), len(b))])
               for j, a in enumerate(proposals) for b in proposals[:j]):
            different.append(key)
        first = cost["updates"][0]
        result[key] = dict(x=starts[0], nonce=nonce, camera=first["before_sha256"],
                           eef=first["before_eef"], jaw=first["before_jaw_qpos"])
    ref.require(different, "No original first-proposal mismatch on this shard")
    return result, different


def verify_gpu(attempt):
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci

    probe = ref.read(attempt / "gpu.json")
    row, = probe["devices"]
    ref.require(probe["status"] == "passed" and probe["purpose"] == "training"
                and probe["exclusive_training_admission"] is True and probe["allocation_type"] == "baige_container"
                and probe["hostname"] == socket.gethostname() and torch.cuda.device_count() == 1
                and row["allocation_ordinal"] == 0 and row["uuid"] == os.environ["NVIDIA_VISIBLE_DEVICES"]
                and row["cuda_uuid"] == str(torch.cuda.get_device_properties(0).uuid)
                and row["cuda_name"] == torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
                and pci(row["pci_bus_id"]) == cuda_pci(0), "Expected one exclusive allocated RTX 4090")
    torch.cuda.set_device(0)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(ref.egl_device(0))


def evaluate(arm, ctx, job, work, reader=None, fixture=False):
    from reground.eval import jobs
    from reground.method import build
    from reground.models import cache
    from opening import file_hash

    if arm == "symmeta":
        stream.run_job(job, ctx, *reader, work, fixture=fixture)
        return
    weights = {}
    if arm == "reground":
        receipt = ref.read(ref.output(dict(kind="fit", seed=job["seed"])))
        dirs = {m["dir"] for m in receipt["models"]}
        ref.require(len(dirs) == 4, "Expected the four released cache entries")

    def load(spec, device, **kw):
        path = cache.spec_dir(spec)
        key = str(path.relative_to(ref.STORE))
        ref.require(arm == "reground" and key in dirs and ref.read(path / "spec.json") == spec,
                    f"Cache differs from fitted receipt: {path}")
        weights[key] = file_hash(path / "model.pt")
        return cache.load_or_fit(spec, device, log=ctx.log, fit_ok=False)

    build._MEMO.clear()
    try:
        with patch.object(jobs, "job_dir", lambda _: work), patch.object(build, "load_or_fit", load):
            jobs.run_eval_job(job, ctx)
        if arm == "reground":
            ref.require(weights.keys() == dirs, "Controller used a different model set")
            ref.write(work / "weights.json", weights)
    finally:
        build._MEMO.clear()
    if not fixture:
        ref.check_eval(work, job)


def verify_controlled(arm, work, base, episodes):
    """The upstream registered prefix tests, on the sensitivity outputs only."""
    from reground.eval.jobs import load_traces
    from opening import identical

    ours, original = load_traces(work), load_traces(base)
    for key in map(tuple, episodes):
        a, b = ours[key], original[key]
        n = min(10, len(a["A"]), len(b["A"]))
        ref.require(n > 0 and identical(a["X"][0], b["X"][0])
                    and identical(a["A_vla"][:n], b["A_vla"][:n]), f"Different controlled proposal: {key}")
        if arm == "symmeta":
            ref.require(n == 10 or len(a["A"]) == len(b["A"]), f"Different opening termination: {key}")
            for field, stop in (("X", n + 1), ("A", n), ("speed", n)):
                ref.require(identical(a[field][:stop], b[field][:stop]), f"Different controlled {field}: {key}")


def stage(arm, ctx, job, original, work, store, reader=None, fixture=False):
    from opening import Opening

    work.mkdir()
    dest = store / arm
    ref.require(not dest.exists(), f"Already published {dest}")
    tape = None if arm == "capture" else store / "capture"
    gate = Opening(ctx, job, original, work, tape)
    with gate.hooks():
        if arm == "capture":
            gate.capture()
        else:
            evaluate(arm, ctx, job, work, reader, fixture)
    gate.finish()
    if arm in ("symmeta", "reground"):
        verify_controlled(arm, work, store / "base", job["episodes"])
    work.rename(dest)  # atomic whole-shard publication; a competing nonempty directory cannot be replaced
    print(f"Published {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("part-a", "capture", "base", "symmeta", "reground"))
    parser.add_argument("seed", type=int, choices=ref.SEEDS)
    parser.add_argument("shard", type=int, choices=range(18))
    parser.add_argument("attempt", type=Path)
    args = parser.parse_args()
    attempt = args.attempt.resolve(strict=True)
    ref.require(attempt.parent == ref.ROOT / "common-opening/attempts", "Expected a fresh sensitivity attempt")
    ledger = ref.ROOT / "reference-native/attempts" / ("common-opening-" + attempt.name)
    r = ref.read(ledger / "reservation.json")
    remaining = r["started_utc"] + r["wall_seconds"] - time.time()
    ref.require(not (ledger / "exit.json").exists() and r["host"] == socket.gethostname()
                and r["phase"] == "common-opening-" + args.phase and r["seeds"] == [args.seed]
                and r["code"] == os.environ["SYMMETA_CODE_REV"] and r["gpus"] == 1
                and r["upstream"] == ref.PIN and r["policy"] == ref.POLICY and 0 < remaining
                and 1200 <= r["wall_seconds"] == r["max_gpu_seconds"] <= 2700
                and r["prior_gpu_seconds"] + r["max_gpu_seconds"] <= 3600 * r["cap_gpuh"] <= 40 * 3600,
                "Incorrect or expired sensitivity reservation")
    signal.setitimer(signal.ITIMER_REAL, remaining)
    ref.configure()
    from reground.eval.jobs import WorkerCtx
    from reground.envs.libero_env import N_SETTLE
    from reground.policies.vla import N_EXEC

    ref.require(N_EXEC == N_SETTLE == 10 and ref.BATCH == 4, "Changed native cadence")
    verify_gpu(args.attempt)
    job = next(j for j in ref.units("base", [args.seed]) if j["shard"] == args.shard)
    original, different = original_inputs(job)
    ref.write(args.attempt / "selection.json", dict(seed=args.seed, shard=args.shard,
              episodes=job["episodes"], original_proposal_mismatches=different))
    store = ref.ROOT / "common-opening" / "shards" / f"seed{args.seed}-shard{args.shard:03d}"
    store.mkdir(parents=True, exist_ok=True)
    stages = ("capture", "base", "symmeta") if args.phase == "part-a" else (args.phase,)
    vla = None
    for arm in stages:
        ref.require(not (store / arm).exists(), f"{arm} exists. Select the next individual stage.")
        if arm != "capture":
            ref.require((store / "capture/opening.json").is_file(), "Missing completed capture")
        if arm in ("symmeta", "reground"):
            ref.require((store / "base/opening.json").is_file(), "Missing controlled base")
        if arm == "reground":
            ref.require(ref.output(dict(kind="fit", seed=args.seed)).is_file(), "Missing fitted seed")
        ctx = WorkerCtx("pi05", "cuda:0", ref.BATCH, partial(print, flush=True))
        try:
            if vla is None:
                ref.strict_policy(ctx)
                vla = ctx.vla
            else:
                ctx._vla = vla
            if arm == "symmeta":
                other = next(j for j in stream.units([args.seed]) if j["shard"] == args.shard)
                ref.require(other["episodes"] == job["episodes"], "Adapter assignment differs")
                with stream.reader_process() as reader:
                    stage(arm, ctx, other, original, args.attempt / arm, store, reader)
            else:
                if arm == "reground":
                    job = next(j for j in ref.units(arm, [args.seed]) if j["shard"] == args.shard)
                stage(arm, ctx, job, original, args.attempt / arm, store)
        finally:
            ctx.close()


if __name__ == "__main__":
    main()
