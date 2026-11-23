"""Partition only the native calibration keys. The reference owns science and layout."""
import contextlib
import errno
import hashlib
import json
import os
from pathlib import Path
import socket
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

import reference as ref


def selected(phase, seeds, bounds, descending):
    jobs = ref.units(phase, seeds)
    first, stop = bounds
    end = max(j["shard"] for j in jobs) + 1
    ref.require(0 <= first < stop <= end, f"Invalid half-open shard range [{first},{stop}) for {phase}")
    if descending:
        jobs.sort(key=lambda j: -j["shard"])  # stable within each shard
    return [j for j in jobs if first <= j["shard"] < stop]


def gpu(rank, attempt, count, render):
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci

    probe = ref.read(attempt / "gpu.json")
    ref.require(probe["status"] == "passed" and probe["hostname"] == socket.gethostname()
                and len(probe["devices"]) == count and torch.cuda.device_count() == count, "GPU probe mismatch")
    row = probe["devices"][rank]
    name = row["cuda_name"].upper()
    ref.require(row["allocation_ordinal"] == rank and pci(row["pci_bus_id"]) == cuda_pci(rank)
                and (name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090") or "RTX PRO 6000" in name),
                f"Unqualified device {row}")
    torch.cuda.set_device(rank)
    if render:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(ref.egl_device(rank))
        from mujoco.egl import GLContext
        from OpenGL import GL
        context = GLContext(16, 16)
        try:
            context.make_current()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            ref.require("NVIDIA" in vendor.upper(), f"Unexpected renderer vendor {vendor}")
            print(f"CUDA {rank}, EGL {os.environ['MUJOCO_EGL_DEVICE_ID']}, {vendor}", flush=True)
        finally:
            context.free()
    return f"cuda:{rank}"


@contextlib.contextmanager
def batches(job, log):
    """Use the canonical replay format. Elect the first complete batch with link(2)."""
    import torch
    from reground.eval import rollout

    directory = ref.STORE / "batches" / ref.label(job)
    directory.mkdir(parents=True, exist_ok=True)
    native = rollout.run_batch
    index = 0

    def batch(vla, pool, episodes, ctrl, cell, **kwargs):
        nonlocal index
        key = [(e.task, e.init_id, e.rep, e.scene_seed) for e in episodes]
        path = directory / f"batch{index:04d}.pt"
        index += 1
        ref.require(cell == job["cell"] and len(episodes) == ref.BATCH and getattr(ctrl, "gen", None) is None,
                    "Expected the native four-slot base collector")
        if not path.exists():
            result = native(vla, pool, episodes, ctrl, cell, **kwargs)
            raw = SimpleNamespace(traces=[{k: tr[k] for k in ("X", "A", "task")} for tr in result.traces])
            saved = dict(episodes=key, upstream=ref.PIN, policy=ref.POLICY, result=raw, rng=None)
            with tempfile.NamedTemporaryFile(dir=directory, prefix=".part-", suffix=".tmp") as tmp:
                torch.save(saved, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())
                try:
                    os.link(tmp.name, path)
                except FileExistsError:
                    pass
                else:
                    log(f"[batch-checkpoint] {path} episodes={len(episodes)}")
                    return result
        saved = torch.load(path, map_location="cpu", weights_only=False)
        ref.require(saved["episodes"] == key and saved["policy"] == ref.POLICY and saved["upstream"] == ref.PIN
                    and saved["rng"] is None, f"Wrong batch identity at {path}")
        raw = saved["result"]
        ref.require([tr["task"] for tr in raw.traces] == [e.task for e in episodes], f"Wrong tasks at {path}")
        log(f"[replay] {path}")
        return raw

    with patch.object(rollout, "run_batch", batch):
        yield


def link_pair(seal, dest):
    """Publish the elected pair, sidecar first. A retry can finish either link."""
    src = seal / dest.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".json", ".npz"):
        source, target = src.with_suffix(suffix), dest.with_suffix(suffix)
        try:
            os.link(source, target)
        except FileExistsError:
            ref.require(os.path.samefile(source, target),
                        f"Foreign publication at {target}. Inspect the legacy writer or orphan before retry.")


def validate_pair(dest, job):
    from dataclasses import asdict
    from reground.data.babble import BabbleCfg

    facts = ref.check_pool_shard(dest, job)
    meta = ref.read(dest.with_suffix(".json"))
    if job["kind"] == "babble":
        ref.require(meta["cfg"] == asdict(BabbleCfg()), f"Wrong babble config at {dest}")
    else:
        ref.require(meta["x0_fixed"] is True, f"Wrong collection metadata at {dest}")
    return facts


def existing(dest, seal, job):
    if seal.is_dir():
        for suffix in (".json", ".npz"):
            ref.require(os.path.samefile((seal / dest.name).with_suffix(suffix), dest.with_suffix(suffix)),
                        f"Public pair differs from the elected pair at {dest}")
    return "existing", validate_pair(dest, job)


def one(job, ctx, attempt):
    dest = ref.output(job)
    seal = ref.STORE / "partition-seals" / ref.label(job)
    if dest.exists():
        return existing(dest, seal, job)
    # Preserve the original lock path for any local co-operating legacy lane.
    with ref.claim(job) as mine:
        if not mine:
            return "busy", {}
        if dest.exists():
            return existing(dest, seal, job)
        event = "recovered"
        if not seal.exists():
            private = attempt / "partial" / ref.label(job)
            pair = private / "pair"
            with patch.object(ref, "resume_batches", batches):
                ref.perform(job, ctx, private / "work", pair / dest.name)
            candidate = pair / dest.name
            hashes = {suffix: hashlib.sha256(candidate.with_suffix(suffix).read_bytes()).hexdigest()
                      for suffix in (".json", ".npz")}
            ref.write(private / "candidate.json", dict(unit=ref.label(job), sha256=hashes,
                                                       host=socket.gethostname(), pid=os.getpid()))
            seal.parent.mkdir(parents=True, exist_ok=True)
            # Both directories live on the project filesystem. An atomic rename to
            # a nonempty directory cannot replace a previous winner on Linux.
            try:
                pair.rename(seal)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                    raise
            else:
                event = "published"
        validate_pair(seal / dest.name, job)
        link_pair(seal, dest)
        return event, validate_pair(dest, job)


def run_lane(args):
    from reground.eval.jobs import WorkerCtx
    device = gpu(args.rank, args.attempt, args.gpus, args.phase == "collect")
    ctx = WorkerCtx("pi05", device, ref.BATCH, lambda msg: print(msg, flush=True))
    jobs = selected(args.phase, args.seeds, args.shards, args.descending)[args.rank::args.gpus]
    try:
        with (args.attempt / f"{args.phase}-gpu{args.rank}.jsonl").open("x", buffering=1) as stream:
            for job in jobs:
                started = time.monotonic()
                key = ref.label(job)
                print(json.dumps(dict(event="check", unit=key, utc=time.time())), file=stream)
                event, facts = one(job, ctx, args.attempt)
                print(json.dumps(dict(event=event, unit=key, utc=time.time(),
                                      wall_s=time.monotonic() - started, **facts)), file=stream)
    finally:
        ctx.close()


def summary(args):
    missing = []
    jobs = selected(args.phase, args.seeds, args.shards, args.descending)
    for job in jobs:
        key = ref.label(job)
        dest = ref.output(job)
        if dest.exists():
            existing(dest, ref.STORE / "partition-seals" / key, job)
        else:
            missing.append(key)
    result = dict(planned=len(jobs), sealed=len(jobs) - len(missing), outstanding=missing)
    ref.write(args.attempt / "summary.json", result)
    print(json.dumps(result), flush=True)
    return 75 if missing else 0
