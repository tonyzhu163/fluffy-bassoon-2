"""Transport for the released ReGround v1 rotation experiment. See REFERENCE.md."""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import fcntl
import gzip
import hashlib
import json
import os
import signal
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("REGROUND_RUNTIME_ROOT", HERE))
SOURCE = Path(os.environ.get("REGROUND_UPSTREAM", ROOT / "upstream")).resolve()
STORE = ROOT / "reference-native"
NATIVE = STORE / "runs"
PIN = "16a98486f560676b9036d1a6d067b9dd51a5917a"
POLICY = "7152f13e34c6c115d808aea0e5d08aca74eac14b"
CELLS = ("none/0", "rot_z/90")
PHASES = ("babble", "collect", "fit", "base", "reground")
BATCH = 4  # main.py --envs; affects collection overshoot AND controller RNG.
SEEDS = tuple(range(5))


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def configure():
    revision = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    require(revision == PIN, f"Wrong upstream revision {revision}")
    changes = subprocess.check_output(["git", "-C", str(SOURCE), "diff", "--name-only", "HEAD",
                                       "--", "*.py", "*.json", "requirements.txt", "README.md"], text=True)
    require(not changes.strip(), f"Non-release files in upstream source/config: {changes}")
    require(os.environ.get("REGROUND_NEXEC", "10") == "10", "Release executes 10 actions per chunk")
    sys.path.insert(0, str(SOURCE))
    import reground.utils as utils
    require(Path(utils.__file__).resolve().parent.parent == SOURCE, "Wrong imported upstream")
    # Set before importing consumers (which use 'from ..utils import RUNS').
    utils.RUNS = NATIVE


def grid():
    with gzip.open(HERE / "evidence/rotation-jobs.json.gz", "rt") as f:
        return json.load(f)


def units(phase, seeds):
    """Use native task order, shard sizing, task batching and controller reset boundaries."""
    from reground.eval.jobs import shard_jobs
    from util import resolve_tasks

    if phase in ("base", "reground"):
        jobs = [{**j, "method": phase, "cfg": {} if phase == "base" else j["cfg"]}
                for j in grid() if j["seed"] in seeds]
        return shard_jobs(jobs, 4 * BATCH)
    if phase == "fit":
        return [dict(kind="fit", seed=s) for s in seeds]
    size = 2000 if phase == "babble" else 4000
    return [dict(kind=phase, policy="pi05", cell=cell, seed=seed, shard=i, n=size,
                 tasks=resolve_tasks("all")) for i in range(100000 // size)
            for cell in CELLS for seed in seeds]


def label(job):
    if job["kind"] == "fit":
        return f"seed{job['seed']}/fit"
    return (f"seed{job['seed']}/{job['cell']}/{job.get('method', job['kind'])}"
            f"/shard{job['shard']:03d}")


def output(job):
    if job["kind"] == "fit":
        return STORE / "fitted" / f"seed{job['seed']}.json"
    if job["kind"] == "eval":
        return STORE / "shards" / label(job)
    kind = "babble" if job["kind"] == "babble" else "vla_pi05"
    return NATIVE / "data" / job["cell"] / f"seed{job['seed']}" / kind / f"shard{job['shard']:03d}.npz"


@contextlib.contextmanager
def claim(job):
    path = STORE / "locks" / (label(job) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        yield True


def check_pool_shard(path, job):
    import numpy as np
    meta = read(path.with_suffix(".json"))
    with np.load(path, allow_pickle=False) as data:
        X, A, lens, tasks = (data[k] for k in ("X", "A", "lens", "task"))
    require(lens.ndim == 1 and lens.dtype.kind in "iu" and (lens > 0).all(), path)
    n = int(lens.sum())
    require(A.shape == (n, 7) and X.shape == (n + len(lens), 25) and len(tasks) == len(lens), path)
    require(np.isfinite(X).all() and np.isfinite(A).all(), path)
    require(meta["n"] == n and n >= job["n"] and meta["cell"] == job["cell"]
            and meta["seed"] == job["seed"], path)
    if job["kind"] == "babble":
        require(meta["shard"] == job["shard"] and meta["tasks"] == job["tasks"], path)
        expected = [job["tasks"][job["shard"] % len(job["tasks"])]] * len(tasks)
        quantum = 1
    else:
        require(meta["policy"] == "pi05" and meta["git_sha"] == PIN and meta["vla_revision"] == POLICY, path)
        quantum = BATCH
        expected = [job["tasks"][(job["shard"] + k) % len(job["tasks"])] for k in range(len(tasks))]
    require(tasks.tolist() == expected and len(lens) % quantum == 0 and
            int(lens[:-quantum].sum()) < job["n"], f"Invalid task order or stopping boundary: {path}")
    return {"transitions": n, "episodes": len(lens)}


def audit_data(seed):
    from reground.data import babble
    from reground.method import build
    report = {}
    for cell, budget in zip(CELLS, (100000, 10000)):
        counts = {}
        for phase in ("babble", "collect"):
            planned = [j for j in units(phase, [seed]) if j["cell"] == cell]
            paths = [output(j) for j in planned]
            require(set(paths[0].parent.glob("*.npz")) == set(paths), f"Incomplete/extra {cell} {phase} shards")
            facts = [check_pool_shard(p, j) for p, j in zip(paths, planned)]
            counts[phase] = {k: sum(x[k] for x in facts) for k in facts[0]}
        held = babble.heldout_vla_pool(cell, seed, "pi05")
        require(len(held) == 5, "Expected the released five-episode VLA holdout")
        used = babble.adaptation_data(cell, seed, budget, 0.6)
        require(babble.n_transitions(used) == budget, "Incomplete fit inputs")
        counts.update(used_babble=int(0.6 * budget), used_vla=budget - int(0.6 * budget),
                      heldout_vla_episodes=len(held), heldout_vla_transitions=babble.n_transitions(held))
        report[cell] = counts
    require((build.NOM_BUDGET, build.DEFAULT_BUDGET, build.NOM_EPS, build.DEFAULT_EPS)
            == (100000, 10000, 0.6, 0.6), "Unexpected fit budget")
    return report


def fit(seed, device, stage, log):
    """Native model construction and fitting. Publish each complete native cache directory."""
    import torch
    from reground.method import build
    from reground.models import cache
    from reground.utils import stable_hash
    facts = audit_data(seed)
    build._MEMO.clear()
    fitted = []

    def load(spec, device, **kw):
        key = stable_hash(spec)
        dest = cache.spec_dir(spec)
        if not dest.exists():
            # Init happens before model.fit calls manual_seed upstream. Retain the actual
            # pre-construction CPU generator state so an interrupted fit can replay it.
            state = STORE / "fit-starts" / f"{key}.pt"
            if not state.exists():
                state.parent.mkdir(parents=True, exist_ok=True)
                tmp = state.with_suffix(".tmp")
                torch.save(torch.get_rng_state(), tmp)
                tmp.replace(state)
            torch.set_rng_state(torch.load(state, map_location="cpu", weights_only=True))
            with patch.object(cache, "RUNS", stage):
                model = cache.load_or_fit(spec, device, **kw)
                src = cache.spec_dir(spec)
            require(read(src / "spec.json") == spec and (src / "fit_metrics.json").is_file(), src)
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
        else:
            require(read(dest / "spec.json") == spec and (dest / "fit_metrics.json").is_file(), dest)
            model = cache.load_or_fit(spec, device, fit_ok=False, log=log)
        fitted.append({"dir": str(dest.relative_to(STORE)), "kind": spec["kind"], "role": spec["role"]})
        log(f"[checkpoint] {dest}")
        return model

    cfg = grid()[seed]["cfg"]
    try:
        with patch.object(build, "load_or_fit", load):
            build.nominal_models(seed, device, log=log)
            build.make_controller("reground", cfg, CELLS[1], seed, device, log=log, policy="pi05")
    finally:
        build._MEMO.clear()
    require(len(fitted) == 4, "Incomplete nominal/deployed model set")
    return {"models": fitted, "data": facts}


def check_eval(path, job):
    import numpy as np
    from reground.envs.libero_env import max_steps, task_suite
    from reground.eval.jobs import FIELDS, load_traces, speed_limit
    from reground.utils import cfg_id, noise_seed
    with (path / "results.csv").open() as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == FIELDS, "Unexpected upstream CSV schema")
        rows = list(reader)
    traces = load_traces(path)
    expected = [(t, i, r) for t, i, r in job["episodes"]]
    keys = [(r["task"], int(r["init_id"]), int(r["rep"])) for r in rows]
    require(keys == expected and set(traces) == set(keys), f"Incomplete or duplicate outcomes/traces: {path}")
    meta = read(path / "manifest.json")
    require(meta["git_sha"] == PIN and meta["vla_revision"] == POLICY and meta["cfg"] == job["cfg"]
            and meta["n_grid"] == len(job["full_episodes"]), path)
    for (t, i, rep), row in zip(keys, rows):
        steps = int(row["steps"])
        require(0 < steps <= max_steps(t) and (row["success"] == "1" or steps == max_steps(t)), row)
        require(row["success"] in ("0", "1") and int(row["noise_seed"]) == noise_seed(i, rep)
                and row["suite"] == task_suite(t) and row["method"] == job["method"]
                and row["policy"] == "pi05" and row["cfg_id"] == cfg_id(job["cfg"])
                and row["split"] == "test" and row["shift"] == "rot_z" and float(row["level"]) == 90
                and int(row["adapt_seed"]) == job["seed"], row)
        trace = traces[t, i, rep]
        for field, shape in (("X", (steps + 1, 25)), ("A", (steps, 7)), ("A_vla", (steps, 7)),
                             ("ref", (steps, 3)), ("speed", (steps,))):
            require(trace[field].shape == shape, f"Wrong {field} trace shape in {path}")
            if field != "ref" or job["method"] != "base":
                require(np.isfinite(trace[field]).all(), f"Nonfinite {field} trace in {path}")
        viol = int((trace["speed"] > speed_limit("pi05")).sum())
        unsafe = any(int(row[k]) > 0 for k in ("displaced", "fallen", "contact_steps", "speed_viol"))
        require(int(row["speed_viol"]) == viol and int(row["unsafe"]) == int(unsafe), row)
    return {"episodes": len(keys), "transitions": sum(int(r["steps"]) for r in rows)}


def strict_policy(ctx):
    import io
    # The stock LeRobot loader can catch a weight-load exception and return an
    # initialized policy. The runtime worker qualified this exact strict-success line.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            policy = ctx.vla
    finally:
        print(buf.getvalue(), flush=True)
    require(policy.revision == POLICY and "All keys loaded successfully!" in buf.getvalue(),
            "Frozen policy revision/strict weight load failed")


def babble_child(job, stage):
    from dataclasses import asdict
    from reground.data import babble
    with patch.object(babble, "RUNS", stage):
        return babble._babble_worker((job["cell"], job["seed"], job["shard"], job["n"],
                                      job["tasks"], asdict(babble.BabbleCfg())))


@contextlib.contextmanager
def resume_batches(job, log):
    """Replay sealed native batches and the controller's post-batch CEM RNG."""
    import torch
    from reground.eval import rollout
    from types import SimpleNamespace

    directory = STORE / "batches" / label(job)
    directory.mkdir(parents=True, exist_ok=True)
    native = rollout.run_batch
    index = 0

    def batch(vla, pool, episodes, ctrl, cell, **kwargs):
        nonlocal index
        key = [(e.task, e.init_id, e.rep, e.scene_seed) for e in episodes]
        path = directory / f"batch{index:04d}.pt"
        index += 1
        generator = getattr(ctrl, "gen", None)
        if path.exists():
            # These are this runner's own private checkpoints. The native result
            # includes NumPy arrays; weights_only cannot deserialize that type.
            saved = torch.load(path, map_location="cpu", weights_only=False)
            require(saved["episodes"] == key and saved["policy"] == POLICY and saved["upstream"] == PIN,
                    f"Batch checkpoint identity differs: {path}")
            if generator is not None:
                generator.set_state(saved["rng"])
            log(f"[replay] {path}")
            return saved["result"]
        result = native(vla, pool, episodes, ctrl, cell, **kwargs)
        if job["kind"] == "collect":
            raw = SimpleNamespace(traces=[{k: t[k] for k in ("X", "A", "task")} for t in result.traces])
        else:
            raw = result
        saved = dict(episodes=key, upstream=PIN, policy=POLICY, result=raw,
                     rng=generator.get_state() if generator is not None else None)
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(saved, tmp)
        tmp.replace(path)
        log(f"[batch-checkpoint] {path} episodes={len(episodes)}")
        return result

    with patch.object(rollout, "run_batch", batch):
        yield


def perform(job, ctx, stage, dest):
    from multiprocessing import get_context
    from reground.eval import jobs
    from reground.method import build
    from reground.models import cache
    stage.mkdir(parents=True, exist_ok=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if job["kind"] == "fit":
        receipt = fit(job["seed"], ctx.device, stage, ctx.log)
        write(dest, receipt)
        return {"models": len(receipt["models"])}
    if job["kind"] == "babble":
        # Match upstream maxtasksperchild=1. LIBERO leaks memory when rebuilding environments.
        with get_context("spawn").Pool(1, maxtasksperchild=1) as pool:
            raw, _ = pool.apply(babble_child, (job, stage))
        src = Path(raw)
    else:
        if job.get("method") == "reground":
            require(output({"kind": "fit", "seed": job["seed"]}).is_file(), "Run the fit phase first")
        if ctx._vla is None:
            strict_policy(ctx)
        # Both locks and all intermediate outputs live in a fresh per-attempt shard directory.
        with patch.object(jobs, "job_dir", lambda _: stage), resume_batches(job, ctx.log):
            if job["kind"] == "eval":
                def fitted_only(spec, device, **kw):
                    return cache.load_or_fit(spec, device, log=ctx.log, fit_ok=False)
                with patch.object(build, "load_or_fit", fitted_only):
                    jobs.run_eval_job(job, ctx)
                facts = check_eval(stage, job)
                stage.rename(dest)
                return facts
            jobs.run_collect_job(job, ctx)
        src = stage / dest.name
        meta = src.with_suffix(".json")
        write(meta, {**read(meta), "vla_revision": ctx.vla.revision})
    facts = check_pool_shard(src, job)
    # NPZ is the commit indicator, and the upstream loader only consumes *.npz.
    src.with_suffix(".json").replace(dest.with_suffix(".json"))
    src.replace(dest)
    return facts


def egl_device(rank):
    """Map explicit EGL enumeration to a CUDA ordinal via EGL_NV_device_cuda."""
    lib = ctypes.CDLL("libEGL.so.1")
    lib.eglGetProcAddress.restype = ctypes.c_void_p
    array = (ctypes.c_void_p * 64)()
    count = ctypes.c_int()
    enum_addr = lib.eglGetProcAddress(b"eglQueryDevicesEXT")
    require(enum_addr, "EGL_EXT_device_enumeration unavailable")
    enum = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                            ctypes.POINTER(ctypes.c_int))(enum_addr)
    require(enum(len(array), array, ctypes.byref(count)), "EGL device enumeration failed")
    addr = lib.eglGetProcAddress(b"eglQueryDeviceAttribEXT")
    require(addr, "EGL_EXT_device_query unavailable")
    query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
                             ctypes.POINTER(ctypes.c_ssize_t))(addr)
    found = []
    for i, dev in enumerate(array[:count.value]):
        cuda = ctypes.c_ssize_t(-1)
        if query(dev, 0x323A, ctypes.byref(cuda)) and cuda.value == rank:  # EGL_CUDA_DEVICE_NV
            found.append(i)
    require(len(found) == 1, f"EGL devices for CUDA ordinal {rank}: {found}")
    return found[0]


def run_lane(args):
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    from reground.eval import jobs
    from reground.method import build
    require(socket.gethostname() == read(args.attempt / "gpu.json")["hostname"], "Wrong host probe")
    device = f"cuda:{args.rank}"
    row = read(args.attempt / "gpu.json")["devices"][args.rank]
    gpu_name = row["cuda_name"].upper()
    qualified = (gpu_name in ("NVIDIA RPBZZZ6", "NVIDIA GEFORCE RTX 4090")
                 or "RTX PRO 6000" in gpu_name)
    require(row["allocation_ordinal"] == args.rank and pci(row["pci_bus_id"]) == cuda_pci(args.rank)
            and qualified, f"Unexpected GPU: {row}")
    torch.cuda.set_device(args.rank)
    egl = egl_device(args.rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    print(f"CUDA {device} PCI {row['pci_bus_id']} EGL {egl}", flush=True)
    if args.phase in ("collect", "base", "reground"):
        from mujoco.egl import GLContext
        from OpenGL import GL
        context = GLContext(16, 16)
        try:
            context.make_current()
            vendor = GL.glGetString(GL.GL_VENDOR).decode()
            renderer = GL.glGetString(GL.GL_RENDERER).decode()
            require("NVIDIA" in vendor.upper(), f"Wrong EGL vendor {vendor}: {renderer}")
            print(f"EGL vendor={vendor} renderer={renderer}", flush=True)
        finally:
            context.free()
    ctx = jobs.WorkerCtx("pi05", device, BATCH, lambda msg: print(msg, flush=True))
    progress = args.attempt / f"{args.phase}-gpu{args.rank}.jsonl"
    planned = units(args.phase, args.seeds)
    # Rotate priority by rank, then steal any unclaimed work. flock also protects
    # independent allocations sharing the store; the kernel releases locks on death.
    order = planned[args.rank::args.gpus] + [j for i, j in enumerate(planned) if i % args.gpus != args.rank]
    current_seed = None
    try:
        with progress.open("a", buffering=1) as f:
            for job in order:
                dest = output(job)
                if dest.exists():
                    continue
                with claim(job) as mine:
                    if not mine or dest.exists():
                        continue
                    if current_seed != job["seed"]:
                        build._MEMO.clear()
                        current_seed = job["seed"]
                    started = time.monotonic()
                    key = label(job)
                    f.write(json.dumps(dict(event="start", unit=key, utc=time.time())) + "\n")
                    facts = perform(job, ctx, args.attempt / "partial" / key, dest)
                    seconds = time.monotonic() - started
                    event = dict(event="published", unit=key, utc=time.time(), wall_s=seconds, **facts)
                    event.update({k + "_per_s": facts[k] / seconds for k in facts})
                    f.write(json.dumps(event) + "\n")
                    ctx.log(event)
    finally:
        ctx.close()


def summarize():
    import pandas as pd
    from reground.eval.jobs import FIELDS, job_dir
    from reground.eval.stats import summarise
    results = {}
    for method in ("base", "reground"):
        frames = []
        for seed in SEEDS:
            planned = units(method, [seed])
            seed_frames = []
            for job in planned:
                d = output(job)
                check_eval(d, job)
                seed_frames.append(pd.read_csv(d / "results.csv"))
            frame = pd.concat(seed_frames, ignore_index=True)
            require(len(frame) == len(frame.drop_duplicates(["task", "init_id", "rep"])) == 280,
                    "Incomplete seed grid")
            frames.append(frame)
            d = job_dir(planned[0])
            d.mkdir(parents=True, exist_ok=True)
            tmp = d / f"results.{os.getpid()}.tmp"
            frame.to_csv(tmp, index=False, columns=FIELDS)
            tmp.replace(d / "results.csv")
        frame = pd.concat(frames, ignore_index=True)
        results[method] = summarise(frame)
        results[method]["per_task"] = frame.groupby("task")[["success", "unsafe"]].mean().to_dict("index")
    write(STORE / "summary.json", results)
    print(json.dumps(results, indent=2))


def dispatch(args):
    """Own all lane process groups, including the upstream spawned simulator children."""
    processes = []
    logs = []

    def terminated(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminated)
    try:
        for rank in range(args.gpus):
            log = (args.attempt / f"{args.phase}-gpu{rank}.log").open("a", buffering=1)
            logs.append(log)
            command = [sys.executable, "-Bu", __file__, "lane", "--attempt", str(args.attempt),
                       "--phase", args.phase, "--gpus", str(args.gpus), "--rank", str(rank),
                       "--seeds", *map(str, args.seeds)]
            processes.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True))
        while any(p.poll() is None for p in processes):
            require(all(p.returncode in (None, 0) for p in processes), "Lane failed; see its log")
            time.sleep(1)
        require(all(p.returncode == 0 for p in processes), "Lane failed; see its log")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for p in processes:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(p.pid, sig)
            if sig == signal.SIGTERM:
                time.sleep(1)
        for p in processes:
            p.wait()
        for log in logs:
            log.close()
    missing = [label(j) for j in units(args.phase, args.seeds) if not output(j).exists()]
    require(not missing, f"{len(missing)} units outstanding; first: {missing[:3]}")


def reserve(args):
    """Reserve gross allocated wall time under the parent's reference-only ceiling."""
    attempts = args.attempt.parent
    with (attempts / ".budget.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = 0
        for record in attempts.glob("*/reservation.json"):
            item = read(record)
            finish = record.parent / "exit.json"
            previous += read(finish)["allocated_gpu_seconds"] if finish.exists() else item["max_gpu_seconds"]
        charge = args.gpus * args.wall
        require(args.cap > 0 and args.cap <= 50 and previous + charge <= args.cap * 3600,
                f"Reference reservation {previous}+{charge} GPU-seconds exceeds cap {args.cap} GPUh")
        runtime_files = ("runtime/activate.sh", "runtime/offline/sitecustomize.py", "resolved.txt",
                         "hf-verified.jsonl", "assets-verified.json", "libero-config/config.yaml")
        hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                  for name in runtime_files if (ROOT / name).is_file()}
        write(args.attempt / "reservation.json",
              dict(runtime_files_sha256=hashes, upstream=PIN, policy=POLICY, host=socket.gethostname(), code=os.environ["REFERENCE_CODE_PIN"], gpus=args.gpus,
                   phase=os.environ["REFERENCE_PHASE"], seeds=args.seeds, wall_seconds=args.wall,
                   max_gpu_seconds=charge, prior_gpu_seconds=previous, cap_gpuh=args.cap,
                   started_utc=float(os.environ["REFERENCE_STARTED"])))


def status(seeds):
    result = {}
    for phase in PHASES:
        planned = units(phase, seeds)
        result[phase] = {"published": sum(output(j).exists() for j in planned), "total": len(planned)}
    attempts = []
    for reservation in sorted((STORE / "attempts").glob("*/reservation.json")):
        item = read(reservation)
        folder = reservation.parent
        released = read(folder / "exit.json") if (folder / "exit.json").exists() else {}
        latest = {}
        for stream in sorted(folder.glob("*.jsonl")):
            total_s, done = 0.0, 0
            last = None
            with stream.open() as f:
                for line in f:
                    if not line.endswith("\n"):
                        continue  # active writer
                    event = json.loads(line)
                    last = event
                    if event["event"] == "published":
                        done += 1
                        total_s += event["wall_s"]
            latest[stream.stem] = dict(last=last, published=done,
                                       measured_unit_s=total_s / done if done else None)
        attempts.append(dict(id=folder.name, **item, **released, lanes=latest))
    return {"phases": result, "attempts": attempts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "lane", "dispatch", "reserve", "summarize"))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=SEEDS)
    parser.add_argument("--gpus", type=int, choices=(2, 4, 5))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--wall", type=int)
    parser.add_argument("--cap", type=float)
    args = parser.parse_args()
    require(len(set(args.seeds)) == len(args.seeds), "Repeated seed")
    if args.action == "reserve":
        reserve(args)
        return
    configure()
    if args.action == "status":
        print(json.dumps(status(args.seeds), indent=2))
    elif args.action == "summarize":
        summarize()
    elif args.action == "dispatch":
        dispatch(args)
    else:
        require(args.attempt and args.gpus and args.phase and 0 <= args.rank < args.gpus, "Missing lane args")
        run_lane(args)


if __name__ == "__main__":
    main()
