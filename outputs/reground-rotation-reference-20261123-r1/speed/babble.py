"""One camera-free native babble child per shard, on a Linux CPU process pool."""
import argparse
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import resource
import socket
import sys
import time
import traceback

import reference as ref


def cpu_capacity():
    # Baige may expose the host's full affinity mask alongside a smaller pod quota.
    root = Path("/sys/fs/cgroup")
    count = len(os.sched_getaffinity(0))
    groups = [line.split(":", 2) for line in Path("/proc/self/cgroup").read_text().splitlines()]
    if (root / "cpu.max").exists():
        path = next(p for _, controllers, p in groups if controllers == "")
        quota_file = "cpu.max"
    else:
        root = next(p for p in (root / "cpu", root / "cpu,cpuacct")
                    if (p / "cpu.cfs_quota_us").exists())
        path = next(p for _, controllers, p in groups if "cpu" in controllers.split(","))
        quota_file = "cpu.cfs_quota_us"
    # Some containers mount their own cgroup at the root. Others expose its ancestors.
    group = root / path.lstrip("/")
    if not (group / quota_file).exists():
        group = root
    while True:
        quota = (group / quota_file).read_text().split()
        if quota_file == "cpu.cfs_quota_us":
            quota.append((group / "cpu.cfs_period_us").read_text().strip())
        if quota[0] not in ("max", "-1"):
            count = min(count, int(quota[0]) // int(quota[1]))
        if group == root:
            break
        group = group.parent
    return count


def preload():
    # Never fork a policy, CUDA/GL context, live simulator, or multithreaded parent.
    ref.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU payload requires hidden CUDA")
    start = time.monotonic()
    import torch
    from lerobot.envs.libero import LiberoEnv
    from reground.data import babble
    from reground.envs.libero_env import TaskEnv
    ref.require(not torch.cuda.is_initialized(), "Parent initialized CUDA")
    for name in ("mujoco.egl", "robosuite.renderers.context.egl_context"):
        module = sys.modules.get(name)
        ref.require(getattr(module, "EGL_DISPLAY", None) is None, f"Parent initialized {name}")
    threads = len(os.listdir("/proc/self/task"))
    ref.require(threads == 1, f"Unsafe to fork parent with {threads} native threads")
    return dict(seconds=time.monotonic() - start, threads=threads, cuda_initialized=False,
                torch=torch.__version__, lerobot_env=LiberoEnv.__module__, task_env=TaskEnv.__module__,
                babble_source=babble.__file__, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def one(job, attempt, lane):
    # Take flock in the disposable child so no sibling or parent inherits a claim.
    key = ref.label(job)
    dest = ref.output(job)
    with (attempt / f"babble-cpu{lane}.log").open("a", buffering=1) as log, \
            redirect_stdout(log), redirect_stderr(log):
        try:
            with ref.claim(job) as mine:
                if not mine or dest.exists():
                    return
                stage = attempt / "partial" / key
                stage.mkdir(parents=True)
                dest.parent.mkdir(parents=True, exist_ok=True)
                start = time.monotonic()
                with (attempt / f"babble-cpu{lane}.jsonl").open("a", buffering=1) as events:
                    print(json.dumps(dict(event="start", unit=key, utc=time.time(), pid=os.getpid())), file=events)
                    raw, _ = ref.babble_child(job, stage)
                    src = Path(raw)
                    facts = ref.check_pool_shard(src, job)
                    # Same commit marker and two renames, in the same order as reference.perform.
                    src.with_suffix(".json").replace(dest.with_suffix(".json"))
                    src.replace(dest)
                    wall = time.monotonic() - start
                    print(json.dumps(dict(event="published", unit=key, utc=time.time(), wall_s=wall,
                                          rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                          transitions_per_s=facts["transitions"] / wall, **facts)), file=events)
        except BaseException:
            traceback.print_exc(file=log)
            raise


def run(jobs, attempt, workers):
    capacity = cpu_capacity()
    ref.require(1 <= workers <= min(capacity, 24), f"{workers} workers exceed CPU allowance {capacity}")
    pending = deque(j for j in jobs if not ref.output(j).exists())
    started = time.monotonic()
    active = {}
    if pending:
        info = preload()
        ref.write(attempt / "preload.json", dict(**info, workers=workers, cpu_capacity=capacity))
    ctx = mp.get_context("fork")
    free = list(reversed(range(workers)))
    try:
        while pending or active:
            while pending and free:
                job = pending.popleft()
                if ref.output(job).exists():
                    continue
                lane = free.pop()
                child = ctx.Process(target=one, args=(job, attempt, lane))
                child.start()
                active[child.sentinel] = (lane, job, child)
            if active:
                for fd in wait(active):
                    lane, job, child = active.pop(fd)
                    child.join()
                    rc = child.exitcode
                    child.close()
                    ref.require(rc == 0, f"{ref.label(job)} exited {rc}, see babble-cpu{lane}.log")
                    free.append(lane)
    finally:
        for _, _, child in active.values():
            child.kill()
            child.join()
            child.close()
    # A legacy lane may still hold the last missing shards. Return those to the scheduler.
    missing = []
    for job in jobs:
        path = ref.output(job)
        if path.exists():
            ref.check_pool_shard(path, job)
        else:
            missing.append(ref.label(job))
    result = dict(planned=len(jobs), verified=len(jobs) - len(missing), outstanding=missing,
                  wall_s=time.monotonic() - started)
    ref.write(attempt / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["reserve", "run"])
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--workers", type=int, required=True, choices=range(8, 25))
    parser.add_argument("--wall", type=int, required=True)
    parser.add_argument("--cap", type=float, required=True)
    args = parser.parse_args()
    host = socket.gethostname()
    ref.require(host.startswith("job-") and host.endswith("-master-0") and
                os.environ.get("KUBERNETES_SERVICE_HOST") and os.environ.get("WORLD_SIZE", "1") == "1",
                "Scored work requires a Baige Distributed Training master")
    ref.require(args.attempt.resolve().parent == ref.STORE / "attempts", "Wrong attempt store")
    if args.action == "reserve":
        args.seeds, args.gpus = list(ref.SEEDS), 1
        ref.reserve(args)
        return
    reservation = ref.read(args.attempt / "reservation.json")
    probe = ref.read(args.attempt / "gpu.json")
    ref.require(reservation["host"] == probe["hostname"] == host and reservation["gpus"] == 1 and
                reservation["phase"] == "babble" and probe["status"] == "passed" and len(probe["devices"]) == 1
                and probe["devices"][0]["cuda_name"].upper() == "NVIDIA GEFORCE RTX 4090", "Wrong one-card probe")
    ref.configure()
    # Plan in a separate process. Its pandas/Arrow imports can start background threads.
    import subprocess
    script = ("import json, reference as r; r.configure(); "
              "print(json.dumps(r.units('babble', r.SEEDS)))")
    jobs = json.loads(subprocess.check_output([sys.executable, "-B", "-c", script]))
    result = run(jobs, args.attempt, args.workers)
    print(json.dumps(result), flush=True)
    return 75 if result["outstanding"] else 0


if __name__ == "__main__":
    sys.exit(main())
