"""CPU timings for native four-slot reset/respawn and strict model initialization, without inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import reference as ref


def load_only():
    ref.configure()
    from reground.eval.jobs import WorkerCtx
    ctx = WorkerCtx("pi05", "cpu", ref.BATCH, print)
    t = time.monotonic()
    ref.strict_policy(ctx)
    try:
        print(json.dumps(dict(strict_s=time.monotonic() - t, native_load_s=ctx.vla.load_s,
                              revision=ctx.vla.revision)))
    finally:
        ctx.close()


def load_probe(cache):
    t = time.monotonic()
    env = dict(os.environ, HF_HOME=str(cache.parent), HF_HUB_CACHE=str(cache))
    lines = subprocess.check_output([sys.executable, "-Bu", __file__, "load"], env=env, text=True)
    result = json.loads(lines.splitlines()[-1])
    return dict(process_s=time.monotonic() - t, **result)


def verify_copy(cache, local):
    rows = [json.loads(line) for line in (ref.ROOT / "hf-verified.jsonl").read_text().splitlines()]
    needed = sum(row["bytes"] for row in rows)
    ref.require(shutil.disk_usage(local).free > needed, "Insufficient node-local space")
    t = time.monotonic()
    for row in rows:
        repo = "models--" + row["repo"].replace("/", "--")
        path = Path(repo, "snapshots", row["rev"], row["file"])
        target = local / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache / path, target)
        with target.open("rb") as f:
            ref.require(target.stat().st_size == row["bytes"] and
                        hashlib.file_digest(f, "sha256").hexdigest() == row["sha256"], path)
        head = Path(repo, "refs", "main")
        ref.require((cache / head).read_text() == row["rev"], head)
        (local / head).parent.mkdir(exist_ok=True)
        (local / head).write_text(row["rev"])
    return dict(bytes=needed, files=len(rows), copy_and_verify_s=time.monotonic() - t)


def resets():
    ref.configure()
    from reground.eval.rollout import EnvPool
    from util import resolve_tasks
    tasks = resolve_tasks("all")
    t = time.monotonic()
    pool = EnvPool(4, cameras=False)
    startup = time.monotonic() - t
    times = []
    try:
        for offset in (0, 0, 4, 4, 0):
            before = [p.pid for p in pool.procs]
            batch = [(name, "rot_z/90", 15, 0, False) for name in tasks[offset:offset + 4]]
            t = time.monotonic()
            pool.call(range(4), "reset", batch)
            times.append(dict(tasks=[b[0] for b in batch], seconds=time.monotonic() - t,
                              replaced=sum(a != p.pid for a, p in zip(before, pool.procs))))
    finally:
        pool.close()
    return dict(pool_start_s=startup, resets=times)


def bytecode(output):
    source = Path(os.environ["PYTHONPYCACHEPREFIX"])
    def probe(prefix):
        env = dict(os.environ, PYTHONPYCACHEPREFIX=str(prefix))
        lines = subprocess.check_output([sys.executable, "-Bu", __file__, "resets"], env=env, text=True)
        return json.loads(lines.splitlines()[-1])
    shared = probe(source)
    with tempfile.TemporaryDirectory(prefix="reground-bytecode-") as tmp:
        local = Path(tmp) / "pycache"
        t = time.monotonic()
        ref.require(source.stat().st_dev != Path(tmp).stat().st_dev, "Local scratch on shared mount")
        shutil.copytree(source, local)
        files = [p for p in source.rglob("*") if p.is_file()]
        nbytes = 0
        for p in files:
            dst = local / p.relative_to(source)
            ref.require(p.read_bytes() == dst.read_bytes(), f"Bytecode copy differs at {p}")
            nbytes += dst.stat().st_size
        cost = time.monotonic() - t
        staged = probe(local)
    ref.write(output, dict(shared_before=shared, local=staged, shared_after=probe(source),
                           copy_and_verify_s=cost, files=len(files), bytes=nbytes))


def measure(output):
    result = resets()
    ref.write(output, result)
    cache = Path(os.environ["HF_HUB_CACHE"])
    result["shared_before"] = load_probe(cache)
    ref.write(output, result)
    with tempfile.TemporaryDirectory(prefix="reground-cpu-io-") as tmp:
        local = Path(tmp) / "hub"
        local.mkdir()
        ref.require(cache.stat().st_dev != local.stat().st_dev, "Local scratch on shared mount")
        result["stage"] = verify_copy(cache, local)
        result["local"] = load_probe(local)
    result["shared_after"] = load_probe(cache)
    ref.write(output, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("load", "measure", "resets", "bytecode"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ref.require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU probe requires hidden CUDA")
    if args.action == "load":
        load_only()
    elif args.action == "resets":
        print(json.dumps(resets()))
    elif args.action == "bytecode":
        bytecode(args.output)
    else:
        measure(args.output)
