"""Fit or deploy a sealed native ReGround cell. Use launch.sh in the scheduler."""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import socket
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch

from common import (BUDGETS, CELL, HERE, SOURCE_STORE, WORK, cell_dir, configure, digest, initial_state,
                    recipe, ref, reference_model, sha, subset_hash)


def forbidden(*args, **kwargs):
    raise RuntimeError("Unbudgeted adaptation/calibration read requested")


class Inputs:
    def __init__(self, seed, budget):
        self.seed, self.budget = seed, budget
        self.manifest = ref.read(HERE / "evidence" / f"seed{seed}.json")
        self.record = self.manifest["budgets"][str(budget)]
        self.tape = cell_dir(seed, budget) / "prefix.npz"
        self.cfg = recipe(seed, budget)

    def spec(self, role, kind):
        model = reference_model(self.seed, role, kind)
        spec = dict(model["spec"])
        if role == "dep":
            spec["budget"] = self.budget
        return spec

    def load_tape(self):
        from reground.data import babble
        ref.require(sha(self.tape) == self.record["tape_sha256"], self.tape)
        eps = babble.load_episodes(self.tape)
        ref.require(babble.n_transitions(eps) == self.budget and subset_hash(eps) == self.record["sha256"],
                    "Wrong bounded tape")
        return eps


# The release is trusted pinned Python. Audit unexpected opens in its calibration paths,
# including accidental direct opens bypassing the loader instrumentation below.
_guard = None


def audit(event, args):
    if _guard is None or event != "open" or not isinstance(args[0], (str, bytes)):
        return
    raw = os.fsdecode(args[0])
    if not os.path.isabs(raw):
        raw = os.path.abspath(raw)
    roots, allowed = _guard
    if raw.startswith(roots) and Path(raw).resolve() not in allowed:
        raise RuntimeError(f"Unexpected calibration file read: {raw}")


sys.addaudithook(audit)


@contextmanager
def bounded(inputs, fit):
    global _guard
    from reground.data import babble
    from reground.method import build
    from reground.models import cache
    allowed = {SOURCE_STORE / reference_model(inputs.seed, "nom", k)["dir"] / "model.pt"
               for k in ("fwd", "idm")}
    if fit:
        allowed |= {inputs.tape, *(initial_state(inputs.seed, k) for k in ("fwd", "idm"))}
    _guard = (tuple(str(p) + "/" for p in (SOURCE_STORE, WORK / "inputs", WORK / "private-runs/data")),
              {p.resolve() for p in allowed})
    try:
        with ExitStack() as ctx:
            for mod, name in ((build, "adaptation_data"), (build, "estimated_delay"), (babble, "load_pool"),
                              (babble, "adaptation_data"), (babble, "heldout_vla_pool"),
                              (cache, "adaptation_data"), (cache, "heldout_vla_pool")):
                ctx.enter_context(patch.object(mod, name, forbidden))
            yield
    finally:
        _guard = None


def fit_one(inputs, directory, kind, device):
    from reground.models import cache
    from reground.utils import stable_hash
    import torch
    spec = inputs.spec("dep", kind)
    init = initial_state(inputs.seed, kind)
    model_init = inputs.manifest["reference_models"]["dep_" + kind]["init_sha256"]
    identity = dict(spec=spec, upstream=ref.PIN, prefix_sha256=inputs.record["sha256"],
                    init_sha256=model_init, fit_device=device)
    target = directory / "models" / stable_hash(spec)
    if target.exists():
        return verify_fitted(inputs, directory, kind, device)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=directory) as tmp, bounded(inputs, fit=True):
        eps = inputs.load_tape()
        accesses = []
        def adaptation(cell, seed, budget, fraction, policy):
            ref.require((cell, seed, budget, fraction, policy) == (CELL, inputs.seed, inputs.budget, .6, "pi05"),
                        "Unexpected fit data request")
            accesses.append("prefix")
            return eps
        def postfit(cell, seed, policy):
            ref.require((cell, seed, policy) == (CELL, inputs.seed, "pi05"), "Unexpected diagnostic request")
            accesses.append("postfit_diagnostic_omitted")
            return []
        ref.require(sha(init) == model_init, "Changed initialization")
        torch.set_rng_state(torch.load(init, map_location="cpu", weights_only=True))
        with patch.object(cache, "RUNS", Path(tmp)), patch.object(cache, "adaptation_data", adaptation), \
             patch.object(cache, "heldout_vla_pool", postfit):
            cache.load_or_fit(spec, device)
            staged = cache.spec_dir(spec)
        ref.require(accesses == ["prefix", "postfit_diagnostic_omitted"], accesses)
        metrics = ref.read(staged / "fit_metrics.json")
        json.dumps(metrics, allow_nan=False)
        r = inputs.record
        ref.require(set(metrics) == {"fit", "heldout", "fit_s", "n_train_eps", "n_val_eps"}
                    and (metrics["n_train_eps"], metrics["n_val_eps"]) == (len(r["train_indices"]), len(r["val_indices"]))
                    and (kind != "fwd" or metrics["fit"]["n_train"] == r["windows"]["fwd"]["training"]),
                    "Released fit disagrees with split")
        ref.write(staged / "bounded.json", dict(**identity, model_sha256=sha(staged / "model.pt"),
                  metrics_sha256=sha(staged / "fit_metrics.json"), accesses=accesses))
        staged.rename(target)
    return verify_fitted(inputs, directory, kind, device)


def verify_fitted(inputs, directory, kind, device):
    from reground.utils import stable_hash
    spec = inputs.spec("dep", kind)
    d = directory / "models" / stable_hash(spec)
    receipt = ref.read(d / "bounded.json")
    ref.require(ref.read(d / "spec.json") == receipt["spec"] == spec and receipt["fit_device"] == device
                and receipt["upstream"] == ref.PIN and receipt["prefix_sha256"] == inputs.record["sha256"]
                and receipt["init_sha256"] == inputs.manifest["reference_models"]["dep_"+kind]["init_sha256"]
                and receipt["metrics_sha256"] == sha(d / "fit_metrics.json") and receipt["model_sha256"] == sha(d / "model.pt"),
                f"Fit receipt mismatch: {d}")
    return receipt


@contextmanager
def deployment(inputs, directory, device):
    """Only cached weights are available to make_controller and its native rollout."""
    from reground.method import build
    from reground.models import cache
    receipts = {k: verify_fitted(inputs, directory, k, device) for k in ("fwd", "idm")}
    seen = []
    def loaded(spec, dev, **kw):
        role, kind = spec["role"], spec["kind"]
        ref.require(dev == device and kind in receipts and role in ("nom", "dep")
                    and spec == inputs.spec(role, kind), "Unexpected deployment model")
        root = SOURCE_STORE / "runs" if role == "nom" else directory
        with patch.object(cache, "RUNS", root):
            if role == "nom":
                path = cache.spec_dir(spec) / "model.pt"
                ref.require(sha(path) == reference_model(inputs.seed, role, kind)["model_sha256"], path)
            model = cache.load_or_fit(spec, dev, fit_ok=False)
        seen.append([role, kind])
        return model
    build._MEMO.clear()
    try:
        with bounded(inputs, fit=False), patch.object(build, "load_or_fit", loaded):
            yield
            ref.require(seen == [["nom","fwd"],["nom","idm"],["dep","fwd"],["dep","idm"]], "Model access mismatch")
    finally:
        build._MEMO.clear()


def panel_jobs(inputs, panel):
    jobs = ref.units("reground", [inputs.seed])
    if panel == "panel32":
        seen = set()
        selected = []
        for job in jobs:
            first = job["episodes"][:ref.BATCH]
            tasks = {t for t, i, r in first}
            if len(tasks) == 1 and tasks.isdisjoint(seen):
                seen |= tasks
                selected.append({**job, "episodes": first})
        ref.require(len(selected) == len(seen) == 8, "Expected eight distinct native opening batches")
        jobs = selected
    return [{**j, "cfg": inputs.cfg} for j in jobs]


def evaluate(inputs, directory, device, panel, ctx):
    from reground.eval import jobs
    from reground.method import build
    receipts = [verify_fitted(inputs, directory, k, device) for k in ("fwd","idm")]
    # Immutable weights + recipe namespace the replayable CEM states and completed shards.
    contract = digest(dict(upstream=ref.PIN, policy=ref.POLICY, cfg=inputs.cfg,
                           weights=[r["model_sha256"] for r in receipts], nominal=inputs.manifest["reference_models"]["nom_fwd"],
                           transport=sha(Path(ref.__file__)), runner=sha(Path(__file__)), common=sha(HERE / "common.py")))
    store = directory / "eval" / contract
    store.mkdir(parents=True, exist_ok=True)
    for job in panel_jobs(inputs, panel):
        path = store / panel / f"shard{job['shard']:03d}"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=store) as temp:
                stage = Path(temp) / "shard"
                with deployment(inputs, directory, device), patch.object(jobs, "job_dir", lambda _: stage), \
                     patch.object(ref, "STORE", store), ref.resume_batches(job, print):
                    jobs.run_eval_job(job, ctx)
                ref.check_eval(stage, job)
                stage.rename(path)
        facts = ref.check_eval(path, job)
        print(json.dumps(dict(shard=job["shard"], panel=panel, **facts)), flush=True)
    return store


@contextmanager
def claim(directory):
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "active"
    lock.mkdir()  # atomic even where cross-host flock is unreliable
    try:
        ref.write(lock / "owner.json", dict(host=socket.gethostname(), pid=os.getpid(), utc=time.time(),
                                          attempt=os.environ.get("LOW_ATTEMPT")))
        yield
    finally:
        (lock / "owner.json").unlink(missing_ok=True)
        lock.rmdir()


def gpu(device, phase):
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    import torch
    probe = ref.read(Path(os.environ["LOW_ATTEMPT"]) / "gpu.json")
    row, = probe["devices"]
    name = row["cuda_name"].upper()
    ref.require(torch.cuda.device_count() == 1 and probe["hostname"] == socket.gethostname()
                and probe["status"] == "passed" and probe["exclusive_training_admission"]
                and pci(row["pci_bus_id"]) == cuda_pci(0) and row["allocation_ordinal"] == 0
                and (name in ("NVIDIA GEFORCE RTX 4090","NVIDIA RPBZZZ6") or "RTX PRO 6000" in name), "GPU probe mismatch")
    torch.cuda.set_device(0)
    if phase != "fit":
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(ref.egl_device(0))
        from mujoco.egl import GLContext
        from OpenGL import GL
        context = GLContext(16,16)
        try:
            context.make_current()
            ref.require(b"NVIDIA" in GL.glGetString(GL.GL_VENDOR).upper(), "EGL vendor mismatch")
        finally:
            context.free()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["fit", "panel32", "full", "fit-panel32", "cpu-qualify"])
    parser.add_argument("seed", type=int, choices=ref.SEEDS)
    parser.add_argument("budgets", type=int, nargs="+", choices=BUDGETS)
    args = parser.parse_args()
    configure()
    from reground.eval.jobs import WorkerCtx
    import torch
    cpu = args.phase == "cpu-qualify"
    device = "cpu" if cpu else "cuda:0"
    if cpu:
        ref.require(not torch.cuda.is_available() and len(os.sched_getaffinity(0)) <= 2
                    and torch.get_num_threads() <= 2, "CPU qualification must use at most two cores")
    else:
        gpu(device, args.phase)
    ctx = None
    try:
        for n in args.budgets:
            inputs = Inputs(args.seed, n)
            directory = WORK / ("cpu" if cpu else "production") / f"seed{args.seed}" / f"b{n}"
            with claim(directory):
                if cpu or args.phase.startswith("fit"):
                    for kind in ("fwd", "idm"):
                        fit_one(inputs, directory, kind, device)
                if cpu:
                    from qualify import cpu_check
                    cpu_check(inputs, directory)
                elif args.phase != "fit":
                    if ctx is None:
                        ctx = WorkerCtx("pi05", device, ref.BATCH, lambda msg: print(msg, flush=True))
                        ref.strict_policy(ctx)
                    evaluate(inputs, directory, device, args.phase.removeprefix("fit-"), ctx)
    finally:
        if ctx is not None:
            ctx.close()


if __name__ == "__main__":
    main()
