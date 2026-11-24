"""Operational dispatch for the frozen wrist study. Native execution lives in retry.py."""

from contextlib import contextmanager, ExitStack
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback
from zipfile import ZipFile

import reference as ref

DRAIN = tuple(range(55, 70))
PARKED = 54
PIXEL = r"raw/[0-3]/obs/pixels/(?:image|image2)"
JOURNAL_OPS = {"reset", "step", "internal_reset_retry", "vla_batch", "replayed_opening_batch"}


def assigned(grid, order):
    ref.require(len(grid) == 70 and len({k for batch in grid for k in batch}) == 280
                and all(len(batch) == 4 for batch in grid)
                and sorted(order) == list(range(70))
                and tuple(i for i in order if i in DRAIN) == DRAIN,
                "Frozen enrollment or dispatch order differs")
    return DRAIN


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def journal(path, stage):
    counts = {}
    pending = None
    for line in path.read_bytes().splitlines(keepends=True):
        ref.require(line.endswith(b"\n"), f"Incomplete journal line at {path}")
        event, name, op, n = json.loads(line)
        ref.require(name == stage and op in JOURNAL_OPS and type(n) is int and n > 0,
                    f"Invalid journal record at {path}")
        if event == "begin" and pending is None:
            pending = (op, n)
        else:
            ref.require(event == "end" and pending == (op, n), f"Invalid journal sequence at {path}")
            counts[op] = counts.get(op, 0) + n
            pending = None
    ref.require(pending is None, f"Unfinished native call at {path}")
    return counts


def validate_batch(path, keys, retry, *, published):
    ref.require(path.is_dir() and not path.is_symlink(), f"Invalid batch directory {path}")
    required = {
        "contract.json", "support.json", "support.npz", "support.jsonl", "opening.npz",
        *(arm + suffix for arm in retry.ARMS for suffix in (".json", ".jsonl", "/traces/retry.npz")),
    }
    if published:
        required.add("producer.json")
    extra = {"dispatch.json"} if published and os.path.lexists(path / "dispatch.json") else set()
    actual = {str(file.relative_to(path)) for file in path.rglob("*")
              if file.is_file() or file.is_symlink()}
    ref.require(actual == required | extra, f"Paired artifact census differs at {path}")
    for name in required:
        file = path / name
        ref.require(file.is_file() and not file.is_symlink(), f"Missing or invalid paired file {file}")
        if file.suffix == ".npz":
            with ZipFile(file) as archive:
                ref.require(archive.testzip() is None, f"Invalid archive bytes at {file}")
    retry.validate(path, keys)
    budgets = {"support": ref.read(path / "contract.json")["support_budget"]}
    budgets.update({arm: ref.read(path / f"{arm}.json")["budget"] for arm in retry.ARMS})
    for stage, budget in budgets.items():
        ref.require(journal(path / f"{stage}.jsonl", stage) == budget, f"Native accounting differs at {path}")
    if published:
        producer = ref.read(path / "producer.json")
        ref.require(producer["code"] == os.environ["RETRY_PIN"] and producer["ended"] >= producer["started"],
                    f"Invalid scientific producer at {path}")
        sidecar = path / "dispatch.json"
        ref.require(re.fullmatch(r"retry[0-9]+", producer["attempt"]), f"Invalid attempt at {path}")
        source = path.parent.parent / "attempts" / producer["attempt"]
        if os.path.lexists(sidecar):
            receipt = ref.read(sidecar)
            ref.require(not sidecar.is_symlink() and receipt["batch"] == int(path.name[5:])
                        and re.fullmatch(r"job-[a-z0-9]+", receipt["scheduler_id"])
                        and receipt["files"] == {name: sha(path / name) for name in sorted(required)}
                        and receipt["keys"] == list(map(list, keys))
                        and receipt["attempt"] == producer["attempt"]
                        and receipt["science_pin"] == producer["code"],
                        f"Dispatch receipt or artifact bytes differ at {path}")
        else:
            ref.require(not (source / "dispatch.json").exists(), f"Missing dispatch receipt at {path}")
    return required


@contextmanager
def verified_pool(factory, work, stage, n):
    # Keep the frozen pool and its teardown. Confirm its children have exited before
    # an exception can return to dispatch. A teardown failure remains structural.
    children = ()
    try:
        with factory(work, stage, n) as pair:
            try:
                yield pair
            finally:
                children = tuple(pair[0].native.procs)
    finally:
        for child in children:
            child.join(timeout=5)
            ref.require(child.exitcode == 0, f"Native child {child.pid} did not exit cleanly in {stage}")


def raw_guard(exc, work, retry):
    """Identify the exact frozen guard and check the remaining opening leaves."""
    import numpy as np
    from opening import identical

    if type(exc) is not ValueError:
        return False
    tb = exc.__traceback__
    while tb is not None and tb.tb_frame.f_code is not retry.Retry.predict_chunk.__code__:
        tb = tb.tb_next
    if tb is None:
        return False
    loc = tb.tb_frame.f_locals
    key = loc.get("key", "")
    if (not re.fullmatch(PIXEL, key) or str(exc) != f"Paired opening differs at {key}"
            or loc["self"].arm != retry.ARMS[1] or loc["query"] != 0 or tb.tb_next is None
            or tb.tb_next.tb_frame.f_code is not ref.require.__code__):
        return False
    wrist = retry.ARMS[1]
    counts = journal(work / f"{wrist}.jsonl", wrist)
    ref.require(counts.get("reset") == 4 and counts.keys() <= {"reset", "internal_reset_retry"}
                and not (work / f"{wrist}.json").exists(), "Guard journal is not a pre-control wrist reset")
    with np.load(work / "opening.npz", allow_pickle=False) as tape:
        inputs = loc["inputs"]
        ref.require(set(tape.files) == set(inputs) | {"proposal"}, "Opening fields differ after raw guard")
        for name, current in inputs.items():
            stored = tape[name]
            if not identical(stored, current):
                raw_name = "raw/" + name[7:] if name.startswith("mapped/") else name
                ref.require(re.fullmatch(PIXEL, raw_name), f"Structural opening difference at {name}")
                ref.require(stored.dtype == current.dtype and stored.shape == current.shape,
                            f"Structural opening dtype or shape difference at {name}")
    return True


def preflight(store, grid, units, validate, states):
    batches = store / "batches"
    ref.require(batches.is_dir() and not batches.is_symlink(), "Missing scored batch store")
    census = {path.name for path in batches.iterdir()}
    ref.require(census <= {f"batch{i:03d}" for i in range(70)}
                and {f"batch{i:03d}" for i in range(PARKED)} <= census
                and not any(not (batches / name).is_dir() or (batches / name).is_symlink() for name in census),
                "Scored batch census differs from frozen enrollment")
    if f"batch{PARKED:03d}" in census:
        states[PARKED] = "present_parked_for_parent_readback"
    for b in units:
        dest = batches / f"batch{b:03d}"
        if os.path.lexists(dest):
            states[b] = "validating_existing"
            validate(dest, grid[b], published=True)
            states[b] = "validated_existing"
    return batches


def drain(attempt, rank, egl, grid, units, ctx, retry, stream, provenance):
    states = {i: "original_sealed" for i in range(PARKED)}
    states[PARKED] = "enrolled_parked_for_qualification"
    states.update(dict.fromkeys(units, "unstarted"))
    progress = attempt / "dispatch.json"
    case_keys = {i: grid[i] for i in range(len(grid))}

    def checkpoint():
        ref.write(progress, dict(provenance, attempt=attempt.name, cell=retry.CELL,
                                 scene_seed=0, lane_order=case_keys, assigned=units, statuses=states))

    validate = partial(validate_batch, retry=retry)
    checkpoint()
    try:
        batches = preflight(attempt.parent.parent, grid, units, validate, states)
    except Exception:
        for b in units:
            if states[b] == "validating_existing":
                states[b] = "structural_error"
        ref.write(attempt / "preflight-error.json", {"traceback": traceback.format_exc()})
        checkpoint()
        raise
    checkpoint()

    with ExitStack() as reader_scope:
        pipe = reader = None
        for b in units:
            if states[b] == "validated_existing":
                continue
            name = f"batch{b:03d}"
            work = attempt / name
            states[b] = "started"
            checkpoint()
            try:
                work.mkdir()
                if pipe is None:
                    pipe, reader = reader_scope.enter_context(stream.reader_process())
                if ctx._vla is None:
                    ref.strict_policy(ctx)
                begin = time.time()
                pool_factory = partial(verified_pool, retry.fresh)
                retry.run_pair(ctx.vla, pipe, reader, grid[b], work, pool_factory=pool_factory)
                files = validate(work, grid[b], published=False)
                ref.write(work / "producer.json", dict(rank=rank, egl=egl, host=socket.gethostname(),
                          attempt=attempt.name, code=os.environ["RETRY_PIN"], started=begin, ended=time.time()))
                files.add("producer.json")
                ref.write(work / "dispatch.json", dict(provenance, attempt=attempt.name, batch=b, keys=grid[b],
                          cell=retry.CELL, scene_seed=0, support_keys=[retry.support_key(key) for key in grid[b]],
                          files={name: sha(work / name) for name in sorted(files)}))
                work.rename(batches / name)
            except Exception as exc:
                # Keep native journals and the exact exception in the attempt tree.
                states[b] = "structural_error"
                ref.write(attempt / f"{name}-error.json",
                          {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
                checkpoint()
                if not raw_guard(exc, work, retry):
                    raise
                guard_message = str(exc)
            else:
                states[b] = "published"
                checkpoint()
                print(f"Published {name} in {time.time() - begin:.1f}s", flush=True)
                continue
            # Close outside the handler so reader_process checks its child exit status.
            reader_scope.close()
            pipe = reader = None
            states[b] = "unpublished_raw_guard"
            checkpoint()
            print(f"Unpublished {name}: {guard_message}", flush=True)
    return 3 if "unpublished_raw_guard" in states.values() else 0


def provenance(retry, stream):
    import opening
    from reground.eval import jobs, rollout
    from reground.envs import libero_env
    from reground.policies import vla
    import worker

    modules = [retry, stream, opening, ref, jobs, rollout, libero_env, vla, worker]
    hashes = {module.__name__: sha(Path(module.__file__)) for module in modules}
    hashes["extended_recovery_launcher"] = "c46a5996c99f28b93c965873ed1fd7e40f58e782a972f9afaea1127bfaeb4041"
    hashes["dispatch_launcher"] = sha(Path(__file__).with_name("launch.sh"))
    hashes["dispatch_worker"] = sha(Path(__file__))
    return dict(science_pin=os.environ["RETRY_PIN"], recovery_pin=os.environ["WRIST_RECOVERY_PIN"],
                dispatch_pin=os.environ["WRIST_DISPATCH_PIN"], scheduler_id=os.environ["WRIST_BAIGE_JOB_ID"],
                upstream_pin=ref.PIN, reader_weights_sha256=stream.WEIGHTS, policy_revision=ref.POLICY,
                native_hashes=hashes)


def run(attempt, rank):
    ref.configure()
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    from reground.eval.jobs import WorkerCtx
    import worker

    admission = ref.read(attempt / "gpu.json")
    devices = admission["devices"]
    ref.require(admission["status"]=="passed" and admission["allocation_type"]=="baige_container"
                and admission["exclusive_training_admission"] and admission["purpose"]=="training" and
                admission["hostname"]==socket.gethostname() and torch.cuda.device_count()==len(devices),
                "Expected exclusive scheduler 4090 allocation")
    ref.require(rank == 0 and len(devices) == 1, "Expected one exclusive RTX 4090")
    row = devices[rank]
    ref.require(row["allocation_ordinal"]==rank and pci(row["pci_bus_id"])==cuda_pci(rank)
                and row["cuda_name"]==torch.cuda.get_device_name(rank)=="NVIDIA GeForce RTX 4090"
                and row["cuda_uuid"]==str(torch.cuda.get_device_properties(rank).uuid),
                "CUDA ordinal, UUID, PCI or device differs")
    torch.cuda.set_device(rank)
    egl = ref.egl_device(rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    retry, stream = worker.retry, worker.stream
    grid = retry.schedule()
    units = assigned(grid, worker.selected(grid, "full"))
    ref.write(attempt / f"rank{rank}.json", dict(rank=rank, egl=egl, phase="full", batches=units,
                                               parked=[PARKED], device=row))
    ctx = WorkerCtx("pi05", f"cuda:{rank}", ref.BATCH, lambda m: print(m, flush=True))
    try:
        return drain(attempt, rank, egl, grid, units, ctx, retry, stream, provenance(retry, stream))
    finally:
        ctx.close()


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), int(sys.argv[2])))
