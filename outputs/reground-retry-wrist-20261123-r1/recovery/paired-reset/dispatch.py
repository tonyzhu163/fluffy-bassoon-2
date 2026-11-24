"""Run only the explicit outcome-blind pairing selection, leaving sealed and partial work parked."""
from contextlib import ExitStack
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback

import admission


def dispatch(attempt, store, plan, grid, retry, run_pair, context, reader_factory, load_model, provenance):
    states = admission.statuses(store, plan)
    progress = attempt / "paired-reset-progress.json"
    deadline = plan["payload_deadline_monotonic"]
    started = time.monotonic()

    def checkpoint():
        counts = {}
        for status in states.values():
            counts[status] = counts.get(status, 0) + 1
        quarantine = sum(n for status, n in counts.items() if status.endswith("_quarantine"))
        admission.write(progress, dict(selection_sha256=plan["sha256"], statuses=states, counts=counts,
                          quarantined_batches=quarantine, quarantined_starts=quarantine * 4,
                          elapsed_seconds=time.monotonic()-started))

    checkpoint()
    units = [b for b, state in states.items() if state == "unstarted"]
    with ExitStack() as reader_scope:
        pipe = reader = None
        for position, batch in enumerate(units):
            # Reserve baseline scoring time for all as-yet unstarted batches. An in-flight
            # pair is decided when its RPC returns, subject to the hard launcher cutoff.
            def admit():
                return time.monotonic() + plan["pair_seconds"] + (
                    len(units) - position) * plan["score_seconds_per_batch"] <= deadline

            if not admit():
                states[batch] = "budget_deferred"
                checkpoint()
                continue
            work = attempt / f"batch{batch:03d}"
            dest = store / "batches" / work.name
            work.mkdir()
            states[batch] = "started"
            checkpoint()
            begin = time.time()
            try:
                if pipe is None:
                    pipe, reader = reader_scope.enter_context(reader_factory())
                load_model(context)
                states[batch] = run_pair(context.vla, pipe, reader, grid[batch], work, plan["pair_cap"], admit)
                if states[batch] == "publishable":
                    # Exactly the original native validator on this newly complete pair.
                    retry.validate(work, grid[batch])
                    admission.write(work / "producer.json", dict(rank=0, egl=provenance["egl"],
                                    host=socket.gethostname(), attempt=attempt.name, code=provenance["science_pin"],
                                    started=begin, ended=time.time()))
                    admission.write(work / "paired-reset.json", dict(provenance, batch=batch, keys=grid[batch],
                                    attempt=attempt.name, selection_sha256=plan["sha256"], pair_cap=plan["pair_cap"]))
                    work.rename(dest)
                    states[batch] = "published"
                    print(f"Published {work.name}", flush=True)
                else:
                    print(f"{work.name}: {states[batch]}", flush=True)
            except Exception:
                states[batch] = "structural_error"
                admission.write(attempt / "paired-reset-error.json",
                                dict(batch=batch, traceback=traceback.format_exc()))
                checkpoint()
                raise
            checkpoint()
    return 0 if all(state in {"published", "sealed_parked"} for state in states.values()) else 3


def run(attempt):
    import reference as ref
    ref.configure()
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    from opening import file_hash
    from reground.eval import jobs
    import pairing
    import retry
    import symmeta_adapter as stream

    plan = admission.read(attempt / "paired-reset-selection.json")
    proof = ref.read(attempt / "gpu.json")
    ref.require(proof["status"] == "passed" and proof["allocation_type"] == "baige_container"
                and proof["exclusive_training_admission"] and proof["purpose"] == "training"
                and proof["hostname"] == socket.gethostname()
                and len(proof["devices"]) == torch.cuda.device_count() == 1,
                "Expected one exclusive Baige training GPU")
    device, = proof["devices"]
    ref.require(device["allocation_ordinal"] == 0 and pci(device["pci_bus_id"]) == cuda_pci(0)
                and device["cuda_name"] == torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4090"
                and device["cuda_uuid"] == str(torch.cuda.get_device_properties(0).uuid),
                "CUDA ordinal, UUID, PCI or device differs")
    scheduler = re.fullmatch(r"(job-[a-z0-9]+)-master-0", socket.gethostname())
    ref.require(scheduler is not None, "Expected a Baige scheduler master")
    torch.cuda.set_device(0)
    egl = ref.egl_device(0)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    grid = retry.schedule()
    ref.require(len(grid) == 70 and all(len(keys) == 4 for keys in grid), "Frozen enrollment differs")
    ref.write(attempt / "rank0.json", dict(rank=0, egl=egl, phase="full", batches=plan["batches"], device=device))
    hashes = {module.__name__: file_hash(Path(module.__file__)) for module in (ref, jobs, retry, pairing, stream)}
    provenance = dict(rule=pairing.RULE, operational_pin=plan["pin"], science_pin=os.environ["RETRY_PIN"],
                      upstream_pin=ref.PIN, policy_revision=ref.POLICY, egl=egl, scheduler_id=scheduler[1],
                      launcher_sha256=file_hash(Path(__file__).with_name("launch.sh")), native_hashes=hashes)
    context = jobs.WorkerCtx("pi05", "cuda:0", ref.BATCH, lambda message: print(message, flush=True))

    def load_model(ctx):
        if ctx._vla is None:
            ref.strict_policy(ctx)

    try:
        return dispatch(attempt, attempt.parent.parent, plan, grid, retry, pairing.run_pair, context,
                        stream.reader_process, load_model, provenance)
    finally:
        context.close()


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1])))
