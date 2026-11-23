"""Evaluate whole pairs in fresh native pools while retaining the frozen VLA per GPU."""
import os
from pathlib import Path
import socket
import time

import reference as ref
import symmeta_adapter as stream
import retry


def selected(grid, phase):
    ref.require(phase in ("goals","full"),phase)
    goal = [i for i,b in enumerate(grid) if any(k[0] in ("goal_1","goal_2") for k in b)]
    return goal if phase == "goals" else goal + [i for i in range(len(grid)) if i not in goal]


def run(attempt, rank, phase):
    import torch
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    from reground.eval.jobs import WorkerCtx
    admission = ref.read(attempt / "gpu.json")
    devices = admission["devices"]
    ref.require(admission["status"]=="passed" and admission["allocation_type"]=="baige_container"
                and admission["exclusive_training_admission"] and admission["purpose"]=="training" and
                admission["hostname"]==socket.gethostname() and torch.cuda.device_count()==len(devices),
                "Expected exclusive scheduler 4090 allocation")
    row = devices[rank]
    ref.require(row["allocation_ordinal"]==rank and pci(row["pci_bus_id"])==cuda_pci(rank)
                and row["cuda_name"]==torch.cuda.get_device_name(rank)=="NVIDIA GeForce RTX 4090"
                and row["cuda_uuid"]==str(torch.cuda.get_device_properties(rank).uuid),
                "CUDA ordinal, UUID, PCI or device differs")
    torch.cuda.set_device(rank)
    egl = ref.egl_device(rank)
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    grid = retry.schedule()
    units = selected(grid, phase)[rank::len(devices)]
    ref.write(attempt/f"rank{rank}.json", dict(rank=rank, egl=egl, phase=phase, batches=units, device=row))
    store = attempt.parent.parent/"batches"
    store.mkdir(exist_ok=True)
    ctx = WorkerCtx("pi05",f"cuda:{rank}",ref.BATCH,lambda m: print(m,flush=True))
    try:
        with stream.reader_process() as (pipe,reader):
            for b in units:
                name = f"batch{b:03d}"
                dest = store/name
                if dest.exists():
                    retry.validate(dest,grid[b])
                    continue
                if ctx._vla is None:
                    ref.strict_policy(ctx)
                work = attempt/name
                work.mkdir()
                begin = time.time()
                retry.run_pair(ctx.vla,pipe,reader,grid[b],work)
                retry.validate(work,grid[b])
                ref.write(work/"producer.json",dict(rank=rank,egl=egl,host=socket.gethostname(),
                          attempt=attempt.name, code=os.environ["RETRY_PIN"], started=begin, ended=time.time()))
                work.rename(dest)
                print(f"Published {name} in {time.time()-begin:.1f}s",flush=True)
    finally:
        ctx.close()
