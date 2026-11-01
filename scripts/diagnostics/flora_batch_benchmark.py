#!/usr/bin/env python
"""Finite throughput replay of frozen videos, with unchanged official scoring."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from fluffy_bassoon.meta_rl.flora_terminal_worker import TerminalWorker
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    args = parser.parse_args()
    profile = json.loads(Path(args.profile).read_text())
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One allocated 96GB card per official TP1 replica")
    selected_uuid = verify_selected_cuda(torch)
    output = Path(profile["output"])
    # Validate the real distributed scheduler identity first. Each independent
    # TP1 engine must then start without a four-node model rendezvous. This is
    # a child-process environment; the parent scheduler binding stays intact.
    distributed_keys = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
                        "MASTER_ADDR", "MASTER_PORT", "NPROC_PER_NODE")
    visibility = {key: os.environ.get(key) for key in ("NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
    scheduler_environment = {key: os.environ[key] for key in distributed_keys if key in os.environ}
    write_new(output / "scheduler-node-environment.json", dict(
        verified_gpu_uuid=selected_uuid, scheduler_node_environment=scheduler_environment,
        preserved_visibility=visibility, independent_tensor_parallel_size=1))
    for key in distributed_keys:
        os.environ.pop(key, None)
    if any(key in os.environ for key in distributed_keys) or any(
            os.environ.get(key) != value for key, value in visibility.items()):
        raise RuntimeError("TP1 isolation must preserve the verified GPU visibility")
    if (len(profile["jobs"]) != 64 or len({row["id"] for row in profile["jobs"]}) != 64
            or profile["batch_size"] not in (1, 2, 4, 6, 8, 12, 16, 24)):
        raise ValueError("The fixed throughput replay requires64 unique requests and a declared batch size")
    stop = threading.Event()
    lock = threading.Lock()
    started = time.monotonic()
    state = dict(stage="engine_loading", batch_size=profile["batch_size"], videos=64,
                 nvml_peak_used_mib=None, nvml_total_mib=None, nvml_samples=0, nvml_errors=0)

    def heartbeat():
        # vLLM owns child processes: parent torch allocator statistics do not
        # measure their total card use. These cards have one exclusive replica.
        with (output / "gpu-memory.jsonl").open("x") as samples:
            while not stop.is_set():
                sample = dict(elapsed_seconds=time.monotonic() - started)
                try:
                    row = subprocess.check_output(["nvidia-smi", "--id=" + selected_uuid,
                        "--query-gpu=uuid,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        text=True, timeout=5).strip().split(",")
                    if len(row) != 3 or row[0].strip() != selected_uuid:
                        raise ValueError("NVML sample does not identify the selected allocated card")
                    used, total = float(row[1]), float(row[2])
                    sample.update(memory_used_mib=used, memory_total_mib=total)
                    with lock:
                        state["nvml_samples"] += 1
                        state["nvml_peak_used_mib"] = max(used, state["nvml_peak_used_mib"] or 0)
                        state["nvml_total_mib"] = total
                except Exception as error:
                    sample["error"] = f"{type(error).__name__}: {error}"
                    with lock:
                        state["nvml_errors"] += 1
                with lock:
                    snapshot = dict(state, elapsed_seconds=time.monotonic() - started)
                    sample["stage"] = state["stage"]
                samples.write(json.dumps(sample) + "\n")
                samples.flush()
                temporary = output / "status.tmp"
                temporary.write_text(json.dumps(snapshot))
                temporary.replace(output / "status.json")
                stop.wait(1)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result = dict(completed=False, batch_size=profile["batch_size"], tensor_parallel_size=1,
                  new_robot_actions=0, purpose="Throughput only; no calibration or checkpoint selection",
                  scheduler_environment_receipt=str(output / "scheduler-node-environment.json"),
                  independent_tp1_environment=True, removed_distributed_variables=list(distributed_keys))
    try:
        worker = TerminalWorker(profile["model"], batch_size=profile["batch_size"])
        ready = time.monotonic()
        result["cold_engine_seconds"] = ready - started
        with lock:
            state["stage"] = "scoring_frozen_replay64"
        rows = worker.score_jobs(profile["jobs"], output / "scores.jsonl")
        ended = time.monotonic()
        if [row["id"] for row in rows] != [row["id"] for row in profile["jobs"]]:
            raise ValueError("Throughput result changed request count or ordering")
        result.update(completed=True, measured_videos=len(rows), steady_inference_seconds=ended - ready,
                      videos_per_second=len(rows) / (ended - ready), model_load_count=worker.model_load_count,
                      parent_torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      parent_torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        with lock:
            result["failure_stage"] = state["stage"]
        result["OOM"] = isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()
        raise
    finally:
        stop.set()
        thread.join(timeout=6)
        result["total_seconds"] = time.monotonic() - started
        with lock:
            result.update({key: state[key] for key in
                ("nvml_peak_used_mib", "nvml_total_mib", "nvml_samples", "nvml_errors")})
        result["memory_measurement_available"] = result["nvml_samples"] > 0
        write_new(output / "result.json", result)


if __name__ == "__main__":
    main()
