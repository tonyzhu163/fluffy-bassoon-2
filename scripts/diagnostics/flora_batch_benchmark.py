#!/usr/bin/env python
"""Finite throughput replay of frozen videos, with unchanged official scoring."""
import argparse
import json
from pathlib import Path
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
    verify_selected_cuda(torch)
    output = Path(profile["output"])
    stop = threading.Event()
    started = time.monotonic()
    state = dict(stage="engine_loading", batch_size=profile["batch_size"], videos=64)

    def heartbeat():
        while not stop.wait(20):
            temporary = output / "status.tmp"
            temporary.write_text(json.dumps(dict(state, elapsed_seconds=time.monotonic() - started)))
            temporary.replace(output / "status.json")

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    result = dict(completed=False, batch_size=profile["batch_size"], tensor_parallel_size=1,
                  new_robot_actions=0, purpose="Throughput only; no calibration or checkpoint selection")
    try:
        worker = TerminalWorker(profile["model"], batch_size=profile["batch_size"])
        ready = time.monotonic()
        result["cold_engine_seconds"] = ready - started
        state["stage"] = "scoring_frozen_replay64"
        rows = worker.score_jobs(profile["jobs"], output / "scores.jsonl")
        ended = time.monotonic()
        result.update(completed=True, measured_videos=len(rows), steady_inference_seconds=ended - ready,
                      videos_per_second=len(rows) / (ended - ready), model_load_count=worker.model_load_count,
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        result["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        stop.set()
        thread.join(timeout=2)
        result["total_seconds"] = time.monotonic() - started
        write_new(output / "result.json", result)


if __name__ == "__main__":
    main()
