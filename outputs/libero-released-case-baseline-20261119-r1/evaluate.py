"""Submit a development subset to the released LIBERO-Pro2.0 evaluator."""
import argparse
from collections import deque
import json
from multiprocessing import Pipe
import os
from pathlib import Path
import subprocess

import numpy as np

from libero_remote import RemoteRunClient
from libero_remote.benchmark import build_benchmark_plan, export_benchmark_results, submit_benchmark_plan
from libero_remote.benchmark_split import BenchmarkSplit
from libero_remote.runner import run_scheduled_run

HERE = Path(__file__).resolve().parent
SPLITS = HERE.parent / "libero-meta-learning-20261117-r1" / "SPLITS.json"
CASES = ("BASE", "S15", "S20", "D17", "D18")


class OpenPIAdapter:
    def __init__(self, python, output):
        self.pipe, peer = Pipe()
        try:
            self.worker = subprocess.Popen(
                [python, "-Bu", HERE / "motor.py", str(peer.fileno())],
                pass_fds=(peer.fileno(),),
                env=dict(os.environ, PYTHONPATH=os.environ["OPENPI_WORKER_PYTHONPATH"]),
            )
        finally:
            peer.close()
        self.pending = deque()
        self.output = output
        self.commands = 0
        self.clipped_commands = 0
        self.clipped_coordinates = np.zeros(7, dtype=np.int64)
        self.max_correction = np.zeros(7, dtype=np.float32)

    def reset(self, context):
        self.pending.clear()
        self.rng = np.random.default_rng(int(context.episode_spec["seed"]) + 817117)
        self.low = np.asarray(context.action_schema["low"], dtype=np.float32)
        self.high = np.asarray(context.action_schema["high"], dtype=np.float32)

    def act(self, observation):
        if not self.pending:
            noise = self.rng.standard_normal((10, 32)).astype(np.float32)
            self.pipe.send((observation, noise))
            self.pending.extend(self.pipe.recv()[:5])
        proposed = self.pending.popleft()
        bounded = np.clip(proposed, self.low, self.high)
        correction = np.abs(proposed - bounded)
        self.commands += 1
        self.clipped_commands += bool(correction.any())
        self.clipped_coordinates += correction > 0
        np.maximum(self.max_correction, correction, out=self.max_correction)
        return bounded

    def close(self):
        self.pipe.close()
        try:
            self.worker.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.worker.terminate()
            self.worker.wait(timeout=30)
        (self.output / "action-bounds.json").write_text(json.dumps(dict(
            commands=self.commands, clipped_commands=self.clipped_commands,
            clipped_coordinates=self.clipped_coordinates.tolist(),
            max_correction=self.max_correction.tolist()), indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--motor-python", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    splits = json.loads(SPLITS.read_text())
    release = BenchmarkSplit.load(args.release / "assets/benchmark_split_v1.json")
    development = {(suite, task) for suite, ids in splits["development"].items() for task in ids}
    with RemoteRunClient(args.base_url, api_key=os.environ["LIBERO_REMOTE_API_KEY"]) as client:
        catalog = release.filter_catalog(client.get_task_catalog(), "train")
        catalog["tasks"] = [row for row in catalog["tasks"] if (row["suite"], row["task_id"]) in development]
        plan = build_benchmark_plan(catalog, cases=CASES, seeds=(142, 143), init_state_ids=(0, 3))
        planned_tasks = {(row["suite"], row["task_id"]) for row in plan.episodes}
        if planned_tasks != development or plan.selected_task_count != 38 or plan.rollout_count != 76:
            raise ValueError("Released catalog differs from the registered development panel")
        submitted = submit_benchmark_plan(client, plan, args.output)
        run_id = submitted["run_id"]
        print(f"Released scheduler run {run_id}, {plan.rollout_count} episodes", flush=True)
        adapter = OpenPIAdapter(args.motor_python, args.output)
        try:
            run_scheduled_run(client, run_id, "openpi-libero-development", adapter)
        finally:
            exported = export_benchmark_results(client, run_id, args.output)
        summary = exported["summary"]
        print(json.dumps(summary, indent=2), flush=True)
        if exported["run"]["status"] != "completed" or summary["overall"]["completed"] != plan.rollout_count:
            raise RuntimeError("The release scheduler did not complete every planned episode")


if __name__ == "__main__":
    main()
