"""Parallel native rollouts for the existing residual learner.

Each allocated GPU runs complete episodes with the same frozen motor. Draws
are assigned in serial episode/chunk order before dispatch; the learner and
its global minibatches are unchanged. Worker results contain no rewards.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .rlinf_native_ppo import cpu_tree, sha
from .warmstart import atomic_save


def draw_schedule(motor, policy, generator, count, *, deterministic):
    """Retain one original-shaped RNG call per chunk, independent of world size."""
    reference = next(policy.parameters())
    seeds = [int(motor.seed_rng.integers(0, 2**63 - 1)) for _ in range(count)]
    noise = None if deterministic else torch.cat([torch.randn(
        (1, policy.config.residual_dim), device=reference.device, dtype=reference.dtype,
        generator=generator).cpu() for _ in range(count)], 0)
    return seeds, noise


class ParallelActionJournal:
    """Read the actual per-worker escrows, including unresolved intents."""
    def __init__(self, root, world):
        self.root, self.world = Path(root), int(world)

    def charged(self):
        total = 0
        for rank in range(self.world):
            for path in (self.root / f"rank-{rank:02d}").glob("*.json"):
                value = json.loads(path.read_text())
                if value["rank_action_ceiling"] != 1_000_000 // self.world:
                    raise ValueError("Preserve the parallel action partition")
                total += sum(row.get("executed_or_conservative", row["requested"])
                             for row in value["calls"])
        return total


class ParallelResidualEnvironment:
    """Boundary-compatible case cursor delegating to native complete episodes."""
    def __init__(self, cases, *, horizon, output, monitoring_root, profile, code, report):
        self.cases, self.horizon = deepcopy(cases), int(horizon)
        self.output, self.monitoring_root = Path(output), Path(monitoring_root)
        self.profile, self.code, self.report = profile, Path(code), report
        self.cursor, self.active, self.batch = 0, False, 0
        self.worker_recovery = []

    def state_dict(self):
        if self.active:
            raise ValueError("Only a complete parallel episode bank is resumable")
        return dict(cases=deepcopy(self.cases), cursor=self.cursor, horizon=self.horizon,
                    worker_recovery=deepcopy(self.worker_recovery))

    def load_state_dict(self, state):
        if self.active or state["cases"] != self.cases or state["horizon"] != self.horizon:
            raise ValueError("Preserve the exact native case bank on migration")
        self.cursor = int(state["cursor"])
        self.worker_recovery = deepcopy(state.get("worker_recovery", []))

    def collect(self, policy, bound, motor, *, episodes, generator, deterministic=False, verify_zero=False):
        devices = self.profile["rollout_devices"]
        world = len(devices)
        if world != 8 or episodes % world or self.cursor + episodes > len(self.cases):
            raise ValueError("Eight allocated workers and an exact complete episode bank required")
        bank = self.output / f"parallel-{self.cursor:06d}"
        bank.mkdir(parents=True, exist_ok=False)  # Never recollect an attempted bank.
        per_rank = episodes // world
        chunks = per_rank * (self.horizon // 50)
        seeds, noise = draw_schedule(motor, policy, generator, episodes * self.horizon // 50,
                                     deterministic=deterministic)
        children, streams, requests = [], [], []
        self.active = True
        distributed = self.profile.get("distributed_rollout_root")
        try:
            for rank, device in enumerate(devices):
                directory = bank / f"rank-{rank:02d}"
                directory.mkdir()
                request = dict(profile=self.profile, config=asdict(policy.config),
                    policy=cpu_tree(policy.state_dict()), bound=asdict(bound), horizon=self.horizon,
                    cases=self.cases[self.cursor + rank * per_rank:self.cursor + (rank + 1) * per_rank],
                    seeds=seeds[rank * chunks:(rank + 1) * chunks],
                    noise=None if noise is None else noise[rank * chunks:(rank + 1) * chunks],
                    deterministic=deterministic, verify_zero=verify_zero and rank == 0,
                    rank=rank, world=world, output=str(directory), monitoring_root=str(self.monitoring_root),
                    episode_id_prefix=f"case-{self.cursor + rank * per_rank:06d}-rank-{rank:02d}-",
                    attempt=f"{self.profile['attempt']}-{self.output.parent.name}-{self.cursor:06d}")
                request_path = directory / "request.pt"
                atomic_save(request_path, request)
                requests.append(request)
                if distributed:
                    from .rlinf_flora_ppo import write_new
                    queue = Path(distributed) / f"rank-{rank:02d}"
                    queue.mkdir(parents=True, exist_ok=True)
                    write_new(queue / (self.output.parent.name + f"-{self.cursor:06d}.json"),
                        dict(request=str(request_path), sha256=sha(request_path), output=str(directory),
                             source_commit=self.profile["source_commit"]))
                    continue
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=device["visibility_token"],
                    LYCO_BAIGE_ISOLATED_UUID=device["uuid"],
                    MUJOCO_EGL_DEVICE_ID=str(device["allocation_ordinal"]),
                    OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
                stream = (directory / "console.log").open("x")
                streams.append(stream)
                children.append(subprocess.Popen([sys.executable, "-B",
                    str(self.code / "scripts/diagnostics/robocasa_residual_rollout.py"),
                    "--request", str(request_path)], env=env, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True))
            def complete():
                if distributed:
                    return all((bank / f"rank-{rank:02d}/result.pt").exists() for rank in range(world))
                return all(child.poll() is not None for child in children)

            while not complete():
                if any(child.poll() not in (None, 0) for child in children):
                    raise RuntimeError("Native rollout worker failed; preserve every request/result, no recollection")
                if any((bank / f"rank-{rank:02d}/error.json").exists() for rank in range(world)):
                    raise RuntimeError("Distributed native worker failed; preserve every request and partial result")
                if time.time() >= self.profile["deadline_unix"] - self.profile["closure_seconds"]:
                    raise TimeoutError("Parallel pending bank retained before finite closure")
                self.report("parallel_native_rollout", workers=world,
                            complete_workers=sum((bank / f"rank-{rank:02d}/result.pt").exists() for rank in range(world)))
                time.sleep(1)
            if any(child.returncode != 0 for child in children):
                raise RuntimeError("Parallel bank incomplete; no labels or optimizer update")
            records, finished, recovery = [], [], []
            for rank, request in enumerate(requests):
                result = torch.load(bank / f"rank-{rank:02d}/result.pt", map_location="cpu", weights_only=False)
                if (result["cases"] != request["cases"] or len(result["records"]) != chunks
                        or len(result["episodes"]) != per_rank or result["environment"]["cursor"] != per_rank
                        or [row["noise_seed"] for row in result["records"]] != request["seeds"]
                        or any(row["reward"] != 0 for row in result["records"])):
                    raise ValueError("Exact complete unlabeled worker bank required")
                records.extend(result["records"])
                finished.extend(result["episodes"])
                result_path = bank / f"rank-{rank:02d}/result.pt"
                request_path = bank / f"rank-{rank:02d}/request.pt"
                recovery.append(dict(rank=rank, environment=result["environment"],
                    global_rng=result["global_rng"], motor_rng=result["motor_rng"],
                    request=dict(path=str(request_path), sha256=sha(request_path)),
                    result=dict(path=str(result_path), sha256=sha(result_path))))
            if len({row["id"] for row in finished}) != episodes or not records[-1]["done"]:
                raise ValueError("Global episode identities and chronological full bank required")
            self.cursor += episodes
            self.worker_recovery = recovery
            self.active = False
            return records, finished
        finally:
            for child in children:
                if child.poll() is None:
                    # The worker's own native children belong to this session.
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
            for child in children:
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    import signal
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            for stream in streams:
                stream.close()

    def close(self):
        pass  # collect owns and closes each complete native worker session.
