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


def episode_partitions(episodes, world=8):
    """Contiguous complete episodes, with no padding or discarded samples."""
    if type(episodes) is not int or type(world) is not int or world not in (4, 8) or episodes < world:
        raise ValueError("At least one complete episode on each of four or eight workers required")
    base, extra = divmod(episodes, world)
    cursor, result = 0, []
    for rank in range(world):
        stop = cursor + base + (rank < extra)
        result.append((cursor, stop))
        cursor = stop
    return result


def native_worker_rank(node_rank, local_rank, local_gpu_count):
    """Map one- or two-GPU nodes to the same contiguous global worker IDs."""
    if (any(type(value) is not int for value in (node_rank, local_rank, local_gpu_count))
            or node_rank < 0 or local_gpu_count not in (1, 2)
            or not 0 <= local_rank < local_gpu_count or node_rank * local_gpu_count + local_rank >= 8):
        raise ValueError('One or two local GPUs and a valid global worker rank required')
    return node_rank * local_gpu_count + local_rank


def native_chunk_offsets(cases, horizon=None):
    """Bind every complete episode to its registered native horizon."""
    from .rlinf_robocasa_ppo import TASKS
    offsets = [0]
    for case in cases:
        native = TASKS[case['task']][0]
        if horizon is not None and native != horizon:
            raise ValueError('A uniform bank cannot shorten another native task')
        offsets.append(offsets[-1] + native // 50)
    return offsets


def rollout_envs_per_gpu(profile):
    """Missing fields retain the legacy singleton checkpoint/sampler contract."""
    value = profile.get('rollout_lanes_per_gpu', 1)
    if type(value) is not int or value not in (1, 2, 4, 8):
        raise ValueError('Bind one of the measured native rollout lane counts')
    return value


def native_rollout_memory_reserve_mib(profile):
    # Measured standalone peaks plus the same8GiB operational reserve.
    return {1: 17536, 2: 10701+8192, 4: 13781+8192, 8: 19216+8192}[rollout_envs_per_gpu(profile)]


class ParallelActionJournal:
    """Read the actual per-worker escrows, including unresolved intents."""
    def __init__(self, root, world, maximum_actions=1_000_000):
        self.root, self.world = Path(root), int(world)
        self.ceiling = None if maximum_actions is None else int(maximum_actions) // self.world

    def charged(self):
        total = 0
        for rank in range(self.world):
            for path in (self.root / f"rank-{rank:02d}").glob("*.json"):
                value = json.loads(path.read_text())
                if value["rank_action_ceiling"] != self.ceiling:
                    raise ValueError("Preserve the parallel action partition")
                total += sum(row.get("executed_or_conservative", row["requested"])
                             for row in value["calls"])
        return total


class ParallelResidualEnvironment:
    """Boundary-compatible case cursor delegating to native complete episodes."""
    def __init__(self, cases, *, horizon, output, monitoring_root, profile, code, report):
        self.cases, self.horizon = deepcopy(cases), None if horizon is None else int(horizon)
        native_chunk_offsets(self.cases, self.horizon)
        self.output, self.monitoring_root = Path(output), Path(monitoring_root)
        self.profile, self.code, self.report = profile, Path(code), report
        self.environments_per_gpu = rollout_envs_per_gpu(profile)
        self.world_size = len(profile['rollout_devices']) if profile.get('rollout_devices') else 8
        if self.world_size not in (4, 8):raise ValueError('Four or eight distinct native workers required')
        self.cursor, self.active, self.batch = 0, False, 0
        self.worker_recovery = []

    def state_dict(self):
        if self.active:
            raise ValueError("Only a complete parallel episode bank is resumable")
        state = dict(cases=deepcopy(self.cases), cursor=self.cursor, horizon=self.horizon,
                     worker_recovery=deepcopy(self.worker_recovery))
        if self.world_size != 8:state['rollout_world_size'] = self.world_size
        if self.environments_per_gpu != 1:
            state['rollout_lanes_per_gpu'] = self.environments_per_gpu
        return state

    def load_state_dict(self, state):
        if (self.active or state["cases"] != self.cases or state["horizon"] != self.horizon
                or state.get('rollout_lanes_per_gpu', 1) != self.environments_per_gpu
                or state.get('rollout_world_size', 8) != self.world_size):
            raise ValueError("Preserve the exact native case bank on migration")
        self.cursor = int(state["cursor"])
        self.worker_recovery = deepcopy(state.get("worker_recovery", []))

    def collect(self, policy, bound, motor, *, episodes, generator, deterministic=False, verify_zero=False):
        devices = self.profile["rollout_devices"]
        world = len(devices)
        if world != self.world_size or self.cursor + episodes > len(self.cases):
            raise ValueError("The bound worker count and an exact complete episode bank are required")
        partitions = episode_partitions(episodes, world)
        bank = self.output / f"parallel-{self.cursor:06d}"
        bank.mkdir(parents=True, exist_ok=False)  # Never recollect an attempted bank.
        offsets = native_chunk_offsets(self.cases[self.cursor:self.cursor + episodes], self.horizon)
        seeds, noise = draw_schedule(motor, policy, generator, offsets[-1],
                                     deterministic=deterministic)
        children, streams, requests = [], [], []
        self.active = True
        distributed = self.profile.get("distributed_rollout_root")
        try:
            for rank, device in enumerate(devices):
                first, stop = partitions[rank]
                first_chunk, stop_chunk = offsets[first], offsets[stop]
                directory = bank / f"rank-{rank:02d}"
                directory.mkdir()
                request = dict(profile=self.profile, config=asdict(policy.config),
                    policy=cpu_tree(policy.state_dict()), bound=asdict(bound), horizon=self.horizon,
                    cases=self.cases[self.cursor + first:self.cursor + stop],
                    seeds=seeds[first_chunk:stop_chunk],
                    noise=None if noise is None else noise[first_chunk:stop_chunk],
                    deterministic=deterministic, verify_zero=verify_zero and rank == 0,
                    rank=rank, world=world, output=str(directory), monitoring_root=str(self.monitoring_root),
                    episode_id_prefix=(self.profile.get('episode_id_namespace','')
                        + f"case-{self.cursor + first:06d}-rank-{rank:02d}-"),
                    attempt=f"{self.profile['attempt']}-{self.output.parent.name}-{self.cursor:06d}")
                request_path = directory / "request.pt"
                atomic_save(request_path, request)
                requests.append(request)
                if distributed:
                    from .rlinf_flora_ppo import write_new
                    queue = Path(distributed) / f"rank-{rank:02d}"
                    queue.mkdir(parents=True, exist_ok=True)
                    write_new(queue / (self.profile['attempt'] + '-' + self.output.parent.name
                                       + f"-{self.cursor:06d}.json"),
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
                if (result["cases"] != request["cases"] or len(result["records"]) != len(request["seeds"])
                        or len(result["episodes"]) != len(request["cases"])
                        or result["environment"]["cursor"] != len(request["cases"])
                        or result['environment'].get('rollout_lanes_per_gpu', 1) != self.environments_per_gpu
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
