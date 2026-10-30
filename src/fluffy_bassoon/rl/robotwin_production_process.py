"""Production-native isolated simulators served by a parent-owned motor.

Only CPU NumPy observations/actions cross the pipes. Each spawned process
owns its simulator, global RNG and allocated-GPU renderer; it never loads a
VLA. There is no automatic restart or reuse of a failed output identity.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np


class ProductionRPCError(RuntimeError):
    def __init__(self,message,completed=None):
        super().__init__(message)
        self.completed = completed or {}


def _cpu_observation(raw):
    result = {}
    for key in ("main_images", "wrist_images", "states"):
        value = raw[key]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        result[key] = np.ascontiguousarray(value)
    result["task_descriptions"] = list(raw["task_descriptions"])
    return result


class _NativeSimulator:
    """Production single-lane native semantics; no motor is loaded here."""
    def __init__(self, config):
        from .robotwin_noise import RLINF_COMMIT, ROBOTWIN_COMMIT, _install_allocated_renderer
        from fluffy_bassoon.gpu_execution import validate_execution
        validate_execution("baige", os.environ)
        if sys.platform != "linux":
            raise RuntimeError("Production simulators require allocated Linux")
        self.config = config
        for key, expected in (("rlinf_root", RLINF_COMMIT), ("robotwin_root", ROBOTWIN_COMMIT)):
            root = config[key]
            actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
            if actual != expected or dirty:
                raise RuntimeError(f"Unqualified production native source: {key}")
        sys.path[:0] = [config["rlinf_root"], config["robotwin_root"]]
        import torch
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Worker requires the parent's one-GPU allocation")
        self.torch = torch
        self.render_pci = _install_allocated_renderer(config["output_dir"])
        self.env = None
        self.task = None
        self.elapsed = 0
        self.active = False

    def reset(self, task, seed):
        from .robotwin_noise import (environment_config, _install_reset_retry_limit,
            _install_synchronous_vector_executor, _install_final_observation_capture)
        from fluffy_bassoon.baige_gpu import pci
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
        limit = self.config["task_episode_limits"][task]
        random.seed(seed); np.random.seed(seed); self.torch.manual_seed(seed)
        os.chdir(self.config["robotwin_root"])
        _install_reset_retry_limit()
        if self.env is not None and (self.task != task or self.episode_limit != limit):
            self.close()
        if self.env is None:
            cfg = environment_config(self.config["rlinf_root"], self.config["robotwin_root"],
                task, seed, limit, task_config_name=self.config["task_config_name"],
                final_observation_only=self.config["final_observation_only"])
            self.env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
            _install_synchronous_vector_executor(self.env.venv)
            if self.config["final_observation_only"]:
                _install_final_observation_capture(self.env.venv)
            self.task, self.episode_limit = task, limit
        raw, _ = self.env.reset(env_seeds=[seed])
        native = self.env.venv.envs[0].task
        renderer = native.scene.render_system.device
        actual = pci(renderer.pci_string.removeprefix("pci:"))
        if actual != self.render_pci or not renderer.can_render():
            raise RuntimeError("Worker renderer differs from allocated GPU")
        self.elapsed, self.active = 0, True
        return _cpu_observation(raw), dict(task=task, requested_seed=seed,
            actual_seed=int(native.ep_num), actual_render_pci=actual,
            instruction=list(raw["task_descriptions"]), worker_pid=os.getpid(),
            episode_limit=limit, reset_policy_actions=0,
            reset_internal_physics_steps="not separately instrumented")

    def step(self, actions):
        from .robotwin_noise import _capture_sparse_feedback
        if not self.active or len(actions) > self.episode_limit-self.elapsed:
            raise RuntimeError("Step requires an active episode and its complete remaining allowance")
        started = time.monotonic()
        with _capture_sparse_feedback(self.env.venv.envs[0].task) as feedback:
            obs, rewards, terms, truncs, infos = self.env.chunk_step(
                self.torch.from_numpy(actions[None]))
        mask = infos[-1].get("valid_action_mask")
        if mask is None or tuple(mask.shape) != (1,len(actions)):
            raise RuntimeError("Missing exact production valid-action mask")
        mask = _cpu_tree(mask)
        count = int(mask.sum())
        terminated = bool(self.torch.as_tensor(terms).any())
        truncated = bool(self.torch.as_tensor(truncs).any())
        if not 0 <= count <= len(actions) or (count == 0 and not (terminated or truncated)):
            raise RuntimeError("Invalid native production action count")
        self.elapsed += count
        self.active = not (terminated or truncated or self.elapsed >= self.episode_limit)
        reward = float(self.torch.as_tensor(rewards).sum())
        if not np.isfinite(reward):
            raise RuntimeError("Nonfinite production reward")
        return _cpu_observation(obs[-1]), reward, terminated, truncated, dict(
            primitive_steps=count, issued_primitive_steps=len(actions),
            episode_primitive_steps=self.elapsed, valid_action_mask=mask,
            success=bool(self.torch.as_tensor(infos[-1]["episode"]["success_once"]).any()),
            worker_pid=os.getpid(), duration_seconds=time.monotonic()-started,
            adapter_rewards=_cpu_tree(rewards), adapter_terminations=_cpu_tree(terms),
            adapter_truncations=_cpu_tree(truncs), **_cpu_tree(feedback))

    def close(self):
        if self.env is not None:
            self.env.venv.close()
            self.env.venv.env_thread_pool.shutdown(wait=True)
            self.env = None
        self.active = False


def _cpu_tree(value):
    if hasattr(value,"detach"):
        return value.detach().cpu().numpy().copy()
    if isinstance(value,np.ndarray): return value.copy()
    if isinstance(value,dict): return {key:_cpu_tree(item) for key,item in value.items()}
    if isinstance(value,list): return [_cpu_tree(item) for item in value]
    if isinstance(value,tuple): return tuple(_cpu_tree(item) for item in value)
    return value


def _worker_main(connection, config, factory):
    simulator = None
    request_id = 0
    try:
        simulator = factory(config)
        connection.send((0, True, dict(pid=os.getpid(), lane=config["lane"])))
        while True:
            request_id, command, payload = connection.recv()
            if command == "close":
                simulator.close()
                simulator = None
                connection.send((request_id, True, None))
                return
            if command == "reset":
                result = simulator.reset(*payload)
            elif command == "step":
                result = simulator.step(payload)
            else:
                raise ValueError("Unknown simulator-worker command")
            connection.send((request_id, True, result))
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send((request_id, False, dict(error=f"{type(error).__name__}: {error}",
                                                     traceback=traceback.format_exc())))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if simulator is not None:
            try:
                simulator.close()
            except BaseException:
                pass
        connection.close()


class ProductionSimulatorPool:
    """A bounded synchronous request API with concurrent independent workers.

    Selected-lane operations retain the caller's index order. `step` returns a
    list of `(raw, reward, terminated, truncated, info)` tuples. Supply either
    a dense `[lanes,H,14]` array or a list of variable-length `[H_i,14]` arrays;
    `action_counts` may additionally trim each lane's allowance. A worker's
    episode must be reset after it terminates. Motor/model work stays outside
    this class, in the parent process.
    """
    def __init__(self, num_workers, *, rlinf_root, robotwin_root, output_dir,
                 task_episode_limits, final_observation_only=False,
                 task_config_name="demo_clean", timeout_seconds=120,
                 _simulator_factory=None):
        if (type(num_workers) is not int or num_workers < 1 or not task_episode_limits
                or any(type(limit) is not int or limit < 1 for limit in task_episode_limits.values()) or not np.isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Invalid simulator-pool size, horizon or timeout")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.timeout_seconds = float(timeout_seconds)
        self.num_workers = num_workers
        self.processes = []
        self.connections = []
        self.active = [False] * num_workers
        self.failed = False
        self.closed = False
        self._request_id = 0
        context = mp.get_context("spawn")
        factory = _simulator_factory or _NativeSimulator
        common = dict(rlinf_root=str(Path(rlinf_root).resolve()),
                      robotwin_root=str(Path(robotwin_root).resolve()),
                      task_episode_limits=dict(task_episode_limits), final_observation_only=bool(final_observation_only),
                      task_config_name=task_config_name)
        try:
            for lane in range(num_workers):
                parent, child = context.Pipe()
                config = dict(common, lane=lane, output_dir=str(self.output_dir / f"lane-{lane}"))
                process = context.Process(target=_worker_main, args=(child, config, factory),
                                          name=f"robotwin-sim-{os.getpid()}-{lane}")
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            ready = self._receive(list(range(num_workers)), 0)
            (self.output_dir / "workers.json").write_text(json.dumps(ready, indent=2) + "\n")
        except BaseException as error:
            self._abort("startup", error)
            raise

    def _indices(self, indices):
        if self.closed or self.failed:
            raise ProductionRPCError("Simulator pool is closed/failed; use a fresh output identity")
        indices = list(range(self.num_workers)) if indices is None else list(indices)
        if (not indices or len(set(indices)) != len(indices)
                or any(type(index) is not int or not 0 <= index < self.num_workers for index in indices)):
            raise ValueError("Require unique valid simulator lane indices")
        return indices

    def _receive(self, indices, request_id):
        pending = {self.connections[index]: index for index in indices}
        deadline = time.monotonic() + self.timeout_seconds
        result = {}
        errors = []
        while pending:
            remaining = deadline - time.monotonic()
            try:
                ready = wait(list(pending), timeout=max(0., remaining))
            except BaseException as error:
                errors.append(f"Simulator request interrupted: {type(error).__name__}: {error}")
                break
            if not ready:
                errors.append(f"Simulator request timed out; pending lanes {list(pending.values())}")
                break
            for connection in ready:
                lane = pending.pop(connection)
                try:
                    actual_id, success, value = connection.recv()
                except (EOFError, OSError) as error:
                    errors.append(f"Simulator lane {lane} exited without a reply: {error}")
                    continue
                if actual_id != request_id:
                    errors.append(f"Simulator lane {lane} returned the wrong request identity")
                    continue
                if not success:
                    errors.append(f"Simulator lane {lane} failed: {value['error']}\n{value['traceback']}")
                    continue
                result[lane] = value
        if errors:
            raise ProductionRPCError("\n".join(errors), completed=result)
        return [result[index] for index in indices]

    def _request(self, command, payloads, indices):
        self._request_id += 1
        try:
            for index, payload in zip(indices, payloads):
                self.connections[index].send((self._request_id, command, payload))
            return self._receive(indices, self._request_id)
        except BaseException as error:
            self._abort(command, error)
            raise

    def reset(self, tasks_or_task, seeds, *, indices=None):
        from .robotwin_noise import SOURCE_TASKS, HELDOUT_TASKS

        indices = self._indices(indices)
        seeds = list(seeds)
        tasks = [tasks_or_task] * len(indices) if isinstance(tasks_or_task, str) else list(tasks_or_task)
        if (len(seeds) != len(indices) or len(tasks) != len(indices)
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or any(task not in SOURCE_TASKS + HELDOUT_TASKS for task in tasks)):
            raise ValueError("Require one explicit inventory task and nonnegative seed per selected lane")
        results = self._request("reset", list(zip(tasks, seeds)), indices)
        for index in indices:
            self.active[index] = True
        return [result[0] for result in results], [result[1] for result in results]

    def step(self, actions, *, indices=None, action_counts=None):
        indices = self._indices(indices)
        if len(actions) != len(indices) or any(not self.active[index] for index in indices):
            raise ValueError("Require one action chunk per selected active lane")
        if action_counts is not None and len(action_counts) != len(indices):
            raise ValueError("Require one optional action allowance per selected lane")
        payloads = []
        for lane, value in enumerate(actions):
            value = np.asarray(value, dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != 14 or len(value) < 1 or not np.isfinite(value).all():
                raise ValueError("Require finite native [H,14] action chunks")
            count = len(value) if action_counts is None else action_counts[lane]
            if type(count) is not int or not 1 <= count <= len(value):
                raise ValueError("Invalid selected-lane action allowance")
            payloads.append(np.ascontiguousarray(value[:count]))
        results = self._request("step", payloads, indices)
        for index, result in zip(indices, results):
            self.active[index] = not (result[2] or result[3])
        return results

    def _abort(self, operation, error):
        self.failed = True
        try:
            (self.output_dir / "failure.json").write_text(json.dumps(dict(
                operation=operation, error=f"{type(error).__name__}: {error}",
                worker_pids=[p.pid for p in self.processes]), indent=2) + "\n")
        finally:
            self._stop_owned_children()

    def _stop_owned_children(self):
        for connection in self.connections:
            connection.close()
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self.closed = True

    def close(self):
        if self.closed:
            return
        try:
            self._request("close", [None] * self.num_workers, list(range(self.num_workers)))
        finally:
            self._stop_owned_children()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
