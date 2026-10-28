"""Optional persistent CPU subprocess lanes for the common Meta-World trials.

Call batch_reset/batch_step to dispatch every lane group before waiting. Calling
each proxy's step in a loop remains serial. No policy or CUDA object is passed
to workers; all processes use the multiprocessing spawn context.
"""

import atexit
from multiprocessing import get_context
from multiprocessing.connection import wait
import os
import threading
import time
import traceback

from .metaworld_env import BENCHMARKS, make_trial


class MetaWorldWorkerError(RuntimeError):
    """A worker failed; the entire owned batch is closed and cannot be reused."""


def _worker_main(connection, specs):
    lanes = {}
    try:
        for index, family, factory, tasks, options in specs:
            env = make_trial(factory, tasks, **options)
            env.family = family
            lanes[index] = env
        connection.send(("ready", {index: dict(observation_space=env.observation_space,
            action_space=env.action_space) for index, env in lanes.items()}))
        while True:
            operation, requests = connection.recv()
            if operation == "close":
                connection.send(("closed", None))
                return
            if operation not in ("reset", "step"):
                raise ValueError("Unknown Meta-World worker operation")
            results = {}
            for index, argument in requests:
                env = lanes[index]
                result = env.reset(seed=argument) if operation == "reset" else env.step(argument)
                boundary = operation == "reset" or bool(result[-1]["attempt_end"])
                # Large task bytes and outcome lists change only at resets or
                # episode boundaries. Avoid retransmitting them every decision.
                snapshot = dict(finished=env.finished, episode=env.episode,
                                inner_step=env.inner_step, current_return=env.current_return)
                if boundary:
                    snapshot.update(task=env.task, outcomes=list(env.outcomes),
                                    native_returns=list(env.native_returns))
                    if hasattr(env, "episode_metrics"):
                        snapshot["episode_metrics"] = list(env.episode_metrics)
                results[index] = result, snapshot
            connection.send(("result", results))
    except EOFError:
        pass
    except BaseException as exc:
        try:
            connection.send(("error", dict(type=type(exc).__name__, message=str(exc),
                                           traceback=traceback.format_exc()[-8000:])))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        for env in lanes.values():
            try:
                env.close()
            except BaseException:
                pass
        connection.close()


def _cpu_limit(allocated_cpus):
    scheduled = os.environ.get("SLURM_CPUS_PER_TASK")
    limits = []
    if scheduled is not None:
        try:
            limits.append(int(scheduled))
        except ValueError as exc:
            raise ValueError("SLURM_CPUS_PER_TASK must be a positive integer") from exc
    if allocated_cpus is not None:
        if type(allocated_cpus) is not int:
            raise ValueError("Allocated CPU count must be a positive integer")
        limits.append(allocated_cpus)
    if hasattr(os, "sched_getaffinity"):
        limits.append(len(os.sched_getaffinity(0)))
    if not limits or any(value < 1 for value in limits):
        raise ValueError("Supply the allocated CPU count or SLURM_CPUS_PER_TASK")
    # Affinity alone may include a whole login host, not an allocation.
    if scheduled is None and allocated_cpus is None:
        raise ValueError("Supply the allocated CPU count or SLURM_CPUS_PER_TASK")
    return min(limits)


class MetaWorldLaneProxy:
    """Environment-shaped lane backed by a shared persistent batch coordinator.

    Closing any proxy closes its complete owning batch. This keeps existing
    `for env in envs: env.close()` cleanup safe and idempotent.
    """

    def __init__(self, owner, index, family, tasks, options, spaces):
        self._worker_batch, self._lane_index = owner, index
        self.family, self.tasks = family, tuple(tasks)
        self.attempts, self.horizon = options["attempts"], options["horizon"]
        self.goal_visible, self.resample_instance = options["goal_visible"], options["resample_instance"]
        self.variable_length_trials = options["benchmark"] == "pointmaze"
        self.reward_scale = options["reward_scale"]
        self.observation_space, self.action_space = spaces["observation_space"], spaces["action_space"]
        self.finished, self.outcomes, self.native_returns = True, [], []

    def reset(self, *, seed=None, options=None):
        if options is not None:
            raise ValueError("Meta-World worker resets do not use options")
        return self._worker_batch.execute("reset", [self], [seed])[0]

    def step(self, action):
        return self._worker_batch.execute("step", [self], [action])[0]

    def close(self):
        self._worker_batch.close()


class MetaWorldWorkerBatch:
    """Persistent lane groups with process affinity and ordered batch replies."""

    def __init__(self, pool, count, *, benchmark, attempts=5, horizon=500,
                 workers=1, allocated_cpus=None, timeout_seconds=120.):
        classes, tasks = pool
        names = list(classes)
        if (benchmark not in BENCHMARKS or not names or type(count) is not int
                or count < len(names) or count % len(names) or type(workers) is not int
                or not 1 <= workers <= min(count, _cpu_limit(allocated_cpus))
                or not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float("inf")
                or type(attempts) is not int or attempts < 1 or type(horizon) is not int or not 1 <= horizon <= 500):
            raise ValueError("Invalid Meta-World worker count, allocation or trial contract")
        self.worker_count, self.allocated_cpus = workers, _cpu_limit(allocated_cpus)
        self.timeout_seconds, self.closed = float(timeout_seconds), False
        self._lock, self._connections, self._processes, self._group_by_lane = threading.RLock(), [], [], {}
        self._group_indices, self.lanes = [], []
        options = dict(benchmark=benchmark, attempts=attempts, horizon=horizon, goal_visible=benchmark == "ml10",
                       resample_instance=benchmark == "ml10", reward_scale=.1)
        family_tasks = {name: tuple(task for task in tasks if task.env_name == name) for name in names}
        if any(not family_tasks[name] for name in names):
            raise ValueError("Every Meta-World family needs at least one task")
        specs = [(i, names[i % len(names)], classes[names[i % len(names)]],
                  family_tasks[names[i % len(names)]], options) for i in range(count)]
        context = get_context("spawn")
        try:
            for group in range(workers):
                members = specs[group::workers]
                parent, child = context.Pipe()
                process = context.Process(target=_worker_main, args=(child, members),
                                          name=f"metaworld-env-{group}", daemon=True)
                try:
                    process.start()
                except BaseException:
                    parent.close()
                    child.close()
                    raise
                child.close()
                self._connections.append(parent)
                self._processes.append(process)
                indices = tuple(spec[0] for spec in members)
                self._group_indices.append(indices)
                self._group_by_lane.update({index: group for index in indices})
            ready = self._receive(range(workers), expected="ready")
            spaces = {index: value for group in ready.values() for index, value in group.items()}
            self.lanes = [MetaWorldLaneProxy(self, index, family, lane_tasks, lane_options, spaces[index])
                          for index, family, _, lane_tasks, lane_options in specs]
            atexit.register(self.close)
        except BaseException:
            self._shutdown(graceful=False)
            raise

    @property
    def manifest(self):
        return dict(backend="spawn_subprocess", workers=self.worker_count,
                    allocated_cpus=self.allocated_cpus, start_method="spawn",
                    lanes_by_worker=[list(indices) for indices in self._group_indices],
                    worker_pids=[process.pid for process in self._processes],
                    timeout_seconds=self.timeout_seconds)

    def _receive(self, groups, *, expected):
        pending = {self._connections[group]: group for group in groups}
        deadline, replies = time.monotonic() + self.timeout_seconds, {}
        while pending:
            ready = wait(list(pending), timeout=max(0., deadline - time.monotonic()))
            if not ready:
                raise MetaWorldWorkerError(f"Meta-World workers timed out: groups {list(pending.values())}")
            for connection in ready:
                group = pending.pop(connection)
                try:
                    kind, value = connection.recv()
                except (EOFError, OSError) as exc:
                    raise MetaWorldWorkerError(f"Meta-World worker {group} exited during {expected}") from exc
                if kind == "error":
                    raise MetaWorldWorkerError(f"Meta-World worker {group}, lanes {self._group_indices[group]}: "
                                               f"{value['type']}: {value['message']}\n{value['traceback']}")
                if kind != expected:
                    raise MetaWorldWorkerError(f"Unexpected reply from Meta-World worker {group}: {kind}")
                replies[group] = value
        return replies

    def execute(self, operation, lanes, arguments):
        with self._lock:
            if self.closed:
                raise MetaWorldWorkerError("Meta-World worker batch is closed")
            if (not lanes or len(lanes) != len(arguments)
                    or any(lane._worker_batch is not self for lane in lanes)
                    or len({lane._lane_index for lane in lanes}) != len(lanes)):
                raise ValueError("Batch operation needs distinct lanes from one worker batch")
            requests = {}
            for lane, argument in zip(lanes, arguments, strict=True):
                requests.setdefault(self._group_by_lane[lane._lane_index], []).append((lane._lane_index, argument))
            try:
                # Every group starts before we wait, unlike a per-proxy step loop.
                for group, items in requests.items():
                    self._connections[group].send((operation, items))
                grouped = self._receive(requests, expected="result")
                results = {index: item for reply in grouped.values() for index, item in reply.items()}
                ordered = []
                for lane in lanes:
                    result, snapshot = results[lane._lane_index]
                    lane.__dict__.update(snapshot)
                    ordered.append(result)
                return ordered
            except BaseException:
                # A partial batch cannot be recovered in place: its lanes may
                # have advanced by different amounts. Resume at a trial checkpoint.
                self._shutdown(graceful=False)
                raise

    def _shutdown(self, *, graceful):
        if self.closed:
            return
        self.closed = True
        atexit.unregister(self.close)
        if graceful:
            for connection, process in zip(self._connections, self._processes, strict=True):
                if process.is_alive():
                    try:
                        connection.send(("close", None))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
            deadline = time.monotonic() + min(self.timeout_seconds, 5.)
            for process in self._processes:
                process.join(timeout=max(0., deadline - time.monotonic()))
        for connection in self._connections:
            connection.close()
        # These are exclusively the CPU worker children created by this batch.
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=1.)
            if process.is_alive():
                process.kill()
                process.join(timeout=1.)

    def close(self):
        with self._lock:
            self._shutdown(graceful=True)


def make_worker_lanes(pool, count, **kwargs):
    return MetaWorldWorkerBatch(pool, count, **kwargs).lanes


def _owner(envs):
    owners = {env._worker_batch for env in envs if isinstance(env, MetaWorldLaneProxy)}
    if not owners:
        return None
    if len(owners) != 1 or any(not isinstance(env, MetaWorldLaneProxy) for env in envs):
        raise ValueError("Do not mix worker coordinators or serial environments in one batch")
    return next(iter(owners))


def batch_reset(envs, seeds):
    if not envs or len(envs) != len(seeds):
        raise ValueError("Require one reset seed per environment")
    values = [None if seed is None else int(seed) for seed in seeds]
    owner = _owner(envs)
    return (owner.execute("reset", envs, values) if owner is not None else
            [env.reset(seed=seed) for env, seed in zip(envs, values, strict=True)])


def batch_step(envs, actions):
    if not envs or len(envs) != len(actions):
        raise ValueError("Require one executed action per environment")
    owner = _owner(envs)
    return (owner.execute("step", envs, actions) if owner is not None else
            [env.step(action) for env, action in zip(envs, actions, strict=True)])


def close_lanes(envs):
    owner = _owner(envs)
    if owner is not None:
        owner.close()
    else:
        for env in envs:
            env.close()
