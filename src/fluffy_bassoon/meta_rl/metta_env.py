"""Versioned, hidden-task PointMaze and HalfCheetahVel trial adapters.

These are METTA-informed tasks, not a reproduction of its unreleased code.
Only public state/action/reward/boundary/clock fields reach the policy.
"""

from collections import deque
from dataclasses import dataclass
import json

import gymnasium as gym
import numpy as np


# An explicit 7x7 S maze. The paper does not publish its maze in machine-readable
# form; do not silently call this layout an exact METTA reproduction.
MAZE = ((1, 1, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 0, 1),
        (1, 0, 0, 0, 0, 0, 1),
        (1, 0, 1, 1, 1, 1, 1),
        (1, 0, 0, 0, 0, 0, 1),
        (1, 1, 1, 1, 1, 1, 1))
FREE_CELLS = frozenset((r, c) for r, row in enumerate(MAZE) for c, v in enumerate(row) if not v)


@dataclass(frozen=True)
class MemoryTask:
    env_name: str
    data: bytes


def task(name, **values):
    return MemoryTask(name, json.dumps(values, sort_keys=True, separators=(",", ":")).encode())


def distant_pairs():
    pairs = []
    for start in sorted(FREE_CELLS):
        distances, queue = {start: 0}, deque([start])
        while queue:
            r, c = queue.popleft()
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in FREE_CELLS and neighbor not in distances:
                    distances[neighbor] = distances[(r, c)] + 1
                    queue.append(neighbor)
        pairs.extend((start, goal, distance) for goal, distance in sorted(distances.items()) if distance >= 10)
    return pairs


def pointmaze_factory():
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    return gym.make("PointMaze_UMaze-v3", maze_map=[list(row) for row in MAZE],
                    reward_type="sparse", continuing_task=False, reset_target=False,
                    max_episode_steps=500)


def halfcheetah_factory():
    return gym.make("HalfCheetah-v5", ctrl_cost_weight=.05,
                    exclude_current_positions_from_observation=True, max_episode_steps=500)


def memory_pools(benchmark, seed=0):
    if benchmark not in ("pointmaze", "halfcheetah-vel") or type(seed) is not int or seed < 0:
        raise ValueError("Unknown memory benchmark or invalid seed")
    rng = np.random.default_rng(seed)
    if benchmark == "pointmaze":
        # Each seed fixes continuous reset AND goal offsets across a trial.
        # Held-out configurations can share cells; this is not held-out-map testing.
        pairs = distant_pairs()
        order = rng.permutation(len(pairs))
        values = []
        for index in range(148):
            start, goal, distance = pairs[int(order[index % len(order)])]
            values.append(task(benchmark, start=start, goal=goal, distance=distance,
                               reset_seed=int(rng.integers(2**31))))
        factory = pointmaze_factory
    else:
        values = [task(benchmark, velocity=float(v)) for v in rng.uniform(0., 2., 148)]
        factory = halfcheetah_factory
    splits = {"train": values[:100], "train_eval": values[100:124], "test": values[124:]}
    if benchmark == "halfcheetah-vel":
        splits["ood"] = [task(benchmark, velocity=float(v)) for v in rng.uniform(2., 3., 24)]
    if len({value.data for split in splits.values() for value in split}) != sum(map(len, splits.values())):
        raise RuntimeError("Task splits overlap")
    return {name: ({benchmark: factory}, tuple(tasks)) for name, tasks in splits.items()}


def benchmark_contract(benchmark):
    common = dict(goal_visible=False, resample_instance=False,
                  replication="METTA-informed independent adapter; exact upstream parity unverified")
    if benchmark == "pointmaze":
        return dict(**common, reward="+1 once at goal; zero otherwise; native success terminates each episode",
                    maze=MAZE, minimum_cell_path_distance=10,
                    split="100/24/24 disjoint seeded continuous start/goal configurations; shared layout/cells",
                    termination="early physical reset at success; masked padding only AFTER complete trial",
                    padding="no simulator action or PPO actor/critic/entropy loss; excluded from state moments")
    if benchmark == "halfcheetah-vel":
        return dict(**common, reward="-abs(x_velocity - hidden_target) - 0.05 * sum(action**2)",
                    target_train_range=[0., 2.], target_ood_range=[2., 3.],
                    split="100 training, 24 fresh training-range, 24 IID test, 24 extrapolation velocities",
                    primary_metric="native return and absolute velocity error; success threshold is diagnostic only")
    raise ValueError("Unknown memory benchmark")


class MemoryTrial(gym.Env):
    """Task-fixed episodes, with optional absorbing suffix after early trial end.

    An absorbing suffix contains no real interaction and cannot affect any later
    real action: all K physical episodes have already finished. PPO masks it.
    """

    metadata = {"render_modes": []}

    def __init__(self, factory, tasks, *, benchmark, attempts=5, horizon=100,
                 goal_visible=False, resample_instance=False, reward_scale=.1):
        if (benchmark not in ("pointmaze", "halfcheetah-vel") or not tasks
                or {t.env_name for t in tasks} != {benchmark}
                or type(attempts) is not int or attempts < 1
                or type(horizon) is not int or not 1 <= horizon <= 500
                or goal_visible or resample_instance):
            raise ValueError("Invalid hidden-task trial contract")
        self.benchmark, self.tasks = benchmark, tuple(tasks)
        self.attempts, self.horizon = attempts, horizon
        self.goal_visible, self.resample_instance, self.reward_scale = False, False, 1.
        self.variable_length_trials = benchmark == "pointmaze"
        self.env = factory()
        self.action_space = gym.spaces.Box(-1., 1., self.env.action_space.shape, np.float32)
        source = self.env.observation_space
        state_shape = source["observation"].shape if benchmark == "pointmaze" else source.shape
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, state_shape, np.float32),
            "previous_action": self.action_space,
            "previous_reward": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "attempt_end": gym.spaces.Box(0., 1., (1,), np.float32),
            "clock": gym.spaces.Box(np.zeros(2, np.float32),
                                     np.array([max(attempts - 1, 1), horizon], np.float32))})
        self.finished, self._initialized = True, False

    def _state(self, observation):
        # Never flatten the GoalEnv dict: desired_goal would reveal the answer.
        state = np.asarray(observation["observation"] if self.variable_length_trials else observation,
                           np.float32).copy()
        if state.shape != self.observation_space["state"].shape or not np.isfinite(state).all():
            raise ValueError("Invalid simulator state")
        return state

    def _cell(self, state):
        return (int(np.floor(3.5 - float(state[1]))), int(np.floor(3.5 + float(state[0]))))

    def _reset_inner(self):
        self.inner_step, self.current_return, self.velocity_errors = 0, 0., []
        if self.variable_length_trials:
            raw, _ = self.env.reset(seed=self.setting["reset_seed"], options={
                "reset_cell": np.asarray(self.setting["start"], dtype=int),
                "goal_cell": np.asarray(self.setting["goal"], dtype=int)})
        else:
            raw, _ = self.env.reset(seed=int(self.np_random.integers(2**31)))
        self._current_state = self._state(raw)
        self.visited = {self._cell(self._current_state)} if self.variable_length_trials else set()
        self.start_cell = next(iter(self.visited), None)
        return self._current_state

    def _observation(self, state, action, reward, boundary):
        return dict(state=state.copy(), previous_action=np.asarray(action, np.float32).copy(),
                    previous_reward=np.array([reward], np.float32),
                    attempt_end=np.array([boundary], np.float32),
                    clock=np.array([min(self.episode, self.attempts - 1), self.inner_step], np.float32))

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Hidden-task overrides are not public reset options")
        super().reset(seed=seed)
        self.task = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        self.setting = json.loads(self.task.data)
        self.episode, self.finished, self._initialized = 0, False, True
        self.outcomes, self.native_returns, self.episode_metrics = [], [], []
        self._previous_visited = None
        state = self._reset_inner()
        return self._observation(state, np.zeros(self.action_space.shape), 0., False), {}

    def step(self, action):
        if not self._initialized:
            raise gym.error.ResetNeeded("Reset the trial before stepping")
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite and bounded")
        if self.finished:
            if not self.variable_length_trials:
                raise gym.error.ResetNeeded("Reset the completed trial")
            return (self._observation(self._current_state, np.zeros_like(action), 0., False),
                    0., True, False, dict(attempt_end=False, simulator_step=False,
                                         attempt_index=self.attempts - 1, is_success=False))
        raw, _, terminated, truncated, info = self.env.step(action)
        state = self._state(raw)
        self.inner_step += 1
        if self.variable_length_trials:
            success = bool(info["success"])
            reward = float(success)
            self.visited.add(self._cell(state))
            boundary = success or self.inner_step == self.horizon
            if terminated != success or (truncated and self.inner_step != self.horizon):
                raise RuntimeError("Unexpected PointMaze termination")
        else:
            velocity = float(info["x_velocity"])
            error = abs(velocity - self.setting["velocity"])
            reward = -error - .05 * float(np.square(action).sum())
            self.velocity_errors.append(error)
            boundary, success = self.inner_step == self.horizon, False
            if terminated or (truncated and not boundary):
                raise RuntimeError("Unexpected HalfCheetah termination")
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite benchmark reward")
        self.current_return += reward
        completed_episode = self.episode
        if boundary:
            details = dict(steps=self.inner_step)
            if self.variable_length_trials:
                cells = self.visited - {self.start_cell}
                previous = self._previous_visited
                details.update(coverage=len(self.visited & FREE_CELLS) / len(FREE_CELLS),
                    episode_jaccard=(None if previous is None else
                        len(cells & previous) / max(1, len(cells | previous))))
                self._previous_visited = cells
            else:
                details["absolute_velocity_error"] = float(np.mean(self.velocity_errors))
                success = details["absolute_velocity_error"] <= .2
            self.outcomes.append(success)
            self.native_returns.append(self.current_return)
            self.episode_metrics.append(details)
            self.episode += 1
            self.finished = self.episode == self.attempts
            if not self.finished:
                state = self._reset_inner()
        self._current_state = state
        return (self._observation(state, action, reward, boundary), reward, self.finished, False,
                dict(attempt_end=boundary, attempt_index=completed_episode,
                     is_success=success if boundary else False, simulator_step=True))

    def close(self):
        self.env.close()
