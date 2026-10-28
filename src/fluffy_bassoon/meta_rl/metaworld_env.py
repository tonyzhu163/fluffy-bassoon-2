"""Fixed-horizon Meta-World trials, without task identity in policy inputs."""

import hashlib
import pickle

import gymnasium as gym
import numpy as np


BENCHMARKS = ("ml1-push", "ml10", "pointmaze", "halfcheetah-vel")


def make_trial(factory, tasks, *, benchmark, **options):
    if benchmark in ("pointmaze", "halfcheetah-vel"):
        from .metta_env import MemoryTrial
        return MemoryTrial(factory, tasks, benchmark=benchmark, **options)
    if benchmark not in ("ml1-push", "ml10"):
        raise ValueError("Unknown benchmark")
    return MetaWorldTrial(factory, tasks, **options)


def task_digest(tasks):
    return hashlib.sha256(b"".join(t.env_name.encode() + t.data for t in tasks)).hexdigest()


def benchmark_pools(benchmark, seed=0):
    if benchmark in ("pointmaze", "halfcheetah-vel"):
        from .metta_env import memory_pools
        return memory_pools(benchmark, seed)
    import metaworld
    factory = (lambda s: metaworld.ML1("push-v3", seed=s)) if benchmark == "ml1-push" else (
        (lambda s: metaworld.ML10(seed=s)) if benchmark == "ml10" else None)
    if factory is None:
        raise ValueError("Supported benchmarks: ml1-push, ml10")
    source = factory(seed)
    fresh = factory(seed + 1_000_000)
    pools = {"train": (source.train_classes, source.train_tasks),
             "train_eval": (fresh.train_classes, fresh.train_tasks),
             "test": (source.test_classes, source.test_tasks)}
    train_keys = {(t.env_name, t.data) for t in source.train_tasks}
    if train_keys.intersection((t.env_name, t.data) for t in fresh.train_tasks):
        raise RuntimeError("Training and train-family evaluation tasks overlap")
    return pools


class MetaWorldTrial(gym.Env):
    """An inner time limit is a physical reset, not the end of a meta-trial.

    ML1 holds a task instance fixed across episodes; goal-visible ML10 resamples
    the instance within the same family. Dense native reward is scaled by a
    declared constant. Binary episode success is recorded separately.
    """

    metadata = {"render_modes": []}

    def __init__(self, factory, tasks, *, attempts=5, horizon=500,
                 goal_visible=False, resample_instance=False, reward_scale=.1):
        if (not tasks or len({t.env_name for t in tasks}) != 1
                or type(attempts) is not int or attempts < 1
                or type(horizon) is not int or not 1 <= horizon <= 500
                or not np.isfinite(reward_scale) or reward_scale <= 0):
            raise ValueError("Invalid Meta-World trial contract")
        self.env = factory()
        self.tasks = tuple(tasks)
        self.attempts, self.horizon = attempts, horizon
        self.goal_visible, self.resample_instance = goal_visible, resample_instance
        self.reward_scale = reward_scale
        self.action_space = gym.spaces.Box(-1., 1., (4,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, (39,), dtype=np.float32),
            "previous_action": self.action_space,
            "previous_reward": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
            "attempt_end": gym.spaces.Box(0., 1., (1,), dtype=np.float32),
            "clock": gym.spaces.Box(np.zeros(2, np.float32),
                                    np.array([max(attempts - 1, 1), horizon], np.float32))})
        self.finished = True

    def _reset_inner(self):
        if self.episode == 0 or self.resample_instance:
            self.task = self.tasks[int(self.np_random.integers(len(self.tasks)))]
        # set_task is the upstream interface for configuring goal visibility.
        data = pickle.loads(self.task.data)
        data["partially_observable"] = not self.goal_visible
        self.env.set_task(type(self.task)(env_name=self.task.env_name, data=pickle.dumps(data)))
        # Meta-World V3 reset(seed=...) explicitly ignores its seed argument.
        self.env.seed(int(self.np_random.integers(2**31)))
        state, _ = self.env.reset()
        self.episode_success, self.inner_step = False, 0
        return self._state(state)

    def _state(self, state):
        state = np.asarray(state, dtype=np.float32).copy()
        if state.shape != (39,) or not np.isfinite(state).all():
            raise ValueError("Expected finite Meta-World V3 observation of length 39")
        if not self.goal_visible and np.any(state[-3:] != 0):
            raise RuntimeError("Hidden-goal observation contains goal coordinates")
        return state

    def _observation(self, state, action, reward, boundary):
        return dict(state=state.copy(), previous_action=np.asarray(action, np.float32).copy(),
                    previous_reward=np.array([reward], np.float32),
                    attempt_end=np.array([boundary], np.float32),
                    clock=np.array([min(self.episode, self.attempts - 1), self.inner_step], np.float32))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode, self.finished = 0, False
        self.outcomes, self.native_returns, self.current_return = [], [], 0.
        state = self._reset_inner()
        return self._observation(state, np.zeros(4), 0., False), {}

    def step(self, action):
        if self.finished:
            raise gym.error.ResetNeeded("Reset the completed trial")
        action = np.asarray(action, np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite and bounded by [-1, 1]")
        state, reward, terminated, truncated, info = self.env.step(action)
        state = self._state(state)
        if not np.isfinite(reward):
            raise FloatingPointError("Nonfinite native Meta-World reward")
        self.inner_step += 1
        self.episode_success |= bool(info["success"])
        self.current_return += float(reward)
        boundary = self.inner_step == self.horizon
        if terminated or (truncated and not boundary):
            raise RuntimeError("Unexpected early Meta-World termination")
        completed_episode = self.episode
        if boundary:
            self.outcomes.append(self.episode_success)
            self.native_returns.append(self.current_return)
            self.current_return = 0.
            self.episode += 1
            self.finished = self.episode == self.attempts
            if not self.finished:
                state = self._reset_inner()
        scaled_reward = float(reward) * self.reward_scale
        return (self._observation(state, action, scaled_reward, boundary), scaled_reward,
                self.finished, False, dict(attempt_end=boundary, attempt_index=completed_episode,
                                          is_success=self.outcomes[-1] if boundary else False))

    def close(self):
        self.env.close()


def make_lanes(pool, count, *, benchmark, attempts=5, horizon=500):
    classes, tasks = pool
    names = list(classes)
    if count < len(names) or count % len(names):
        raise ValueError("Lane count must be a positive multiple of the family count")
    lanes = []
    try:
        for index in range(count):
            name = names[index % len(names)]
            env = make_trial(classes[name], [t for t in tasks if t.env_name == name], benchmark=benchmark,
                                 attempts=attempts, horizon=horizon,
                                 goal_visible=benchmark == "ml10",
                                 resample_instance=benchmark == "ml10")
            env.family = name
            lanes.append(env)
    except BaseException:
        for env in lanes:
            env.close()
        raise
    return lanes
