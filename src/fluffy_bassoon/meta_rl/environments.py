"""Small Gymnasium trials with persistent hidden settings and fixed-length attempts."""

import gymnasium as gym
import numpy as np


class _TrialEnv(gym.Env):
    metadata = {"render_modes": []}
    dense_reward = False  # Sparse tasks pay only at attempt ends; dense tasks pay on every step.

    def __init__(self, *, attempts, horizon, state_dim, action_space):
        for name, value in (("attempts", attempts), ("horizon", horizon)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.attempts, self.horizon = attempts, horizon
        self.action_space = action_space
        self._discrete = isinstance(action_space, gym.spaces.Discrete)
        action_dim = action_space.n if self._discrete else action_space.shape[0]
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-2, 2, (state_dim,), dtype=np.float32),
            "transition_state": gym.spaces.Box(-2, 2, (state_dim,), dtype=np.float32),
            "previous_action": gym.spaces.Box(0 if self._discrete else -1, 1,
                                             (action_dim,), dtype=np.float32),
            "previous_reward": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "attempt_end": gym.spaces.Box(0, 1, (1,), dtype=np.float32),
            "clock": gym.spaces.Box(np.zeros(2, dtype=np.float32),
                                    np.array([attempts - 1, horizon], dtype=np.float32)),
        })
        self._done = True

    def _observation(self, transition_state, action, reward, attempt_end):
        return {
            "state": self._state.copy(),
            "transition_state": transition_state.copy(),
            "previous_action": action.copy(),
            "previous_reward": np.array([reward], dtype=np.float32),
            "attempt_end": np.array([attempt_end], dtype=np.float32),
            "clock": np.array([self._attempt, self._elapsed], dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Trial reset does not accept hidden-setting overrides")
        super().reset(seed=seed)
        self._sample_hidden()
        self._attempt = self._elapsed = 0
        self._done = False
        self._state = self._start_attempt()
        previous_action = np.zeros(self.observation_space["previous_action"].shape,
                                   dtype=np.float32)
        return self._observation(self._state, previous_action, 0., False), {}

    def step(self, action):
        if self._done:
            raise gym.error.ResetNeeded("Reset the trial before stepping")
        if not self._discrete:
            action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError("Action must be finite and inside the declared action space")
        transition_state, success = self._transition(action)
        attempt_success = self._attempt_success(success)
        if self._discrete:
            previous_action = np.zeros(self.action_space.n, dtype=np.float32)
            previous_action[int(action)] = 1.
        else:
            previous_action = action.copy()
        self._elapsed += 1
        attempt_end = self._elapsed == self.horizon
        reward = float(success) if attempt_end or self.dense_reward else 0.
        acted_attempt = self._attempt
        self._done = attempt_end and self._attempt == self.attempts - 1
        self._state = transition_state
        if attempt_end and not self._done:
            self._attempt += 1
            self._elapsed = 0
            self._state = self._start_attempt()
        observation = self._observation(transition_state, previous_action, reward, attempt_end)
        info = {"attempt_end": attempt_end, "attempt_index": acted_attempt,
                "is_success": attempt_success if attempt_end else False}
        return observation, reward, self._done, False, info

    def _attempt_success(self, success):
        return bool(success)


class HiddenRewardBandit(_TrialEnv):
    """One pull per attempt; the rewarded arm stays fixed for the entire trial."""

    def __init__(self, *, attempts=4, arms=2):
        if type(arms) is not int or arms < 2:
            raise ValueError("arms must be an integer of at least two")
        super().__init__(attempts=attempts, horizon=1, state_dim=1,
                         action_space=gym.spaces.Discrete(arms))

    def _sample_hidden(self):
        self._target = int(self.np_random.integers(self.action_space.n))

    def _start_attempt(self):
        return np.zeros(1, dtype=np.float32)

    def _transition(self, action):
        return self._state.copy(), int(action) == self._target


class BernoulliBandit(HiddenRewardBandit):
    """Trial-fixed uniform arm probabilities, with independent Bernoulli outcomes."""

    def __init__(self, *, attempts=32, arms=6):
        super().__init__(attempts=attempts, arms=arms)

    def _sample_hidden(self):
        self._means = self.np_random.uniform(0, 1, self.action_space.n)

    def _transition(self, action):
        return self._state.copy(), self.np_random.random() < self._means[int(action)]


class NonstationaryBernoulliBandit(BernoulliBandit):
    """Bernoulli bandit whose arm probabilities are all redrawn before a pull with a fixed hazard."""

    def __init__(self, *, attempts=64, arms=6, hazard=.05):
        if not 0 < hazard < 1:
            raise ValueError("hazard must lie strictly between zero and one")
        self.hazard = float(hazard)
        super().__init__(attempts=attempts, arms=arms)

    def _sample_hidden(self):
        super()._sample_hidden()
        self._mean_history = np.empty((self.attempts, self.action_space.n))

    def _start_attempt(self):
        if self._attempt and self.np_random.random() < self.hazard:
            self._means = self.np_random.uniform(0, 1, self.action_space.n)
        self._mean_history[self._attempt] = self._means
        return super()._start_attempt()


class HiddenGainReach(_TrialEnv):
    """Planar reaching with a trial-fixed signed actuator gain and fresh starts/goals."""

    def __init__(self, *, attempts=4, horizon=8, gains=(-1., 1.), step_size=.2, tolerance=.1):
        gains = np.asarray(gains, dtype=np.float32)
        if gains.ndim != 1 or not gains.size or not np.isfinite(gains).all() or (gains == 0).any():
            raise ValueError("gains must be a nonempty vector of finite nonzero values")
        for name, value in (("step_size", step_size), ("tolerance", tolerance)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.gains, self.step_size, self.tolerance = gains.copy(), float(step_size), float(tolerance)
        super().__init__(attempts=attempts, horizon=horizon, state_dim=4,
                         action_space=gym.spaces.Box(-1, 1, (2,), dtype=np.float32))

    def _sample_hidden(self):
        self._gain = float(self.np_random.choice(self.gains))

    def _start_attempt(self):
        position = self.np_random.uniform(-.5, .5, 2)
        angle = self.np_random.uniform(0, 2 * np.pi)
        goal = position + .75 * np.array([np.cos(angle), np.sin(angle)])
        return np.concatenate((position, goal)).astype(np.float32)

    def _transition(self, action):
        position = np.clip(self._state[:2] + self.step_size * self._gain * action, -2, 2)
        goal = self._state[2:]
        state = np.concatenate((position, goal)).astype(np.float32)
        return state, np.linalg.norm(position - goal) <= self.tolerance


class _GridTrial(_TrialEnv):
    """Square grid with a centre start; the state is the agent cell scaled to [-1, 1]."""

    dense_reward = True
    MOVES = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])  # stay, up, down, left, right

    def __init__(self, *, attempts, horizon, size):
        if type(size) is not int or size < 3:
            raise ValueError("size must be an integer of at least three")
        self.size = size
        super().__init__(attempts=attempts, horizon=horizon, state_dim=2,
                         action_space=gym.spaces.Discrete(len(self.MOVES)))

    def _random_cells(self, count):
        return [np.array(divmod(int(cell), self.size))
                for cell in self.np_random.choice(self.size ** 2, count, replace=False)]

    def _encode(self):
        return (2 * self._position / (self.size - 1) - 1).astype(np.float32)

    def _at(self, cell):
        return bool((self._position == cell).all())

    def _start_attempt(self):
        self._position = np.full(2, self.size // 2)
        self._reset_progress()
        return self._encode()

    def _transition(self, action):
        self._position = np.clip(self._position + self.MOVES[int(action)], 0, self.size - 1)
        return self._encode(), self._reward()


class DarkRoom(_GridTrial):
    """Find a trial-fixed hidden goal cell; every step spent on it pays one."""

    def __init__(self, *, attempts=4, horizon=20, size=9):
        super().__init__(attempts=attempts, horizon=horizon, size=size)

    def _sample_hidden(self):
        self._goal, = self._random_cells(1)

    def _reset_progress(self):
        self._reached = False

    def _reward(self):
        self._reached |= self._at(self._goal)
        return self._at(self._goal)

    def _attempt_success(self, success):
        return self._reached


class KeyToDoor(_GridTrial):
    """Collect a hidden key, then reach a hidden door; each pays one, once per attempt."""

    def __init__(self, *, attempts=4, horizon=50, size=9):
        super().__init__(attempts=attempts, horizon=horizon, size=size)

    def _sample_hidden(self):
        self._key, self._door = self._random_cells(2)

    def _reset_progress(self):
        self._has_key = self._opened = False

    def _reward(self):
        if not self._has_key and self._at(self._key):
            self._has_key = True
            return True
        if self._has_key and not self._opened and self._at(self._door):
            self._opened = True
            return True
        return False

    def _attempt_success(self, success):
        return self._opened
