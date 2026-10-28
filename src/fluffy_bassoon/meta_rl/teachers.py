"""Feedback-driven compact teachers. No teacher receives an environment instance."""

import numpy as np


TEACHER_VERSION = "public_feedback_teachers_v1"
DPT_LABEL_VERSION = "training_task_action_labels_v1"


def expert_action(env, task, observation, *, rng, rotation_rates=None):
    """TRAINING LABEL ONLY: access privileged task state; never call from a student policy.

    Labels answer what to do if the task were known. Histories keep the actually
    executed behavior action and reward. No privileged value is an input feature.
    """
    if task == "bandit":
        return int(env._target)
    if task in ("bernoulli", "nonstationary"):
        scores = env._means
        return int(rng.choice(np.flatnonzero(scores == scores.max())))
    if task == "cached_rotation":
        from .maniskill_rotation import ANGLES
        scores = rotation_rates[ANGLES.index(env._theta)]
        return int(rng.choice(np.flatnonzero(scores == scores.max())))
    if task == "reach":
        state = observation["state"]
        return np.clip((state[2:] - state[:2]) / (env.step_size * env._gain), -.999, .999).astype(np.float32)
    if task in ("darkroom", "keydoor"):
        if task == "keydoor" and env._opened:
            return 0
        target = env._goal if task == "darkroom" else (env._door if env._has_key else env._key)
        delta = target - env._position
        choices = []
        if delta[0]:
            choices.append(4 if delta[0] > 0 else 3)
        if delta[1]:
            choices.append(1 if delta[1] > 0 else 2)
        return int(rng.choice(choices)) if choices else 0
    raise ValueError("Unsupported task-specific action label")


class FeedbackTeacher:
    def __init__(self, task, *, seed, arms=2, size=9, step_size=.2, rotation_rates=None):
        self.task, self.arms, self.size, self.step_size = task, arms, size, step_size
        self.rng = np.random.default_rng(seed)
        self.alpha, self.beta = np.ones(arms), np.ones(arms)
        self.candidates = set(range(arms))
        self.best = None
        self.last_position = self.last_action = self.gain = None
        self.targets = [None, None]
        self.visited = [set(), set()]
        self.stage = 0
        self.rotation_rates = rotation_rates
        self.posterior = np.ones(3) / 3

    def _cell(self, state):
        return tuple(np.rint((state[:2] + 1) * (self.size - 1) / 2).astype(int))

    def action(self, observation):
        obs = observation
        has_previous = self.last_action is not None
        reward = float(obs["previous_reward"][0])
        if self.task == "bandit":
            if has_previous:
                if reward:
                    self.best = self.last_action
                else:
                    self.candidates.discard(self.last_action)
            action = self.best if self.best is not None else int(self.rng.choice(sorted(self.candidates)))
        elif self.task in ("bernoulli", "nonstationary"):
            if has_previous:
                # Discounted Thompson is a declared heuristic, not an exact changepoint posterior.
                if self.task == "nonstationary":
                    self.alpha = 1 + .95 * (self.alpha - 1)
                    self.beta = 1 + .95 * (self.beta - 1)
                self.alpha[self.last_action] += reward
                self.beta[self.last_action] += 1 - reward
            action = int(self.rng.beta(self.alpha, self.beta).argmax())
        elif self.task == "cached_rotation":
            if has_previous:
                likelihood = self.rotation_rates[:, self.last_action]
                self.posterior *= likelihood if reward else 1 - likelihood
                self.posterior /= self.posterior.sum()
            theta = self.rng.choice(3, p=self.posterior)
            scores = self.rotation_rates[theta]
            action = int(self.rng.choice(np.flatnonzero(scores == scores.max())))
        elif self.task == "reach":
            if has_previous and np.linalg.norm(self.last_action) > 1e-6:
                displacement = obs["transition_state"][:2] - self.last_position
                self.gain = float(np.sign(displacement @ self.last_action))
            position, goal = obs["state"][:2], obs["state"][2:]
            # The first bounded probe identifies signed gain from the public transition.
            action = np.array([.5, .5]) if self.gain is None else (
                (goal - position) / (self.step_size * self.gain))
            action = np.clip(action, -.999, .999).astype(np.float32)
            self.last_position = position.copy()
        elif self.task in ("darkroom", "keydoor"):
            if has_previous:
                cell = self._cell(obs["transition_state"])
                self.visited[min(self.stage, 1)].add(cell)
                if reward and self.stage < (2 if self.task == "keydoor" else 1):
                    self.targets[self.stage] = cell
                    if self.task == "keydoor":
                        self.stage += 1
            if obs["attempt_end"][0]:
                self.stage = 0
            position = self._cell(obs["state"])
            if self.stage == 2:
                action = 0
            else:
                target = self.targets[self.stage]
                if target is None:
                    candidates = [(x, y) for x in range(self.size) for y in range(self.size)
                                  if (x, y) not in self.visited[self.stage]]
                    distances = np.array([abs(x-position[0])+abs(y-position[1]) for x,y in candidates])
                    if not len(candidates):
                        raise RuntimeError("Teacher explored every cell without finding the hidden target")
                    target = candidates[int(self.rng.choice(np.flatnonzero(distances == distances.min())))]
                dx, dy = target[0] - position[0], target[1] - position[1]
                action = (4 if dx > 0 else 3) if dx else ((1 if dy > 0 else 2) if dy else 0)
        else:
            raise ValueError("Unsupported feedback teacher")
        self.last_action = action
        return action


def rotation_teacher_rates(cache):
    """Training-scene population likelihood only; no validation rows or private trial plan."""
    from .maniskill_rotation import ANGLES
    rates = []
    for theta in ANGLES:
        row = []
        for phi in ANGLES:
            outcomes = [value for scene in cache.split_scenes["train"]
                        for value in cache.outcomes[(scene, theta + phi)]]
            # Beta(1,1) smoothing keeps every posterior update well defined.
            row.append((sum(outcomes) + 1) / (len(outcomes) + 2))
        rates.append(row)
    return np.asarray(rates)
