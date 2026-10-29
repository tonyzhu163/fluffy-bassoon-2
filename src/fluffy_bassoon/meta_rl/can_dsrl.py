"""Released DSRL Can motor and environment, with explicit retry-trial boundaries.

The external authors' repositories remain unmodified. This adapter fixes reset
seed plumbing and records actual simulator success independently of the shifted
training reward. No physics, observations, goals or simulator rewards are changed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import gymnasium as gym
import numpy as np
import torch


DSRL_COMMIT = "8d21b9cf55459f022c0e769bca1343edd9d30e5a"
DPPO_COMMIT = "86ce51834055c02f9013e60dd4c4275606d82df7"
SB3_COMMIT = "10e5d311b5c42fe571ff3eeb72f622ca616a149d"
CHUNK, ACTION_DIM, OBS_DIM, HORIZON = 4, 7, 23, 300
NOISE_DIM = CHUNK * ACTION_DIM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def configure_external(root):
    root = Path(root).resolve()
    pins = {".": DSRL_COMMIT, "dppo": DPPO_COMMIT, "stable-baselines3": SB3_COMMIT}
    for name, expected in pins.items():
        path = root / name
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).strip()
        if actual != expected or dirty:
            raise ValueError(f"Require clean pinned external source: {path}")
    # The release uses both `model.*` and `dppo.*` imports.
    sys.path[:0] = [str(root), str(root / "dppo"), str(root / "stable-baselines3")]
    return root


def load_motor(root, checkpoint, normalization, device="cuda"):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    root = configure_external(root)
    config = OmegaConf.load(root / "cfg/robomimic/dsrl_can.yaml")
    config.base_policy_path = str(Path(checkpoint).resolve())
    config.normalization_path = str(Path(normalization).resolve())
    config.device = device
    config.model.device = device
    config.model.network.cond_dim = OBS_DIM
    # Resolve only the model: the release's log-directory interpolations use
    # Hydra resolvers that are irrelevant to motor construction.
    model = instantiate(config.model).eval().requires_grad_(False)
    from utils import DPPOBasePolicyWrapper
    return CountedMotor(DPPOBasePolicyWrapper(model)), config


class CountedMotor:
    """Count decoder work, including DSRL target and distillation queries."""
    def __init__(self, wrapped):
        self.wrapped, self.base_policy = wrapped, wrapped.base_policy
        self.calls, self.samples = 0, 0

    def __call__(self, obs, noise, return_numpy=True):
        self.calls += 1
        self.samples += len(obs)
        return self.wrapped(obs, noise, return_numpy=return_numpy)


class CanChunkEnv(gym.Env):
    """Original 300-step Can episode, chunked into four primitive actions.

    A reset seed denotes an initial scene, never a privileged policy input.
    Trial callers can retry that scene or draw iid scenes. SB3 callers use the
    normal seeded reset stream. `reward` matches release reward_offset=1.
    """
    metadata = {"render_modes": []}

    def __init__(self, root=None, normalization=None, *, seed=0, low_env=None, retries=1):
        super().__init__()
        if type(retries) is not int or retries < 1:
            raise ValueError("Require a positive retry count")
        if low_env is None:
            configure_external(root)
            from env_utils import make_robomimic_env
            low_env = make_robomimic_env(
                env="can", normalization_path=str(normalization), dppo_path=str(Path(root) / "dppo"),
                low_dim_keys=["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "object"],
            )
        self.low = low_env
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (NOISE_DIM,), np.float32)
        self.stream = np.random.default_rng(seed)
        self.retries, self.retries_left, self.retry_seed = retries, 0, None
        self.count, self.success, self.scene_seed = 0, False, None
        self.previous_observation = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            if self.retries_left == 0:
                self.retry_seed = int(self.stream.integers(0, 1_000_000))
                self.retries_left = self.retries
            seed = self.retry_seed
            self.retries_left -= 1
        self.scene_seed = int(seed)
        # The upstream lowdim wrapper deliberately ignores reset(seed=...).
        obs = self.low.reset(options={"seed": self.scene_seed})["state"]
        self.count, self.success = 0, False
        self.previous_observation = np.asarray(obs, dtype=np.float32).reshape(OBS_DIM)
        return self.previous_observation.copy(), {"scene_seed": self.scene_seed}

    def step(self, action):
        if self.count >= HORIZON:
            raise RuntimeError("Reset the completed episode before stepping")
        action = np.asarray(action, dtype=np.float32).reshape(CHUNK, ACTION_DIM)
        if not np.isfinite(action).all():
            raise ValueError("Nonfinite decoded action")
        reward, raw_reward = 0., 0.
        for command in action:
            obs, r, _, _ = self.low.step(command)
            raw_reward += float(r)
            reward += float(r) - 1.
            self.success |= bool(r > 0)
            self.count += 1
        self.previous_observation = np.asarray(obs["state"], dtype=np.float32).reshape(OBS_DIM)
        info = dict(success_once=self.success, raw_reward=raw_reward, primitive_steps=CHUNK,
                    scene_seed=self.scene_seed)
        return self.previous_observation.copy(), reward, self.count == HORIZON, False, info

    def snapshot(self):
        if self.count != 0:
            raise RuntimeError("Simulator recovery requires an initial-episode boundary")
        return dict(boundary="initial episode", numpy_rng=np.random.get_state(),
                    python_rng=random.getstate(), stream=self.stream.bit_generator.state,
                    count=self.count, success=self.success, scene_seed=self.scene_seed,
                    retries_left=self.retries_left, retry_seed=self.retry_seed,
                    previous_observation=self.previous_observation)

    def restore(self, state):
        if state["boundary"] != "initial episode" or state["count"] != 0:
            raise ValueError("Cannot restore a partial simulator episode")
        obs, _ = self.reset(seed=state["scene_seed"])
        if not np.array_equal(obs, state["previous_observation"]):
            raise RuntimeError("Initial simulator observation differs on recovery")
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.stream.bit_generator.state = state["stream"]
        self.count, self.success = state["count"], state["success"]
        self.scene_seed = state["scene_seed"]
        self.retries_left, self.retry_seed = state["retries_left"], state["retry_seed"]
        self.previous_observation = state["previous_observation"].copy()

    def close(self):
        raw = getattr(self.low, "env", None)
        if raw is not None and hasattr(raw, "env") and hasattr(raw.env, "close"):
            raw.env.close()


def trial_inputs(obs, previous_noise, previous_reward, attempt_end, attempt, chunk, attempts):
    """Policy-visible history only; no scene seed or simulator state."""
    batch = len(obs)
    clock = np.tile([attempt / max(attempts - 1, 1), chunk / (HORIZON // CHUNK)], (batch, 1))
    return np.concatenate([obs, previous_noise, previous_reward[:, None],
                           np.full((batch, 1), attempt_end), clock], axis=-1).astype(np.float32)


@torch.no_grad()
def collect_trial(policy, motor, envs, seeds, *, attempts=4, reset_memory=False,
                  reward_blind=False, deterministic=False, scene_mode="retry"):
    if attempts < 1 or len(envs) != len(seeds) or scene_mode not in ("retry", "iid"):
        raise ValueError("Invalid trial shape or scene mode")
    device = next(policy.parameters()).device
    n = len(envs)
    state, previous_noise, previous_reward = None, np.zeros((n, NOISE_DIM)), np.zeros(n)
    records = {key: [] for key in ("observations", "raw_actions", "log_probs", "values", "rewards",
                                   "trial_ends", "attempt_ends", "resets")}
    successes, initial_hashes = [], []
    policy.eval()
    for attempt in range(attempts):
        if reset_memory:
            # Clear boundary feedback too: otherwise a reset controller can
            # retain the preceding episode's final action and reward.
            previous_noise, previous_reward = np.zeros((n, NOISE_DIM)), np.zeros(n)
        attempt_seeds = [int(s) if scene_mode == "retry" else int(s) + attempt * 1_000_000 for s in seeds]
        obs = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, attempt_seeds, strict=True)])
        initial_hashes.append([hashlib.sha256(o.tobytes()).hexdigest() for o in obs])
        for chunk in range(HORIZON // CHUNK):
            inputs = trial_inputs(obs, previous_noise, np.zeros(n) if reward_blind else previous_reward,
                                  float(attempt > 0 and chunk == 0), attempt, chunk, attempts)
            inputs = torch.as_tensor(inputs, device=device)
            resets = torch.full((n,), reset_memory and chunk == 0, dtype=torch.bool, device=device)
            action, raw, logprob, _, values, state = policy.step(inputs, state, reset=resets,
                                                               deterministic=deterministic)
            decoded = motor(torch.as_tensor(obs, device=device), action.reshape(n, CHUNK, ACTION_DIM))
            result = [e.step(a) for e, a in zip(envs, decoded, strict=True)]
            obs = np.stack([r[0] for r in result])
            previous_noise, previous_reward = action.cpu().numpy(), np.array([r[1] for r in result])
            final_chunk = chunk == HORIZON // CHUNK - 1
            values_to_store = dict(observations=inputs, raw_actions=raw, log_probs=logprob, values=values,
                                  rewards=torch.as_tensor(previous_reward, dtype=torch.float32, device=device),
                                  trial_ends=torch.full((n,), final_chunk and attempt == attempts - 1,
                                                        dtype=torch.bool, device=device),
                                  attempt_ends=torch.full((n,), final_chunk, dtype=torch.bool, device=device),
                                  resets=resets)
            for key, value in values_to_store.items():
                records[key].append(value.detach())
        successes.append([r[4]["success_once"] for r in result])
    rollout = {key: torch.stack(value) for key, value in records.items()}
    return rollout, dict(successes=np.asarray(successes).T.tolist(), initial_observation_sha256=initial_hashes,
                         scene_seeds=list(map(int, seeds)), attempts=attempts, scene_mode=scene_mode,
                         primitive_steps=n * attempts * HORIZON)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
