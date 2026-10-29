"""Causal noise-controller distillation over real frozen-VLA experience.

Each stored row is (observation before action, executed motor noise, reward
AFTER action, done). Targets are separate labels, never history inputs. The
GTrXL core is shared with recurrent PPO; weights stay frozen at deployment.
"""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .gtrxl import GatedTransformerXLCore
from .warmstart import atomic_save, checked_load, validate_weights


DATA_FORMAT = "frozen_vla_history_dataset_v1"
MODEL_FORMAT = "frozen_vla_history_policy_v1"
MODES = ("full", "episode", "static")
OBJECTIVES = ("ad", "dpt", "behavior_cloning")
FIELDS = ("observations", "executed_noise", "target_noise", "rewards", "dones", "episode_starts", "teacher_steps")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_trial(trial, *, objective):
    """Check chronology and labels before writing or training any trial."""
    if objective not in OBJECTIVES or not isinstance(trial, dict) or set(trial) != set(FIELDS):
        raise ValueError("Require the declared objective and exactly the public trial fields")
    obs, executed, target = (np.asarray(trial[key]) for key in FIELDS[:3])
    if (obs.ndim != 2 or min(obs.shape) < 1 or executed.ndim != 2
            or executed.shape[0] != obs.shape[0] or executed.shape[1] < 1
            or target.shape != executed.shape):
        raise ValueError("Require observations [T,F] and executed/target noise [T,A]")
    length = len(obs)
    for key in FIELDS[:4]:
        value = np.asarray(trial[key])
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Require finite float32 observations, noise, labels and actual rewards")
    if np.asarray(trial["rewards"]).shape != (length,):
        raise ValueError("Require one actual reward after every executed noise")
    done, starts = (np.asarray(trial[key]) for key in ("dones", "episode_starts"))
    if (done.shape != (length,) or starts.shape != (length,) or done.dtype != np.bool_
            or starts.dtype != np.bool_ or not starts[0] or not done[-1]
            or not np.array_equal(starts[1:], done[:-1])):
        raise ValueError("Require complete episodes, with each next start exactly after done")
    steps = np.asarray(trial["teacher_steps"])
    if steps.dtype != np.int64 or steps.shape != (length,) or (steps < 0).any() or (np.diff(steps) < 0).any():
        raise ValueError("Require chronological nonnegative int64 teacher environment-step identities")
    if objective in ("ad", "behavior_cloning") and not np.array_equal(executed, target):
        raise ValueError("AD/behavior-cloning targets must equal the actually executed noise")
    if objective == "ad" and steps[-1] == steps[0]:
        raise ValueError("AD requires a learning trajectory; a fixed teacher is behavior cloning")
    return length, obs.shape[1], executed.shape[1]


def save_dataset(path, trials, *, objective, metadata):
    """Write concatenated ragged trials plus a hash-bound JSON manifest.

    metadata carries collection provenance and a split declaration; task IDs,
    privileged state and teacher identities are never model input fields.
    """
    path = Path(path)
    if path.exists() or path.with_suffix(".json").exists() or not trials or not metadata:
        raise ValueError("Require a fresh dataset path, nonempty complete trials and provenance")
    shapes = [validate_trial(trial, objective=objective) for trial in trials]
    if len({shape[1:] for shape in shapes}) != 1:
        raise ValueError("All trials must share observation/noise dimensions")
    portable_metadata = json.loads(json.dumps(metadata, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: np.concatenate([trial[key] for trial in trials], axis=0) for key in FIELDS}
    arrays["lengths"] = np.asarray([shape[0] for shape in shapes], dtype=np.int64)
    # Opening explicitly prevents numpy from silently appending another suffix.
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    manifest = dict(format=DATA_FORMAT, objective=objective, trials=len(trials),
        transitions=sum(shape[0] for shape in shapes), observation_dim=shapes[0][1],
        noise_dim=shapes[0][2], metadata=portable_metadata, sha256=file_sha256(path))
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def load_dataset(path, *, expected_sha256=None):
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    digest = file_sha256(path)
    if (manifest.get("format") != DATA_FORMAT or digest != manifest.get("sha256")
            or (expected_sha256 is not None and digest != expected_sha256)):
        raise ValueError("Dataset format or checksum differs")
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != {*FIELDS, "lengths"}:
            raise ValueError("Dataset contains unknown/missing fields")
        arrays = {key: saved[key].copy() for key in saved.files}
    lengths = arrays.pop("lengths")
    if (lengths.dtype != np.int64 or lengths.ndim != 1 or len(lengths) < 1
            or (lengths < 1).any() or int(lengths.sum()) != len(arrays["observations"])):
        raise ValueError("Invalid concatenated trial offsets")
    if any(len(value) != int(lengths.sum()) for value in arrays.values()):
        raise ValueError("Trial fields differ in length")
    offsets = np.concatenate(([0], lengths.cumsum()))
    trials = [{key: value[start:end] for key, value in arrays.items()}
              for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    shapes = [validate_trial(trial, objective=manifest["objective"]) for trial in trials]
    if (len(trials) != manifest["trials"] or int(lengths.sum()) != manifest["transitions"]
            or any(shape[1:] != (manifest["observation_dim"], manifest["noise_dim"]) for shape in shapes)):
        raise ValueError("Dataset manifest dimensions differ")
    if file_sha256(path) != digest:
        raise ValueError("Dataset changed while reading")
    return trials, manifest


def trial_tokens(trial, *, mode="full"):
    """Shift real transitions by one. Current label/reward cannot reach query t."""
    if mode not in MODES:
        raise ValueError("Unknown history reset mode")
    obs = trial["observations"]
    previous_noise = np.zeros_like(trial["executed_noise"])
    previous_reward = np.zeros((len(obs), 1), dtype=np.float32)
    previous_done = np.zeros((len(obs), 1), dtype=np.float32)
    if mode != "static":
        previous_noise[1:] = trial["executed_noise"][:-1]
        previous_reward[1:, 0] = trial["rewards"][:-1]
        previous_done[1:, 0] = trial["dones"][:-1]
        if mode == "episode":
            for value in (previous_noise, previous_reward, previous_done):
                value[trial["episode_starts"]] = 0
    return np.concatenate((obs, previous_noise, previous_reward, previous_done,
                           trial["episode_starts"][:, None].astype(np.float32)), axis=-1)


@dataclass(frozen=True)
class HistoryConfig:
    observation_dim: int
    noise_dim: int
    width: int = 128
    layers: int = 2
    heads: int = 4
    context_length: int = 300
    # Unbounded motor-noise coordinates; adapters own any fixed scaling/map.
    initial_std: float = .5

    def __post_init__(self):
        if (any(type(value) is not int or value < 1 for value in
                (self.observation_dim, self.noise_dim, self.width, self.layers, self.heads, self.context_length))
                or self.width % self.heads or self.width % 2
                or not math.isfinite(self.initial_std) or self.initial_std <= 0):
            raise ValueError("Invalid history-controller dimensions or initial Gaussian standard deviation")


class HistoryNoisePolicy(nn.Module):
    """Functional causal Transformer with an unbounded diagonal Gaussian head."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.core = GatedTransformerXLCore(config.observation_dim + config.noise_dim + 3,
            width=config.width, layers=config.layers, heads=config.heads, memory_length=config.context_length)
        self.actor = nn.Linear(config.width, config.noise_dim)
        self.log_std = nn.Parameter(torch.full((config.noise_dim,), math.log(config.initial_std)))
        self.register_buffer("observation_mean", torch.zeros(config.observation_dim))
        self.register_buffer("observation_std", torch.ones(config.observation_dim))

    @torch.no_grad()
    def fit_observation_normalization(self, trials):
        """Fit on training observations only; fixed thereafter and exported."""
        values = torch.as_tensor(np.concatenate([trial["observations"] for trial in trials]),
                                 dtype=self.observation_mean.dtype, device=self.observation_mean.device)
        self.observation_mean.copy_(values.mean(0))
        self.observation_std.copy_(values.std(0, unbiased=False).clamp(min=1e-4))

    def initial_state(self, batch):
        return self.core.initial_state(batch)

    def step(self, token, state=None, *, reset=None, deterministic=False, generator=None):
        """Consume one public query token; return noise, log probability, state."""
        if token.ndim != 2 or token.shape[1] != self.core.input_dim:
            raise ValueError("Require public query token [B,F+A+3]")
        if reset is not None:
            if reset.shape != token.shape[:1] or reset.dtype != torch.bool or reset.device != token.device:
                raise ValueError("Require boolean per-lane reset on the token device")
            if state is not None:
                initial = self.initial_state(len(token))
                state = {key: torch.where(reset.reshape(-1, *([1] * (value.ndim - 1))),
                                          initial[key], value) for key, value in state.items()}
        d = self.config.observation_dim
        encoded = torch.cat(((token[:, :d] - self.observation_mean) / self.observation_std,
                             token[:, d:]), dim=-1)
        features, state = self.core(encoded, state)
        mean = self.actor(features)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        noise = mean if deterministic else mean + std * torch.randn(mean.shape, device=mean.device,
                                                                    dtype=mean.dtype, generator=generator)
        return noise, Normal(mean, std).log_prob(noise).sum(-1), state

    def sequence_nll(self, tokens, targets, valid, episode_starts, *, mode="full"):
        """Return mean NLL across valid chronological positions, never pad labels."""
        if mode not in MODES or tokens.ndim != 3 or targets.shape != (*tokens.shape[:2], self.config.noise_dim):
            raise ValueError("Invalid sequence shape/mode")
        if (valid.shape != tokens.shape[:2] or valid.dtype != torch.bool or not valid.any()
                or episode_starts.shape != valid.shape or episode_starts.dtype != torch.bool):
            raise ValueError("Require boolean valid/start masks with at least one query")
        state, losses = None, []
        # Sequence layout [time,batch,...], matching the existing recurrent API.
        for index in range(len(tokens)):
            reset = (torch.ones_like(valid[index]) if mode == "static" else
                     episode_starts[index] if mode == "episode" else None)
            token = torch.where(valid[index, :, None], tokens[index], 0.)
            mean, _, state = self.step(token, state, reset=reset, deterministic=True)
            std = self.log_std.clamp(-5, 2).exp()
            # Padding is sanitized before evaluation, so masked NaNs cannot pollute gradients.
            target = torch.where(valid[index, :, None], targets[index], mean.detach())
            losses.append(-Normal(mean, std).log_prob(target).sum(-1))
        return torch.stack(losses)[valid].mean()


def sample_batch(trials, policy, rng, *, batch_size, max_context=None, mode="full"):
    """Sample complete trials or random-length causal prefixes ending at a query.

    Random windows train different context lengths. Each window starts from a
    fresh state, retains only available previous feedback, and never crosses
    trial identities. All supervised positions in a window carry real labels.
    """
    if mode not in MODES or batch_size < 1 or (max_context is not None and max_context < 1):
        raise ValueError("Invalid batch/context/reset mode")
    windows = []
    for index in rng.integers(len(trials), size=batch_size):
        trial = trials[int(index)]
        end, start = len(trial["observations"]), 0
        if max_context is not None:
            end = int(rng.integers(1, end + 1))
            length = int(rng.integers(1, min(max_context, end) + 1))
            start = end - length
        windows.append((trial_tokens(trial, mode=mode)[start:end], trial["target_noise"][start:end],
                        trial["episode_starts"][start:end]))
    longest = max(len(window[0]) for window in windows)
    device = next(policy.parameters()).device
    tokens = torch.zeros(longest, batch_size, policy.core.input_dim, device=device)
    targets = torch.zeros(longest, batch_size, policy.config.noise_dim, device=device)
    valid = torch.zeros(longest, batch_size, device=device, dtype=torch.bool)
    starts = torch.zeros_like(valid)
    for lane, (observations, labels, boundaries) in enumerate(windows):
        length = len(observations)
        tokens[:length, lane] = torch.as_tensor(observations, device=device)
        targets[:length, lane] = torch.as_tensor(labels, device=device)
        valid[:length, lane] = True
        starts[:length, lane] = torch.as_tensor(boundaries, device=device)
    return tokens, targets, valid, starts


class FrozenHistoryController:
    """Single-trial deployment helper. Record only outcomes of executed noise.

    Call ``act`` with the current public observation, then ``observe`` with
    the actual executed noise/reward/done. Call ``reset_trial`` at each new
    task/trial; episode resets inside a trial preserve context in full mode.
    """

    def __init__(self, policy, *, mode="full", generator=None):
        if mode not in MODES:
            raise ValueError("Unknown history mode")
        self.policy, self.mode, self.generator = policy.eval(), mode, generator
        self.policy.requires_grad_(False)
        self.reset_trial()

    def reset_trial(self):
        self.state = None
        self.previous_noise = np.zeros(self.policy.config.noise_dim, np.float32)
        self.previous_reward, self.previous_done = 0., False
        self.episode_start, self.pending = True, False

    @torch.no_grad()
    def act(self, observation, *, deterministic=False):
        if self.pending:
            raise RuntimeError("Observe the previous executed transition before querying again")
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.policy.config.observation_dim,) or not np.isfinite(observation).all():
            raise ValueError("Require one finite public observation vector")
        clear = self.mode == "static" or (self.mode == "episode" and self.episode_start)
        if clear:
            self.state = None
        token = np.concatenate((observation, np.zeros_like(self.previous_noise) if clear else self.previous_noise,
            np.asarray([0. if clear else self.previous_reward, 0. if clear else self.previous_done,
                        self.episode_start], np.float32)))
        weight = next(self.policy.parameters())
        noise, _, self.state = self.policy.step(torch.as_tensor(token[None], device=weight.device,
            dtype=weight.dtype), self.state, deterministic=deterministic, generator=self.generator)
        self.pending = True
        return noise[0].cpu().numpy()

    def observe(self, executed_noise, reward, done):
        if not self.pending:
            raise RuntimeError("Query an action before recording its actual outcome")
        noise = np.asarray(executed_noise, dtype=np.float32)
        if (noise.shape != self.previous_noise.shape or not np.isfinite(noise).all()
                or not math.isfinite(float(reward)) or type(done) not in (bool, np.bool_)):
            raise ValueError("Require actual finite executed noise/reward and boolean done")
        self.previous_noise = noise.copy()
        self.previous_reward, self.previous_done = float(reward), bool(done)
        self.episode_start, self.pending = bool(done), False


def export_policy(path, policy, *, manifest):
    if not manifest or any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Require finite candidate weights and provenance")
    payload = dict(format=MODEL_FORMAT, config=asdict(policy.config),
        manifest=json.loads(json.dumps(manifest, allow_nan=False)),
        policy={key: value.detach().cpu().clone() for key, value in policy.state_dict().items()})
    return atomic_save(path, payload)


def load_policy(path, *, expected_sha256, device="cpu"):
    saved = checked_load(path, expected_sha256)
    if saved.get("format") != MODEL_FORMAT or not saved.get("manifest"):
        raise ValueError("Require a history-controller checkpoint and its provenance")
    policy = HistoryNoisePolicy(HistoryConfig(**saved["config"]))
    validate_weights(policy, saved["policy"])
    policy.load_state_dict(saved["policy"], strict=True)
    return policy.to(device).eval().requires_grad_(False), saved["manifest"]
