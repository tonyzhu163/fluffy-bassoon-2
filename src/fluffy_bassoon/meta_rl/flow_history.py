"""Direct physical-action histories from a trained Flow-SDE action expert.

The action expert is bypassed by students. Its Gaussian draws are neither
student labels nor steering codes. Visual features come from the original
frozen VLM, alongside all 25 public state coordinates.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual, _add, velocity_with_kv_residual
from .causal_decoder import CausalDecoderCore
from .history_value import discounted_returns, importance_weights
from .vla_history import MODES, save_dataset, trial_tokens, validate_trial
from .warmstart import atomic_save, checked_load, validate_weights


BINARY = (6, 11)
FEATURE_DIM, ACTION_DIM = 2073, 600
FLOW_STEERING_INTERFACES = ("kv_bridge", "velocity_bridge")
FLOW_META_OBJECTIVES = ("ad", "dpt", "dit", "ic_iql")
BRIDGE_MODEL_FORMAT = "flow_steering_bridge_policy_v1"


@dataclass(frozen=True)
class DirectActionCodec:
    mean: tuple
    std: tuple
    steering_dim: int = ACTION_DIM
    kind: str = "flow_physical_action"

    def __post_init__(self):
        mean, std = np.asarray(self.mean), np.asarray(self.std)
        if (mean.shape != (12,) or std.shape != (12,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()
                or self.steering_dim != ACTION_DIM or self.kind != "flow_physical_action"):
            raise ValueError("Require the fixed twelve-coordinate native action normalization")

    @classmethod
    def from_stats(cls, stats):
        mean = np.asarray(stats["actions"]["mean"], np.float32)[:12].copy()
        std = np.asarray(stats["actions"]["std"], np.float32)[:12].copy() + np.float32(1e-6)
        mean[list(BINARY)], std[list(BINARY)] = 0., 1.
        return cls(tuple(map(float, mean)), tuple(map(float, std)))

    def record(self):
        return dict(kind=self.kind, steering_dim=ACTION_DIM, mean=list(self.mean), std=list(self.std),
            labels="actual mapped physical H50x12; binary coordinates -1/+1",
            deployment="continuous inverse normalization; binary sign to native 0/1 input before native +/-1 map",
            native_gaussian="not used by direct-action student")

    def encode(self, physical):
        physical = np.asarray(physical, np.float32)
        if (physical.shape[-2:] != (50, 12) or not np.isfinite(physical).all()
                or not np.isin(physical[..., list(BINARY)], [-1., 1.]).all()):
            raise ValueError("Require actual finite mapped physical commands, including binary +/-1")
        return ((physical - np.asarray(self.mean, np.float32)) / np.asarray(self.std, np.float32)).reshape(
            *physical.shape[:-2], ACTION_DIM).astype(np.float32)

    def decode(self, controls):
        controls = np.asarray(controls, np.float32)
        if controls.shape[-1:] != (ACTION_DIM,) or not np.isfinite(controls).all():
            raise ValueError("Require finite normalized H50x12 direct-action coordinates")
        chunk = controls.reshape(*controls.shape[:-1], 50, 12)
        commands = chunk * np.asarray(self.std, np.float32) + np.asarray(self.mean, np.float32)
        commands[..., list(BINARY)] = (chunk[..., list(BINARY)] > 0).astype(np.float32)
        return commands.astype(np.float32)

    @staticmethod
    def mapped_physical(commands):
        physical = np.asarray(commands, np.float32).copy()
        if physical.shape[-2:] != (50, 12) or not np.isfinite(physical).all():
            raise ValueError("Require finite native H50x12 commands")
        physical[..., list(BINARY)] = np.where(physical[..., list(BINARY)] > .5, 1., -1.)
        return physical

    def feedback_control(self, commands):
        # History records the command that physically executed, including the
        # binary map. It must not feed the raw Gaussian prediction back as if
        # it were the behavior action in the training corpus.
        return self.encode(self.mapped_physical(commands))


def teacher_noise(reference_sha, row):
    identity = dict(version="flow_final18_query_noise_v1", source_chunk_sha256=reference_sha, row=int(row))
    seed = int.from_bytes(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).digest()[:8], "little")
    return np.random.Generator(np.random.PCG64(seed)).standard_normal((50, 32)).astype(np.float32)


class FlowDirectMotor:
    def __init__(self, native, codec):
        self.native, self.codec, self.model = native, codec, native.model

    def prepare_batch(self, observation):
        features, prepared = self.native.prepare_batch(observation)
        state = observation["states"]
        if torch.is_tensor(state):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, np.float32)
        result = np.concatenate((state, features[:, 16:]), -1)
        if result.shape != (len(state), FEATURE_DIM) or not np.isfinite(result).all():
            raise ValueError("Require original frozen prefix2048 plus public state25")
        return result.astype(np.float32), prepared

    def prepare(self, observation):
        features, prepared = self.prepare_batch(observation)
        if len(features) != 1:
            raise ValueError("Serial port requires one observation")
        return features[0], prepared

    def execute_batch(self, prepared, *, noise, steering, spec):
        if (prepared is not self.native._prepared or spec.record() != self.codec.record()
                or np.shape(steering) != (prepared["batch_size"], ACTION_DIM)):
            raise ValueError("Require the unconsumed same-observation direct-action token")
        try:
            return self.codec.decode(steering)
        finally:
            self.native._prepared = None

    def execute(self, prepared, *, noise, steering, spec):
        return self.execute_batch(prepared, noise=noise, steering=np.asarray(steering)[None], spec=spec)[0]


class FlowReferenceMotor(FlowDirectMotor):
    """Original/final frozen Flow policy with the same exogenous native draws."""
    def __init__(self, native, codec, reference):
        super().__init__(native, codec)
        self.reference = reference.eval().requires_grad_(False)

    @torch.no_grad()
    def execute_batch(self, prepared, *, noise, steering, spec):
        if prepared is not self.native._prepared or spec.record() != self.codec.record():
            raise ValueError("Reference must consume the current original frozen VLM prefix")
        draw = torch.as_tensor(noise, dtype=torch.float32, device=self.reference.device)
        if draw.shape != (prepared["batch_size"], 50, 32) or not torch.isfinite(draw).all():
            raise ValueError("Use the prospective common native draw for every reference query")
        original = self.reference.build_prefix_cache
        try:
            self.reference.build_prefix_cache = lambda *args, **kwargs: prepared["prefix"]
            commands, _ = self.reference.predict_action_batch(prepared["observation"], mode="eval", noise=draw)
            commands = commands.detach().float().cpu().numpy()
            if commands.shape != (prepared["batch_size"], 50, 12) or not np.isfinite(commands).all():
                raise ValueError("Require complete finite native reference commands")
            return commands
        finally:
            self.reference.build_prefix_cache = original
            self.native._prepared = None


class ReferencePlaceholder(torch.nn.Module):
    """Protocol state only; the native motor supplies all anchor commands."""
    def __init__(self):
        from .vla_history import HistoryConfig
        super().__init__()
        self.config = HistoryConfig(FEATURE_DIM, ACTION_DIM, context_length=128)
        self.register_buffer("protocol_identity", torch.zeros(1))
        self.anchor_parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def step(self, token, state=None, **kwargs):
        length = torch.zeros(len(token), dtype=torch.long, device=token.device) if state is None else state["length"]
        return torch.zeros((len(token), ACTION_DIM), device=token.device), torch.zeros(len(token), device=token.device), dict(length=length+1)


def export_corpus(output, raw, features, teacher_actions, codec, *, provenance, validation_slots=tuple(range(112, 128))):
    """Identical complete chronological streams for all four methods (PPO or GRPO)."""
    states, actions = np.asarray(raw["public_states"]), np.asarray(raw["physical_actions"])
    features, teacher_actions = np.asarray(features), np.asarray(teacher_actions)
    if (states.ndim != 3 or states.shape[2] != 25 or actions.ndim != 4
            or actions.shape[:2] != states.shape[:2] or actions.shape[2:] != (50, 12)
            or features.shape != (*states.shape[:2], FEATURE_DIM) or teacher_actions.shape != actions.shape
            or not np.array_equal(features[..., :25], states)
            or not np.isfinite(features).all() or not provenance):
        raise ValueError("Bind the fully audited physical source and frozen public features")
    num_streams, total_chunks = states.shape[:2]
    raw_starts = np.asarray(raw["episode_starts"], dtype=bool)
    if raw_starts.shape != (num_streams, total_chunks) or not raw_starts[:, 0].all():
        raise ValueError("Do not change chronology or physical boundaries")
    first_starts = np.flatnonzero(raw_starts[0])
    chunks_per_episode = int(first_starts[1]) if len(first_starts) > 1 else total_chunks
    if chunks_per_episode < 1 or total_chunks % chunks_per_episode:
        raise ValueError("Do not change chronology or physical boundaries")
    num_versions = total_chunks // chunks_per_episode
    if num_versions < 2:
        raise ValueError("Require at least two chronological policy versions")
    starts = np.broadcast_to(np.arange(total_chunks) % chunks_per_episode == 0, (num_streams, total_chunks))
    ends = np.broadcast_to(np.arange(total_chunks) % chunks_per_episode == chunks_per_episode - 1, (num_streams, total_chunks))
    versions = np.broadcast_to(np.repeat(np.arange(num_versions), chunks_per_episode), (num_streams, total_chunks))
    if (not np.array_equal(raw["episode_starts"], starts)
            or not np.array_equal(raw["physical_episode_ends"], ends)
            or not np.array_equal(raw["producer_versions"], versions)):
        raise ValueError("Do not change chronology or physical boundaries")
    actual, labels = codec.encode(actions), codec.encode(teacher_actions)
    rewards = np.asarray(raw["primitive_rewards"].sum(-1), np.float32)
    if (not np.isin(rewards, [0., 1.]).all()
            or (rewards.reshape(num_streams, num_versions, chunks_per_episode).sum(-1) > 1).any()):
        raise ValueError("Keep actual first-success reward once per physical episode")
    validation = set(validation_slots)
    if not validation or not validation < set(range(num_streams)):
        raise ValueError("Require a fixed nonempty held-out collection-slot partition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for partition, slots in (("train", sorted(set(range(num_streams)) - validation)), ("validation", sorted(validation))):
        for objective in ("ad", "dpt", "dit", "ic_iql"):
            trials = [dict(observations=features[s].astype(np.float32), executed_noise=actual[s],
                target_noise=labels[s] if objective == "dpt" else actual[s], rewards=rewards[s],
                dones=ends[s].copy(), episode_starts=starts[s].copy(), teacher_steps=versions[s].copy()) for s in slots]
            metadata = dict(provenance, trial_ids=[f"flow-source-slot-{s:03d}" for s in slots],
                schedule=dict(streams=num_streams, policy_versions=num_versions, chunks_per_episode=chunks_per_episode),
                feature_contract=dict(dimension=FEATURE_DIM, meaning="original frozen VLM prefix2048 RMS plus raw public state25"),
                noise_contract=codec.record(), value_contract=dict(discount=.99, boundary="physical_done",
                    rewards="actual native first-success once", terminal_bootstrap=0,
                    next_observation="following saved query within episode; no invented terminal RGB"),
                validation="held-out source stream imitation/value diagnostic; source PPO saw all source observations")
            path = output / f"{objective}-{partition}.npz"
            saved = save_dataset(path, trials, objective=objective, metadata=metadata)
            manifests[f"{objective}_{partition}"] = dict(path=str(path), sha256=saved["sha256"])
    return manifests


def export_grpo_group_trials(
    output, raw, features, teacher_actions, codec, *, provenance,
    group_size=8, attempts_per_trial=5, validation_groups=None,
):
    """Export within-scene multi-attempt trials from GRPO grouped rollouts."""
    states, actions = np.asarray(raw["public_states"]), np.asarray(raw["physical_actions"])
    features, teacher_actions = np.asarray(features), np.asarray(teacher_actions)
    if (states.ndim != 3 or states.shape[2] != 25 or actions.ndim != 4
            or actions.shape[:2] != states.shape[:2] or actions.shape[2:] != (50, 12)
            or features.shape != (*states.shape[:2], FEATURE_DIM) or teacher_actions.shape != actions.shape
            or not np.array_equal(features[..., :25], states)
            or not np.isfinite(features).all() or not provenance):
        raise ValueError("Bind the fully audited physical source and frozen public features")
    num_streams, total_chunks = states.shape[:2]
    if (type(group_size) is not int or group_size < 2 or num_streams % group_size
            or type(attempts_per_trial) is not int or not 2 <= attempts_per_trial <= group_size):
        raise ValueError("Require integer group_size >= 2 dividing num_streams and 2 <= attempts_per_trial <= group_size")
    raw_starts = np.asarray(raw["episode_starts"], dtype=bool)
    if raw_starts.shape != (num_streams, total_chunks) or not raw_starts[:, 0].all():
        raise ValueError("Do not change chronology or physical boundaries")
    first_starts = np.flatnonzero(raw_starts[0])
    chunks_per_episode = int(first_starts[1]) if len(first_starts) > 1 else total_chunks
    if chunks_per_episode < 1 or total_chunks % chunks_per_episode:
        raise ValueError("Do not change chronology or physical boundaries")
    num_versions = total_chunks // chunks_per_episode
    if num_versions < 1:
        raise ValueError("Require at least one policy version")
    starts = np.broadcast_to(np.arange(total_chunks) % chunks_per_episode == 0, (num_streams, total_chunks))
    ends = np.broadcast_to(np.arange(total_chunks) % chunks_per_episode == chunks_per_episode - 1, (num_streams, total_chunks))
    versions = np.broadcast_to(np.repeat(np.arange(num_versions), chunks_per_episode), (num_streams, total_chunks))
    if (not np.array_equal(raw["episode_starts"], starts)
            or not np.array_equal(raw["physical_episode_ends"], ends)
            or not np.array_equal(raw["producer_versions"], versions)):
        raise ValueError("Do not change chronology or physical boundaries")
    actual, labels = codec.encode(actions), codec.encode(teacher_actions)
    rewards = np.asarray(raw["primitive_rewards"].sum(-1), np.float32)
    if (not np.isin(rewards, [0., 1.]).all()
            or (rewards.reshape(num_streams, num_versions, chunks_per_episode).sum(-1) > 1).any()):
        raise ValueError("Keep actual first-success reward once per physical episode")
    num_groups = num_streams // group_size
    if validation_groups is None:
        validation_groups = tuple(range(max(1, num_groups - max(1, num_groups // 8)), num_groups))
    validation = set(validation_groups)
    if not validation or not validation < set(range(num_groups)):
        raise ValueError("Require a fixed nonempty held-out scene-group partition")
    trial_len = attempts_per_trial * chunks_per_episode
    trial_starts = np.arange(trial_len) % chunks_per_episode == 0
    trial_dones = np.arange(trial_len) % chunks_per_episode == chunks_per_episode - 1
    trial_steps = np.repeat(np.arange(attempts_per_trial, dtype=np.int64), chunks_per_episode)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for partition, groups in (("train", sorted(set(range(num_groups)) - validation)), ("validation", sorted(validation))):
        for objective in ("ad", "dpt", "dit", "ic_iql"):
            trials, trial_ids = [], []
            for v in range(num_versions):
                c0, c1 = v * chunks_per_episode, (v + 1) * chunks_per_episode
                for g in groups:
                    s0, s1 = g * group_size, (g + 1) * group_size
                    group_states = states[s0:s1, c0:c1]
                    for m in range(1, group_size):
                        if not np.array_equal(group_states[0, 0], group_states[m, 0]):
                            raise ValueError("GRPO group members must share identical initial public state at reset")
                    ep_rewards = rewards[s0:s1, c0:c1]
                    ep_returns = ep_rewards.sum(-1)
                    failed = [m for m in range(group_size) if ep_returns[m] == 0.]
                    succeeded = [m for m in range(group_size) if ep_returns[m] > 0.]
                    if failed and succeeded:
                        num_fail = min(len(failed), max(1, attempts_per_trial - len(succeeded)))
                        chosen = failed[:num_fail] + succeeded[:attempts_per_trial - num_fail]
                    else:
                        chosen = (failed + succeeded)[:attempts_per_trial]
                    idx = [s0 + m for m in chosen]
                    obs_seq = features[idx, c0:c1].reshape(trial_len, FEATURE_DIM).astype(np.float32)
                    act_seq = actual[idx, c0:c1].reshape(trial_len, ACTION_DIM)
                    lbl_seq = labels[idx, c0:c1].reshape(trial_len, ACTION_DIM)
                    rew_seq = rewards[idx, c0:c1].reshape(trial_len)
                    trials.append(dict(
                        observations=obs_seq, executed_noise=act_seq,
                        target_noise=lbl_seq if objective == "dpt" else act_seq,
                        rewards=rew_seq, dones=trial_dones.copy(),
                        episode_starts=trial_starts.copy(), teacher_steps=trial_steps.copy(),
                    ))
                    trial_ids.append(f"grpo-v{v:03d}-group{g:02d}")
            metadata = dict(provenance, trial_ids=trial_ids,
                schedule=dict(streams=num_streams, group_size=group_size, scene_groups=num_groups,
                              attempts_per_trial=attempts_per_trial, policy_versions=num_versions,
                              chunks_per_episode=chunks_per_episode),
                feature_contract=dict(dimension=FEATURE_DIM, meaning="original frozen VLM prefix2048 RMS plus raw public state25"),
                noise_contract=codec.record(), value_contract=dict(discount=.99, boundary="physical_done",
                    rewards="actual native first-success once", terminal_bootstrap=0,
                    next_observation="following saved query within episode; no invented terminal RGB"),
                validation="held-out GRPO scene-group partition; within-scene failure-to-success ordering")
            path = output / f"{objective}-grpo-groups-{partition}.npz"
            saved = save_dataset(path, trials, objective=objective, metadata=metadata)
            manifests[f"{objective}_{partition}"] = dict(path=str(path), sha256=saved["sha256"])
    return manifests


@dataclass(frozen=True)
class FlowBridgeCodec:
    mean: tuple
    std: tuple
    interface: str = "kv_bridge"
    steering_dim: int = ACTION_DIM

    def __post_init__(self):
        if self.interface not in FLOW_STEERING_INTERFACES:
            raise ValueError(f"Unsupported flow bridge interface: {self.interface!r}")
        DirectActionCodec(self.mean, self.std, steering_dim=self.steering_dim)

    @classmethod
    def from_stats(cls, stats, *, interface="kv_bridge"):
        direct = DirectActionCodec.from_stats(stats)
        return cls(direct.mean, direct.std, interface=interface)

    def record(self):
        return dict(
            kind=f"flow_{self.interface}",
            interface=self.interface,
            steering_dim=ACTION_DIM,
            mean=list(self.mean),
            std=list(self.std),
            labels="actual mapped physical H50x12; binary coordinates -1/+1",
            deployment=f"{self.interface} over frozen VLA prefix/suffix with actual physical-action history feedback",
        )

    def feedback_control(self, commands):
        return DirectActionCodec(self.mean, self.std).feedback_control(commands)


@dataclass(frozen=True)
class FlowBridgeConfig:
    interface: str = "kv_bridge"
    observation_dim: int = FEATURE_DIM
    history_action_dim: int = ACTION_DIM
    width: int = 128
    layers: int = 2
    heads: int = 4
    context_length: int = 300
    kv_heads: int = 1
    head_dim: int = 256
    kv_rank: int = 4
    kv_layers: tuple[int, ...] = (-1,)
    scale: float = .1
    action_horizon: int = 50
    model_action_dim: int = 32
    active_action_dim: int = 12

    def __post_init__(self):
        if (self.interface not in FLOW_STEERING_INTERFACES
                or any(type(v) is not int or v < 1 for v in (
                    self.observation_dim, self.history_action_dim, self.width, self.layers,
                    self.heads, self.context_length, self.kv_heads, self.head_dim,
                    self.kv_rank, self.action_horizon, self.model_action_dim, self.active_action_dim))
                or self.width > self.history_action_dim
                or self.width % self.heads or self.width % 2
                or self.active_action_dim > self.model_action_dim
                or not math.isfinite(self.scale) or self.scale <= 0):
            raise ValueError("Invalid flow-steering bridge configuration")

    @property
    def noise_dim(self) -> int:
        return self.history_action_dim


class HistoryFlowSteeringPolicy(nn.Module):
    """Causal history decoder driving a KV or velocity bridge on a frozen VLA.

    Supports all four meta-learning objectives (AD, DPT, DIT, IC-IQL) via
    advantage-weighted flow-matching supervision through the frozen action
    expert. Both steering interfaces are zero-initialized at the output layer,
    preserving exact native frozen-motor parity before training.
    """

    def __init__(self, config: FlowBridgeConfig):
        super().__init__()
        self.config = config
        self.core = CausalDecoderCore(
            config.observation_dim + config.history_action_dim + 3,
            width=config.width, layers=config.layers, heads=config.heads,
            memory_length=config.context_length,
        )
        if config.interface == "kv_bridge":
            self.kv_adapter = ReactiveKVResidual(
                public_dim=config.width, kv_heads=config.kv_heads, head_dim=config.head_dim,
                rank=config.kv_rank, layers=config.kv_layers, scale=config.scale,
                detach_features=False,
            )
            self.velocity_head = None
        else:
            self.kv_adapter = None
            self.velocity_head = nn.Linear(config.width, config.model_action_dim)
            nn.init.zeros_(self.velocity_head.weight)
            nn.init.zeros_(self.velocity_head.bias)
        self.register_buffer("observation_mean", torch.zeros(config.observation_dim))
        self.register_buffer("observation_std", torch.ones(config.observation_dim))

    @torch.no_grad()
    def fit_observation_normalization(self, trials):
        values = torch.as_tensor(np.concatenate([trial["observations"] for trial in trials]),
                                 dtype=self.observation_mean.dtype, device=self.observation_mean.device)
        self.observation_mean.copy_(values.mean(0))
        self.observation_std.copy_(values.std(0, unbiased=False).clamp(min=1e-4))

    def initial_state(self, batch):
        return self.core.initial_state(batch)

    def encode_step(self, token, state=None, *, reset=None):
        if token.ndim != 2 or token.shape[1] != self.core.input_dim:
            raise ValueError("Require public query token [B,F+A+3]")
        if reset is not None and state is not None:
            initial = self.core.initial_state(len(token))
            state = {key: torch.where(reset.reshape(-1, *([1] * (value.ndim - 1))),
                                      initial[key], value) for key, value in state.items()}
        d = self.config.observation_dim
        encoded = torch.cat(((token[:, :d] - self.observation_mean) / self.observation_std,
                             token[:, d:]), dim=-1)
        return self.core(encoded, state)

    def step(self, token, state=None, *, reset=None, deterministic=False, generator=None):
        """Consume one public query token; return padded context [B, A], zero logp, state."""
        if reset is not None and (reset.shape != token.shape[:1] or reset.dtype != torch.bool or reset.device != token.device):
            raise ValueError("Require boolean per-lane reset on the token device")
        context, state = self.encode_step(token, state, reset=reset)
        pad = self.config.history_action_dim - self.config.width
        control = torch.nn.functional.pad(context, (0, pad)) if pad > 0 else context
        return control, torch.zeros(len(token), dtype=control.dtype, device=control.device), state

    def encode_sequence(self, tokens, valid, episode_starts, *, mode="full"):
        if mode not in MODES or tokens.ndim != 3 or valid.shape != tokens.shape[:2]:
            raise ValueError("Invalid sequence shape/mode")
        state, contexts = None, []
        for index in range(len(tokens)):
            reset = (torch.ones_like(valid[index]) if mode == "static" else
                     episode_starts[index] if mode == "episode" else None)
            token = torch.where(valid[index, :, None], tokens[index], 0.)
            context, state = self.encode_step(token, state, reset=reset)
            contexts.append(context)
        return torch.stack(contexts)

    def predict_velocity(self, model, observation, noisy_actions, timestep, cache, prefix_mask, context):
        if self.config.interface == "kv_bridge":
            return velocity_with_kv_residual(
                model, observation, noisy_actions, timestep, cache, prefix_mask, context, self.kv_adapter)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError("Require an eval-mode motor with every parameter frozen")
        detached_cache = tuple((k.detach(), v.detach()) for k, v in cache)
        with torch.no_grad():
            suffix = model.run_suffix(observation, noisy_actions.detach(), timestep.detach(),
                                      detached_cache, prefix_mask)
            base_velocity = model.velocity_from_suffix(suffix).detach()
        delta = self.config.scale * torch.tanh(self.velocity_head(context))[:, None, :]
        return _add(base_velocity, delta.expand_as(base_velocity).to(base_velocity.dtype))

    def flow_matching_loss(self, model, observation, target_actions, noise, timestep,
                           cache, prefix_mask, context, *, sample_weights=None):
        """Flow-matching MSE over active coordinates, supporting AD/DPT/DIT/IC-IQL weights."""
        batch = len(target_actions)
        expected = (batch, self.config.action_horizon, self.config.model_action_dim)
        if (target_actions.shape != expected or noise.shape != expected
                or timestep.shape != (batch,) or context.shape != (batch, self.config.width)):
            raise ValueError("Require batch-aligned target_actions, noise, timestep and history context")
        tau = timestep[:, None, None].to(target_actions.dtype)
        noisy_actions = (1. - tau) * target_actions.detach() + tau * noise.detach()
        target_velocity = (noise.detach() - target_actions.detach())[..., :self.config.active_action_dim]
        predicted = self.predict_velocity(
            model, observation, noisy_actions, timestep, cache, prefix_mask, context
        )[..., :self.config.active_action_dim]
        per_sample = (predicted - target_velocity).square().mean(dim=(-1, -2))
        if sample_weights is not None:
            weights = sample_weights.reshape(batch).detach().to(per_sample.dtype)
            if not torch.isfinite(weights).all() or (weights < 0).any():
                raise ValueError("Require finite nonnegative sample weights")
            per_sample = per_sample * weights
        return per_sample.mean()

    @staticmethod
    def meta_objective_weights(objective, *, q_values=None, v_values=None, value_config=None, batch_size=None, device="cpu"):
        """Return per-sample weights for AD/DPT (1.0) or DIT/IC-IQL (exp-clamped Q-V)."""
        if objective not in FLOW_META_OBJECTIVES:
            raise ValueError(f"Unsupported meta-learning objective: {objective!r}")
        if objective in ("ad", "dpt"):
            if batch_size is None or batch_size < 1:
                raise ValueError("AD and DPT require a positive batch_size")
            return torch.ones(batch_size, dtype=torch.float32, device=device)
        if q_values is None or v_values is None or value_config is None:
            raise ValueError(f"{objective} requires q_values, v_values and value_config")
        return importance_weights(q_values.detach() - v_values.detach(), value_config)


class FlowBridgeMotor(FlowDirectMotor):
    """Closed-loop KV or velocity bridge motor with frozen VLM and action expert."""
    def __init__(self, native, codec: FlowBridgeCodec, policy: HistoryFlowSteeringPolicy):
        if not isinstance(codec, FlowBridgeCodec) or policy.config.interface != codec.interface:
            raise ValueError("Bind matching FlowBridgeCodec and HistoryFlowSteeringPolicy interface")
        super().__init__(native, codec)
        self.policy = policy.eval().requires_grad_(False)

    @torch.no_grad()
    def execute_batch(self, prepared, *, noise, steering, spec):
        if (prepared is not self.native._prepared or spec.record() != self.codec.record()
                or np.shape(steering) != (prepared["batch_size"], ACTION_DIM)):
            raise ValueError("Require the unconsumed same-observation bridge steering token")
        batch = prepared["batch_size"]
        draw = torch.as_tensor(noise, dtype=torch.float32, device=self.model.device)
        if draw.shape != (batch, 50, 32) or not torch.isfinite(draw).all():
            raise ValueError("Use the prospective common native draw for every bridge query")
        steering_tensor = torch.as_tensor(steering, dtype=torch.float32, device=self.model.device)
        if not torch.isfinite(steering_tensor).all():
            raise ValueError("Require finite bridge steering tensor")
        context = steering_tensor[:, :self.policy.config.width]
        hidden, mask, cache = prepared["prefix"]
        original_prefix = self.model.build_prefix_cache
        original_velocity = self.model.velocity_from_suffix
        try:
            if self.policy.config.interface == "kv_bridge":
                edited_cache = self.policy.kv_adapter(cache, mask, context)
                self.model.build_prefix_cache = lambda *args, **kwargs: (hidden, mask, edited_cache)
            else:
                self.model.build_prefix_cache = lambda *args, **kwargs: prepared["prefix"]
                delta = self.policy.config.scale * torch.tanh(self.policy.velocity_head(context))[:, None, :]

                def altered_velocity(suffix):
                    base = original_velocity(suffix)
                    return _add(base, delta.expand_as(base).to(base.dtype))

                self.model.velocity_from_suffix = altered_velocity
            commands, _ = self.model.predict_action_batch(prepared["observation"], mode="eval", noise=draw)
            commands = commands.detach().float().cpu().numpy() if torch.is_tensor(commands) else np.asarray(commands, np.float32)
            if commands.shape != (batch, 50, 12) or not np.isfinite(commands).all():
                raise ValueError("Require complete finite native bridge commands")
            return commands
        finally:
            self.model.build_prefix_cache = original_prefix
            self.model.velocity_from_suffix = original_velocity
            self.native._prepared = None


def native_flow_targets(encoded_actions, history_codec, native_action_stats):
    """Convert saved physical-history labels to the native model's FM space.

    History uses canonical binary +/-1 with unit scale; the native normalizer
    retains its learned means/scales for all32 coordinates. Padding therefore
    happens in physical space before applying the official transform.
    """
    encoded = np.asarray(encoded_actions, np.float32)
    if encoded.shape[-1:] != (ACTION_DIM,) or not np.isfinite(encoded).all():
        raise ValueError("Require finite encoded physical H50x12 target labels")
    direct = DirectActionCodec(history_codec.mean, history_codec.std)
    physical = (encoded.reshape(*encoded.shape[:-1], 50, 12)
                * np.asarray(direct.std, np.float32) + np.asarray(direct.mean, np.float32))
    if not np.isin(physical[..., list(BINARY)], [-1., 1.]).all():
        raise ValueError("Require actual canonical binary commands, not continuous predictions")
    mean, std = (np.asarray(native_action_stats[key]) for key in ('mean', 'std'))
    if (mean.shape != (32,) or std.shape != (32,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std < 0).any()):
        raise ValueError("Require the official32-coordinate native action statistics")
    padded = np.pad(physical, [(0,0)] * (physical.ndim-1) + [(0,20)])
    # openpi.transforms.Normalize._normalize, with quantiles disabled.
    return ((padded - mean) / (std + 1e-6)).astype(np.float32)


def sample_bridge_query_batch(
    trials, policy: HistoryFlowSteeringPolicy, rng, *,
    action_codec, native_action_stats,
    batch_size: int, max_episodes: int = 12, mode: str = "full",
    objective: str = "ad", discount: float = .99,
):
    """Sample complete-episode causal windows and select one target query per window."""
    if (mode not in MODES or objective not in FLOW_META_OBJECTIVES
            or type(batch_size) is not int or batch_size < 1
            or type(max_episodes) is not int or max_episodes < 1):
        raise ValueError("Require valid mode/objective and positive batch/episode window counts")
    windows = []
    for index in rng.integers(len(trials), size=batch_size):
        trial_idx = int(index)
        trial = trials[trial_idx]
        validate_trial(trial, objective=objective)
        boundaries = np.flatnonzero(trial["episode_starts"])
        end_episode = int(rng.integers(1, len(boundaries) + 1))
        count = int(rng.integers(1, min(max_episodes, end_episode) + 1))
        begin = int(boundaries[end_episode - count])
        end = int(boundaries[end_episode]) if end_episode < len(boundaries) else len(trial["rewards"])
        token = trial_tokens(trial, mode=mode)[begin:end].copy()
        token[0, policy.config.observation_dim:-1] = 0
        query_rel = int(rng.integers(0, end - begin))
        query_abs = begin + query_rel
        target_flat = np.asarray(trial["target_noise"][query_abs], np.float32)
        target_padded = native_flow_targets(target_flat, action_codec, native_action_stats)
        if target_padded.shape != (policy.config.action_horizon, policy.config.model_action_dim):
            raise ValueError("Native flow targets differ from the bound model action shape")
        windows.append(dict(
            tokens=token, actions=trial["executed_noise"][begin:end],
            rewards=trial["rewards"][begin:end], dones=trial["dones"][begin:end],
            starts=trial["episode_starts"][begin:end],
            returns=discounted_returns(trial["rewards"], trial["dones"], discount)[begin:end],
            query_rel=query_rel, query_abs=query_abs, trial_idx=trial_idx,
            target_actions=target_padded,
        ))
    length = max(len(row["tokens"]) for row in windows)
    if length > policy.config.context_length:
        raise ValueError("Sampled complete-episode window exceeds bridge context_length")
    reference = next(policy.parameters())
    device, dtype = reference.device, reference.dtype
    shapes = dict(tokens=(policy.core.input_dim,), actions=(policy.config.history_action_dim,),
                  rewards=(), dones=(), starts=(), returns=(), valid=())
    result = {key: torch.zeros((length, batch_size, *shape), device=device,
        dtype=torch.bool if key in ("dones", "starts", "valid") else dtype)
        for key, shape in shapes.items()}
    for lane, window in enumerate(windows):
        count = len(window["tokens"])
        result["valid"][:count, lane] = True
        for key in ("tokens", "actions", "rewards", "dones", "starts", "returns"):
            result[key][:count, lane] = torch.as_tensor(window[key], device=device)
    result["query_rel"] = torch.tensor([w["query_rel"] for w in windows], dtype=torch.long, device=device)
    result["target_actions"] = torch.as_tensor(
        np.stack([w["target_actions"] for w in windows]), dtype=dtype, device=device)
    result["query_indices"] = [(w["trial_idx"], w["query_abs"]) for w in windows]
    return result


def export_bridge_policy(path, policy: HistoryFlowSteeringPolicy, *, manifest):
    if not manifest or any(not torch.isfinite(value).all() for value in policy.state_dict().values()):
        raise ValueError("Require finite bridge candidate weights and provenance")
    cfg = asdict(policy.config)
    cfg["kv_layers"] = list(policy.config.kv_layers)
    payload = dict(
        format=BRIDGE_MODEL_FORMAT, config=cfg,
        manifest=json.loads(json.dumps(manifest, allow_nan=False)),
        policy={key: value.detach().cpu().clone() for key, value in policy.state_dict().items()},
    )
    return atomic_save(path, payload)


def load_bridge_policy(path, *, expected_sha256, device="cpu"):
    saved = checked_load(path, expected_sha256)
    if saved.get("format") != BRIDGE_MODEL_FORMAT or not saved.get("manifest"):
        raise ValueError("Require a flow-steering bridge checkpoint and its provenance")
    cfg_dict = dict(saved["config"])
    cfg_dict["kv_layers"] = tuple(cfg_dict["kv_layers"])
    policy = HistoryFlowSteeringPolicy(FlowBridgeConfig(**cfg_dict))
    validate_weights(policy, saved["policy"])
    policy.load_state_dict(saved["policy"], strict=True)
    return policy.to(device).eval().requires_grad_(False), saved["manifest"]
