"""Direct physical-action histories from a trained Flow-SDE action expert.

The action expert is bypassed by students. Its Gaussian draws are neither
student labels nor steering codes. Visual features come from the original
frozen VLM, alongside all 25 public state coordinates.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .vla_history import save_dataset


BINARY = (6, 11)
FEATURE_DIM, ACTION_DIM = 2073, 600


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
    """Identical complete chronological streams for all four methods."""
    states, actions = raw["public_states"], raw["physical_actions"]
    if (states.shape != (128, 144, 25) or actions.shape != (128, 144, 50, 12)
            or features.shape != (128, 144, FEATURE_DIM) or teacher_actions.shape != actions.shape
            or not np.array_equal(features[..., :25], states)
            or not np.isfinite(features).all() or not provenance):
        raise ValueError("Bind the fully audited 18-version physical source and frozen public features")
    starts = np.broadcast_to(np.arange(144) % 8 == 0, (128, 144))
    ends = np.broadcast_to(np.arange(144) % 8 == 7, (128, 144))
    versions = np.broadcast_to(np.repeat(np.arange(18), 8), (128, 144))
    if (not np.array_equal(raw["episode_starts"], starts)
            or not np.array_equal(raw["physical_episode_ends"], ends)
            or not np.array_equal(raw["producer_versions"], versions)):
        raise ValueError("Do not change chronology or physical boundaries")
    actual, labels = codec.encode(actions), codec.encode(teacher_actions)
    rewards = np.asarray(raw["primitive_rewards"].sum(-1), np.float32)
    if (not np.isin(rewards, [0., 1.]).all() or (rewards.reshape(128, 18, 8).sum(-1) > 1).any()):
        raise ValueError("Keep actual first-success reward once per physical episode")
    validation = set(validation_slots)
    if not validation or not validation < set(range(128)):
        raise ValueError("Require a fixed nonempty held-out collection-slot partition")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifests = {}
    for partition, slots in (("train", sorted(set(range(128)) - validation)), ("validation", sorted(validation))):
        for objective in ("ad", "dpt", "dit", "ic_iql"):
            trials = [dict(observations=features[s].astype(np.float32), executed_noise=actual[s],
                target_noise=labels[s] if objective == "dpt" else actual[s], rewards=rewards[s],
                dones=ends[s].copy(), episode_starts=starts[s].copy(), teacher_steps=versions[s].copy()) for s in slots]
            metadata = dict(provenance, trial_ids=[f"flow-source-slot-{s:03d}" for s in slots],
                feature_contract=dict(dimension=FEATURE_DIM, meaning="original frozen VLM prefix2048 RMS plus raw public state25"),
                noise_contract=codec.record(), value_contract=dict(discount=.99, boundary="physical_done",
                    rewards="actual native first-success once", terminal_bootstrap=0,
                    next_observation="following saved query within episode; no invented terminal RGB"),
                validation="held-out source stream imitation/value diagnostic; source PPO saw all source observations")
            path = output / f"{objective}-{partition}.npz"
            saved = save_dataset(path, trials, objective=objective, metadata=metadata)
            manifests[f"{objective}_{partition}"] = dict(path=str(path), sha256=saved["sha256"])
    return manifests
