"""Opt-in audits fingerprint the one actually sampled/executed action path."""

from dataclasses import dataclass, replace
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_bridge_train import CHUNKS, collect_live_trials


@dataclass
class Observation:
    state: torch.Tensor
    images: dict
    image_masks: dict
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor


class Motor:
    def __init__(self, changed=None):
        self.changed, self.queries, self.decodes = changed, 0, 0

    def env_obs_to_observation(self, obs):
        return Observation(torch.zeros(2, 32) + (1 if self.changed == "state" else 0),
            {"base": torch.full((2, 4, 4, 3), 1 if self.changed == "images" else 0, dtype=torch.uint8)},
            {"base": torch.ones(2, dtype=torch.bool)},
            torch.ones(2, 4, dtype=torch.long) + (1 if self.changed == "prompt" else 0),
            torch.ones(2, 4, dtype=torch.bool))

    def sample_actions(self, observation, *, noise, num_steps):
        self.queries += 1
        return torch.full((2, 8, 7), 2.)

    def decode_actions(self, actions, state):
        self.decodes += 1
        return actions[:, :5, :7]


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))
        self.calls = 0

    def sample_chunk(self, motor, observation, feedback, memory, *, rng, **kwargs):
        self.calls += 1
        noise = torch.randn(2, 8, 32, generator=rng)
        key = observation.state[:, None, None, :8].expand(-1, 3, 1, -1)
        memory = torch.zeros(2, 1) if memory is None else memory
        return noise[..., :7], memory + 1, dict(kv_key=key, kv_value=key+.1,
            prefix_mask=torch.ones(2, 3, dtype=torch.bool), noisy_actions=noise[None].expand(4, -1, -1, -1),
            old_log_prob=torch.zeros(2), old_value=torch.zeros(2))


class Env:
    device = "cpu"

    def __init__(self):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.tensor([0, 1])))
        self.commands = []

    def reset(self, **kwargs):
        return {}, {}

    def chunk_step(self, command):
        self.commands.append(command.clone())
        return [{}] * 5, None, None, None, [dict(success=torch.tensor([True, False]))] * 5


def collect(*, audit=False, changed=None, labels=False):
    motor, policy, env, rows = Motor(changed), Policy(), Env(), []
    plan = dict(theta_degrees=[0., 0.], reset_seeds=[7],
        scenes=[[dict(object=0, episode_id=0), dict(object=1, episode_id=1)]])
    rollout = collect_live_trials(policy, motor, env, plan, generator=torch.Generator().manual_seed(41),
        deterministic=True, execute_labels=labels,
        preprocess=lambda obs: replace(obs, state=obs.state + .25,
            images={key: value.float()/255 for key, value in obs.images.items()}),
        prepare_actions=lambda actions: actions * .5,
        trajectory_audit_callback=rows.append if audit else None)
    return rollout, motor, policy, env, rows


@pytest.mark.parametrize("labels", [False, True])
def test_audit_adds_no_samples_or_native_queries_and_records_executed_override(labels):
    plain, plain_motor, plain_policy, plain_env, _ = collect(labels=labels)
    audited, motor, policy, env, rows = collect(audit=True, labels=labels)
    assert policy.calls == plain_policy.calls == CHUNKS
    assert motor.queries == plain_motor.queries == (CHUNKS if labels else 0)
    assert motor.decodes == plain_motor.decodes
    torch.testing.assert_close(audited["outcomes"], plain["outcomes"], rtol=0, atol=0)
    assert len(rows) == CHUNKS
    for chunk, row in enumerate(rows):
        assert row["attempt"] == 0 and row["chunk"] == chunk and row["reset_seed"] == 7
        assert row["state"] == torch.zeros(2, 32).tolist()
        assert row["prepared_state"] == torch.full((2, 32), .25).tolist()
        assert row["state_sha256"] != row["prepared_state_sha256"]
        assert set(row["observation_fields"]["fields"]) == {
            "state", "images", "image_masks", "tokenized_prompt", "tokenized_prompt_mask"}
        torch.testing.assert_close(torch.tensor(row["issued_commands"]), env.commands[chunk], rtol=0, atol=0)
        torch.testing.assert_close(env.commands[chunk], plain_env.commands[chunk], rtol=0, atol=0)
        torch.testing.assert_close(torch.tensor(row["prepared_commands"]),
                                   torch.tensor(row["decoded_actions"]) * .5, rtol=0, atol=0)
        assert row["success_per_step"] == [[True, False]] * 5
        assert row["success_once"] == [True, False]
        if labels:
            assert torch.tensor(row["decoded_actions"]).eq(2).all()
    json.dumps(rows, allow_nan=False)
    rows[0]["issued_commands"][0][0][0] = 999.
    assert env.commands[0][0, 0, 0] != 999.


def test_repeated_zero_path_audits_are_exact_and_hash_every_native_input_field():
    *_, first = collect(audit=True)
    *_, repeated = collect(audit=True)
    assert first == repeated
    for changed in ("images", "prompt", "state"):
        *_, other = collect(audit=True, changed=changed)
        assert first[0]["observation_sha256"] != other[0]["observation_sha256"]
        assert first[0]["prepared_observation_sha256"] != other[0]["prepared_observation_sha256"]
        assert first[0]["initial_noise_sha256"] == other[0]["initial_noise_sha256"]
        if changed != "state":
            assert first[0]["native_kv_sha256"] == other[0]["native_kv_sha256"]
