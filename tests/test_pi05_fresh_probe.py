"""Complete state cloning, valid history swaps and paired fresh probe accounting."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from fluffy_bassoon.meta_rl.pi05_fresh_probe import (
    ATTEMPTS, CHUNKS, ProbeBoundary, clone_boundary, evaluate_fresh_probes,
    run_live_attempt, validate_probe_plans,
)


class MockPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.zeros(()))
        self.bridge = nn.Module()
        self.bridge.residual_output = nn.Linear(1, 1)
        with torch.no_grad():
            self.bridge.residual_output.weight.fill_(.01)
            self.bridge.residual_output.bias.fill_(.02)
        self.calls = []

    def sample_chunk(self, motor, observation, feedback, memory, *, rng, deterministic, reset, num_steps):
        assert deterministic and not reset.any() and num_steps == 4
        batch = len(feedback)
        memory = {"summary": torch.zeros(batch, 1)} if memory is None else memory
        self.calls.append(dict(memory=memory["summary"].clone(), feedback=feedback.clone()))
        # Deliberately mutate input memory: evaluator must still isolate branches.
        memory["summary"].add_(1 + feedback[:, 7:8] * 10)
        torch.randn(batch, 8, 32, generator=rng)
        actions = torch.ones(batch, 8, 32)
        actions[..., :7] += self.bridge.residual_output(memory["summary"])[:, None]
        key = observation.state[:, None, None].clone()
        return actions, memory, dict(recurrence_updates=1, num_steps=4, kv_key=key,
                                     kv_value=key + 1, prefix_mask=torch.ones(batch, 1, dtype=torch.bool))


class MockMotor:
    def env_obs_to_observation(self, obs):
        return SimpleNamespace(state=obs["state"])

    def decode_actions(self, actions, state):
        return actions[:, :5, :7]


class MockEnv:
    device = "cpu"

    def __init__(self, lanes=2):
        self.env = SimpleNamespace(unwrapped=SimpleNamespace(select_carrot_ids=torch.zeros(lanes, dtype=torch.long)))
        self.resets, self.commands = [], []
        self.lanes = lanes

    def reset(self, *, seed, options):
        ids = options["episode_id"].float()
        self.state = ids[:, None].expand(-1, 4).clone()
        self.resets.append((seed, ids.tolist()))
        return {"state": self.state}, {}

    def chunk_step(self, actions):
        self.commands.append(actions.clone())
        successes = torch.zeros(self.lanes, dtype=torch.bool)
        successes[0] = True
        infos = [dict(success=successes.clone()) for _ in range(5)]
        return [{"state": self.state}] * 5, None, None, None, infos


def plans(replicas=2):
    def scenes(first):
        return [dict(object=0, episode_id=first + lane) for lane in range(2)]
    support = dict(theta_degrees=[-30., 30.], scenes=[scenes(10), scenes(20), scenes(30)],
                   reset_seeds=[11, 12, 13])
    probes = [dict(scenes=scenes(100 + 10 * index), reset_seed=1000 + index) for index in range(replicas)]
    return support, probes


def evaluate(policy, env, *, conditions=("normal", "reset_memory", "opposite_history", "zero_bridge"),
             replicas=2, **kwargs):
    support, probes = plans(replicas)
    return evaluate_fresh_probes(policy, MockMotor(), env, support, probes,
        support_generator=torch.Generator().manual_seed(17), probe_seeds=list(range(90, 90 + replicas)),
        history_batch_id="independent-batch-7", opposite_indices=[1, 0], conditions=conditions,
        preprocess=lambda observation: observation, prepare_actions=lambda actions: actions, **kwargs)


def test_boundary_clone_transplants_every_field_and_never_aliases():
    state = ProbeBoundary(dict(summary=torch.tensor([[1.], [2.]]), length=torch.tensor([3, 4])),
        torch.arange(14.).reshape(2, 7), torch.tensor([1., 0.]), 3, ("a", "b"), attempt_end=True)
    copied = clone_boundary(state, indices=[1, 0])
    assert copied.history_ids == ("b", "a")
    assert copied.attempt_index == 3 and copied.attempts == 4 and copied.attempt_end
    assert copied.pending_reward.tolist() == [0., 1.]
    torch.testing.assert_close(copied.previous_command[0], state.previous_command[1])
    assert copied.memory["length"].tolist() == [4, 3]
    copied.memory["summary"].zero_()
    copied.previous_command.zero_()
    copied.pending_reward.zero_()
    assert state.memory["summary"].sum() == 3 and state.previous_command.sum() == 91
    assert state.pending_reward.sum() == 1
    reset = clone_boundary(state, reset_memory=True)
    assert reset.memory is None and torch.equal(reset.pending_reward, state.pending_reward)
    with pytest.raises(ValueError, match="permutation"):
        clone_boundary(state, indices=[0, 0])


def test_attempt_consumes_pending_terminal_feedback_once_and_keeps_k4_clocks():
    policy, env = MockPolicy(), MockEnv()
    state = ProbeBoundary(dict(summary=torch.tensor([[50.], [60.]])),
        torch.tensor([[.25] * 7, [-.25] * 7]), torch.tensor([1., 0.]), 3,
        ("a", "b"), attempt_end=True)
    before = clone_boundary(state)
    _, probes = plans(1)
    result = run_live_attempt(policy, MockMotor(), env, probes[0]["scenes"], [-30., 30.], 51, state,
        generator=torch.Generator().manual_seed(4), preprocess=lambda obs: obs,
        prepare_actions=lambda actions: actions)
    first, second = policy.calls[0]["feedback"], policy.calls[1]["feedback"]
    torch.testing.assert_close(first[:, :7], state.previous_command)
    assert first[:, 7].tolist() == [1., 0.] and first[:, 8].eq(1).all()
    assert first[:, 9].eq(3 / ATTEMPTS).all() and first[:, 10].eq(0).all()
    assert second[:, 7:9].eq(0).all()
    assert second[:, 9].eq(.75).all() and second[:, 10].eq(1 / CHUNKS).all()
    assert len(policy.calls) == CHUNKS and result["state"].attempt_index == 4
    torch.testing.assert_close(state.memory["summary"], before.memory["summary"])
    assert result["counts"]["simulator_steps"] == 160
    assert result["counts"]["native_prefix_queries"] == 32
    assert result["counts"]["native_suffix_queries"] == 128
    # The public feedback action is unrotated; only the simulator receives the rotation.
    assert not torch.equal(env.commands[0][:, -1], policy.calls[1]["feedback"][:, :7])


def test_fresh_replicas_restore_same_boundary_and_all_conditions_are_paired():
    policy, env = MockPolicy(), MockEnv()
    before = {key: value.clone() for key, value in policy.state_dict().items()}
    boundaries = []
    result = evaluate(policy, env, on_boundary=lambda state, metadata: boundaries.append((state, metadata)))
    assert result["completed"] and not result["stopped_at_attempt_boundary"]
    assert len(boundaries) == 3 and boundaries[-1][0].pending_reward.tolist() == [1., 0.]
    assert boundaries[-1][0].attempt_index == 3
    assert all(torch.equal(value, before[key]) for key, value in policy.state_dict().items())
    # A replica starts from the same support state despite in-place mock memory updates.
    for lane in range(2):
        normal = [row for row in result["rows"] if row["condition"] == "normal" and row["lane"] == lane]
        assert normal[0]["first_feedback"] == normal[1]["first_feedback"]
        opposite = [row for row in result["rows"] if row["condition"] == "opposite_history" and row["lane"] == lane]
        assert opposite[0]["donor_history_id"] == f"independent-batch-7/lane-{1 - lane}"
        assert opposite[0]["first_feedback"][7] == 1 - normal[0]["first_feedback"][7]
    assert all(check["first_prefix_identical"] and check["native_noise_streams_identical"]
               for check in result["pairing_checks"])
    assert result["independent_clusters"] == 1 and len(result["rows"]) == 2 * 4 * 2
    assert {row["cluster_id"] for row in result["rows"]} == {"independent-batch-7"}
    assert result["support_counts"]["simulator_steps"] == 3 * 2 * 80
    assert result["probe_counts"]["simulator_steps"] == 2 * 4 * 2 * 80
    assert result["simulator_steps"] == 1760 and result["policy_chunks"] == 352
    assert result["total_counts"]["native_suffix_queries"] == 1408
    assert all(counts["simulator_steps"] == 320 for counts in result["probe_counts_by_condition"].values())
    json.dumps(result, allow_nan=False)


def test_reward_blind_recollects_support_and_counts_its_actual_extra_work():
    policy = MockPolicy()
    result = evaluate(policy, MockEnv(), conditions=("normal", "reward_blind"), replicas=1)
    assert set(result["support_counts_by_collection"]) == {"normal", "reward_blind"}
    assert result["support_counts"]["simulator_steps"] == 2 * 3 * 2 * 80
    assert result["probe_counts"]["simulator_steps"] == 2 * 2 * 80
    blinded = [row for row in result["rows"] if row["condition"] == "reward_blind"]
    assert all(row["first_feedback"][7] == 0 for row in blinded)
    # Reward masking covers support recurrence too, rather than just the probe boundary.
    assert all(call["feedback"][:, 7].eq(0).all() for call in policy.calls[3 * CHUNKS:6 * CHUNKS])


def test_stop_after_support_returns_complete_receipt_without_starting_a_probe():
    stopped, progress, boundaries = {"value": False}, [], []
    def on_progress(snapshot):
        progress.append(deepcopy(snapshot))
        if len(snapshot["support_rows"]) == 6:
            stopped["value"] = True
        snapshot["rows"].append({"must_not_mutate_evaluator": True})
    result = evaluate(MockPolicy(), MockEnv(), should_stop=lambda: stopped["value"],
                      on_progress=on_progress, on_boundary=lambda state, metadata: boundaries.append(state))
    assert not result["completed"] and result["stopped_at_attempt_boundary"]
    assert result["rows"] == [] and result["probe_counts"]["simulator_steps"] == 0
    assert result["simulator_steps"] == 480 and len(boundaries) == 3
    assert boundaries[-1].attempt_index == 3
    assert result["support_boundary_metadata"]["normal"]["pending_reward"] == [1., 0.]
    assert all("must_not_mutate_evaluator" not in row for row in result["rows"])


def test_stop_between_probe_conditions_keeps_completed_rows_and_costs():
    stopped = {"value": False}
    def progress(snapshot):
        if snapshot["completed_probe_branches"]:
            stopped["value"] = True
    result = evaluate(MockPolicy(), MockEnv(), should_stop=lambda: stopped["value"], on_progress=progress)
    assert not result["completed"] and len(result["rows"]) == 2
    assert result["completed_probe_branches"] == [dict(probe_index=0, condition="normal")]
    assert result["probe_counts"]["simulator_steps"] == 160
    assert result["simulator_steps"] == 640


@pytest.mark.parametrize("change", ["zero_theta", "different_object", "reused_scene", "bad_permutation"])
def test_invalid_history_and_freshness_controls_are_rejected_before_execution(change):
    support, probes = plans(1)
    indices = [1, 0]
    if change == "zero_theta":
        support["theta_degrees"] = [0., 0.]
    elif change == "different_object":
        support["scenes"][1][1]["object"] = 1
    elif change == "reused_scene":
        probes[0]["scenes"][0]["episode_id"] = support["scenes"][0][0]["episode_id"]
    else:
        indices = [0, 0]
    with pytest.raises(ValueError):
        validate_probe_plans(support, probes, [3], indices)
