"""Run in an allocated Linux step using the isolated AMAGO requirements."""

from collections import namedtuple
from copy import deepcopy
import os
import pickle
import random
import sys
import time

import numpy as np
import pytest
import torch

pytest.importorskip("gin")
pytest.importorskip("wandb")
pytest.importorskip("gym")
from fluffy_bassoon.meta_rl.metaworld_amago import (
    ReplayStore, TorchSDPAAttention, UPSTREAM_COMMIT, collect_trials,
    exploration_random_stream, extend_evaluation_positions, install_torch_sdpa,
    load_upstream, make_experiment, make_explorers, make_timestep, restore_snapshot,
    round_grad_updates, sampled_transition_count, save_snapshot, summarize,
)
from fluffy_bassoon.meta_rl.metaworld_env import MetaWorldTrial, benchmark_pools, make_lanes


pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Model tests require allocated Linux")
Task = namedtuple("Task", "env_name data")


class FakeSawyer:
    def set_task(self, task):
        self.task = pickle.loads(task.data)

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def observation(self):
        state = np.zeros(39, np.float32)
        state[0] = self.rng.uniform()
        state[-3:] = self.task["goal"]
        return state

    def reset(self):
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        return self.observation(), float(self.steps), False, False, {"success": self.steps == 1}

    def close(self):
        pass


def fake_lanes():
    tasks = [Task("push-v3", pickle.dumps(dict(partially_observable=False, goal=[x, 2, 3])))
             for x in (1, 2, 3)]
    envs = [MetaWorldTrial(FakeSawyer, tasks, attempts=2, horizon=3,
                          goal_visible=True, resample_instance=True) for _ in range(2)]
    for env in envs:
        env.family = "push-v3"
    return envs


def experiment(attention="vanilla"):
    return make_experiment(load_upstream(), horizon=6, max_seq_len=6,
                           memory_size=64, memory_layers=1, attention=attention,
                           batch_size=2, grad_updates=1, warmup_epochs=2, device="cpu")


def seed():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


def assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_tree_equal(a, b)
    else:
        assert left == right


def test_original_version_and_off_policy_method_are_used():
    seed()
    exp = experiment()
    assert UPSTREAM_COMMIT == "514c1c2a471418aaf7829ba60ad0582b7764e643"
    assert type(exp.policy).__module__ == "amago.agent"
    assert type(exp).train_step.__module__ == "amago.learning"
    assert exp.policy.num_critics == 4
    assert exp.policy.online_coeff == 1. and exp.policy.offline_coeff == .1
    assert exp.policy.reward_multiplier == 10.
    assert len(exp.policy.gammas) == 7
    assert isinstance(exp.optimizer, torch.optim.AdamW)


def test_original_utd_cumulative_rounding_and_optimizer_step_warmup():
    from fractions import Fraction
    assert [round_grad_updates(i, 125000, ratio=Fraction(1, 12)) for i in range(3)] == [10416, 10417, 10417]
    assert sum(round_grad_updates(i, 125000, ratio=Fraction(1, 12)) for i in range(80)) == 833333
    assert round_grad_updates(5, 125000, fixed=1000) == 1000
    exp = make_experiment(load_upstream(), horizon=6, max_seq_len=6,
        memory_size=64, memory_layers=1, attention="vanilla", batch_size=1,
        grad_updates=2083, warmup_epochs=10, warmup_grad_updates=10000, device="cpu")
    assert exp.warmup_scheduler.total_iters == 10000


def test_matched_replay_passes_use_cumulative_sampled_transition_exposure():
    from fractions import Fraction
    ratio = Fraction(10, 24 * 2000)
    steps = [round_grad_updates(i, 125000, ratio=ratio) for i in range(80)]
    assert steps[:3] == [26, 26, 26] and steps[23] == 27
    assert sum(steps) == 2083
    assert sum(steps) * 24 * 2000 == 99984000
    assert 10 * 10000000 - sum(steps) * 24 * 2000 < 24 * 2000
    assert sum(steps[:10]) == 260
    seed()
    exp = experiment()
    from amago.loading import RLData, RLData_pad_collate
    trajectories = collect_trials(exp.policy, fake_lanes(), [3, 4],
                                  max_seq_len=6, attention="vanilla")
    batch = RLData_pad_collate([RLData(t) for t in trajectories])
    assert sampled_transition_count(batch) == 12
    batch.rl2s[0, -1] = 0  # Explicitly excluded padding state.
    assert sampled_transition_count(batch) == 11


def test_boundary_reward_action_alignment_and_no_task_identity():
    seed()
    exp = experiment()
    envs = fake_lanes()
    data = collect_trials(exp.policy, envs, [12, 13], max_seq_len=6,
                          attention="vanilla", explorers=make_explorers(envs))
    trajectory = data[0]
    assert trajectory.obs.keys() == {"observation"}
    assert trajectory.obs["observation"].shape == (7, 41)
    np.testing.assert_array_equal(trajectory.rl2s[:, 0], [1, 0, 0, 1, 0, 0, 1])
    np.testing.assert_allclose(trajectory.rews[:, 0], [.1, .2, .3, .1, .2, .3])
    np.testing.assert_array_equal(trajectory.rl2s[1:, 1:2], trajectory.rews)
    np.testing.assert_array_equal(trajectory.rl2s[1:, 3:], trajectory.actions)
    np.testing.assert_array_equal(trajectory.dones[:, 0], [0, 0, 0, 0, 0, 1])
    # An inner reset receives the terminal reward/action and the next state.
    assert trajectory.rl2s[3, 1] == pytest.approx(.3)
    assert trajectory.obs["observation"][3, -1] == 0.
    assert summarize(envs)["success_by_episode"] == [1., 1.]
    assert summarize(envs)["native_returns"] == [[6., 6.], [6., 6.]]


def test_reward_blind_intervention_retains_boundary_and_action():
    load_upstream()
    env = fake_lanes()[0]
    observation, _ = env.reset(seed=9)
    for _ in range(3):
        observation, *_ = env.step(np.full(4, .4, np.float32))
    normal = make_timestep(observation, time_idx=3, length=6, attempts=2, horizon=3)
    blind = make_timestep(observation, time_idx=3, length=6, attempts=2, horizon=3,
                          reward_blind=True)
    assert normal.real_reward == pytest.approx(.3) and blind.real_reward == 0.
    assert normal.reset and blind.reset
    np.testing.assert_array_equal(normal.prev_action, blind.prev_action)


def test_extended_eval_preserves_training_prefix_clocks_and_parameter_schema():
    seed()
    exp = experiment("torch-sdpa")
    short = fake_lanes()[0]
    long = MetaWorldTrial(FakeSawyer, short.tasks, attempts=5, horizon=3,
                         goal_visible=True, resample_instance=True)
    long.family = short.family
    weights = deepcopy(exp.policy.state_dict())
    first = collect_trials(exp.policy, [short], [91], max_seq_len=6,
                           attention="torch-sdpa", deterministic=True)[0]
    with extend_evaluation_positions(exp.policy):
        extended = collect_trials(exp.policy, [long], [91], max_seq_len=6,
            attention="torch-sdpa", deterministic=True, clock_attempts=2)[0]
    # Compare six actual decisions, before the shorter trial's terminal state
    # differs from a physical reset into the longer trial's third episode.
    np.testing.assert_array_equal(first.obs["observation"][:6], extended.obs["observation"][:6])
    np.testing.assert_array_equal(first.rl2s[:6], extended.rl2s[:6])
    np.testing.assert_array_equal(first.actions, extended.actions[:6])
    np.testing.assert_array_equal(extended.time_idxs, np.arange(16))
    assert extended.rl2s[-1, 2] == pytest.approx(15 / 6)
    assert extended.obs["observation"][-1, -2] == 4.
    assert len(long.outcomes) == 5
    assert_tree_equal(weights, exp.policy.state_dict())
    assert not exp.policy.traj_encoder.tformer.position_embedding._forward_pre_hooks


def test_batch_step_seam_preserves_order_and_executed_feedback():
    seed()
    exp = experiment("torch-sdpa")
    left, right = fake_lanes(), fake_lanes()
    expected = collect_trials(exp.policy, left, [6, 7], max_seq_len=6,
                              attention="torch-sdpa", deterministic=True)
    seen = []
    def batch_step(actions):
        seen.append(actions.copy())
        return [env.step(action) for env, action in zip(right, actions)]
    actual = collect_trials(exp.policy, right, [6, 7], max_seq_len=6,
        attention="torch-sdpa", deterministic=True, batch_step=batch_step)
    assert len(seen) == 6
    for a, b in zip(expected, actual):
        assert_tree_equal(a.to_dict(), b.to_dict())
    for lane in range(2):
        np.testing.assert_array_equal(np.stack(seen)[:, lane], actual[lane].actions)


def test_worker_lanes_match_serial_stochastic_exploration_and_replay():
    from fluffy_bassoon.meta_rl.checkpoint import rng_state, restore_rng
    from fluffy_bassoon.meta_rl.metaworld_workers import make_worker_lanes
    seed()
    exp = experiment("torch-sdpa")
    serial = fake_lanes()
    pool = ({"push-v3": FakeSawyer}, serial[0].tasks)
    workers = make_worker_lanes(pool, 2, benchmark="ml10", attempts=2, horizon=3,
                                workers=2, allocated_cpus=2)
    try:
        initial = rng_state()
        expected = collect_trials(exp.policy, serial, [91, 92], max_seq_len=6,
            attention="torch-sdpa", explorers=make_explorers(serial))
        restore_rng(initial)
        actual = collect_trials(exp.policy, workers, [91, 92], max_seq_len=6,
            attention="torch-sdpa", explorers=make_explorers(workers))
        for a, b in zip(expected, actual):
            assert_tree_equal(a.to_dict(), b.to_dict())
        assert summarize(serial) == summarize(workers)
    finally:
        for env in serial + workers:
            env.close()


def test_exploration_numpy_draws_ignore_simulator_global_numpy_perturbations():
    seed()
    left, right = make_explorers(fake_lanes(), seed=123), make_explorers(fake_lanes(), seed=123)
    for explorers in (left, right):
        for explorer in explorers:
            explorer.global_multiplier = .7
    with exploration_random_stream(left):
        expected = [explorer.action(np.zeros(4, np.float32)) for explorer in left]
    # Simulate environment constructor/reset use of the driver's NumPy global RNG.
    np.random.seed(901)
    np.random.randn(100)
    parent_state = np.random.get_state()
    with exploration_random_stream(right):
        actual = [explorer.action(np.zeros(4, np.float32)) for explorer in right]
    np.testing.assert_array_equal(expected, actual)
    assert_tree_equal(parent_state, np.random.get_state())


@pytest.mark.parametrize("attention", ["vanilla", "torch-sdpa"])
def test_exact_next_collection_replay_update_and_recovery(tmp_path, attention):
    seed()
    exp = experiment(attention)
    envs = fake_lanes()
    explorers = make_explorers(envs)
    sampler = np.random.default_rng(123)
    replay = ReplayStore(tmp_path / "replay", 3)
    for traj in collect_trials(exp.policy, envs, sampler.integers(0, 1000, 2),
            max_seq_len=6, attention=attention, explorers=explorers):
        replay.append(traj)
    exp.policy.train()
    losses = exp.train_step(replay.batch(2, 6), log_step=False)
    assert all(torch.isfinite(losses[k]) for k in ("actor_loss", "critic_loss"))
    exp.warmup_scheduler.step(0)
    contract = {"test": "official_amago1", "upstream_commit": UPSTREAM_COMMIT}
    save_snapshot(tmp_path / "latest.pt", experiment=exp, replay=replay,
        sampler=sampler, explorers=explorers, iteration=1, counters={"optimizer_steps": 1},
        contract=contract)

    def next_update():
        trajectories = collect_trials(exp.policy, envs, sampler.integers(0, 1000, 2),
            max_seq_len=6, attention=attention, explorers=explorers)
        for traj in trajectories:
            replay.append(traj)
        exp.policy.train()
        batch = replay.batch(2, 6)
        exp.train_step(batch, log_step=False)
        exp.warmup_scheduler.step(1)
        return trajectories, deepcopy(exp.policy.state_dict()), deepcopy(exp.optimizer.state_dict()), \
            deepcopy(exp.warmup_scheduler.state_dict()), deepcopy(replay.entries), deepcopy(batch.rl2s)

    expected = next_update()
    iteration, counters = restore_snapshot(tmp_path / "latest.pt", experiment=exp,
        replay=replay, sampler=sampler, explorers=explorers, contract=contract)
    assert iteration == 1 and counters == {"optimizer_steps": 1}
    actual = next_update()
    for left, right in zip(expected[0], actual[0]):
        assert_tree_equal(left.to_dict(), right.to_dict())
    for left, right in zip(expected[1:], actual[1:]):
        assert_tree_equal(left, right)
    with pytest.raises(ValueError, match="contract differs"):
        restore_snapshot(tmp_path / "latest.pt", experiment=exp, replay=replay,
            sampler=sampler, explorers=explorers, contract={"changed": True})


def test_replay_checks_hashes_and_retains_checkpoint_files(tmp_path):
    seed()
    exp = experiment()
    replay = ReplayStore(tmp_path / "replay", 1)
    trajectories = collect_trials(exp.policy, fake_lanes(), [3, 4],
                                  max_seq_len=6, attention="vanilla")
    replay.append(trajectories[0])
    first = deepcopy(replay.entries)
    replay.append(trajectories[1])
    assert len(replay.entries) == 1
    replay.restore(first)
    from pathlib import Path
    Path(first[0]["path"]).write_bytes(b"corrupt replay")
    with pytest.raises(ValueError, match="Replay bytes changed"):
        replay.restore(first)


def test_native_sdpa_projection_output_matches_frozen_attention_math():
    from einops import rearrange
    seed()
    exp = experiment("flash")  # Construction does not call the optional extension.
    before = deepcopy(exp.policy.state_dict())
    assert install_torch_sdpa(exp.policy) == 1
    assert_tree_equal(before, exp.policy.state_dict())
    exp.policy.eval()
    layer = exp.policy.traj_encoder.tformer.layers[0].self_attention
    sequence = torch.randn(2, 7, 64)
    qkv = rearrange(layer.qkv_projection(sequence),
        "b l (three d heads) -> b l three heads d", three=3, heads=8).to(torch.bfloat16)
    q, k, v = qkv.float().unbind(2)
    scores = torch.einsum("blhd,bshd->bhls", q, k) / q.shape[-1] ** .5
    mask = torch.ones(7, 7, dtype=torch.bool).triu(1)
    weights = scores.masked_fill(mask, -torch.inf).softmax(-1)
    reference = torch.einsum("bhls,bshd->blhd", weights, v).to(torch.bfloat16)
    reference = layer.out_projection(rearrange(
        layer.head_scaler * reference, "b l h d -> b l (h d)"))
    actual = layer(sequence)
    torch.testing.assert_close(actual, reference, rtol=8e-3, atol=8e-3)
    changed = sequence.clone()
    changed[:, 4:] += 100.
    torch.testing.assert_close(layer(changed)[:, :4], actual[:, :4], rtol=0, atol=0)
    layer.train()
    value = layer(sequence.requires_grad_()).square().mean()
    value.backward()
    assert torch.isfinite(sequence.grad).all()


def test_native_sdpa_cache_handles_unequal_lanes_and_unused_nan_slots():
    seed()
    primitive = TorchSDPAAttention().eval()
    qkv = torch.randn(2, 6, 3, 2, 8)
    keys = torch.full((2, 8, 2, 8), torch.nan, dtype=torch.bfloat16)
    values = torch.full_like(keys, torch.nan)
    lengths = torch.zeros(2, dtype=torch.int32)
    for index in range(6):
        if index == 3:
            lengths[1] = 0  # One lane starts a new trial; its old bytes remain.
        actual = primitive(qkv[:, index:index + 1], keys, values, lengths)
        for lane in range(2):
            start = 3 if lane == 1 and index >= 3 else 0
            full = primitive(qkv[lane:lane + 1, start:index + 1])
            torch.testing.assert_close(actual[lane:lane + 1], full[:, -1:], rtol=8e-3, atol=8e-3)
        assert torch.isfinite(actual).all()
        lengths += 1
    primitive.train()
    with pytest.raises(ValueError, match="eval-mode one-step"):
        primitive(qkv[:, :1], keys, values, lengths)


@pytest.mark.parametrize("attention", ["torch-sdpa", "flash"])
def test_cached_actions_match_full_sequence_and_reset_history(attention):
    if attention == "flash":
        if os.environ.get("FLUFFY_AMAGO_ATTENTION") != "flash":
            pytest.skip("Flash cache parity requires its qualified CUDA extension")
        pytest.importorskip("flash_attn")
    device = "cuda" if attention == "flash" else "cpu"
    seed()
    exp = make_experiment(load_upstream(), horizon=6, max_seq_len=8,
        memory_size=64, memory_layers=1, attention=attention, batch_size=1,
        grad_updates=1, warmup_epochs=2, device=device)
    policy = exp.policy.eval()
    from amago.hindsight import Trajectory
    env = fake_lanes()[0]
    obs, _ = env.reset(seed=5)
    trajectory = Trajectory(max_goals=1, timesteps=[make_timestep(
        obs, time_idx=0, length=6, attempts=2, horizon=3)])
    for step in range(6):
        obs, _, terminated, _, _ = env.step(np.full(4, .2, np.float32))
        trajectory.add_timestep(make_timestep(obs, time_idx=step + 1, length=6,
            attempts=2, horizon=3, terminal=terminated))
    obs, goals, rl2s = trajectory.make_sequence()
    hidden = policy.traj_encoder.init_hidden_state(1, torch.device(device))
    with torch.no_grad():
        for index in range(6):
            def inputs(start, stop):
                return dict(obs={"observation": torch.as_tensor(
                    obs["observation"][None, start:stop], device=device)},
                    goals=torch.as_tensor(goals[None, start:stop], device=device),
                    rl2s=torch.as_tensor(rl2s[None, start:stop], device=device),
                    seq_lengths=torch.full((1, 1, 1), stop - start,
                                           dtype=torch.long, device=device),
                    time_idxs=torch.arange(start, stop, device=device)[None], sample=False)
            full, _ = policy.get_actions(**inputs(0, index + 1))
            cached, hidden = policy.get_actions(**inputs(index, index + 1), hidden_state=hidden)
            np.testing.assert_allclose(cached, full, rtol=2e-3, atol=2e-3)
        reset = policy.traj_encoder.reset_hidden_state(hidden, np.array([True]))
        first, reset = policy.get_actions(**inputs(3, 4), hidden_state=reset)
        isolated, _ = policy.get_actions(**inputs(3, 4))
        np.testing.assert_allclose(first, isolated, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(os.environ.get("FLUFFY_METAWORLD_REAL_QUALIFICATION") != "1",
                    reason="Real simulator qualification is opt-in in an allocated Linux job")
def test_real_ml10_complete_trial_and_off_policy_update(tmp_path):
    seed()
    attention = os.environ.get("FLUFFY_AMAGO_ATTENTION", "vanilla")
    device = "cuda" if attention in ("flash", "torch-sdpa") else "cpu"
    exp = make_experiment(load_upstream(), horizon=2500, max_seq_len=32,
        memory_size=64, memory_layers=1, attention=attention, batch_size=1,
        grad_updates=1, warmup_epochs=2, device=device)
    pools = benchmark_pools("ml10", 17)
    # One complete actual K5H500 trial qualifies boundaries; ten-family curves
    # require the full balanced evaluator in the executable driver.
    envs = make_lanes(pools["train"], 10, benchmark="ml10", attempts=5, horizon=500)
    try:
        trajectories = collect_trials(exp.policy, envs[:1], [987], max_seq_len=32,
                                      attention=attention, deterministic=True)
        replay = ReplayStore(tmp_path / "replay", 1)
        replay.append(trajectories[0])
        exp.policy.train()
        losses = exp.train_step(replay.batch(1, 32), log_step=False)
        assert len(envs[0].outcomes) == 5
        assert all(torch.isfinite(losses[k]) for k in ("actor_loss", "critic_loss"))
    finally:
        for env in envs:
            env.close()


@pytest.mark.skipif(os.environ.get("FLUFFY_AMAGO_DEFAULT_CAPACITY") != "1",
                    reason="Default B24 L2000 capacity is a separate allocated-GPU admission gate")
def test_default_b24_l2000_real_replay_capacity(tmp_path):
    """One genuine default-shape update; never infer this from the tiny smoke test."""
    from pathlib import Path
    import json
    seed()
    attention = os.environ.get("FLUFFY_AMAGO_ATTENTION", "torch-sdpa")
    assert attention in ("torch-sdpa", "flash")
    assert torch.cuda.is_available()
    exp = make_experiment(load_upstream(), horizon=2500, max_seq_len=2000,
        memory_size=256, memory_layers=3, attention=attention, batch_size=24,
        grad_updates=1000, warmup_epochs=10, device="cuda")
    pools = benchmark_pools("ml10", 17)
    envs = make_lanes(pools["train"], 10, benchmark="ml10", attempts=5, horizon=500)
    try:
        started = time.monotonic()
        trajectories = collect_trials(exp.policy, envs[:1], [987], max_seq_len=2000,
                                      attention=attention, deterministic=True)
        torch.cuda.synchronize()
        collection_seconds = time.monotonic() - started
        replay = ReplayStore(tmp_path / "replay", 1)
        replay.append(trajectories[0])
        batch = replay.batch(24, 2000)
        assert batch.rl2s.shape == (24, 2001, 7)
        assert sampled_transition_count(batch) == 48000
        exp.policy.train()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        losses = exp.train_step(batch, log_step=False)
        torch.cuda.synchronize()
        step_seconds = time.monotonic() - started
        assert all(torch.isfinite(losses[k]) for k in ("actor_loss", "critic_loss", "Global Grad Norm"))
        result = dict(qualification="one default-shape off-policy update, not a learning result",
            attention=attention, batch_size=24, transition_length=2000, state_length=2001,
            memory_size=256, memory_layers=3, attempts=5, horizon=500,
            replay_source="one real K5H500 trial; independent random windows sampled with replacement",
            sampled_replay_transitions=48000,
            upstream_commit=UPSTREAM_COMMIT, torch=torch.__version__,
            cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            parameters=sum(p.numel() for p in exp.policy.parameters()),
            trainable_parameters=sum(p.numel() for p in exp.policy.trainable_params if p.requires_grad),
            collection_seconds=collection_seconds, optimization_seconds=step_seconds,
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            losses={k: float(losses[k].detach().cpu()) for k in ("actor_loss", "critic_loss")})
        destination = Path(os.environ.get("FLUFFY_AMAGO_CAPACITY_OUTPUT",
                                         str(tmp_path / "default-capacity.json")))
        with destination.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps(result, allow_nan=False))
    finally:
        for env in envs:
            env.close()
