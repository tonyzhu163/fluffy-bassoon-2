"""Core numerical, causality and recovery checks; run on allocated Linux."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.flora_success import FloraSuccessRule, first_success, timing_metrics
from fluffy_bassoon.meta_rl.interface_ppo import InterfaceLearner, InterfaceSpec, collect
from fluffy_bassoon.meta_rl.paired_evaluation import EvaluationPlan, evaluate, paired_readout
from fluffy_bassoon.meta_rl.residual_history import HistoryResidualActor
from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface
from fluffy_bassoon.meta_rl.vla_history import HistoryConfig, HistoryNoisePolicy


def test_success_hold_is_causal_and_confirmed_without_backdating():
    rule = FloraSuccessRule(consecutive=2)
    score = lambda step: dict(step=step, observed_through=step, reward=.99, p5=.99)
    assert first_success([score(50), score(100)], rule) == 100
    assert first_success([score(50), score(150)], rule) is None
    with pytest.raises(ValueError, match="causal"):
        first_success([dict(score(50), observed_through=100)], rule)
    metrics = timing_metrics([50, 150, 100, None], [200, 100, None, 50])
    assert metrics["prematurely_detected"] == metrics["false_on_never_success"] == 1
    assert metrics["first_success_precision"] == pytest.approx(1 / 3)


def test_joint_policy_scores_every_noise_coordinate_and_preserves_native_initial_std():
    spec = InterfaceSpec()
    learner = InterfaceLearner(3, spec, width=8, layers=1)
    control, log_prob, _ = learner.policy.act(torch.zeros(1, 3), generator=learner.generator)
    native, steering = spec.decode(control[0].numpy())
    assert native.shape == (1, 50, 32) and steering.shape == (6,)
    torch.testing.assert_close(learner.policy.log_std[:1600], torch.zeros(1600))
    distribution, _ = learner.policy._distribution(torch.zeros(1, 3))
    torch.testing.assert_close(log_prob, distribution.log_prob(control).sum(-1))
    changed = control.clone(); changed[0, 1000] += 2
    assert learner.policy.evaluate(torch.zeros(1, 3), changed)[0] != log_prob


class Motor:
    def prepare(self, observation):
        return np.array([observation], np.float32), observation

    def execute(self, prepared, *, noise, steering, spec):
        assert noise.shape == (1, 50, 32) and not len(steering)
        return noise[0, :, :12]


class Environment:
    def __init__(self):
        self.ended = self.counter = 0

    def reset(self):
        self.counter = 0
        return 0.

    def step_chunk(self, chunk):
        self.counter += 50
        return 1., self.counter == 100, dict(executed_controls=50, time=self.counter)

    def end_episode(self):
        self.ended += 1

    def state_dict(self):
        return dict(counter=self.counter, ended=self.ended)

    def load_state_dict(self, state):
        self.counter, self.ended = state["counter"], state["ended"]


def test_first_success_masks_later_chunks_and_checkpoint_restores_next_adam_step():
    spec = InterfaceSpec(kind="noise", steering_dim=0)
    learner = InterfaceLearner(1, spec, width=8, layers=1, epochs=1, minibatch_size=8)
    env = Environment()
    reward = lambda payload: dict(first_success_step=payload["time"], reward=1.)
    rows = collect(learner.policy, Motor(), env, spec, reward, episodes=3, generator=learner.generator)
    assert len(rows) == env.ended == 3
    assert all(r["done"] and r["episode_start"] and r["first_success_step"] == 50 for r in rows)
    sampler = Environment()
    learner.train_batch(rows)
    saved = learner.state_dict(sampler=sampler, environment=env)
    learner.train_batch(rows)
    expected = deepcopy(learner.policy.state_dict())
    resumed = InterfaceLearner(1, spec, seed=99, width=8, layers=1, epochs=1, minibatch_size=8)
    resumed.load_state_dict(saved, sampler=sampler, environment=env)
    resumed.train_batch(rows)
    assert resumed.updates == 2 and resumed.controls == 300
    for key, value in expected.items():
        torch.testing.assert_close(resumed.policy.state_dict()[key], value, rtol=0, atol=0)
    for key, state in learner.optimizer.state_dict()["state"].items():
        for field, value in state.items():
            torch.testing.assert_close(resumed.optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0)


def test_reward_blind_removes_only_reward_and_memory_reset_clears_attempts():
    torch.manual_seed(2)
    policy = HistoryNoisePolicy(HistoryConfig(3, 2, width=8, layers=1, heads=2, context_length=128))
    actor = HistoryResidualActor(policy, reset_each_episode=False, reward_blind=True)
    actor.begin_trial(); actor.begin_attempt()
    control = actor.act(torch.ones(1, 3), deterministic=True)[0]
    actor.observe(control[0].numpy(), 1., True)
    actor.install_terminal_reward(1.)
    actor.begin_attempt()
    captured = []
    original = policy.step
    def step(token, *args, **kwargs):
        captured.append(token.clone())
        return original(token, *args, **kwargs)
    policy.step = step
    actor.act(torch.ones(1, 3), deterministic=True)
    assert captured[0][0, 5] == 0 and captured[0][0, 6] == 1
    actor = HistoryResidualActor(policy, mode="episode", reset_each_episode=False)
    actor.begin_trial(); actor.begin_attempt()
    first = actor.act(torch.ones(1, 3), deterministic=True)[0]
    actor.observe(first[0].numpy(), 1., True); actor.begin_attempt()
    second = actor.act(torch.ones(1, 3), deterministic=True)[0]
    torch.testing.assert_close(first, second)


def test_paired_harness_shares_cases_streams_and_clusters_draws():
    plan = EvaluationPlan(("OpenDrawer",), cases=2, draws=2, attempts=2, horizon=100)
    assert plan.manifest() == plan.manifest()
    seen = []
    def factory(mode):
        return SimpleNamespace(mode=mode, begin_trial=lambda: None, begin_attempt=lambda: None)
    def rollout(actor, case, seeds):
        seen.append((actor.mode, case["native_seed"], seeds))
        success = actor.mode == "full"
        return dict(completed=True, controls=100, pairing_sha256=str(case["native_seed"]),
                    simulator_success_once=success, simulator_final_success=success,
                    first_success_step=50 if success else None)
    rows = evaluate(plan, factory, rollout)
    assert len(rows) == 24
    for index in range(0, len(seen), 6):
        assert seen[index][1:] == seen[index + 2][1:] == seen[index + 4][1:]
    report = paired_readout(rows, "full", "memory_reset", attempt=2, bootstraps=100)
    assert report["cases"] == report["draws_per_case"] == 2
    assert report["difference"] == 1
    assert report["exact_case_sign_randomization_p"] == .5
    with pytest.raises(ValueError, match="every paired"):
        paired_readout(rows[:-2], "full", "reward_blind", attempt=2)


def test_native_adapter_extracts_prefix_before_sampling_and_reuses_it_without_rng():
    class Model(torch.nn.Module):
        device = "cpu"
        def __init__(self):
            super().__init__(); self.weight = torch.nn.Parameter(torch.ones(1))
            self.prefixes = self.draws = 0
        def build_prefix_cache(self, *args):
            self.prefixes += 1
            return torch.ones(1, 2, 4), torch.ones(1, 2, dtype=torch.bool), ((torch.ones(1,1,2,4), torch.ones(1,1,2,4)),)
        def velocity_from_suffix(self, suffix):
            return suffix
        def predict_action_batch(self, observation, *, mode, noise):
            assert mode == "eval" and noise is not None
            self.build_prefix_cache(observation)
            self.draws += 1
            return self.velocity_from_suffix(noise)[..., :12], None
    model = Model()
    stats = dict(state=dict(mean=np.zeros(16),std=np.ones(16)),
                 actions=dict(mean=np.zeros(12),std=np.ones(12)))
    adapter = FrozenRoboCasaInterface(SimpleNamespace(model=model,stats=stats))
    before = torch.get_rng_state().clone()
    features, prepared = adapter.prepare(dict(states=np.ones((1,25),np.float32)))
    assert features.shape == (20,) and model.prefixes == 1 and model.draws == 0
    with pytest.raises(RuntimeError, match="Execute"):
        adapter.prepare(dict(states=np.ones((1,25),np.float32)))
    spec = InterfaceSpec(kind="noise",steering_dim=0)
    noise, steering = spec.decode(np.ones(32,np.float32))
    chunk = adapter.execute(prepared,noise=noise,steering=steering,spec=spec)
    np.testing.assert_array_equal(chunk,noise[0,:,:12])
    assert model.prefixes == model.draws == 1
    assert not model.weight.requires_grad
    torch.testing.assert_close(before,torch.get_rng_state(),rtol=0,atol=0)



class BatchNativeModel(torch.nn.Module):
    """Small deterministic native-shaped prefix/cache/velocity reference."""
    device = "cpu"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))
        self.prefixes = self.draws = 0
        self.empty_row = self.fail_draw = False
        self.layers = 1

    def build_prefix_cache(self, observation):
        self.prefixes += 1
        row = torch.as_tensor(observation["states"][:, 0])
        first = (row[:, None] + 1) * torch.arange(1., 5.)
        second = (row[:, None] + 2) * torch.arange(4., 0., -1)
        hidden = torch.stack((first, second, torch.full_like(first, float("nan"))), 1)
        mask = torch.stack((torch.ones_like(row, dtype=torch.bool), row > 0,
                            torch.zeros_like(row, dtype=torch.bool)), 1)
        if self.empty_row:
            mask[-1] = False
        value = (row[:, None, None, None] + 1) * torch.arange(1., 9.).reshape(1, 1, 2, 4)
        self.built = (hidden, mask, tuple((torch.ones_like(value), value * (layer + 1))
                                          for layer in range(self.layers)))
        return self.built

    def velocity_from_suffix(self, suffix):
        return .75 * suffix

    @torch.no_grad()
    def predict_action_batch(self, observation, *, mode, noise):
        assert mode == "eval" and noise is not None and not torch.is_grad_enabled()
        assert set(observation) <= {"states", "main_images", "wrist_images", "extra_view_images", "task_descriptions"}
        self.used = self.build_prefix_cache(observation)
        self.draws += 1
        if self.fail_draw:
            raise RuntimeError("native draw failed")
        offset = self.used[2][-1][1].mean((1, 2, 3))[:, None, None]
        return self.velocity_from_suffix(noise)[..., :12] + offset, None


def batch_native():
    stats = dict(state=dict(mean=np.arange(16) / 10, std=np.arange(16) / 20 + 1),
                 actions=dict(mean=np.arange(12), std=np.ones(12) * 2))
    native = SimpleNamespace(model=BatchNativeModel(), stats=stats)
    observation = dict(states=np.arange(50, dtype=np.float32).reshape(2, 25),
                       task_descriptions=["drawer", "fridge"], simulator_success=True, task_id=123)
    return native, observation


@pytest.mark.parametrize("kind", ["conditioning", "velocity"])
def test_batched_interface_zero_parity_public_features_prefix_reuse_and_no_rng(kind):
    native, observation = batch_native()
    adapter = FrozenRoboCasaInterface(native)
    noise = np.linspace(-1, 1, 2 * 50 * 32, dtype=np.float32).reshape(2, 50, 32)
    torch_rng, numpy_rng = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
    features, prepared = adapter.prepare_batch(observation)
    assert features.shape == (2, 20) and native.model.prefixes == 1 and native.model.draws == 0
    hidden, mask, cache = prepared["prefix"]
    expected = hidden.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True)
    expected /= expected.square().mean(-1, keepdim=True).sqrt()
    np.testing.assert_allclose(features[:, 16:], expected.numpy())
    state = observation["states"][:, [*range(11, 25), 7, 8]]
    np.testing.assert_allclose(features[:, :16], (state - native.stats["state"]["mean"]) /
                               (native.stats["state"]["std"] + 1e-6), rtol=1e-6)
    original_value = cache[-1][1].clone()
    spec = InterfaceSpec(kind=kind, steering_dim=32, fraction=.25)
    result = adapter.execute_batch(prepared, noise=noise, steering=np.zeros((2, 32)), spec=spec)
    assert result.shape == (2, 50, 12) and native.model.prefixes == native.model.draws == 1
    assert native.model.used[0] is hidden and native.model.used[1] is mask
    assert native.model.used[2][-1][0] is cache[-1][0]
    torch.testing.assert_close(native.model.used[2][-1][1], original_value, rtol=0, atol=0)
    torch.testing.assert_close(cache[-1][1], original_value, rtol=0, atol=0)
    reference, _ = batch_native()
    expected, _ = reference.model.predict_action_batch(dict(states=observation["states"]),
                                                      mode="eval", noise=torch.from_numpy(noise))
    np.testing.assert_array_equal(result, expected.numpy())
    assert not native.model.training and not native.model.weight.requires_grad and native.model.weight.grad is None
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and numpy_rng[2:] == after[2:]
    np.testing.assert_array_equal(numpy_rng[1], after[1])


@pytest.mark.parametrize("kind", ["conditioning", "velocity"])
def test_batched_interface_steering_matches_fixed_projection_and_serial_rows(kind):
    native, observation = batch_native()
    adapter = FrozenRoboCasaInterface(native)
    noise = np.linspace(-1, 1, 2 * 50 * 32, dtype=np.float32).reshape(2, 50, 32)
    code = np.linspace(-2, 2, 64, dtype=np.float32).reshape(2, 32)
    spec = InterfaceSpec(kind=kind, steering_dim=32, fraction=.17)
    _, prepared = adapter.prepare_batch(observation)
    original = prepared["prefix"][2][-1][1].clone()
    result = adapter.execute_batch(prepared, noise=noise, steering=code, spec=spec)
    tanh = torch.from_numpy(code).tanh()
    if kind == "conditioning":
        projection = np.random.default_rng(921003).choice((-1., 1.), (32, 4)) / np.sqrt(32)
        delta = (tanh @ torch.as_tensor(projection, dtype=torch.float32)).reshape(2, 1, 1, 4)
        rms = original.square().mean((1, 2, 3), keepdim=True).sqrt()
        expected = original + spec.fraction * rms * delta
        torch.testing.assert_close(native.model.used[2][-1][1], expected, rtol=0, atol=0)
        velocity = .75 * torch.from_numpy(noise)
    else:
        expected = original
        velocity = .75 * torch.from_numpy(noise) + spec.fraction * tanh[:, None]
    chunks = velocity[..., :12] + expected.mean((1, 2, 3))[:, None, None]
    np.testing.assert_allclose(result, chunks.numpy(), rtol=0, atol=0)
    # Per-row RMS and steering must agree with independent serial execution.
    serial = []
    for row in range(2):
        one, _ = batch_native()
        port = FrozenRoboCasaInterface(one)
        _, token = port.prepare(dict(states=observation["states"][row:row + 1]))
        serial.append(port.execute(token, noise=noise[row:row + 1], steering=code[row], spec=spec))
    np.testing.assert_allclose(result, serial, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(prepared["prefix"][2][-1][1], original, rtol=0, atol=0)


def test_all_layer_conditioning_steers_each_layer_and_keeps_final_layer_map():
    noise = np.linspace(-1, 1, 2 * 50 * 32, dtype=np.float32).reshape(2, 50, 32)
    code = np.linspace(-2, 2, 64, dtype=np.float32).reshape(2, 32)
    runs = {}
    for kind in ("conditioning", "conditioning_all"):
        native, observation = batch_native()
        native.model.layers = 3
        adapter = FrozenRoboCasaInterface(native)
        _, prepared = adapter.prepare_batch(observation)
        cache = [entry[1].clone() for entry in prepared["prefix"][2]]
        zero = InterfaceSpec(kind=kind, steering_dim=32, fraction=.17)
        result = adapter.execute_batch(prepared, noise=noise, steering=np.zeros((2, 32)), spec=zero)
        for used, original in zip(native.model.used[2], cache):
            torch.testing.assert_close(used[1], original, rtol=0, atol=0)
        _, prepared = adapter.prepare_batch(observation)
        runs[kind] = (adapter.execute_batch(prepared, noise=noise, steering=code, spec=zero),
                      [entry[1] for entry in native.model.used[2]], cache, result)
    tanh = torch.from_numpy(code).tanh()
    single, every = runs["conditioning"][1], runs["conditioning_all"][1]
    cache = runs["conditioning"][2]
    for layer, original in enumerate(cache):
        projection = np.random.default_rng(921003 + 2 - layer).choice((-1., 1.), (32, 4)) / np.sqrt(32)
        delta = (tanh @ torch.as_tensor(projection, dtype=torch.float32)).reshape(2, 1, 1, 4)
        rms = original.square().mean((1, 2, 3), keepdim=True).sqrt()
        torch.testing.assert_close(every[layer], original + .17 * rms * delta, rtol=0, atol=0)
        if layer < 2:
            torch.testing.assert_close(single[layer], original, rtol=0, atol=0)
    torch.testing.assert_close(every[-1], single[-1], rtol=0, atol=0)
    np.testing.assert_array_equal(runs["conditioning"][3], runs["conditioning_all"][3])


def test_batched_interface_rejects_bad_rows_and_restores_hooks_after_failures():
    native, observation = batch_native()
    port = FrozenRoboCasaInterface(native)
    prefix, velocity = native.model.build_prefix_cache, native.model.velocity_from_suffix
    native.model.empty_row = True
    with pytest.raises(ValueError, match="nonempty native prefix per batch row"):
        port.prepare_batch(observation)
    assert native.model.build_prefix_cache == prefix and port._prepared is None
    native.model.empty_row = False
    _, prepared = port.prepare_batch(observation)
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    noise, code = np.zeros((2, 50, 32)), np.ones((2, 32))
    with pytest.raises(ValueError, match="current pre-action prefix"):
        port.execute_batch(dict(prepared), noise=noise, steering=code, spec=spec)
    assert port._prepared is prepared
    with pytest.raises(ValueError, match="explicit native noise"):
        port.execute_batch(prepared, noise=noise[:1], steering=code, spec=spec)
    assert native.model.draws == 0 and port._prepared is None
    _, prepared = port.prepare_batch(observation)
    native.model.fail_draw = True
    with pytest.raises(RuntimeError, match="native draw failed"):
        port.execute_batch(prepared, noise=noise, steering=code, spec=spec)
    assert native.model.build_prefix_cache == prefix and native.model.velocity_from_suffix == velocity
    assert port._prepared is None
    with pytest.raises(ValueError, match="current pre-action prefix"):
        port.execute_batch(prepared, noise=noise, steering=code, spec=spec)


@pytest.mark.parametrize("kind,dim", [("select", 2), ("direct", 600), ("action", 6)])
def test_serial_interface_retains_candidate_direct_and_action_ports(kind, dim):
    native, observation = batch_native()
    bound = SimpleNamespace(residual_dim=6, apply=lambda chunk, code: chunk + code.sum())
    port = FrozenRoboCasaInterface(native, bound=bound)
    spec = InterfaceSpec(kind=kind, steering_dim=dim, candidates=2)
    control = np.linspace(-1, 1, spec.control_dim, dtype=np.float32)
    noise, code = spec.decode(control)
    _, prepared = port.prepare(dict(states=observation["states"][:1]))
    result = port.execute(prepared, noise=noise, steering=code, spec=spec)
    if kind == "direct":
        expected = code.reshape(50, 12) * (np.asarray(native.stats["actions"]["std"]) + 1e-6)
        expected += native.stats["actions"]["mean"]
        assert native.model.draws == 0
    else:
        selected = code.argmax() if kind == "select" else 0
        expected = .75 * noise[selected, :, :12] + 4.5
        if kind == "action":
            expected += code.sum()
        assert native.model.draws == (2 if kind == "select" else 1)
    assert native.model.prefixes == 1
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-6)

def test_training_windows_reach_128_actual_tokens():
    from fluffy_bassoon.meta_rl.vla_history import sample_batch
    from fluffy_bassoon.meta_rl.vla_history_train import supervised_update
    policy = HistoryNoisePolicy(HistoryConfig(3,2,width=8,layers=1,heads=2,context_length=128))
    length = 128
    trial = dict(observations=np.zeros((length,3),np.float32),
                 executed_noise=np.zeros((length,2),np.float32), target_noise=np.zeros((length,2),np.float32),
                 rewards=np.zeros(length,np.float32),dones=np.array([False]*127+[True]),
                 episode_starts=np.array([True]+[False]*127),teacher_steps=np.arange(length,dtype=np.int64))
    class LongestWindow:
        def integers(self, low, high=None, size=None):
            return np.zeros(size,dtype=np.int64) if size is not None else high - 1
    batch = sample_batch([trial],policy,LongestWindow(),batch_size=1,max_context=128)
    assert batch[0].shape[0] == 128 and batch[2].sum() == 128
    result = supervised_update(policy,torch.optim.AdamW(policy.parameters(),lr=1e-3),batch,mode="full")
    assert np.isfinite(result["nll"]) and result["valid_positions"] == 128


@pytest.mark.parametrize("arm",["S","F"])
def test_native_first_success_closes_early_and_keeps_truth_out_of_flora_payload(tmp_path,arm):
    import json
    from test_robocasa_residual import Journal, Native, Writer
    from fluffy_bassoon.meta_rl.robocasa_interface import RoboCasaInterfaceEnvironment
    class Prefix:
        def reset(self):
            self.frames = 0
        def append(self, frames):
            assert len(frames) == 3 and all(x.dtype == np.uint8 for x in frames)
            self.frames += 1
        def score(self, step):
            assert self.frames == step + 1
            return dict(first_success_step=100 if step >= 100 else None,reward=float(step >= 100))
    prefix = Prefix() if arm == "F" else None
    env = RoboCasaInterfaceEnvironment(lambda case:Native(horizon=400,success=lambda step:step>=73),
        [dict(native_seed=1,task="OpenDrawer")],interface_horizon=400,reward_arm=arm,
        prefix_observer=prefix,output=tmp_path/'videos',monitoring_root=tmp_path/'monitor',
        journal=Journal(),writer_factory=Writer)
    observation = env.reset()
    assert 'success' not in observation
    for index in range(2):
        observation,done,payload=env.step_chunk(np.zeros((50,12),np.float32))
        assert not done and set(payload)=={'executed_controls','end_step','outcome'}
        assert payload['outcome']['first_success_step'] == (None if index==0 else 73 if arm=='S' else 100)
    env.end_episode()
    assert env.native.steps==100 and not env.active
    truth=json.loads((tmp_path/'monitor/episode-000000.json').read_text())
    assert truth['first_success_step']==73 and truth['simulator_final_success']
    assert env.writer.count==101 and env.state_dict()['horizon']==400
