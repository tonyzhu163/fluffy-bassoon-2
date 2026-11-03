"""Core first-success collector, deterministic streams and episode recovery checks."""
from copy import deepcopy
import gzip
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.interface_ppo import InterfaceSpec
from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import collect_bank, draw_seed, _qualify_cached, _calibrate_request
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, stack_records
from fluffy_bassoon.meta_rl.robocasa_interface import FrozenRoboCasaInterface


class Motor:
    def __init__(self):
        self.pending = None
        self.batch_sizes = []

    def prepare_batch(self, observation):
        assert self.pending is None and set(observation) == {"states"}
        features = observation["states"][:, :2].copy()
        self.pending = object()
        self.batch_sizes.append(len(features))
        return features, self.pending

    def execute_batch(self, prepared, *, noise, steering, spec):
        assert prepared is self.pending
        assert noise.shape == (len(steering), 50, 32)
        assert steering.shape[1] == spec.steering_dim
        self.pending = None
        return noise.numpy()[..., :12] + .25 * np.tanh(steering.cpu().numpy())[:, None, :12]


class Environment:
    def __init__(self, cases, root, lane):
        self.cases, self.cursor = deepcopy(cases), 0
        self.output, self.monitoring_root = root / f"videos-{lane}", root / f"monitor-{lane}"
        self.output.mkdir(parents=True)
        self.monitoring_root.mkdir(parents=True)
        self.native = SimpleNamespace(current_raw_obs=None)
        self.active = self.closed = False
        self.dispatched_commands, self.completed_commands = [], []

    def seek(self, slot):
        assert not self.active and slot >= self.cursor
        self.cursor = slot

    def observation(self):
        state = np.zeros((1, 25), np.float32)
        state[0, :2] = self.case["native_seed"], self.steps
        self.native.current_raw_obs = [dict(image=np.full((2, 2, 3), self.steps, np.uint8),
                                             private_success=bool(self.success_steps))]
        return dict(states=state)

    def reset(self):
        self.case = self.cases[self.cursor]
        self.key = f"episode-{self.cursor:06d}"
        self.cursor += 1
        self.steps, self.horizon = 0, self.case["horizon"]
        self.success_steps, self.active = [], True
        return self.observation()

    def step_chunk(self, chunk):
        self.dispatched_commands, self.completed_commands = [], []
        count = 4 if self.case.get("fail_at") == self.steps + 50 else 50
        actual = chunk.copy()
        actual[:, [6, 11]] = np.where(actual[:, [6, 11]] < .5, -1, 1)
        self.dispatched_commands = list(actual[:count])
        self.completed_commands = list(actual[:count])
        begin = self.steps
        self.steps += count
        first = self.case.get("first_success")
        last = self.case.get("last_success", self.horizon)
        if first is not None:
            self.success_steps.extend(step for step in range(begin + 1, self.steps + 1) if first <= step <= last)
        observation = self.observation()
        if count != 50:
            raise RuntimeError("owned lane failed")
        time.sleep(self.case.get("delay", 0))
        done = self.steps == self.horizon
        self.active = not done
        return observation, done, dict(id=self.key)

    def _counts(self):
        return dict(controls=self.steps + 3, reset_controls=3)

    def state_dict(self):
        assert not self.active
        return dict(cases=self.cases, cursor=self.cursor, horizon=None)

    def close(self):
        self.closed = True


def policy():
    return ResidualActorCritic(ResidualPPOConfig(2, 32, width=8, layers=1, gamma=.99))


def request(cases, **overrides):
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    return dict(sequence=1, phase="train", iteration=1, cases=cases,
                spec=dict(kind=spec.kind, steering_dim=spec.steering_dim),
                source_commit="source", runtime_commit="runtime", **overrides)


def load_gz(path):
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def collect(tmp_path, cases, *, lanes=2, actor=None, **flags):
    environments = [Environment(cases, tmp_path, lane) for lane in range(lanes)]
    result = collect_bank(actor or policy(), Motor(), environments,
        request=request(cases, **flags), output=tmp_path / "history", rank=0,
        deadline_unix=time.time() + 30)
    assert all(env.closed for env in environments)
    return result


def test_full_physical_horizon_and_first_success_only_ppo_records_are_retained(tmp_path):
    cases = [dict(task="one", native_seed=1, horizon=150, first_success=73, last_success=73),
             dict(task="two", native_seed=2, horizon=100),
             dict(task="three", native_seed=3, horizon=100, first_success=1)]
    actor = policy()
    result = collect(tmp_path, cases, actor=actor)
    rows = result["records"]
    assert [row["case_slot"] for row in rows] == [0, 0, 1, 1, 2]
    assert [row["reward"] for row in rows] == [0, 1, 0, 0, 1]
    assert [row["done"] for row in rows] == [False, True, False, True, True]
    assert [row["episode_start"] for row in rows] == [True, False, True, False, True]
    assert [episode["physical_chunks"] for episode in result["episodes"]] == [3, 2, 2]
    assert [episode["success_at_end"] for episode in result["episodes"]] == [False, False, True]
    rollout = stack_records(rows)
    assert rollout["behavior_mean"].shape == rollout["behavior_std"].shape == rollout["controls"].shape
    reference = torch.distributions.Normal(rollout["behavior_mean"], rollout["behavior_std"])
    torch.testing.assert_close(reference.log_prob(rollout["controls"]).sum(-1), rollout["log_probs"],
                               rtol=1e-6, atol=1e-6)
    for episode in result["episodes"]:
        checkpoint = torch.load(episode["checkpoint"]["path"], weights_only=False)
        assert len(checkpoint["records"]) == episode["valid_ppo_chunks"]
        final = load_gz(episode["final"]["path"])
        assert final["native_counts"]["controls"] - 3 == episode["horizon"]
        for chunk in episode["chunks"]:
            record = load_gz(chunk["path"])
            assert record["status"] == "completed" and record["executed_controls"] == 50
            assert record["iteration"] == 1 and record["policy_iteration"] == 0
            assert record["requested_commands"].shape == (50, 12)
            assert np.asarray(record["executed_commands"]).shape == (50, 12)
            assert "raw_observation" in record and "raw_next_observation" in record
            assert "private_success" not in record["observation"]
            assert record["native_noise"].shape == (50, 32)
            assert record["actor_standard_normal"].shape == (32,)
    suffix = load_gz(result["episodes"][0]["chunks"][-1]["path"])
    assert not suffix["valid_ppo"] and suffix["physical_done"] and suffix["reward"] == 0
    log_prob = actor.evaluate(torch.as_tensor(np.asarray([row["features"] for row in rows])),
                              torch.stack([row["control"] for row in rows]))[0]
    np.testing.assert_allclose(log_prob.detach().numpy(), [row["log_prob"] for row in rows], rtol=1e-6)


def test_case_draws_and_controls_are_independent_of_lane_arrival_order(tmp_path):
    cases = [dict(task="one", native_seed=seed, horizon=100, delay=.002 * (seed % 2)) for seed in range(1, 5)]
    actor = policy()
    serial = collect(tmp_path / "serial", cases, lanes=1, actor=actor)
    parallel = collect(tmp_path / "parallel", cases, lanes=4, actor=actor)
    for left, right in zip(serial["records"], parallel["records"]):
        assert (left["case"], left["chunk_index"]) == (right["case"], right["chunk_index"])
        torch.testing.assert_close(left["control"], right["control"], rtol=0, atol=0)
    for left, right in zip(serial["episodes"], parallel["episodes"]):
        for a, b in zip(left["chunks"], right["chunks"]):
            a, b = load_gz(a["path"]), load_gz(b["path"])
            assert a["native_noise_seed"] == b["native_noise_seed"]
            np.testing.assert_array_equal(a["native_noise"], b["native_noise"])
            np.testing.assert_array_equal(a["requested_commands"], b["requested_commands"])
    assert draw_seed(cases[0], 0, "native_gaussian") != draw_seed(cases[0], 0, "actor_standard_normal")


def test_completed_sibling_checkpoint_survives_another_lane_failure(tmp_path):
    cases = [dict(task="good", native_seed=1, horizon=100),
             dict(task="bad", native_seed=2, horizon=100, fail_at=100)]
    environments = [Environment(cases, tmp_path, lane) for lane in range(2)]
    with pytest.raises(RuntimeError, match="owned lane failed"):
        collect_bank(policy(), Motor(), environments, request=request(cases), output=tmp_path / "history",
                     rank=0, deadline_unix=time.time() + 30)
    complete = torch.load(tmp_path / "history/episode-000000/checkpoint.pt", weights_only=False)
    assert complete["episode"]["completed"] and complete["records"][-1]["done"]
    failed = load_gz(tmp_path / "history/episode-000001/chunk-0001.pkl.gz")
    assert failed["status"] == "failed_during_dispatch" and len(failed["executed_commands"]) == 4
    assert all(env.closed for env in environments)


@pytest.mark.parametrize("phase,zero", [("test_before", True), ("dev", False), ("test_after", False)])
def test_eval_uses_fixed_native_noise_and_zero_or_mean_steering_without_actor_draw(tmp_path, phase, zero):
    cases = [dict(task="one", native_seed=1, horizon=100)]
    actor = policy()
    with torch.no_grad():
        actor.actor[-1].bias.fill_(.6)
    state = torch.get_rng_state().clone()
    specification = request(cases, deterministic=True, zero_steering=zero)
    specification["phase"] = phase
    environment = Environment(cases, tmp_path, 0)
    result = collect_bank(actor, Motor(), [environment], request=specification,
        output=tmp_path / "history", rank=0, deadline_unix=time.time() + 30)
    for row in result["records"]:
        torch.testing.assert_close(row["control"], torch.full((32,), 0. if zero else .6), rtol=0, atol=0)
    for artifact in result["episodes"][0]["chunks"]:
        record = load_gz(artifact["path"])
        assert record["actor_standard_normal"] is None and record["actor_noise_seed"] is None
        assert record["native_noise_seed"] == draw_seed(cases[0], record["chunk_index"], "native_gaussian")
    torch.testing.assert_close(state, torch.get_rng_state(), rtol=0, atol=0)


def test_cached_qualification_reuses_one_prefix_and_preserves_real_pending_draw():
    from test_interface_ppo import batch_native
    native, observation = batch_native()
    motor = FrozenRoboCasaInterface(native)
    _, prepared = motor.prepare_batch(observation)
    noise = torch.ones((2, 50, 32))
    spec = InterfaceSpec(kind="velocity", steering_dim=32)
    result = _qualify_cached(motor, prepared, noise, spec, .5)
    assert result["zero_native_exact"] and result["extra_physical_controls"] == 0
    assert result["action_delta_rms"] > 0 and motor._prepared is prepared
    motor.execute_batch(prepared, noise=noise, steering=np.zeros((2, 32)), spec=spec)
    assert native.model.prefixes == 1 and native.model.draws == 4


def test_native_scale_calibration_uses_only_one_reset_and_no_policy_controls(tmp_path):
    from test_interface_ppo import batch_native
    native, _ = batch_native()
    motor = FrozenRoboCasaInterface(native)
    case = dict(task="one", native_seed=1, horizon=100)
    class ResetOnly(Environment):
        def reset(self):
            observation = super().reset()
            self.initial_evidence = dict(physical="observed-reset")
            self.writer = SimpleNamespace(count=1, refs=lambda: [])
            return observation
        def step_chunk(self, chunk):
            raise AssertionError("Calibration cannot dispatch policy controls")
    env = ResetOnly([case], tmp_path, 0)
    specification = request([case])
    specification["phase"] = "calibrate"
    result = _calibrate_request(motor, env, request=specification, output=tmp_path / "calibration")
    assert result["records"] == result["episodes"] == []
    assert result["calibration"]["zero_native_bitwise_equal"]
    assert result["calibration_scope"]["policy_controls"] == env.steps == 0
    assert result["calibration_scope"]["frame_count"] == 1 and env.closed


@pytest.mark.parametrize("change", [None, "source", "weights"])
@pytest.mark.parametrize("lanes", [4, 48])
def test_worker_restart_binds_saved_initializer_and_requalifies_new_allocation(tmp_path, monkeypatch, change, lanes):
    """A completed bank is never replayed or accepted under a different motor."""
    from dataclasses import asdict
    import fluffy_bassoon.meta_rl.joint_steering_ppo_worker as worker
    directory = tmp_path / "workers/rank-00"
    directory.mkdir(parents=True)
    configuration = ResidualPPOConfig(2064, 32, width=8, layers=1, gamma=.99)
    actor = ResidualActorCritic(configuration)
    case = dict(task="OpenDrawer", native_seed=1700000)
    previous_request = dict(sequence=1, phase="calibrate", iteration=0, lanes=lanes,
        cases=[case], deterministic=True, zero_steering=False,
        policy_config=asdict(configuration), policy=actor.state_dict(),
        spec=dict(kind="velocity", steering_dim=32, fraction=.05), canonical_root=str(tmp_path / "canonical"),
        source_commit="source", runtime_commit="runtime", model_directory="public-model", norm_path="public-norm")
    worker._save(directory / "request-000001.pt", previous_request)
    native_model = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    native_model.device = torch.device("cpu")
    actual_signature = dict(model_sha256="public-weights", model_config_sha256="public-config",
        norm_sha256="public-normalization", parameter_count=2, cuda_uuid="new-gpu",
        all_parameters_frozen=True, parameter_versions_sha256=worker.tensor_hash(worker._versions(native_model)))
    prior_signature = dict(actual_signature, cuda_uuid="old-gpu", parameter_versions_sha256="old-reload-versions")
    previous = dict(completed=True, phase="calibrate", sequence=1,
        request_sha256=worker.sha(directory / "request-000001.pt"),
        worker_rng=worker.snapshot_rng(), native_model_frozen_signature=prior_signature,
        request_binding=worker._request_binding(previous_request))
    worker._save(directory / "result-000001.pt", previous)
    worker._json(directory / "qualification.json", dict(allocation_id="old-allocation", zero_native_exact=True))
    current = dict(previous_request, sequence=2, phase="train", iteration=1, deterministic=False)
    current["spec"] = dict(previous_request["spec"], fraction=.025)  # Chosen after calibration.
    if change == "source":
        current["source_commit"] = "different-source"
    worker._save(directory / "request-000002.pt", current)
    loads, collections = [], []

    def load_native(request, code):
        loads.append(request)
        signature = dict(actual_signature)
        if change == "weights":
            signature["model_sha256"] = "different-weights"
        return SimpleNamespace(model=native_model), None, signature

    def collect(policy, motor, environments, **kwargs):
        collections.append(kwargs)
        assert len(environments) == lanes
        receipt_path = kwargs["qualification_path"]
        assert receipt_path == directory / "qualification.json" and not receipt_path.exists()
        assert list(directory.glob("qualification-prior-*.json"))
        worker._json(receipt_path, dict(zero_native_exact=True, prefix_reused=True,
            extra_physical_controls=0, action_delta_rms=.1,
            allocation_id=kwargs["request"]["allocation_id"]))
        (tmp_path / "STOP").write_text("complete synthetic boundary")
        return dict(episodes=[], records=[])

    monkeypatch.setattr(worker, "_load_native", load_native)
    monkeypatch.setattr(worker, "_RecordingEnvironment", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(worker, "collect_bank", collect)
    if change is None:
        worker.run_worker(tmp_path, 0, tmp_path / "code", time.time() + 30)
        assert len(loads) == len(collections) == 1
        result = torch.load(directory / "result-000002.pt", weights_only=False)
        assert result["completed"] and result["qualification_receipt"]["path"] == str(directory / "qualification.json")
        assert result["allocation_id"] != "old-allocation"
        assert result["native_model_frozen_signature"]["cuda_uuid"] == "new-gpu"
    else:
        message = "Persistent motor/interface" if change == "source" else "Reloaded frozen initializer"
        with pytest.raises(ValueError, match=message):
            worker.run_worker(tmp_path, 0, tmp_path / "code", time.time() + 30)
        assert len(loads) == (0 if change == "source" else 1) and not collections
        assert not (directory / "result-000002.pt").exists()
        assert (directory / "workererror.json").exists()


def test_all_layer_cached_qualification_observes_every_consumed_key_and_value_without_extra_draws():
    from test_interface_ppo import AllLayerNativeModel, batch_native
    native, observation = batch_native()
    native.model = AllLayerNativeModel()
    motor = FrozenRoboCasaInterface(native)
    _, prepared = motor.prepare_batch(observation)
    originals = [[tensor.clone() for tensor in pair] for pair in prepared["prefix"][2]]
    prefix, predict = native.model.build_prefix_cache, native.model.predict_action_batch
    torch_rng, numpy_rng = torch.get_rng_state().clone(), deepcopy(np.random.get_state())
    spec = InterfaceSpec(kind="kv_all_layers", kv_layers=3, steering_dim=192, fraction=.17)
    result = _qualify_cached(motor, prepared, torch.ones((2, 50, 32)), spec, .5)
    assert result["zero_native_exact"] and result["extra_physical_controls"] == 0
    assert result["consumed_cache_layers"] == 3 and result["original_cache_unchanged"]
    assert result["action_delta_rms"] > 0
    targets = result["cache_targets"]
    assert [(row["layer"], row["target"]) for row in targets] == [
        (layer, target) for layer in range(3) for target in ("key", "value")]
    for row in targets:
        layer, target = row["layer"], ("key", "value").index(row["target"])
        actual, original = native.model.used[2][layer][target], originals[layer][target]
        difference = actual.float() - original.float()
        assert row["shape"] == list(original.shape) and row["dtype"] == str(original.dtype)
        assert row["differing_components"] == int(torch.count_nonzero(difference)) > 0
        assert row["delta_rms"] == float(difference.square().mean().sqrt()) > 0
        assert row["finite"] and row["original_cache_unchanged"]
        torch.testing.assert_close(prepared["prefix"][2][layer][target], original, rtol=0, atol=0)
    assert native.model.prefixes == 1 and native.model.draws == 3
    assert native.model.build_prefix_cache == prefix and native.model.predict_action_batch == predict
    assert motor._prepared is prepared and not native.model.weight.requires_grad
    torch.testing.assert_close(torch_rng, torch.get_rng_state(), rtol=0, atol=0)
    after = np.random.get_state()
    assert numpy_rng[0] == after[0] and numpy_rng[2:] == after[2:]
    np.testing.assert_array_equal(numpy_rng[1], after[1])


def test_48_synchronized_lanes_preserve_exact_rotating_case_order_and_every_physical_chunk(tmp_path):
    cases = [dict(task=f"task-{(slot + 2) % 5}", native_seed=100 + slot,
                  horizon=150 if slot % 7 == 0 else 100,
                  **(dict(first_success=1) if slot % 3 == 0 else {})) for slot in range(48)]
    environments = [Environment(cases, tmp_path, lane) for lane in range(48)]
    motor = Motor()
    specification = request(cases, lanes=48, synchronize_initial_lanes=True)
    result = collect_bank(policy(), motor, environments, request=specification,
        output=tmp_path / "history", rank=0, deadline_unix=time.time() + 60)
    assert motor.batch_sizes[0] == 48 and result["execution"]["peak_active_lanes"] == 48
    assert result["execution"]["requested_lanes"] == result["execution"]["maximum_native_batch"] == 48
    assert result["execution"]["initial_lane_barrier"]
    assert [episode["case"] for episode in result["episodes"]] == cases
    assert [episode["physical_chunks"] for episode in result["episodes"]] == [case["horizon"] // 50 for case in cases]
    for episode in result["episodes"]:
        assert Path(episode["checkpoint"]["path"]).exists()
        assert len(episode["chunks"]) == episode["horizon"] // 50
    assert all(environment.closed for environment in environments)


def test_ready_episode_is_checkpointed_before_waiting_for_a_later_failing_lane(tmp_path, monkeypatch):
    import threading
    import fluffy_bassoon.meta_rl.joint_steering_ppo_worker as worker
    durable = threading.Event()
    save = worker._save
    def observed_save(path, value, **kwargs):
        reference = save(path, value, **kwargs)
        if str(path).endswith("episode-000001/checkpoint.pt"):
            durable.set()
        return reference
    class DelayedFailure(Environment):
        def step_chunk(self, chunk):
            if self.case.get("fail_at") == self.steps + 50:
                assert durable.wait(5), "Ready sibling evidence was held behind the failing future"
            return super().step_chunk(chunk)
    monkeypatch.setattr(worker, "_save", observed_save)
    cases = [dict(task="bad", native_seed=1, horizon=100, fail_at=100),
             dict(task="good", native_seed=2, horizon=100)]
    environments = [DelayedFailure(cases, tmp_path, lane) for lane in range(2)]
    with pytest.raises(RuntimeError, match="owned lane failed"):
        collect_bank(policy(), Motor(), environments, request=request(cases, synchronize_initial_lanes=True),
            output=tmp_path / "history", rank=0, deadline_unix=time.time() + 30)
    assert durable.is_set() and (tmp_path / "history/episode-000001/checkpoint.pt").exists()


def test_all_native_lanes_are_closed_even_when_one_close_fails(tmp_path):
    class CloseFailure(Environment):
        def close(self):
            super().close()
            raise RuntimeError("close failed")
    cases = [dict(task="one", native_seed=1, horizon=100)]
    environments = [CloseFailure(cases, tmp_path, 0), Environment(cases, tmp_path, 1)]
    with pytest.raises(RuntimeError, match="Native lane cleanup failed"):
        collect_bank(policy(), Motor(), environments, request=request(cases),
            output=tmp_path / "history", rank=0, deadline_unix=time.time() + 30)
    assert all(environment.closed for environment in environments)
    assert (tmp_path / "history/episode-000000/checkpoint.pt").exists()


def test_recording_environment_releases_native_child_after_encoder_close_failure():
    from fluffy_bassoon.meta_rl.joint_steering_ppo_worker import _RecordingEnvironment
    closed = []
    def fail_encoder():
        raise RuntimeError("encoder close failed")
    env = object.__new__(_RecordingEnvironment)
    env.active = True
    env.writer = SimpleNamespace(close=fail_encoder)
    env.native = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="encoder close failed"):
        env.close()
    assert closed == [True] and env.native is None
