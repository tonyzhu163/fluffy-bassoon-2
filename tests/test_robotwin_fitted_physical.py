"""Prepared allocated-Linux CPU fixtures; no frontend/data admission implied."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import pytest
import torch

from fluffy_bassoon.rl import robotwin_fitted_physical as hybrid
from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual, ReactiveALOHAActionResidual
from fluffy_bassoon.rl.robotwin_noise import _RoboTwinBackend
from fluffy_bassoon.rl.robotwin_steering import CoefficientSteering, FixedActionResidual, validate_interface
from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual
from fluffy_bassoon.rl.steering_actuator_condition import CalibrationDistribution, calibration_pair
from fluffy_bassoon.meta_rl import robotwin_fitted_source as entry
from test_robotwin_steering import interface

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Allocated Linux CPU only")


def fixture_interface(tmp_path, interface, kind):
    value = deepcopy(interface)
    value.update(head_dim=256, motor_sha256=hybrid.file_sha(write(tmp_path/"model.safetensors", b"synthetic motor")),
        normalization_sha256=hybrid.file_sha(write(tmp_path/"norm.json", b"synthetic normalization")))
    if kind == "action":
        head = WideActionResidual(joint_lower=value["joint_lower"], joint_upper=value["joint_upper"],
            motor_sha256=value["motor_sha256"], normalization_sha256=value["normalization_sha256"],
            joint_convention_sha256=value["joint_convention_sha256"], base_interface_sha256="f"*64)
        with torch.no_grad():
            head.output.bias.fill_(.1)
    else:
        head = ReactiveKVResidual(public_dim=2062, kv_heads=1, head_dim=256, rank=4, layers=(-1,), scale=.1)
        with torch.no_grad():
            head.token_projections[0].bias.fill_(.2)
            head.output_bases[0].weight.fill_(.01)
    examples = json.loads((Path(__file__).resolve().parents[1]/
        "outputs/fitted-physical-bridge-20261101-r1/PRODUCER-PROFILES.json").read_text())
    profile = examples["original_profile"]
    contract = dict(source_commit="d"*40, profile=profile, complete_cache_sha256="e"*64, synthetic_test_only=True)
    endpoint = tmp_path/"endpoint-005000.pt"
    torch.save(dict(kind=kind, iteration=5000, config=head.config(), policy=head.state_dict(),
        contract=contract, diagnostics=dict(train={}, validation={})), endpoint)
    code = Path(hybrid.__file__)
    physical = FixedActionResidual(value)
    value["fitted_frontend"] = dict(format=hybrid.FORMAT, kind=kind, iteration=5000,
        checkpoint=dict(path=str(endpoint), sha256=hybrid.file_sha(endpoint)), fit_source_commit="d"*40,
        fit_profile_format=profile["format"], fit_provenance=deepcopy(contract),
        fit_contract_sha256=hybrid.content_sha(contract), fit_profile_sha256=hybrid.content_sha(profile),
        fit_complete_cache_sha256="e"*64, head_config_sha256=hybrid.content_sha(head.config()),
        head_state_sha256=hybrid.state_sha(head.state_dict()), motor_sha256=value["motor_sha256"],
        normalization_sha256=value["normalization_sha256"], joint_convention_sha256=value["joint_convention_sha256"],
        base_interface_sha256="f"*64, physical_basis_sha256=physical.basis_sha256,
        wrapper_sha256=hybrid.file_sha(code), backend_sha256=hybrid.file_sha(code.with_name("robotwin_noise.py")),
        fixed_control_sha256=hybrid.file_sha(code.with_name("robotwin_steering.py")),
        frontend_implementation_sha256=hybrid.file_sha(code.with_name("robotwin_wide_action.py" if kind == "action" else "pi05_residual.py")),
        residual_primitives_sha256=hybrid.file_sha(code.with_name("pi05_residual.py")))
    return validate_interface(value)


def write(path, data):
    path.write_bytes(data)
    return path


class TinyMotor(torch.nn.Module):
    device, num_steps = "cpu", 5
    def __init__(self):
        super().__init__()

    def _sample_actions_from_prefix_cache(self, prepared, mask, cache, *, noise):
        offset = cache[-1][0][mask].mean()
        result = torch.zeros_like(noise) + offset
        result[..., 6] = 1.2 + offset
        result[..., 13] = -.0
        return result
    def decode_actions(self, targets, state):
        return targets[..., :14]


class Env:
    noise_shape, observation_dim, native_horizon = (32, 32), 2062, 32
    corpus_writer, active, elapsed_steps = None, False, 0
    def __init__(self, tmp_path):
        backend = object.__new__(_RoboTwinBackend)
        backend.torch, backend.model = torch, TinyMotor().eval().requires_grad_(False)
        backend._versions = {name: parameter._version for name, parameter in backend.model.named_parameters()}
        backend.noise_shape = tuple(self.noise_shape)
        backend.native_horizon = backend.noise_shape[0]
        backend.corpus_writer = None
        backend._decision_metadata = {}
        backend.contract = dict(model_path=str(tmp_path), norm_stats_path=str(tmp_path/"norm.json"),
            noise_shape=list(backend.noise_shape), native_horizon=backend.native_horizon,
            observation_dim=self.observation_dim)
        backend._prepared = backend._prefix_mask = backend._kv_cache = None
        backend._public_features = backend._last_residual_metadata = backend._residual_prepared = None
        backend._raw_obs = backend._paired_instruction = backend._instruction_binding = None
        backend.kv_adapter = backend.action_residual = backend.post_decode_residual = None
        backend.post_command_transform, backend.record_agent_requested_actions = None, False
        backend._post_decode_installed = False
        backend.prefix_seconds = backend.motor_seconds = 0.
        backend.prefix_calls = backend.motor_queries = 0
        self.backend, self.contract = backend, backend.contract
    def reset(self, task, seed):
        backend = self.backend
        backend._prepared = SimpleNamespace(state=torch.zeros(1, 14))
        backend._residual_prepared = backend._prepared
        backend._prefix_mask = torch.tensor([[True, True, False]])
        backend._kv_cache = tuple((torch.zeros(1, 3, 1, 256), torch.zeros(1, 3, 1, 256)) for _ in range(2))
        backend._public_features = torch.zeros(1, 2062)
        self.last_reset_info = dict(task=task, actual_seed=seed)
        return np.zeros(2062, np.float32)
    def record_decision_metadata(self, metadata):
        self.metadata = metadata
    def step(self, noise):
        self.backend.decode(noise)
        public = self.backend._last_residual_metadata["public_agent_requested_actions"].numpy()[0, -1].copy()
        return np.zeros(2062, np.float32), 0., False, False, dict(last_valid_agent_request=public)


@pytest.mark.parametrize("kind", ["kv", "action"])
def test_zero_is_fitted_baseline_raw_native_is_separate_and_nonzero_preserves_frozen_hook(tmp_path, interface, monkeypatch, kind):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    configured = fixture_interface(tmp_path, interface, kind)
    env = Env(tmp_path)
    control = hybrid.create_control(env, configured, interface_sha256="c"*64)
    frontend = control.frontend
    control.set_coefficients("action", np.zeros(4))
    control.reset("place_phone_stand", 1, motor_seed=2)
    noise = np.zeros((32, 32), np.float32)
    env.backend.post_decode_residual = None
    fitted = env.backend.decode(noise)
    env.backend.post_decode_residual = control.action
    zero = env.backend.decode(noise)
    np.testing.assert_array_equal(zero.view(np.uint32), fitted.view(np.uint32))
    control.set_coefficients("native", np.zeros(4))
    native = env.backend.decode(noise)
    assert not np.array_equal(native, zero)
    control.set_coefficients("action", np.zeros(4))
    assert (env.backend.kv_adapter if kind == "kv" else env.backend.action_residual) is frontend
    control.step(np.array([1., -.5, .3, -.2], np.float32))
    control.assert_frontend_frozen()
    assert not frontend.training and all(not p.requires_grad for p in frontend.parameters())
    assert hybrid.state_sha(frontend.state_dict()) == configured["fitted_frontend"]["head_state_sha256"]
    assert env.metadata["steering_coefficients"].shape == (4,)
    with torch.no_grad():
        next(frontend.parameters()).flatten()[0].add_(1)
    with pytest.raises(RuntimeError, match="changed"):
        control.assert_frontend_frozen()


def test_physical4d_uses_exact_basis_and_headroom_without_gripper_or_zero_bit_changes(interface):
    head = hybrid.HeadroomPhysical4D(interface).eval()
    assert head.basis_sha256 == FixedActionResidual(interface).basis_sha256
    base = torch.zeros(1, 32, 14)
    base[..., list(hybrid.ARMS)] = 2.9999
    base[..., 6], base[..., 13] = -0., 1.3
    features = torch.zeros(1, 2062)
    with torch.no_grad():
        zero = head(base, features).actions
        assert torch.equal(zero.view(torch.int32), base.view(torch.int32))
        head.coefficients.fill_(1)
        result = head(base, features)
    assert torch.equal(result.actions[..., [6, 13]].view(torch.int32), base[..., [6, 13]].view(torch.int32))
    assert result.requested_delta.abs().max() <= interface["arm_joint_bound"]
    assert result.applied_delta.abs().max() <= interface["arm_joint_bound"]+1e-6
    assert (result.actions[..., list(hybrid.ARMS)] <= 3).all() and result.clipped.any()


def test_recorded_request_and_public2076_are_after_frontend_physical_before_hidden_calibration(tmp_path, interface, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    configured = fixture_interface(tmp_path, interface, "action")
    env = Env(tmp_path)
    control = hybrid.create_control(env, configured)
    control.set_coefficients("action", np.zeros(4))
    control.reset("place_phone_stand", 1, motor_seed=2)
    def condition(base, requested):
        final = requested.clone(); final[..., 0] += .01
        return final, dict(condition_id="audit-only")
    env.backend.post_command_transform = condition
    control.step(np.array([.4, 0., 0., 0.], np.float32))
    record = env.backend._last_residual_metadata
    request = record["public_agent_requested_actions"]
    torch.testing.assert_close(request, record["post_decode_physical"]["actions"], rtol=0, atol=0)
    final = record["audit_only_post_command_transform"]["actuator_commands"]
    torch.testing.assert_close(final[..., 0], request[..., 0]+.01)
    assert not torch.equal(final[..., 0], request[..., 0])
    obs = control.student_observation(np.zeros(2062, np.float32))
    assert obs.shape == (2076,)
    np.testing.assert_array_equal(obs[-14:], request.numpy()[0, -1])
    np.testing.assert_array_equal(env.metadata["student_previous_agent_command"], np.zeros(14))
    control.reset("place_phone_stand", 1, motor_seed=2)
    np.testing.assert_array_equal(control.student_observation(np.zeros(2062))[-14:], np.zeros(14))


def test_source_and_readout_use_one_factory_and_identical_mapping_including_calibration(tmp_path, interface, monkeypatch):
    configured = fixture_interface(tmp_path, interface, "action")
    source = hybrid.create_control(Env(tmp_path), configured, interface_sha256="c"*64)
    source.set_coefficients("action", np.zeros(4))
    pair = calibration_pair(physical_basis=source.physical_basis(), physical_basis_sha256=source.action.basis_sha256,
        arm_joint_bound=interface["arm_joint_bound"], selected_arm="action",
        command_lower=interface["joint_lower"], command_upper=interface["joint_upper"],
        range_receipt_sha256=interface["joint_convention_sha256"], motor_sha256=configured["motor_sha256"],
        mapping_sha256=source.mapping_sha256())
    distribution = CalibrationDistribution(pair)
    source.condition_contract = distribution.contract()
    readout = hybrid.create_control(Env(tmp_path), configured, post_command_transform=distribution,
        condition_contract=distribution.contract(), interface_sha256="c"*64)
    readout.set_coefficients("action", np.zeros(4))
    assert source.mapping_sha256() == readout.mapping_sha256()
    assert source.fitted_binding == readout.fitted_binding
    legacy = CoefficientSteering(Env(tmp_path), configured, interface_sha256="c"*64)
    legacy.set_coefficients("action", np.zeros(4))
    assert legacy.mapping_sha256() != source.mapping_sha256()  # Default cannot silently deploy this mapping.
    from fluffy_bassoon.meta_rl import robotwin_source_parallel, steering_history_readout
    calls = []
    monkeypatch.setattr(robotwin_source_parallel, "main", lambda argv, **kw: calls.append(kw))
    monkeypatch.setattr(steering_history_readout, "main", lambda argv, **kw: calls.append(kw))
    entry.main(["source"]); entry.main(["readout"])
    assert calls[0]["control_factory"] is calls[1]["control_factory"] is hybrid.create_control
    assert calls[0]["selection_validator"] is entry.validate_selection
    from fluffy_bassoon.meta_rl import steering_history
    marker = dict(format="fitted_physical_control_qualification_binding_v1")
    path = tmp_path/"proof-binding.json"
    path.write_text(json.dumps(marker))
    monkeypatch.setattr(entry, "verify_control_qualification", lambda binding, profile: dict(new_proof=True))
    assert steering_history.verify_control_qualification(dict(path=str(path), sha256=hybrid.file_sha(path)), {}) == dict(new_proof=True)
    with pytest.raises(ValueError, match="A2 condition-cross evidence is pending"):
        steering_history.verify_condition_cross({}, {}, objective="ad")


def test_changed_code_checkpoint_state_and_old_A1_cannot_be_admitted(tmp_path, interface):
    configured = fixture_interface(tmp_path, interface, "kv")
    wrong = deepcopy(configured)
    wrong["fitted_frontend"]["wrapper_sha256"] = "0"*64
    with pytest.raises(ValueError, match="Loaded hybrid"):
        hybrid.validate_binding(wrong)
    from test_steering_condition_collection import profile
    with pytest.raises(ValueError, match="old fixed-map"):
        entry.validate_selection(dict(schema="frozen_motus_source_selection_v1"), {}, {}, profile(), configured,
            interface_sha=profile().interface_sha256, selection_rule={})
    binding = configured["fitted_frontend"]
    endpoint = Path(binding["checkpoint"]["path"])
    endpoint.write_bytes(endpoint.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        hybrid.load_frontend(binding, device="cpu")


def test_canonical_intent_records_public_physical_request_and_actual_conditioned_commands(interface, monkeypatch):
    from test_pi05_residual import backend
    instance = backend(monkeypatch)
    instance._public_features = torch.zeros(1, 2062)
    instance.record_agent_requested_actions = True
    frontend = ReactiveALOHAActionResidual(public_dim=2062, arm_joint_bound=.02, gripper_bound=.003).eval()
    with torch.no_grad():
        frontend.output.bias.fill_(.5)
    instance.action_residual = frontend
    physical = hybrid.HeadroomPhysical4D(interface).eval()
    physical.coefficients[0] = .4
    instance.post_decode_residual = physical
    def condition(base, requested):
        final = requested.clone(); final[..., 0] += .01
        return final, dict(condition_id="hidden-fixture")
    instance.post_command_transform = condition
    instance._decision_metadata = dict(steering_coefficients=np.array([.4, 0., 0., 0.], np.float32))
    captured = {}
    raw = dict(main_images=np.zeros((1, 2, 2, 3), np.uint8),
        wrist_images=np.zeros((1, 2, 2, 2, 3), np.uint8), states=np.zeros((1, 14)), task_descriptions=["fixture"])
    def step(commands):
        captured["commands"] = commands.clone()
        return [raw], torch.ones(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), [
            dict(valid_action_mask=torch.tensor([True, False, True]), episode=dict(success_once=True))]
    instance.env = SimpleNamespace(chunk_step=step, venv=SimpleNamespace(envs=[SimpleNamespace(task=object())]))
    instance._encode = lambda observation: np.zeros(2062, np.float32)
    instance.corpus_writer = SimpleNamespace(
        begin_decision=lambda noise, actions, metadata: captured.update(issued=actions.copy(), metadata=metadata),
        complete_native=lambda **values: captured.update(native=values))
    monkeypatch.setattr("fluffy_bassoon.rl.robotwin_noise._capture_sparse_feedback", lambda task: nullcontext({}))
    *_, info = instance.step(np.zeros((3, 32), np.float32), 3)
    record = captured["metadata"]["frozen_pi05_residual"]
    requested = record["public_agent_requested_actions"].numpy()[0]
    np.testing.assert_array_equal(requested, record["post_decode_physical"]["actions"].numpy()[0])
    np.testing.assert_array_equal(captured["issued"], captured["commands"].numpy()[0])
    np.testing.assert_array_equal(captured["issued"], record["audit_only_post_command_transform"]["actuator_commands"].numpy()[0])
    np.testing.assert_array_equal(info["last_valid_agent_request"], requested[2])
    assert not np.array_equal(captured["issued"][:, 0], requested[:, 0])
    assert captured["metadata"]["steering_coefficients"].shape == (4,)
    assert info["primitive_steps"] == 2 and captured["native"]["executed_steps"] == 2


def test_diagnostic_outcomes_remain_verbatim_and_missing_twelfth_identity_is_rejected(tmp_path, interface):
    configured = fixture_interface(tmp_path, interface, "kv")
    binding = configured["fitted_frontend"]
    endpoint = Path(binding["checkpoint"]["path"])
    saved = torch.load(endpoint, map_location="cpu", weights_only=False)
    examples = json.loads((Path(__file__).resolve().parents[1]/
        "outputs/fitted-physical-bridge-20261101-r1/PRODUCER-PROFILES.json").read_text())
    saved["contract"]["profile"] = examples["diagnostic_profile"]
    binding["fit_profile_format"] = examples["diagnostic_profile"]["format"]
    saved["contract"].update(diagnostic_purpose=hybrid.DIAGNOSTIC, source_eligible=False,
        command_geometry=dict(synthetic_test_only=True), outcome_evidence={f"phone-{i:02d}": dict(
            success=i != 4, primitive_steps=1, chunks=1, record_id=f"synthetic-{i}", producer_commit="b"*40,
            producer_result=dict(path="synthetic-only.json", sha256="a"*64),
            episode_manifest_ref=dict(path="synthetic-only-manifest.json", sha256="b"*64)) for i in range(12)})
    def publish_fixture():
        torch.save(saved, endpoint)
        binding.update(fit_provenance=deepcopy(saved["contract"]),
            fit_contract_sha256=hybrid.content_sha(saved["contract"]),
            fit_profile_sha256=hybrid.content_sha(saved["contract"]["profile"]))
        binding["checkpoint"]["sha256"] = hybrid.file_sha(endpoint)
    publish_fixture()
    hybrid.load_frontend(binding, device="cpu")
    assert binding["fit_provenance"]["source_eligible"] is False
    assert binding["fit_provenance"]["outcome_evidence"]["phone-04"]["success"] is False
    wrong = deepcopy(configured)
    wrong["fitted_frontend"]["fit_provenance"]["source_eligible"] = True
    with pytest.raises(ValueError, match="verbatim"):
        hybrid.validate_binding(wrong)
    complete = deepcopy(saved)
    del saved["contract"]["profile"]["whole_split"]["phone-04"]
    publish_fixture()
    with pytest.raises(ValueError, match="all12"):
        hybrid.load_frontend(binding, device="cpu")
    saved.clear(); saved.update(complete)
    del saved["contract"]["outcome_evidence"]["phone-04"]
    publish_fixture()
    with pytest.raises(ValueError, match="all12"):
        hybrid.load_frontend(binding, device="cpu")
