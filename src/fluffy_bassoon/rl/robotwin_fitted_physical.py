"""Opt-in frozen fitted competence frontend followed by physical4D control.

The endpoint ABI is the existing reactive-curve {kind,config,policy,contract}.
This is a new hybrid mapping, not full-head RL or the old fixed-map A1 result.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

from .pi05_residual import ReactiveKVResidual, DecodedActionResidual
from .robotwin_steering import CoefficientSteering, FixedActionResidual, clipped_coefficients, digest
from .robotwin_wide_action import WideActionResidual


FORMAT = "frozen_fitted_frontend_physical4d_v1"
ARMS = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
DIAGNOSTIC = "fixed12_original_plan_command_capacity_diagnostic_only"


def content_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def state_sha(state):
    from fluffy_bassoon.meta_rl.robotwin_history_corpus import parameter_digest
    return parameter_digest(state)


def validate_fit_provenance(contract, profile_format):
    profile = contract["profile"]
    identities = {f"phone-{i:02d}" for i in range(12)}
    if (profile.get("format") != profile_format
            or profile.get("whole_split") != {f"phone-{i:02d}": "train" if i < 8 else "validation" for i in range(12)}
            or profile.get("primary_endpoint") != 5000 or profile.get("selection") != "fixed_final_endpoint"):
        raise ValueError("Require the declared all12/8TRAIN4VAL final fit/cache/provenance profile")
    if "diagnostic_purpose" in contract:
        outcomes = contract.get("outcome_evidence", {})
        if (contract["diagnostic_purpose"] != DIAGNOSTIC or contract.get("source_eligible") is not False
                or set(outcomes) != identities or not contract.get("command_geometry")
                or outcomes["phone-04"].get("success") is not False):
            raise ValueError("Preserve diagnostic-only status and actual all12 outcomes, including original failed04")
        for row in outcomes.values():
            if (type(row.get("success")) is not bool or type(row.get("primitive_steps")) is not int
                    or not 0 < row["primitive_steps"] <= 400 or type(row.get("chunks")) is not int or row["chunks"] < 1
                    or not row.get("record_id") or not re.fullmatch("[0-9a-f]{40}", row.get("producer_commit", ""))
                    or any(not isinstance(row.get(key), dict) or not row[key].get("path")
                        or not re.fullmatch("[0-9a-f]{64}", row[key].get("sha256", ""))
                        for key in ("producer_result", "episode_manifest_ref"))):
                raise ValueError("Diagnostic source-plan outcomes/producers/canonical references are incomplete")
    elif profile_format != "robotwin_fixed12_reactive_learning_curve_v1":
        raise ValueError("Unknown undeclared fit purpose; do not reinterpret a diagnostic as the original positive-bank fit")


def validate_binding(interface):
    binding = deepcopy(interface.get("fitted_frontend", {}))
    hashes = ("fit_contract_sha256", "fit_profile_sha256", "fit_complete_cache_sha256",
        "head_config_sha256", "head_state_sha256", "motor_sha256",
        "normalization_sha256", "joint_convention_sha256", "base_interface_sha256",
        "physical_basis_sha256", "wrapper_sha256", "backend_sha256", "fixed_control_sha256",
        "frontend_implementation_sha256", "residual_primitives_sha256")
    if (binding.get("format") != FORMAT or binding.get("kind") not in ("kv", "action")
            or binding.get("iteration") != 5000
            or not isinstance(binding.get("fit_profile_format"), str) or not binding["fit_profile_format"]
            or not isinstance(binding.get("fit_provenance"), dict)
            or not re.fullmatch("[0-9a-f]{40}", binding.get("fit_source_commit", ""))
            or any(not re.fullmatch("[0-9a-f]{64}", binding.get(k, "")) for k in hashes)
            or not re.fullmatch("[0-9a-f]{64}", binding.get("checkpoint", {}).get("sha256", ""))
            or not binding.get("checkpoint", {}).get("path")):
        raise ValueError("Require explicit final-endpoint, frontend/state/physical/wrapper provenance")
    if content_sha(binding["fit_provenance"]) != binding["fit_contract_sha256"]:
        raise ValueError("Preserve the endpoint's complete declared fit purpose/outcome contract verbatim")
    validate_fit_provenance(binding["fit_provenance"], binding["fit_profile_format"])
    for key in ("motor_sha256", "normalization_sha256", "joint_convention_sha256"):
        if binding[key] != interface[key]:
            raise ValueError("Fitted frontend differs from the actual physical interface: " + key)
    modules = {"wrapper_sha256": Path(__file__), "backend_sha256": Path(__file__).with_name("robotwin_noise.py"),
        "fixed_control_sha256": Path(__file__).with_name("robotwin_steering.py"),
        "frontend_implementation_sha256": Path(__file__).with_name(
            "robotwin_wide_action.py" if binding["kind"] == "action" else "pi05_residual.py"),
        "residual_primitives_sha256": Path(__file__).with_name("pi05_residual.py")}
    if any(file_sha(path) != binding[key] for key, path in modules.items()):
        raise ValueError("Loaded hybrid/backend/fixed-control code differs from its explicit binding")
    return binding


def load_frontend(binding, *, device):
    """Reconstruct exactly the fit owner's endpoint; never project it to4D."""
    path = Path(binding["checkpoint"]["path"])
    if file_sha(path) != binding["checkpoint"]["sha256"]:
        raise ValueError("Frozen fitted endpoint bytes changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if (saved.get("kind") != binding["kind"] or saved.get("iteration") != binding["iteration"]
            or content_sha(saved["config"]) != binding["head_config_sha256"]
            or content_sha(saved["contract"]) != binding["fit_contract_sha256"]
            or saved["contract"] != binding["fit_provenance"]
            or content_sha(saved["contract"]["profile"]) != binding["fit_profile_sha256"]
            or saved["contract"].get("complete_cache_sha256") != binding["fit_complete_cache_sha256"]
            or saved["contract"].get("source_commit") != binding["fit_source_commit"]
            or state_sha(saved["policy"]) != binding["head_state_sha256"]):
        raise ValueError("Fitted endpoint kind/config/state/source/contract changed")
    validate_fit_provenance(saved["contract"], binding["fit_profile_format"])
    configuration = saved["config"]
    if binding["kind"] == "action":
        head = WideActionResidual.from_config(configuration)
        if any(configuration["source_bindings"][key] != binding[key] for key in
               ("motor_sha256", "normalization_sha256", "joint_convention_sha256", "base_interface_sha256")):
            raise ValueError("Wide head source files/limits differ from the hybrid binding")
    else:
        expected = dict(kind="reactive_native_kv_residual_v1", public_dim=2062, kv_heads=1,
            head_dim=256, rank=4, layers=[-1], scale=.1, layout="B,P,KVH,HD_post_RoPE")
        if configuration != expected:
            raise ValueError("Require the declared full fitted last-layer rank4 KV frontend")
        head = ReactiveKVResidual(**{k: configuration[k] for k in
            ("public_dim", "kv_heads", "head_dim", "rank", "layers", "scale")})
    head.load_state_dict(saved["policy"], strict=True)
    head = head.to(device).eval().requires_grad_(False)
    if state_sha(head.state_dict()) != binding["head_state_sha256"]:
        raise ValueError("Loaded frontend differs from the original fitted state")
    return head


class HeadroomPhysical4D(FixedActionResidual):
    """Unchanged reviewed4x14 basis/tanh map, clipped to remaining arm headroom."""
    def __init__(self, interface):
        super().__init__(interface)
        self.register_buffer("arm_lower", torch.as_tensor(interface["joint_lower"], dtype=torch.float32)[list(ARMS)])
        self.register_buffer("arm_upper", torch.as_tensor(interface["joint_upper"], dtype=torch.float32)[list(ARMS)])

    def config(self):
        return dict(super().config(), kind="post_frontend_headroom_physical4d_v1",
            arm_lower=self.arm_lower.cpu().tolist(), arm_upper=self.arm_upper.cpu().tolist(),
            zero="exact fitted-frontend command bits", headroom="clip12 arms; preserve raw grippers")

    def forward(self, base_actions, public_features):
        arms = base_actions[..., list(ARMS)]
        lower, upper = self.arm_lower.to(arms), self.arm_upper.to(arms)
        if (not torch.equal(lower.to(self.arm_lower), self.arm_lower)
                or not torch.equal(upper.to(self.arm_upper), self.arm_upper)):
            raise ValueError("Command dtype cannot exactly represent reviewed physical arm limits")
        if ((arms < lower).any() or (arms > upper).any()):
            raise ValueError("Fitted frontend arms exceed source limits; zero parity cannot be repaired")
        result = super().forward(base_actions, public_features)
        proposed = result.actions[..., list(ARMS)]
        bounded = torch.where(proposed < lower, lower, torch.where(proposed > upper, upper, proposed))
        actions = result.actions.clone()
        actions[..., list(ARMS)] = bounded
        return DecodedActionResidual(result.base_actions, result.requested_delta,
            actions-result.base_actions, actions != result.actions, actions)


class FittedPhysicalSteering(CoefficientSteering):
    """Same4D/public2076 protocol with an immutable fitted competence baseline."""
    def __init__(self, env, interface_contract, **kwargs):
        self.fitted_binding = validate_binding(interface_contract)
        self.frontend = load_frontend(self.fitted_binding, device=env.backend.model.device)
        contract = env.backend.contract
        if (file_sha(Path(contract["model_path"])/"model.safetensors") != self.fitted_binding["motor_sha256"]
                or file_sha(contract["norm_stats_path"]) != self.fitted_binding["normalization_sha256"]):
            raise ValueError("Actual runtime motor/normalization files differ from the fitted frontend")
        super().__init__(env, interface_contract, **kwargs)
        self.action = HeadroomPhysical4D(self.interface).to(env.backend.model.device).eval()
        if digest(self.physical_basis()) != self.fitted_binding["physical_basis_sha256"]:
            raise ValueError("Actual reviewed physical basis differs from the composed mapping")
        env.backend.install_post_decode_residual(self.action)
        env.backend.post_decode_residual = None  # Native control is explicit until an arm is selected.
        env.backend.contract = dict(env.backend.contract, coefficient_steering=self.config())
        env.contract = env.backend.contract

    def config(self):
        result = super().config()
        result.update(schema=FORMAT, fitted_frontend=deepcopy(self.fitted_binding),
            supported_arms=["native", "action"],
            order="native decode with frozen fitted frontend -> physical4D -> public request -> latent calibration -> native execution",
            zero_coefficients="fitted competence baseline; raw native requires arm=native",
            physical_action="reviewed4x14 arm-only map; no frontend weight optimization")
        return result

    def set_coefficients(self, arm, proposed):
        if arm not in ("native", "action", "decoded_ALOHA_joint_coefficients"):
            raise ValueError("Hybrid frontend exposes physical4D action or explicit raw-native control only")
        self.proposed, self.coefficients = clipped_coefficients(proposed)
        self.arm = "action" if arm != "native" else arm
        if self.arm == "native" and np.any(self.coefficients):
            raise ValueError("Raw-native control requires zero physical coefficients")
        with torch.no_grad():
            self.action.coefficients.copy_(torch.as_tensor(self.coefficients, device=self.action.coefficients.device))
        enabled = self.arm == "action"
        backend = self.env.backend
        backend.kv_adapter = self.frontend if enabled and self.fitted_binding["kind"] == "kv" else None
        backend.action_residual = self.frontend if enabled and self.fitted_binding["kind"] == "action" else None
        backend.post_decode_residual = self.action if enabled else None

    def step(self, proposed):
        self.assert_frontend_frozen()
        result = super().step(proposed)
        self.assert_frontend_frozen()
        return result

    def assert_frontend_frozen(self):
        if (self.frontend.training or any(p.requires_grad for p in self.frontend.parameters())
                or state_sha(self.frontend.state_dict()) != self.fitted_binding["head_state_sha256"]):
            raise RuntimeError("Frozen competence frontend changed during coefficient control")

    def snapshot(self):
        self.assert_frontend_frozen()
        return dict(super().snapshot(), fitted_frontend=deepcopy(self.fitted_binding))


def create_control(env, interface_contract, **kwargs):
    """One explicit factory shared by the new source and student entry paths."""
    return FittedPhysicalSteering(env, interface_contract, **kwargs)
