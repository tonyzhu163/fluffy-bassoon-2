"""DPT-inspired native histories: actual interaction inputs, separate action labels.

The labeler can access the training task. Neither the recurrent input nor the
frozen action expert receives the private rotation. Offline targets use pi0.5's
noise-to-action flow convention, not a surrogate residual-mean teacher.
"""

import json
from pathlib import Path

import torch

from fluffy_bassoon.evaluation_stages import write_json
from .warmstart import NATIVE, atomic_save, checked_load, compatibility


HISTORY_FORMAT = "native_pi05_dpt_history_v1"


@torch.no_grad()
def corrected_action_label(motor, state, native_actions, theta_degrees):
    """TRAINING LABEL ONLY: undo the hidden actuator rotation in decoded XY space."""
    origin = torch.zeros_like(native_actions)
    zero = motor.decode_actions(origin, state)
    columns = []
    for axis in (0, 1):
        basis = origin.clone()
        basis[..., axis] = 1
        columns.append((motor.decode_actions(basis, state) - zero)[..., :2])
    matrix = torch.stack(columns, dim=-1)
    decoded = motor.decode_actions(native_actions, state)
    steps = decoded.shape[1]
    if not 0 < steps <= native_actions.shape[1]:
        raise ValueError("Decoded action horizon must be a nonempty prefix of the model horizon")
    predicted = zero[..., :2] + torch.einsum("bhij,bhj->bhi", matrix, native_actions[:, :steps, :2])
    if not torch.allclose(predicted, decoded[..., :2], atol=1e-5, rtol=1e-4):
        raise ValueError("Action labels require a verified affine XY decoder")
    angles = -torch.as_tensor(theta_degrees, device=decoded.device, dtype=decoded.dtype).deg2rad()
    if angles.shape != (decoded.shape[0],):
        raise ValueError("Require one training-task rotation per lane")
    x, y = decoded[..., 0], decoded[..., 1]
    desired = decoded.clone()
    desired[..., 0] = angles.cos()[:, None] * x - angles.sin()[:, None] * y
    desired[..., 1] = angles.sin()[:, None] * x + angles.cos()[:, None] * y
    target = native_actions.clone()
    # Native pi0.5 emits H=8 model actions but this decoder exposes only H=5.
    # Relabel the executable prefix; retain the unsupervised tail unchanged.
    target[:, :steps, :2] = torch.linalg.solve(matrix, (desired[..., :2] - zero[..., :2]).unsqueeze(-1)).squeeze(-1)
    if not torch.allclose(motor.decode_actions(target, state), desired, atol=1e-5, rtol=1e-4):
        raise ValueError("Inverse action label failed decoding or changed non-XY commands")
    return target


@torch.no_grad()
def make_dpt_record(motor, prepared, feedback, target_actions, *, generator):
    """Cache one independent flow-matching sample per public history position."""
    _, mask, cache = motor.build_prefix_cache(prepared)
    noise = torch.randn(target_actions.shape, device=target_actions.device, dtype=torch.float32, generator=generator)
    time = torch.rand(target_actions.shape[0], device=target_actions.device, generator=generator)
    noisy = (1 - time[:, None, None]) * target_actions + time[:, None, None] * noise
    suffix = motor.run_suffix(prepared, noisy, time, cache, mask)
    native = motor.velocity_from_suffix(suffix)
    def cpu(value):
        return value.detach().to("cpu", copy=True)
    return {key: cpu(value) for key, value in dict(kv_key=cache[-1][0], kv_value=cache[-1][1],
        prefix_mask=mask, feedback=feedback, noisy_actions=noisy, timesteps=time,
        suffix_hidden=suffix, native_velocity=native, target_velocity=noise-target_actions,
        target_actions=target_actions).items()}


def export_history(directory, rollout, *, contract, iteration):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    schema = compatibility(NATIVE, contract)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        format=HISTORY_FORMAT, compatibility=schema, histories=[],
        labels="training-only inverse-rotation of frozen pi0.5 actions; competence measured separately",
        context="actual requested command, terminal reward, boundary and clocks; no private task inputs",
        source_commit=contract["source_commit"])
    if manifest["format"] != HISTORY_FORMAT or manifest["compatibility"] != schema:
        raise ValueError("Cannot mix benchmark interfaces in a history directory")
    filename = f"history-{iteration:06d}.pt"
    if (directory / filename).exists():
        raise ValueError("Teacher histories are immutable; use a fresh output identity")
    data = dict(records=rollout["dpt_records"], supervised_action_steps=schema["action_chunk"],
                **{key: rollout[key] for key in ("rewards", "resets", "attempt_ends", "trial_ends")})
    validate_history(data, schema)
    digest = atomic_save(directory / filename, dict(format=HISTORY_FORMAT, compatibility=schema, data=data))
    manifest["histories"].append(dict(file=filename, sha256=digest, iteration=iteration,
        simulator_steps=rollout["simulator_steps"], policy_chunks=rollout["policy_chunks"],
        additional_label_motor_calls=rollout["policy_chunks"]))
    write_json(manifest_path, manifest)
    return digest


def validate_history(data, schema):
    chunks = schema["attempts"] * (schema["episode_steps"] // schema["action_chunk"])
    ends, resets = data["trial_ends"], data["resets"]
    if (ends.dtype != torch.bool or ends.ndim != 2 or ends.shape[0] != chunks or not ends.shape[1]
            or ends[:-1].any() or not ends[-1].all() or resets.shape != ends.shape
            or resets.dtype != torch.bool or resets.any() or len(data["records"]) != chunks
            or data["rewards"].shape != ends.shape or not data["rewards"].isfinite().all()
            or data["attempt_ends"].shape != ends.shape):
        raise ValueError("Require complete trials with memory retained across attempts")
    expected = {"kv_key", "kv_value", "prefix_mask", "feedback", "noisy_actions", "timesteps",
                "suffix_hidden", "native_velocity", "target_velocity", "target_actions"}
    if type(data.get("supervised_action_steps")) is not int or data["supervised_action_steps"] != schema["action_chunk"]:
        raise ValueError("Supervise only the benchmark's executed action prefix")
    for record in data["records"]:
        if set(record) != expected or any(not isinstance(v, torch.Tensor) or not v.isfinite().all() for v in record.values()):
            raise ValueError("DPT record must contain only the declared finite fields")
        shape = record["noisy_actions"].shape
        if (len(shape) != 3 or shape[0] != ends.shape[1] or not 0 < data["supervised_action_steps"] <= shape[1]
                or record["feedback"].shape != (shape[0], 11)
                or record["timesteps"].shape != (shape[0],)
                or any(record[key].shape != shape for key in ("native_velocity", "target_velocity", "target_actions"))):
            raise ValueError("DPT flow/action shape mismatch")


def load_history(path, expected_sha256, schema):
    saved = checked_load(path, expected_sha256)
    if saved.get("format") != HISTORY_FORMAT or saved.get("compatibility") != schema:
        raise ValueError("DPT history benchmark/interface differs")
    validate_history(saved["data"], schema)
    return saved["data"]


def distillation_loss(policy, data):
    """Full-trial conditional flow matching; target actions never enter the recurrence."""
    policy.eval()
    memory, losses = None, []
    reference = next(policy.parameters())
    for record in data["records"]:
        def saved(key):
            return record[key].detach().to(reference.device)
        features, memory = policy.bridge.read_context(((saved("kv_key"), saved("kv_value")),),
            saved("prefix_mask"), saved("feedback").to(reference.dtype), memory)
        velocity = policy.bridge.correct_velocity(saved("native_velocity"), saved("suffix_hidden"),
            saved("noisy_actions"), saved("timesteps"), features)
        steps = data["supervised_action_steps"]
        losses.append((velocity[:, :steps, :policy.active_dims]
                       - saved("target_velocity")[:, :steps, :policy.active_dims]).square().mean())
    return torch.stack(losses).mean()


def update_distillation(policy, optimizer, data):
    loss = distillation_loss(policy, data)
    if not loss.isfinite():
        raise RuntimeError("Non-finite native flow-matching loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    if policy.bridge.kind == "persistent_ctm":
        with torch.no_grad():
            policy.bridge.core.decay_params_action.clamp_(0, 15)
            policy.bridge.core.decay_params_out.clamp_(0, 15)
    if any(not parameter.isfinite().all() for parameter in policy.parameters()):
        raise RuntimeError("Non-finite native warm-start parameter")
    return dict(loss=float(loss.detach()), gradient_norm=float(norm))
