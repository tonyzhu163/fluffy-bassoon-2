"""Frozen JAX OpenPI pi0.5 motor wrapper and TCP-compensated rollout utilities."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fluffy_bassoon.xemb.tcp import TcpCompensatedController, TcpOffsetSpec

OPENPI_CKPT = Path("/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1")
SPLITS_PATH = Path(__file__).resolve().parents[1] / "libero-meta-learning-20261117-r1" / "SPLITS.json"
NOISE_SEED_OFFSET = 817117


def write_json(path: Path, data):
    """Atomically write JSON with NaN/Inf disallowed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def array_digest(array) -> str:
    arr = np.ascontiguousarray(array)
    prefix = repr((arr.shape, arr.dtype.str)).encode()
    return hashlib.sha256(prefix + arr.tobytes()).hexdigest()


def public_digest(raw: dict) -> dict:
    return {k: array_digest(v) for k, v in raw.items()}


def to_openpi_observation(raw: dict, prompt: str) -> dict:
    """Convert raw LIBERO observation into the exact OpenPI pi0.5 input dictionary."""
    from openpi_client import image_tools

    rgb = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(raw[key][::-1, ::-1]), 224, 224)
        )
        for key in ("agentview_image", "robot0_eye_in_hand_image")
    ]
    # Preserve reference.observation's dtype and operation order through OpenPI preprocessing.
    quat = raw["robot0_eef_quat_xyzw"]
    w = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - w * w)
    rotation = quat[:3] * 2.0 * np.arccos(w) / den if den > 1e-10 else np.zeros(3)
    state = np.concatenate((raw["robot0_eef_pos"], rotation, raw["robot0_gripper_qpos"]))
    return {
        "observation/image": rgb[0],
        "observation/wrist_image": rgb[1],
        "observation/state": state,
        "prompt": prompt,
    }


def load_frozen_pi05():
    """Load the qualified frozen JAX OpenPI pi0.5 policy on one allocated GPU."""
    import jax
    from openpi.policies import policy_config
    from openpi.training import config

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected exactly one allocated GPU for frozen OpenPI pi0.5")
    return policy_config.create_trained_policy(config.get_config("pi05_libero"), str(OPENPI_CKPT))


def run_tcp_episode(
    simulator,
    policy,
    case: dict,
    spec: TcpOffsetSpec,
    *,
    prompt=None,
    dx_hat_m: float = 0.0,
    dy_hat_m: float = 0.0,
    compensate_release: bool = False,
    max_controls=None,
    check_reset=None,
):
    """Execute one episode under physical TCP offset `spec` with optional TCP compensation."""
    sim_spec, initial = simulator.call(
        "create",
        dict(case, dx_m=spec.dx_m, dy_m=spec.dy_m, dz_m=spec.dz_m),
    )
    raw = initial["observation"]
    released_prompt = initial["language_instruction"]
    if prompt is not None and prompt != released_prompt:
        raise ValueError(f"Expected paired prompt {prompt!r}, got {released_prompt!r} on {case}")
    cur_prompt = released_prompt
    horizon = int(sim_spec["max_steps"])
    limit = horizon if max_controls is None else min(horizon, int(max_controls))
    bounds = initial["action_schema"]
    low = np.asarray(bounds["low"], dtype=np.float32)
    high = np.asarray(bounds["high"], dtype=np.float32)

    rng = np.random.default_rng(int(case["seed"]) + NOISE_SEED_OFFSET)
    noise_bank = rng.standard_normal((horizon, 10, 32)).astype(np.float32)
    first_item = to_openpi_observation(raw, cur_prompt)
    identity = initial["environment_reproducibility"]
    reset_info = dict(
        physics=identity["initial_state_fingerprint"],
        perturbation_plan=identity["perturbation_plan_fingerprint"],
        public=public_digest(raw),
        motor_frame0={k: array_digest(v) for k, v in first_item.items() if k != "prompt"},
        worker_step_index=initial["worker_step_index"],
        action_schema=initial["action_schema"],
        observation_schema=initial["observation_schema"],
        observation_schema_version=initial["observation_schema_version"],
        renderer=initial["renderer"],
        renderer_vendor=initial["renderer_vendor"],
        noise_seed=int(case["seed"]) + NOISE_SEED_OFFSET,
        noise_sha256=array_digest(noise_bank),
        horizon=horizon,
        prompt=released_prompt,
    )
    if check_reset is not None:
        check_reset(reset_info)

    ctrl = TcpCompensatedController(
        raw["robot0_eef_quat_xyzw"],
        dx_hat_m,
        dy_hat_m,
        compensate_release=compensate_release,
    )

    rewards = []
    opening_proposal_sha256 = None
    controls = 0
    query_idx = 0
    done = False
    final_success = False
    terminated = False
    truncated = False
    started = time.monotonic()

    if not ctrl.active:
        while not done and controls < limit:
            item = first_item if controls == 0 else to_openpi_observation(raw, cur_prompt)
            proposal = np.asarray(policy.infer(item, noise=noise_bank[query_idx])["actions"], dtype=np.float32)
            if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
                raise ValueError(f"Invalid pi0.5 proposal on {case} / {spec.name}")
            if controls == 0:
                opening_proposal_sha256 = array_digest(proposal)
            query_idx += 1
            count = min(5, limit - controls)
            issued = np.clip(proposal[:count], low, high).astype(np.float32)
            step = simulator.call("chunk", issued)
            controls += int(step["steps"])
            rewards.extend(step["rewards"])
            raw = step["observation"]
            cur_prompt = step["language_instruction"]
            final_success = bool(step["success"])
            terminated = bool(step["terminated"])
            truncated = bool(step["truncated"])
            done = terminated or truncated
    else:
        proposal_queue = []
        has_full_rgb = True
        while not done and controls < limit:
            if ctrl.needs_motor_proposal:
                if not proposal_queue:
                    if not has_full_rgb:
                        obs_msg = simulator.call("observe")
                        raw = obs_msg["observation"]
                        cur_prompt = obs_msg["language_instruction"]
                        has_full_rgb = True
                    item = first_item if controls == 0 else to_openpi_observation(raw, cur_prompt)
                    proposal = np.asarray(
                        policy.infer(item, noise=noise_bank[query_idx])["actions"],
                        dtype=np.float32,
                    )
                    if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
                        raise ValueError(f"Invalid pi0.5 proposal on {case} / {spec.name}")
                    if controls == 0:
                        opening_proposal_sha256 = array_digest(proposal)
                    query_idx += 1
                    proposal_queue = [np.clip(proposal[i], low, high).astype(np.float32) for i in range(5)]
                prop_step = proposal_queue.pop(0)
                action = ctrl.next_action(
                    raw["robot0_eef_pos"],
                    raw["robot0_eef_quat_xyzw"],
                    raw["robot0_gripper_qpos"],
                    prop_step,
                )
                if not ctrl.needs_motor_proposal:
                    proposal_queue.clear()
            else:
                action = ctrl.next_action(
                    raw["robot0_eef_pos"],
                    raw["robot0_eef_quat_xyzw"],
                    raw["robot0_gripper_qpos"],
                )
            need_rgb = ctrl.needs_motor_proposal and len(proposal_queue) == 0 and (controls + 1 < limit)
            bounded = np.clip(action, low, high).astype(np.float32)
            step = simulator.call("step", dict(action=bounded, need_rgb=need_rgb, need_64=False))
            controls += 1
            rewards.append(float(step["reward"]))
            raw = step["observation"]
            has_full_rgb = bool(step["has_full_rgb"])
            cur_prompt = step["language_instruction"]
            final_success = bool(step["success"])
            terminated = bool(step["terminated"])
            truncated = bool(step["truncated"])
            done = terminated or truncated

    return dict(
        reset=reset_info,
        shift_meta=initial["shift_meta"],
        released_prompt=released_prompt,
        controls=controls,
        queries=query_idx,
        grasp_triggers=ctrl.grasp_trigger_count,
        release_triggers=ctrl.release_trigger_count,
        native_rewards=rewards,
        native_return=float(sum(rewards)),
        opening_proposal_sha256=opening_proposal_sha256,
        success=bool(final_success),
        terminated=bool(terminated),
        truncated=bool(truncated),
        estimate=dict(
            dx_hat_m=float(dx_hat_m),
            dy_hat_m=float(dy_hat_m),
            compensate_release=bool(compensate_release),
        ),
        seconds=round(time.monotonic() - started, 3),
    )
