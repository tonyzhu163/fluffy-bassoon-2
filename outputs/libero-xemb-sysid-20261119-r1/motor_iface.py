"""Frozen JAX OpenPI π0.5 motor wrapper and retargeted rollout utilities."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fluffy_bassoon.xemb.embodiment import (
    ACTIVE_PROBE_BURSTS,
    EmbodimentSpec,
    jaw_width_from_qpos,
    retarget_action_from_motor,
    retarget_observation_for_motor,
)

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
    """Convert raw LIBERO observation into the exact OpenPI π0.5 input dictionary."""
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
    """Load the qualified frozen JAX OpenPI π0.5 policy on one allocated GPU."""
    import jax
    from openpi.policies import policy_config
    from openpi.training import config

    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected exactly one allocated GPU for frozen OpenPI π0.5")
    return policy_config.create_trained_policy(config.get_config("pi05_libero"), str(OPENPI_CKPT))


def run_episode(
    simulator,
    policy,
    case: dict,
    spec: EmbodimentSpec,
    *,
    prompt=None,
    yaw_hat: float = 0.0,
    scale_hat: float = 1.0,
    grip_hat: int = 1,
    online_probe_reader=None,
    max_controls=None,
    record_chunks: bool = False,
    max_record_chunks: int = 24,
    check_reset=None,
    audit_base_views: bool = False,
):
    """Execute one episode under `spec` using retargeted frozen π0.5 or active probing."""
    sim_spec, initial = simulator.call("create", dict(case, embodiment=spec.name))
    raw = initial["observation"]
    anchor = np.asarray(initial["anchor_eef_pos"], dtype=np.float32)
    released_prompt = initial["language_instruction"]
    if prompt is not None and prompt != released_prompt:
        raise ValueError(f"Expected paired prompt {prompt!r}, got {released_prompt!r} on {case}")
    prompt = released_prompt
    horizon = int(sim_spec["max_steps"])
    limit = horizon if max_controls is None else min(horizon, int(max_controls))
    bounds = initial["action_schema"]
    low = np.asarray(bounds["low"], dtype=np.float32)
    high = np.asarray(bounds["high"], dtype=np.float32)

    rng = np.random.default_rng(int(case["seed"]) + NOISE_SEED_OFFSET)
    # The released adapter draws one (10, 32) float32 sample per five-control slot.
    noise_bank = rng.standard_normal(((horizon + 4) // 5, 10, 32)).astype(np.float32)
    frame0_img = initial["agentview_64"]
    first_item = to_openpi_observation(raw, prompt)
    frame0_state = first_item["observation/state"].astype(np.float32)
    identity = initial["environment_reproducibility"]
    reset_info = dict(
        physics=identity["initial_state_fingerprint"],
        perturbation_plan=identity["perturbation_plan_fingerprint"],
        public=public_digest(raw),
        motor_frame0={k: array_digest(v) for k, v in first_item.items() if k != "prompt"},
        reader_frame0=dict(image=array_digest(frame0_img), state=array_digest(frame0_state)),
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

    rewards = []
    view_hashes = []
    controls = 0
    done = False
    step = None
    started = time.monotonic()

    cur_img = frame0_img
    cur_pos = np.asarray(raw["robot0_eef_pos"], dtype=np.float32).copy()
    cur_jaw = float(jaw_width_from_qpos(raw["robot0_gripper_qpos"]))

    if record_chunks:
        img_before = np.zeros((max_record_chunks, 64, 64, 3), dtype=np.uint8)
        img_after = np.zeros((max_record_chunks, 64, 64, 3), dtype=np.uint8)
        issued_mean = np.zeros((max_record_chunks, 7), dtype=np.float32)
        delta_eef = np.zeros((max_record_chunks, 3), dtype=np.float32)
        delta_jaw = np.zeros((max_record_chunks, 2), dtype=np.float32)
        valid = np.zeros((max_record_chunks,), dtype=bool)
    chunk_idx = 0

    def execute_chunk(issued_commands: np.ndarray):
        nonlocal raw, controls, done, step, cur_img, cur_pos, cur_jaw, chunk_idx, prompt
        bounded = np.clip(issued_commands, low, high).astype(np.float32)
        step = simulator.call("chunk", bounded)
        n = int(step["steps"])
        controls += n
        rewards.extend(step["rewards"])
        next_img = np.asarray(step["agentview_64"], dtype=np.uint8)
        next_pos = np.asarray(step["positions"][-1], dtype=np.float32)
        next_jaw = float(jaw_width_from_qpos(step["gripper_positions"][-1]))
        if record_chunks and chunk_idx < max_record_chunks:
            img_before[chunk_idx] = cur_img
            img_after[chunk_idx] = next_img
            issued_mean[chunk_idx] = bounded[:n].mean(axis=0)
            delta_eef[chunk_idx] = next_pos - cur_pos
            delta_jaw[chunk_idx] = np.array([next_jaw - cur_jaw, next_jaw], dtype=np.float32)
            valid[chunk_idx] = True
            chunk_idx += 1
        cur_img = next_img
        cur_pos = next_pos
        cur_jaw = next_jaw
        raw = step["observation"]
        prompt = step["language_instruction"]
        done = bool(step["terminated"] or step["truncated"])

    cur_yaw, cur_scale, cur_grip = float(yaw_hat), float(scale_hat), int(grip_hat)
    if online_probe_reader is not None:
        probe_ib, probe_ia, probe_im, probe_de, probe_dj, probe_v = [], [], [], [], [], []
        for burst in ACTIVE_PROBE_BURSTS:
            if done or controls >= limit:
                break
            count = min(len(burst), limit - controls)
            bounded_burst = np.clip(burst[:count], low, high).astype(np.float32)
            prev_img, prev_pos, prev_jaw = cur_img, cur_pos.copy(), cur_jaw
            execute_chunk(bounded_burst)
            probe_ib.append(prev_img)
            probe_ia.append(cur_img)
            probe_im.append(bounded_burst[:step["steps"]].mean(axis=0))
            probe_de.append(cur_pos - prev_pos)
            probe_dj.append(np.array([cur_jaw - prev_jaw, cur_jaw], dtype=np.float32))
            probe_v.append(True)
        while len(probe_v) < len(ACTIVE_PROBE_BURSTS):
            probe_ib.append(np.zeros((64, 64, 3), dtype=np.uint8))
            probe_ia.append(np.zeros((64, 64, 3), dtype=np.uint8))
            probe_im.append(np.zeros((7,), dtype=np.float32))
            probe_de.append(np.zeros((3,), dtype=np.float32))
            probe_dj.append(np.zeros((2,), dtype=np.float32))
            probe_v.append(False)
        probe_trace = dict(
            img_before=np.stack(probe_ib)[None].astype(np.float32) / 255.0,
            img_after=np.stack(probe_ia)[None].astype(np.float32) / 255.0,
            issued_mean=np.stack(probe_im)[None].astype(np.float32),
            delta_eef=np.stack(probe_de)[None].astype(np.float32),
            delta_jaw=np.stack(probe_dj)[None].astype(np.float32),
            valid=np.asarray(probe_v, dtype=bool)[None],
            frame0_img=frame0_img[None].astype(np.float32) / 255.0,
            frame0_state=frame0_state[None],
        )
        cur_yaw, cur_scale, cur_grip = online_probe_reader(probe_trace)

    while not done and controls < limit:
        if policy is None:
            raise RuntimeError("Policy is required when executing closed-loop motor steps")
        if controls == 0:
            item = first_item  # Every anchored yaw correction has zero displacement at reset.
        else:
            motor_obs = raw if cur_yaw == 0 else retarget_observation_for_motor(raw, anchor, cur_yaw)
            item = to_openpi_observation(motor_obs, prompt)
        noise = noise_bank[controls // 5]
        proposal = np.asarray(policy.infer(item, noise=noise)["actions"], dtype=np.float32)
        if proposal.shape != (10, 7) or not np.isfinite(proposal).all():
            raise ValueError(f"Invalid π0.5 proposal on {case} / {spec.name}")
        count = min(5, limit - controls)
        # OpenPI outputs decoded relative OSC controls, including the native gripper sign.
        canonical = np.clip(proposal[:count], low, high)
        if (cur_yaw, cur_scale, cur_grip) == (0.0, 1.0, 1):
            issued = canonical
        else:
            issued = retarget_action_from_motor(canonical, cur_yaw, cur_scale, cur_grip, low, high)
        if audit_base_views:
            view_hashes.append(dict(control=controls, agent=array_digest(item["observation/image"]),
                                    wrist=array_digest(item["observation/wrist_image"]),
                                    state=array_digest(item["observation/state"]), commands=array_digest(issued)))
        execute_chunk(issued)

    record = dict(
        reset=reset_info,
        released_prompt=released_prompt,
        controls=controls,
        native_return=sum(rewards, 0.0),
        success=bool(step["success"]) if step is not None else False,
        terminated=bool(step["terminated"]) if step is not None else False,
        truncated=bool(step["truncated"]) if step is not None else False,
        inferred=dict(yaw_deg=float(cur_yaw), xy_scale=float(cur_scale), gripper_sign=int(cur_grip)),
        seconds=round(time.monotonic() - started, 3),
    )
    if audit_base_views:
        record["motor_queries"] = view_hashes
    if record_chunks:
        trace = dict(
            img_before=img_before,
            img_after=img_after,
            issued_mean=issued_mean,
            delta_eef=delta_eef,
            delta_jaw=delta_jaw,
            valid=valid,
            frame0_img=frame0_img,
            frame0_state=frame0_state,
        )
        return record, trace
    return record
