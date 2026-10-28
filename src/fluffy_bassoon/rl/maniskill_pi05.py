"""Frozen RLinf pi0.5 on ManiSkill PutOnPlateInScene25Main-v3: shared gate/cache pieces.

Model and environment settings follow RLinf's own pi05_maniskill evaluation. Norm
stats are always passed explicitly: the SFT and RL checkpoints store them at
different relative paths, and RLinf silently skips normalization when its default
lookup misses. The rotation hook implements docs/maniskill-pi05-icrl.md.
"""
import hashlib
import math
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import numpy as np
import torch


TRAIN_OBJECTS, TRAIN_OVERLAYS, OBJECTS = 16, 16, 25
CHUNK, STEPS = 5, 80
# The WidowX agent's only mode. RLinf's YAML says `None`, which loads as the string "None".
CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"


def execution_mode():
    from fluffy_bassoon.gpu_execution import validate_execution
    mode = os.environ.get("FLUFFY_GPU_EXECUTION", "slurm")
    if mode not in ("slurm", "baige"):
        raise ValueError("ManiSkill requires an allocated Slurm step or exclusive Baige container")
    validate_execution(mode, os.environ)
    return mode


def prepare_gpu(root, output):
    mode = execution_mode()
    command = [sys.executable, str(root / "scripts/[train]gpu_probe.py"),
               "--expected-count", "1", "--output", str(output / "gpu-probe.json"),
               "--purpose", "evaluation" if mode == "baige" else "training"]
    if mode == "baige":
        command.append("--baige-container")
    subprocess.run(command, check=True, timeout=90)
    if mode == "baige":
        os.environ['LYCO_BAIGE_GPU_MAP'] = str((output/'gpu-probe.json').resolve())
    return mode


def job_id():
    return socket.gethostname().removesuffix("-master-0") if execution_mode() == "baige" else os.environ["SLURM_JOB_ID"]


def model_fingerprint(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def record_renderer_binding(env, output):
    """Read SAPIEN's actual graphics devices, rather than inferring from a PID."""
    from fluffy_bassoon.baige_gpu import cuda_pci, pci
    expected = cuda_pci(0)
    devices = []
    for scene in env.env.unwrapped.scene.sub_scenes:
        device = scene.render_system.device
        observed = pci(device.pci_string.removeprefix('pci:'))
        if observed != expected or not device.can_render():
            raise RuntimeError('SAPIEN renderer differs from the allocated CUDA PCI device')
        devices.append(dict(name=device.name, pci_bus_id=observed, cuda_id=device.cuda_id))
    simulation_device = torch.device(env.device)
    if not devices or simulation_device.type != 'cuda' or simulation_device.index not in (None, 0):
        raise RuntimeError('Require native CUDA simulation and actual graphics contexts')
    record = dict(job_id=job_id(), pid=os.getpid(), cuda_pci_bus_id=expected,
                  simulation_device=str(env.device), actual_sapien_render_devices=devices)
    (output/'renderer-binding.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def model_config(model_path, norm_stats_path):
    from pathlib import Path
    from omegaconf import OmegaConf
    from .rlinf_checkpoint_compat import checkpoint_has_noise_head, checkpoint_has_value_head
    add_value_head = checkpoint_has_value_head(Path(model_path)/"model.safetensors")
    noise_method = ("flow_noise" if checkpoint_has_noise_head(Path(model_path)/"model.safetensors")
                    else "flow_ode")
    return OmegaConf.create(dict(
        model_type="openpi", model_path=str(model_path), precision=None, pi05=True,
        is_lora=False, num_action_chunks=CHUNK, action_dim=7, use_proprio=True, num_steps=4,
        add_value_head=add_value_head, policy_setup="widowx_bridge",
        openpi_data=dict(norm_stats_path=str(norm_stats_path)),
        openpi=dict(task="rl", config_name="pi05_maniskill", num_images_in_input=1,
                    noise_level=.5, noise_method=noise_method, noise_params=[.16, .12, 200],
                    joint_logprob=True, action_chunk=CHUNK, num_steps=4, train_expert_only=True,
                    action_env_dim=7, add_value_head=add_value_head, value_after_vlm=True,
                    value_vlm_mode="mean_token", detach_critic_input=False, model_action_dim=32,
                    paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                    max_token_len=200, action_horizon=8, torch_compile=False)))


def env_config(rlinf_root, num_envs, seed):
    from pathlib import Path
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(rlinf_root)/"examples/embodiment/config/env/maniskill_put_on_plate_in_scene_25_main.yaml")
    cfg = OmegaConf.merge(cfg, dict(
        total_num_envs=num_envs, auto_reset=False, ignore_terminations=True, seed=seed,
        use_fixed_reset_state_ids=True, group_size=1, max_episode_steps=STEPS,
        max_steps_per_rollout_epoch=STEPS,
        video_cfg=dict(save_video=False, info_on_video=False, video_base_dir="unused"),
        init_params=dict(obj_set="all", use_multiple_plates=False, max_episode_steps=STEPS,
                         control_mode=CONTROL_MODE)))
    if os.environ.get('FLUFFY_GPU_EXECUTION') == 'baige':
        from fluffy_bassoon.baige_gpu import selected_device
        cfg.init_params.render_backend = 'pci:' + selected_device()['pci_bus_id']
    return cfg


def check_object_infos(infos):
    lc, _, lo, _, lp, _, l1, l2 = infos
    if (lc, lo, lp) != (OBJECTS, 21, 1):
        raise ValueError(f"Unexpected object/overlay/plate counts {(lc, lo, lp)}")
    return lo, lp, l1, l2


def episode_id(infos, obj, overlay, position, orientation):
    """RLinf Main-v3 id: ((object*plates)*overlays + overlay)*positions*orientations + ..."""
    lo, lp, l1, l2 = check_object_infos(infos)
    if not (0 <= obj < OBJECTS and 0 <= overlay < lo and 0 <= position < l1 and 0 <= orientation < l2):
        raise ValueError("Scene index out of range")
    return ((obj*lp)*lo + overlay)*l1*l2 + position*l2 + orientation


def sample_scenes(infos, objects, per_object, rng):
    """Training-range overlay, position and orientation for each object."""
    _, _, l1, l2 = check_object_infos(infos)
    scenes = []
    for obj in objects:
        for _ in range(per_object):
            overlay, pos, quat = int(rng.integers(TRAIN_OVERLAYS)), int(rng.integers(l1)), int(rng.integers(l2))
            scenes.append(dict(object=int(obj), overlay=overlay, position=pos, orientation=quat,
                               episode_id=episode_id(infos, int(obj), overlay, pos, quat)))
    return scenes


def rotate_translation(chunk, angles_degrees):
    """Rotate each env's xy translation command about the vertical axis.

    chunk: [envs, steps, 7] RLinf-prepared ManiSkill actions (world_vector, rot_axangle,
    gripper). Only world_vector[:2] changes; rotations about one axis compose additively.
    """
    angles = torch.as_tensor(angles_degrees, dtype=torch.float64)
    if (chunk.ndim != 3 or chunk.shape[-1] != 7 or angles.shape != (chunk.shape[0],)
            or not torch.isfinite(angles).all()):
        raise ValueError("Require chunk[envs,steps,7] and one finite angle per env")
    radians = angles * (math.pi / 180)
    cos, sin = radians.cos().to(chunk)[:, None], radians.sin().to(chunk)[:, None]
    x, y = chunk[..., 0], chunk[..., 1]
    rotated = chunk.clone()
    rotated[..., 0] = cos*x - sin*y
    rotated[..., 1] = sin*x + cos*y
    return rotated


OUTCOME_FIELDS = dict(
    success_once="success at any step", success_at_end="success at step 80",
    first_success_step="first step with success, -1 if none",
    grasped_once="RLinf is_src_obj_grasped at any step",
    stable_grasp_once="RLinf consecutive_grasp (5 grasped steps in a row) at any step",
    first_grasp_step="first step with is_src_obj_grasped, -1 if none",
    object_plate_distance_end_m="RLinf carrot_plate_dist (object-to-plate centre, metres) at step 80",
    object_plate_distance_min_m="minimum carrot_plate_dist over the episode")


class EpisodeOutcomes:
    """Per-env outcomes accumulated from RLinf PutOnPlateInScene25 per-step infos.

    Graded fields (grasp, object-to-plate distance) are recorded for later analysis;
    they are not learner inputs unless a protocol says so.
    """

    def __init__(self, envs):
        self.step = 0
        self.once = torch.zeros(envs, dtype=torch.bool)
        self.last = torch.zeros(envs, dtype=torch.bool)
        self.first = torch.full((envs,), -1, dtype=torch.long)
        self.grasped = torch.zeros(envs, dtype=torch.bool)
        self.stable = torch.zeros(envs, dtype=torch.bool)
        self.first_grasp = torch.full((envs,), -1, dtype=torch.long)
        self.distance = torch.full((envs,), float("nan"), dtype=torch.float64)
        self.distance_min = torch.full((envs,), float("inf"), dtype=torch.float64)

    def update(self, info):
        success = info["success"].detach().cpu().bool()
        grasped = info["is_src_obj_grasped"].detach().cpu().bool()
        distance = info["carrot_plate_dist"].detach().cpu().to(torch.float64)
        if not (success.shape == grasped.shape == distance.shape == self.once.shape) or not torch.isfinite(distance).all():
            raise ValueError("Require one finite outcome per env from RLinf infos")
        self.first[success & (self.first < 0)] = self.step
        self.first_grasp[grasped & (self.first_grasp < 0)] = self.step
        self.once |= success
        self.last = success
        self.grasped |= grasped
        self.stable |= info["consecutive_grasp"].detach().cpu().bool()
        self.distance = distance
        self.distance_min = torch.minimum(self.distance_min, distance)
        self.step += 1

    def rows(self):
        if self.step != STEPS:
            raise RuntimeError(f"Episode ended after {self.step} of {STEPS} steps")
        return [dict(success_once=bool(self.once[i]), success_at_end=bool(self.last[i]),
                     first_success_step=int(self.first[i]), grasped_once=bool(self.grasped[i]),
                     stable_grasp_once=bool(self.stable[i]), first_grasp_step=int(self.first_grasp[i]),
                     object_plate_distance_end_m=float(self.distance[i]),
                     object_plate_distance_min_m=float(self.distance_min[i])) for i in range(len(self.once))]


@torch.no_grad()
def run_batch(model, env, rows, *, seed, generator, angles_degrees=None, audit_path=None):
    """One synchronized batch of 80-step episodes; returns per-row outcome fields."""
    from rlinf.envs.action_utils import prepare_actions
    base = env.env.unwrapped
    ids = torch.tensor([row["episode_id"] for row in rows], device=env.device)
    obs, _ = env.reset(seed=seed, options={"episode_id": ids})
    if base.select_carrot_ids.cpu().tolist() != [row["object"] for row in rows]:
        raise AssertionError("Environment did not reset to the planned objects")
    instructions = list(obs["task_descriptions"])
    outcomes = EpisodeOutcomes(len(rows))
    generator.manual_seed(seed)
    step = 0
    for _ in range(STEPS // CHUNK):
        actions, _ = model.predict_action_batch(obs, mode="eval", rng=generator)
        chunk = prepare_actions(raw_chunk_actions=actions, env_type="maniskill", model_type="openpi",
                                num_action_chunks=CHUNK, action_dim=7, policy="widowx_bridge")
        original = chunk
        if angles_degrees is not None:
            chunk = rotate_translation(original, angles_degrees)
        if not torch.equal(chunk[..., 2:], original[..., 2:]):
            raise AssertionError("Rotation changed z, rotation or gripper")
        if audit_path is not None:
            before, after = original.detach().cpu(), chunk.detach().cpu()
            record = dict(batch_seed=seed, step_start=step, envs=len(rows), action_chunk=CHUNK,
                native_actions=len(rows)*CHUNK, gripper_rotation_z_unchanged=True,
                original_sha256=hashlib.sha256(before.contiguous().numpy().tobytes()).hexdigest(),
                issued_sha256=hashlib.sha256(after.contiguous().numpy().tobytes()).hexdigest(),
                angles_degrees=angles_degrees,
                xy_norm_max_error=float((before[..., :2].norm(dim=-1)-after[..., :2].norm(dim=-1)).abs().max()))
            witnesses = [i for i, angle in enumerate(angles_degrees or []) if abs(angle) == 45][:2]
            record['rotation_witnesses'] = [dict(scene=rows[i], original=before[i].tolist(),
                                                 issued=after[i].tolist()) for i in witnesses]
            if witnesses:
                positions = base.agent.robot.links_map['ee_gripper_link'].pose.p.detach().cpu()
                for i, witness in zip(witnesses, record['rotation_witnesses']):
                    witness['ee_position_before_chunk'] = positions[i].tolist()
        obs_list, _, _, _, infos_list = env.chunk_step(chunk)
        if audit_path is not None:
            if witnesses:
                positions = base.agent.robot.links_map['ee_gripper_link'].pose.p.detach().cpu()
                for i, witness in zip(witnesses, record['rotation_witnesses']):
                    witness['ee_position_after_chunk'] = positions[i].tolist()
            with Path(audit_path).open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
        for info in infos_list:
            outcomes.update(info)
            step += 1
        obs = obs_list[-1]
    return [dict(instruction=instruction, **outcome) for instruction, outcome in zip(instructions, outcomes.rows())]


def rates(rows, key="success_once"):
    values = [r[key] for r in rows]
    return float(np.mean(values)) if values else float("nan")
