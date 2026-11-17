"""Native visual trials for history-conditioned frozen-expert recalibration."""
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

CODE = Path(__file__).resolve().parents[2]
ROOT = Path('/mowenjie/fluffy-bassoon-2')
RUNTIME = ROOT / 'sources/critic-selected-a1229239-r1'
RUNTIME_PIN = '8f9bfa7661928ef8c17494b04f4d7bf3125634e1'
MODEL = ROOT / 'inputs/robocasa365-public-pi05-c484-torch-20261101-r2'
NORM = ROOT / 'inputs/robocasa365-public-pi05-c484-75000-20261101-r1/assets/norm_stats.json'

# Reuse the qualified native reset and camera contract. No JAX model is loaded.
spec = importlib.util.spec_from_file_location('calibration_reset', CODE /
    'outputs/robocasa-feedback-advantage-20261117-r1/collect_panel.py')
reset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reset)


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def public(obs, raw):
    item = reset.inputs(obs)
    # Gym's official policy view is eef7/base7/gripper2 (16D). RLinf's
    # history port requires its raw 25D sensor layout before selecting those
    # same 16 motor coordinates. Preserve public sensors, never pad fake ones.
    # Read the same sensor sample returned by env.step. Forcing an update
    # here would advance observable sampling and desynchronize state/images.
    sensors = raw._get_observations(force_update=False)
    keys = ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos',
            'robot0_gripper_qvel', 'robot0_base_to_eef_pos',
            'robot0_base_to_eef_quat', 'robot0_base_pos', 'robot0_base_quat')
    state = np.concatenate([np.asarray(sensors[k]).reshape(-1) for k in keys]).astype(np.float32)
    if state.shape != (25,):
        raise ValueError(f'Native public state layout differs: {state.shape}')
    np.testing.assert_allclose(state[[*range(11,25),7,8]], item['observation/state'], atol=1e-6, rtol=0)
    return dict(main_images=item['observation/image'][None],
        wrist_images=item['observation/wrist_image'][None],
        extra_view_images=item['observation/right_image'][None, None],
        states=state[None],
        task_descriptions=[item['prompt']])


def load(pin):
    from fluffy_bassoon.robocasa.joint_steering_ppo_worker import _load_native
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec, FlowDirectMotor
    os.environ['RLINF_PATH'] = str(RUNTIME)
    native, _, signature = _load_native(dict(source_commit=pin,
        runtime_commit=RUNTIME_PIN, model_directory=str(MODEL), norm_path=str(NORM)), CODE)
    stats = json.loads(NORM.read_text())
    stats = stats.get('norm_stats', stats)
    codec = DirectActionCodec.from_stats(stats)
    return native, FlowDirectMotor(native, codec), codec, stats, signature


@torch.no_grad()
def base_prediction(native, prepared, noise):
    original = native.model.build_prefix_cache
    try:
        native.model.build_prefix_cache = lambda *a, **kw: prepared['prefix']
        commands, _ = native.model.predict_action_batch(prepared['observation'],
            mode='eval', noise=torch.as_tensor(noise[None], device=native.model.device))
        return commands[0].detach().float().cpu().numpy()
    finally:
        native.model.build_prefix_cache = original


def command_feedback(physical, codec):
    """Only delivered controls enter history; the unused suffix is constant padding."""
    if not 1 <= len(physical) <= 5:
        raise ValueError('One decision delivers between one and five controls')
    result = np.zeros((50, 12), np.float32)
    result[:len(physical)] = ((physical - np.asarray(codec.mean, np.float32)) /
                             np.asarray(codec.std, np.float32))
    return result.reshape(-1)


def rollout(native, motor, codec, *, directory, scene, offset, fraction,
            target_start=None, attempt=1, student=None, log=None, save_images=True):
    """All geometry stays in the environment/teacher, outside student inputs.

    The approach-window flag is held for five physical controls, matching the
    motor execution stride. This makes same-query teacher chunk labels causal.
    A grasp ends the window at the next decision boundary, at most four controls
    later than the old control-rate wrapper; this migration is explicit.
    """
    from robocasa.utils.env_utils import convert_action
    from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    log = log or (lambda *a, **kw: None)
    offset = np.asarray(offset, np.float64)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('Finite base-frame offset required')
    (env, obs), start_key = reset.matched_start(scene, directory.name, target_start, log)
    raw = env.unwrapped.env
    robot = raw.robots[0]
    controller = robot.part_controllers['right']
    assert controller.input_ref_frame == 'base' and controller.input_type == 'delta'
    gain = np.asarray((controller.output_max - controller.output_min) /
                      (controller.input_max - controller.input_min))[:3]
    assert np.all(gain > 0)
    names, centers = reset.handle(raw)
    nominal_goal = controller.compute_goal_pos
    active = False

    def shifted(*args, **kwargs):
        goal = nominal_goal(*args, **kwargs)
        return goal + offset if active else goal

    controller.compute_goal_pos = shifted
    rows, images, states, features, labels, masks, feedback = [], [], [], [], [], [], []
    actions, physical_hashes = [], []
    prompt = None
    decision = first_grasp = None
    clipped = 0
    start_time = time.monotonic()
    # Same query-attempt noise across all arms; separate from support noise.
    noises = np.random.default_rng(scene + 700000 + (attempt - 1) * 1000003).standard_normal(
        (120, 50, 32), dtype=np.float32)
    try:
        for t in range(0, 600, 5):
            if decision is None and first_grasp is None and np.linalg.norm(reset.geometry(raw, robot, centers)) <= .10:
                decision = t
            active = decision is not None and first_grasp is None and t < decision + 100
            item = public(obs, raw)
            feat, prepared = motor.prepare_batch(item)
            noise = noises[t // 5]
            base = base_prediction(native, prepared, noise)
            base_physical = DirectActionCodec.mapped_physical(base)
            target = base_physical.copy()
            if active:
                wanted = target[:5, :3] - offset[None] / gain[None]
                bounded = np.clip(wanted, controller.input_min[:3], controller.input_max[:3])
                clipped += int(np.any(np.abs(wanted - bounded) > 1e-7, axis=1).sum())
                target[:5, :3] = bounded
            if student is None:
                commanded = base_physical.copy()
                if active:
                    commanded[:5, :3] = np.clip(commanded[:5, :3] - fraction * offset[None] / gain[None],
                                                controller.input_min[:3], controller.input_max[:3])
                native._prepared = None
            else:
                commanded = student.predict(feat[0], prepared, noise, start=(t == 0))
                if commanded.shape != (50, 12):
                    raise ValueError('Student must produce the full native action chunk')
            delivered, rewards = [], []
            for j in range(5):
                a = commanded[j].copy()
                physical_hashes.append(reset.physics(raw))
                obs, reward, done, truncated, info = env.step(convert_action(a))
                delivered.append(a)
                rewards.append(float(reward))
                success = bool(info['success'])
                if first_grasp is None and raw._check_grasp(robot.gripper['right'], names):
                    first_grasp = t + j + 1
                if success or done or truncated:
                    break
            terminal = success or done or truncated or t + len(delivered) >= 600
            mask = np.zeros((50, 12), bool)
            mask[:len(delivered)] = True
            actual = np.asarray(delivered, np.float32)
            hist = command_feedback(actual, codec)
            if student is not None:
                student.feedback(hist, sum(rewards), terminal)
            features.append(feat[0]); labels.append(target); masks.append(mask); feedback.append(hist)
            states.append(item['states'][0]); prompt = item['task_descriptions'][0]
            if save_images:
                images.append(np.stack((item['main_images'][0], item['wrist_images'][0], item['extra_view_images'][0, 0])))
            actions.extend(delivered)
            rows.append(dict(control=t, delivered=len(delivered), reward=sum(rewards), done=bool(terminal),
                             success=success, active=bool(active)))
            if terminal:
                break
        payload = dict(features=np.asarray(features, np.float32), states=np.asarray(states, np.float32),
            teacher_actions=np.asarray(labels, np.float32), action_masks=np.asarray(masks),
            feedback=np.asarray(feedback, np.float32), actions=np.asarray(actions, np.float32),
            rewards=np.asarray([r['reward'] for r in rows], np.float32),
            dones=np.asarray([r['done'] for r in rows], bool), noise=noises[:len(rows)])
        if save_images:
            payload['images'] = np.asarray(images, np.uint8)
        np.savez_compressed(directory / 'episode.npz', **payload)
        result = dict(scene=scene, offset=offset.tolist(), fraction=fraction, attempt=attempt,
            start_signature=start_key, success=success, controls=len(actions), decisions=len(rows),
            onset=decision, first_grasp=first_grasp, teacher_clipped_controls=clipped,
            prompt=prompt, rows=rows, seconds=time.monotonic()-start_time,
            terminal_physics=reset.physics(raw), physical_hashes=physical_hashes,
            bytes=(directory / 'episode.npz').stat().st_size,
            window_contract='chunk-boundary onset/end; 100 controls maximum; stop next chunk after grasp')
        write(directory / 'result.json', result)
        log('episode', **{k: result[k] for k in ('scene','success','controls','seconds','onset','first_grasp','bytes')}, path=str(directory))
        return result
    finally:
        native._prepared = None
        controller.compute_goal_pos = nominal_goal
        env.close()
