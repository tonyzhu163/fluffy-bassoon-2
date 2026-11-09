"""2x2 paired SFT ablation for released Motus pi0.5 on place_phone_stand:
1. Prompt mode: raw unseen instruction vs metadata.pt DEFAULT_PROMPT + instruction
2. Controller mode: native 2-point take_action vs continuous 32-step chunk TOPP
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NATIVE_COMMIT = 'fa8d97b0f33bb894eae16df5a8175490f5b977a7'
MODEL_SHA = 'de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb'
NORM_SHA = '1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb'
METADATA_SHA = 'ef52d518167a652b74cc1c4d3a49fc18f758c286fc0ee8485d33a3caf3bd99ee'

DEFAULT_PROMPT = (
    'The whole scene is in a realistic, industrial art style with three views: '
    'a fixed rear camera, a movable left arm camera, and a movable right arm camera. '
    'The aloha robot is currently performing the following task: '
)

ARMS = (
    dict(name='raw_take_action', prompt_mode='raw', controller='take_action'),
    dict(name='default_prompt_take_action', prompt_mode='default_prefix', controller='take_action'),
    dict(name='raw_continuous_chunk', prompt_mode='raw', controller='continuous_chunk'),
    dict(name='default_prompt_continuous_chunk', prompt_mode='default_prefix', controller='continuous_chunk'),
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.writing')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def configuration(model):
    from openpi.training import config
    from openpi.models import pi0_config
    return config.TrainConfig(
        name='motus_release_inference_ablation',
        exp_name='frozen',
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=32,
            action_dim=32,
            max_token_len=200,
            discrete_state_input=True,
            dtype='bfloat16',
            paligemma_variant='gemma_2b',
            action_expert_variant='gemma_300m',
        ),
        data=config.LeRobotAlohaDataConfig(
            repo_id='motus_release',
            assets=config.AssetsConfig(
                assets_dir=str(model / 'assets'),
                asset_id='pi0.5_clean_randomize_joint_training',
            ),
            default_prompt=DEFAULT_PROMPT,
            adapt_to_pi=True,
            use_delta_joint_actions=True,
        ),
        wandb_enabled=False,
    )


def manifest(native, model):
    import openpi
    import transformers
    import curobo
    from transformers.models.siglip import check
    assert transformers.__version__ == '4.53.2'
    assert check.check_whether_transformers_replace_is_installed_correctly()
    commit = subprocess.check_output(['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == NATIVE_COMMIT
    assert not subprocess.check_output(
        ['git', '-C', str(native), 'status', '--porcelain', '--untracked-files=no'],
        text=True,
    ).strip()
    norm = model / 'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(model / 'model.safetensors') == MODEL_SHA
    assert sha(norm) == NORM_SHA
    if (model / 'metadata.pt').exists():
        assert sha(model / 'metadata.pt') == METADATA_SHA
    package = Path(openpi.__file__).parent
    package_files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}
    return dict(
        native_commit=commit,
        native_files={
            p: sha(native / p)
            for p in (
                'envs/_base_task.py',
                'envs/place_phone_stand.py',
                'envs/robot/robot.py',
                'script/eval_policy.py',
                'description/utils/generate_episode_instructions.py',
                'task_config/_eval_step_limit.yml',
                'task_config/demo_clean.yml',
            )
        },
        model_sha256=MODEL_SHA,
        norm_sha256=NORM_SHA,
        default_prompt=DEFAULT_PROMPT,
        openpi_version=importlib.metadata.version('rlinf-openpi'),
        openpi_files=package_files,
        curobo_version=importlib.metadata.version('nvidia-curobo'),
        curobo_core={
            p: sha(Path(curobo.__file__).parent / p)
            for p in (
                'wrap/reacher/motion_gen.py',
                'types/robot.py',
                'cuda_robot_model/cuda_robot_model.py',
            )
        },
        transformers_version=transformers.__version__,
        transformers_replacements={
            p: sha(Path(transformers.__file__).parent / p)
            for p in (
                'models/gemma/modeling_gemma.py',
                'models/gemma/configuration_gemma.py',
                'models/paligemma/modeling_paligemma.py',
                'models/siglip/modeling_siglip.py',
                'models/siglip/check.py',
            )
        },
    )


def environment_args(native, setting='demo_clean'):
    import yaml
    args = yaml.safe_load((native / f'task_config/{setting}.yml').read_text())
    args.update(
        task_name='place_phone_stand',
        task_config=setting,
        ckpt_setting='motus_release_ablation',
        policy_name='motus_openpi_ablation',
        eval_mode=True,
        eval_video_log=False,
    )
    args.pop('step_lim', None)
    embodiment = yaml.safe_load((native / 'task_config/_embodiment_config.yml').read_text())
    assert args['embodiment'] == ['aloha-agilex']
    robot = embodiment['aloha-agilex']['file_path']
    args.update(left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True)
    cfg = yaml.safe_load((Path(robot) / 'config.yml').read_text())
    args.update(left_embodiment_config=cfg, right_embodiment_config=cfg)
    cams = yaml.safe_load((native / 'task_config/_camera_config.yml').read_text())
    cam = cams[args['camera']['head_camera_type']]
    args.update(head_camera_h=cam['h'], head_camera_w=cam['w'])
    return args


def format_prompt(instruction, prompt_mode):
    if prompt_mode == 'raw':
        return instruction
    if prompt_mode == 'default_prefix':
        return DEFAULT_PROMPT + instruction
    raise ValueError(f'Unknown prompt_mode: {prompt_mode}')


def encode(obs, prompt_text):
    import numpy as np
    return dict(
        state=np.asarray(obs['joint_action']['vector']),
        prompt=prompt_text,
        images={
            dest: np.transpose(obs['observation'][source]['rgb'], (2, 0, 1))
            for dest, source in (
                ('cam_high', 'head_camera'),
                ('cam_left_wrist', 'left_camera'),
                ('cam_right_wrist', 'right_camera'),
            )
        },
    )


def panel(obs, path):
    import numpy as np
    from PIL import Image
    Image.fromarray(
        np.concatenate(
            [obs['observation'][k]['rgb'] for k in ('head_camera', 'left_camera', 'right_camera')],
            axis=1,
        )
    ).save(path)


def compress_path(path, atol=1e-3):
    import numpy as np
    compressed = [path[0]]
    for i in range(1, len(path)):
        if not np.allclose(path[i], compressed[-1], atol=atol):
            compressed.append(path[i])
    if len(compressed) == 1 and len(path) > 1:
        compressed.append(path[-1])
    return np.asarray(compressed, dtype=np.float64)


def linear_chunk_trajectory(path, steps_per_target=15, dt=1.0 / 250.0):
    import numpy as np
    pos_segments, vel_segments = [], []
    for k in range(len(path) - 1):
        seg_pos = np.linspace(path[k], path[k + 1], steps_per_target + 1, dtype=np.float64)[1:]
        seg_vel = np.repeat(
            ((path[k + 1] - path[k]) / (steps_per_target * dt))[None, :],
            steps_per_target,
            axis=0,
        )
        pos_segments.append(seg_pos)
        vel_segments.append(seg_vel)
    return np.vstack(pos_segments), np.vstack(vel_segments)


def plan_arm_chunk(planner, path, steps_per_target=15, dt=1.0 / 250.0):
    import numpy as np
    span = float(np.max(np.abs(path - path[0])))
    if span <= 1e-4:
        pos, vel = linear_chunk_trajectory(path, steps_per_target=steps_per_target, dt=dt)
        return pos, vel, 'stationary'
    compressed = compress_path(path, atol=1e-3)
    if len(compressed) >= 2 and float(np.max(np.abs(compressed[-1] - compressed[0]))) > 1e-4:
        try:
            _times, pos, vel, _acc, _duration = planner.TOPP(compressed, dt, verbose=False)
            if (
                pos is not None
                and vel is not None
                and pos.shape[0] > 0
                and pos.shape == vel.shape
                and np.isfinite(pos).all()
                and np.isfinite(vel).all()
            ):
                return np.asarray(pos, dtype=np.float64), np.asarray(vel, dtype=np.float64), 'topp'
        except Exception:
            pass
    pos, vel = linear_chunk_trajectory(path, steps_per_target=steps_per_target, dt=dt)
    return pos, vel, 'linear_15'


def interpolate_gripper(gripper_path, n_step):
    import numpy as np
    num_actions = len(gripper_path) - 1
    assert num_actions >= 1 and n_step >= num_actions
    mod_num = n_step % num_actions
    step_counts = [0] + [
        n_step // num_actions + (1 if i < mod_num else 0)
        for i in range(num_actions)
    ]
    segments = []
    for gripper_step in range(1, gripper_path.shape[0]):
        region = np.linspace(
            gripper_path[gripper_step - 1],
            gripper_path[gripper_step],
            step_counts[gripper_step] + 1,
            dtype=np.float64,
        )[1:]
        segments.append(region)
    return np.concatenate(segments, axis=0)


def execute_continuous_chunk(task, actions):
    """Execute a multi-step action chunk continuously without 2-point stop-and-go TOPP."""
    import numpy as np
    if task.take_action_cnt >= task.step_lim or task.eval_success:
        return dict(requested=0, delivered=0, physics_steps=0, left_mode='noop', right_mode='noop')
    remaining = int(task.step_lim - task.take_action_cnt)
    actions = np.asarray(actions[:remaining], dtype=np.float64)
    num_targets = int(actions.shape[0])
    assert num_targets >= 1

    task._update_render()
    if task.render_freq:
        task.viewer.render()

    left_jointstate = task.robot.get_left_arm_jointState()
    right_jointstate = task.robot.get_right_arm_jointState()
    left_arm_dim = len(left_jointstate) - 1
    right_arm_dim = len(right_jointstate) - 1
    current_jointstate = np.asarray(left_jointstate + right_jointstate, dtype=np.float64)

    left_arm_actions = actions[:, :left_arm_dim]
    left_gripper_actions = actions[:, left_arm_dim]
    right_arm_actions = actions[:, left_arm_dim + 1 : left_arm_dim + right_arm_dim + 1]
    right_gripper_actions = actions[:, left_arm_dim + right_arm_dim + 1]

    left_current_qpos = current_jointstate[:left_arm_dim]
    right_current_qpos = current_jointstate[left_arm_dim + 1 : left_arm_dim + right_arm_dim + 1]
    left_path = np.vstack((left_current_qpos, left_arm_actions))
    right_path = np.vstack((right_current_qpos, right_arm_actions))

    left_pos, left_vel, left_mode = plan_arm_chunk(task.robot.left_mplib_planner, left_path)
    right_pos, right_vel, right_mode = plan_arm_chunk(task.robot.right_mplib_planner, right_path)

    # If one arm is stationary while the other moves under TOPP, match step counts
    # so the stationary arm never stretches or distorts the moving arm's schedule.
    if left_mode == 'stationary' and right_mode != 'stationary' and len(right_pos) >= num_targets:
        left_pos, left_vel = linear_chunk_trajectory(left_path, steps_per_target=max(1, len(right_pos) // num_targets))
    elif right_mode == 'stationary' and left_mode != 'stationary' and len(left_pos) >= num_targets:
        right_pos, right_vel = linear_chunk_trajectory(right_path, steps_per_target=max(1, len(left_pos) // num_targets))

    left_n_step = int(left_pos.shape[0])
    right_n_step = int(right_pos.shape[0])
    assert left_n_step >= num_targets and right_n_step >= num_targets

    left_gripper_path = np.hstack((task.robot.get_left_gripper_val(), left_gripper_actions))
    right_gripper_path = np.hstack((task.robot.get_right_gripper_val(), right_gripper_actions))
    left_gripper = interpolate_gripper(left_gripper_path, left_n_step)
    right_gripper = interpolate_gripper(right_gripper_path, right_n_step)

    before = int(task.take_action_cnt)
    now_left_id, now_right_id, physics_steps = 0, 0, 0
    while now_left_id < left_n_step or now_right_id < right_n_step:
        if now_left_id < left_n_step and now_left_id / left_n_step <= now_right_id / right_n_step:
            task.robot.set_arm_joints(left_pos[now_left_id], left_vel[now_left_id], 'left')
            task.robot.set_gripper(left_gripper[now_left_id], 'left')
            now_left_id += 1

        if now_right_id < right_n_step and now_right_id / right_n_step <= now_left_id / left_n_step:
            task.robot.set_arm_joints(right_pos[now_right_id], right_vel[now_right_id], 'right')
            task.robot.set_gripper(right_gripper[now_right_id], 'right')
            now_right_id += 1

        task.scene.step()
        physics_steps += 1
        task._update_render()

        if task.check_success():
            task.eval_success = True
            progress = max(now_left_id / max(left_n_step, 1), now_right_id / max(right_n_step, 1))
            delivered = max(1, min(num_targets, int(np.ceil(num_targets * progress))))
            task.take_action_cnt = before + delivered
            task.get_obs()
            return dict(
                requested=num_targets,
                delivered=delivered,
                physics_steps=physics_steps,
                left_mode=left_mode,
                right_mode=right_mode,
                left_n_step=left_n_step,
                right_n_step=right_n_step,
            )

    task.take_action_cnt = before + num_targets
    task._update_render()
    if task.render_freq:
        task.viewer.render()
    task.get_obs()
    return dict(
        requested=num_targets,
        delivered=num_targets,
        physics_steps=physics_steps,
        left_mode=left_mode,
        right_mode=right_mode,
        left_n_step=left_n_step,
        right_n_step=right_n_step,
    )


def run(args):
    import numpy as np
    import torch
    native, model = Path(os.environ['NATIVE_SOURCE']), Path(os.environ['NATIVE_PPO_MODEL'])
    receipt = manifest(native, model)
    from envs.place_phone_stand import place_phone_stand
    from generate_episode_instructions import generate_episode_descriptions

    if args.check_only:
        cfg = configuration(model)
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        assert data.use_quantile_norm and data.norm_stats is not None
        from openpi import transforms
        pipeline = transforms.compose([
            *data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=True),
            *data.model_transforms.inputs,
        ])
        raw_sample = dict(
            state=np.zeros(14),
            images={'cam_high': np.zeros((3, 224, 224), np.uint8)},
            prompt=format_prompt('Set the phone on the stand.', 'raw'),
        )
        pref_sample = dict(
            state=np.zeros(14),
            images={'cam_high': np.zeros((3, 224, 224), np.uint8)},
            prompt=format_prompt('Set the phone on the stand.', 'default_prefix'),
        )
        t_raw = pipeline(raw_sample)
        t_pref = pipeline(pref_sample)
        assert t_raw['state'].shape == (32,) and t_raw['tokenized_prompt'].shape == (200,)
        assert t_pref['state'].shape == (32,) and t_pref['tokenized_prompt'].shape == (200,)
        assert not np.array_equal(t_raw['tokenized_prompt'], t_pref['tokenized_prompt'])
        # Synthetic check of continuous chunk helpers
        dummy_path = np.linspace(np.zeros(6), np.ones(6) * 0.5, 33)
        pos, vel = linear_chunk_trajectory(dummy_path, steps_per_target=15)
        assert pos.shape == (480, 6) and vel.shape == (480, 6)
        g = interpolate_gripper(np.linspace(1.0, 0.0, 33), 480)
        assert g.shape == (480,)
        assert environment_args(native, 'demo_clean')['left_embodiment_config']['planner'] == 'curobo'
        save(
            args.output / 'CPU-QUALIFICATION.json',
            dict(
                status='passed',
                manifest=receipt,
                raw_nonzero_tokens=int(np.count_nonzero(t_raw['tokenized_prompt_mask'])),
                default_prefix_nonzero_tokens=int(np.count_nonzero(t_pref['tokenized_prompt_mask'])),
                synthetic_chunk_shape=list(pos.shape),
                model_constructed=False,
                native_simulator_imports_passed=True,
            ),
        )
        return

    qualified = json.loads(Path(__file__).with_name('REFERENCE-CPU-QUALIFICATION.json').read_bytes())
    assert qualified['status'] == 'passed' and qualified['native_simulator_imports_passed']
    assert receipt == qualified['manifest']
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer
    _install_allocated_renderer(args.output / 'renderers')
    from openpi.policies.policy_config import create_trained_policy
    policy = create_trained_policy(
        configuration(model),
        model,
        pytorch_device='cuda:0',
        sample_kwargs={'num_steps': 10},
    )
    original = getattr(policy._model.sample_actions, '_torchdynamo_orig_callable', None)
    if original is not None:
        policy._model.sample_actions = original
        policy._sample_actions = original
    policy._model.requires_grad_(False)
    assert not policy._model.training
    versions = [(p, p._version) for p in policy._model.parameters()]

    setting = 'demo_clean'
    config = environment_args(native, setting)
    arm_summaries = {
        arm['name']: dict(
            prompt_mode=arm['prompt_mode'],
            controller=arm['controller'],
            completed=0,
            successes=0,
            success_rate=0.0,
            episodes=[],
        )
        for arm in ARMS
    }
    rejected = []
    paired_seeds = []
    report = dict(
        status='running',
        source_commit=os.environ['TASK_SOURCE'],
        manifest=receipt,
        setting=setting,
        target_episodes=args.episodes,
        first_candidate_seed=100000,
        arms=arm_summaries,
        paired_seeds=paired_seeds,
        rejected_candidates=rejected,
        started_unix=time.time(),
    )
    save(args.output / 'RESULT.json', report)

    def deadline():
        if time.time() > float(os.environ['TASK_DEADLINE']) - 90:
            raise TimeoutError('Evaluation deadline; preserve completed evidence')

    task = place_phone_stand()
    task.suc = task.test_num = 0
    seed = 100000
    accepted_count = 0
    try:
        while accepted_count < args.episodes:
            deadline()
            assert seed < 110000, 'Exceeded candidate seed budget'
            try:
                task.setup_demo(now_ep_num=accepted_count, seed=seed, is_test=True, **config)
                info = task.play_once()
                feasible = bool(task.plan_success and task.check_success())
            except Exception as exc:
                if isinstance(exc, (FileNotFoundError, ImportError, AttributeError, TypeError, KeyError)):
                    raise
                rejected.append(dict(seed=seed, reason=type(exc).__name__, detail=str(exc)[:1000]))
                if (
                    len(rejected) >= 5
                    and type(exc).__name__ != 'UnStableError'
                    and len({(r['reason'], r.get('detail')) for r in rejected[-5:]}) == 1
                ):
                    save(args.output / 'RESULT.json', report)
                    raise RuntimeError('Repeated identical expert exception') from exc
                task.close_env()
                seed += 1
                save(args.output / 'RESULT.json', report)
                continue
            task.close_env()
            if not feasible:
                rejected.append(dict(seed=seed, reason='expert_not_successful'))
                seed += 1
                save(args.output / 'RESULT.json', report)
                continue

            # Deterministically sample one unseen instruction for this seed so all 4 arms
            # receive the exact same task phrasing.
            rng = np.random.RandomState(seed)
            descriptions = generate_episode_descriptions('place_phone_stand', [info['info']], args.episodes)
            unseen_list = descriptions[0]['unseen']
            instruction = str(unseen_list[int(rng.randint(0, len(unseen_list)))])

            seed_record = dict(
                episode_index=accepted_count,
                seed=seed,
                instruction=instruction,
                outcomes={},
                decision0_chunk_l2={},
            )
            decision0_actions = {}

            for arm in ARMS:
                deadline()
                arm_name = arm['name']
                prompt_text = format_prompt(instruction, arm['prompt_mode'])
                task.setup_demo(now_ep_num=accepted_count, seed=seed, is_test=True, **config)
                assert task.step_lim == 400
                task.set_instruction(instruction=prompt_text)

                episode_dir = args.output / f'{arm_name}-{accepted_count:03d}-seed-{seed}'
                episode_dir.mkdir()
                obs = task.get_obs()
                panel(obs, episode_dir / 'initial.jpg')

                row = dict(
                    arm=arm_name,
                    prompt_mode=arm['prompt_mode'],
                    controller=arm['controller'],
                    seed=seed,
                    instruction=instruction,
                    prompt_text=prompt_text,
                    expert_feasible=True,
                    initial_state=np.asarray(obs['joint_action']['vector']).tolist(),
                    decisions=[],
                    complete=False,
                    started_unix=time.time(),
                )
                arm_summaries[arm_name]['episodes'].append(row)

                decision_idx = 0
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    deadline()
                    noise_seed = int(seed * 1000 + decision_idx)
                    noise = np.random.RandomState(noise_seed).randn(32, 32).astype(np.float32)
                    torch.manual_seed(noise_seed)
                    torch.cuda.manual_seed_all(noise_seed)
                    with torch.inference_mode():
                        actions = np.asarray(policy.infer(encode(obs, prompt_text), noise=noise)['actions'])
                    assert actions.shape == (32, 14) and np.isfinite(actions).all()
                    if decision_idx == 0:
                        decision0_actions[arm_name] = actions.copy()
                    np.save(episode_dir / f'actions-{decision_idx:03d}.npy', actions)

                    if arm['controller'] == 'take_action':
                        before = int(task.take_action_cnt)
                        for action in actions:
                            task.take_action(action)
                            obs = task.get_obs()
                        row['decisions'].append(
                            dict(
                                decision_idx=decision_idx,
                                noise_seed=noise_seed,
                                requested=32,
                                delivered=int(task.take_action_cnt - before),
                            )
                        )
                    elif arm['controller'] == 'continuous_chunk':
                        chunk_info = execute_continuous_chunk(task, actions)
                        obs = task.get_obs()
                        chunk_info.update(decision_idx=decision_idx, noise_seed=noise_seed)
                        row['decisions'].append(chunk_info)
                    else:
                        raise ValueError(f"Unknown controller: {arm['controller']}")

                    decision_idx += 1
                    save(episode_dir / 'episode.json', row)

                row.update(
                    success=bool(task.eval_success),
                    complete=True,
                    delivered_targets=int(task.take_action_cnt),
                    step_limit=int(task.step_lim),
                    ended_unix=time.time(),
                )
                assert row['success'] or row['delivered_targets'] == 400
                panel(obs, episode_dir / 'final.jpg')
                save(episode_dir / 'episode.json', row)

                summary = arm_summaries[arm_name]
                summary['completed'] = sum(1 for e in summary['episodes'] if e['complete'])
                summary['successes'] += int(row['success'])
                summary['success_rate'] = round(summary['successes'] / max(1, summary['completed']), 4)
                seed_record['outcomes'][arm_name] = dict(
                    success=row['success'],
                    delivered_targets=row['delivered_targets'],
                    decisions=len(row['decisions']),
                )
                task.close_env(clear_cache=True)

            # Verify exact decision-0 action identity across same-prompt arms
            assert np.array_equal(
                decision0_actions['raw_take_action'],
                decision0_actions['raw_continuous_chunk'],
            ), 'Decision-0 actions must match for raw prompt arms'
            assert np.array_equal(
                decision0_actions['default_prompt_take_action'],
                decision0_actions['default_prompt_continuous_chunk'],
            ), 'Decision-0 actions must match for default_prompt arms'
            diff_l2 = float(
                np.linalg.norm(
                    decision0_actions['default_prompt_take_action'] - decision0_actions['raw_take_action']
                )
            )
            seed_record['decision0_chunk_l2'] = dict(
                raw_controller_diff=0.0,
                default_prompt_controller_diff=0.0,
                default_vs_raw_l2=round(diff_l2, 6),
            )
            paired_seeds.append(seed_record)
            accepted_count += 1
            seed += 1
            save(args.output / 'RESULT.json', report)
            print(
                json.dumps(
                    dict(
                        completed_seeds=accepted_count,
                        last_seed=seed_record['seed'],
                        outcomes={k: v['success'] for k, v in seed_record['outcomes'].items()},
                        totals={k: f"{v['successes']}/{v['completed']}" for k, v in arm_summaries.items()},
                    )
                ),
                flush=True,
            )
    finally:
        del task

    assert all(p._version == version and p.grad is None for p, version in versions)
    report.update(status='complete', frozen_parameters_unchanged=True, ended_unix=time.time())
    save(args.output / 'RESULT.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, choices=(2, 10, 20, 50), default=10)
    parser.add_argument('--check-only', action='store_true')
    options = parser.parse_args()
    assert options.output.is_dir()
    run(options)
