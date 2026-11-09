"""Direct OpenPI release evaluation through unmodified native RoboTwin control."""
import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

NATIVE_COMMIT = 'fa8d97b0f33bb894eae16df5a8175490f5b977a7'
MODEL_SHA = 'de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb'
NORM_SHA = '1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb'


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
        name='motus_release_inference_reconstruction', exp_name='frozen',
        model=pi0_config.Pi0Config(pi05=True, action_horizon=32,
            action_dim=32, max_token_len=200, discrete_state_input=True,
            dtype='bfloat16', paligemma_variant='gemma_2b', action_expert_variant='gemma_300m'),
        data=config.LeRobotAlohaDataConfig(repo_id='motus_release',
            assets=config.AssetsConfig(assets_dir=str(model/'assets'),
                                      asset_id='pi0.5_clean_randomize_joint_training'),
            adapt_to_pi=True, use_delta_joint_actions=True),
        wandb_enabled=False)


def manifest(native, model):
    import openpi
    import transformers
    import curobo
    import sapien
    from sapien import _oidn_tricks
    from transformers.models.siglip import check
    assert transformers.__version__ == '4.53.2'
    assert check.check_whether_transformers_replace_is_installed_correctly()
    assert _oidn_tricks.oidn_dll._name.endswith('libOpenImageDenoise.so.2.3.3')
    commit = subprocess.check_output(['git', '-C', str(native), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == NATIVE_COMMIT
    assert not subprocess.check_output(['git', '-C', str(native), 'status', '--porcelain',
                                       '--untracked-files=no'], text=True).strip()
    norm = model/'assets/pi0.5_clean_randomize_joint_training/norm_stats.json'
    assert sha(model/'model.safetensors') == MODEL_SHA
    assert sha(norm) == NORM_SHA
    package = Path(openpi.__file__).parent
    package_files = {str(p.relative_to(package)): sha(p) for p in sorted(package.rglob('*.py'))}
    return dict(native_commit=commit, native_files={p: sha(native/p) for p in (
        'envs/_base_task.py', 'envs/place_phone_stand.py', 'envs/robot/robot.py',
        'script/eval_policy.py', 'description/utils/generate_episode_instructions.py',
        'task_config/_eval_step_limit.yml', 'task_config/demo_clean.yml', 'task_config/demo_randomized.yml')},
        model_sha256=MODEL_SHA, norm_sha256=NORM_SHA,
        openpi_version=importlib.metadata.version('rlinf-openpi'), openpi_files=package_files,
        sapien_version=importlib.metadata.version('sapien'), oidn_version='2.3.3',
        oidn_loader_sha256=sha(Path(sapien.__file__).parent/'_oidn_tricks.py'),
        oidn_libraries={p: sha(Path(sapien.__file__).parent/'oidn_library'/p) for p in (
            'libOpenImageDenoise.so.2.3.3', 'libOpenImageDenoise_core.so.2.3.3',
            'libOpenImageDenoise_device_cuda.so.2.3.3')},
        curobo_version=importlib.metadata.version('nvidia-curobo'),
        curobo_core={p: sha(Path(curobo.__file__).parent/p) for p in (
            'wrap/reacher/motion_gen.py', 'types/robot.py', 'cuda_robot_model/cuda_robot_model.py')},
        transformers_version=transformers.__version__, transformers_replacements={p:
            sha(Path(transformers.__file__).parent/p) for p in (
                'models/gemma/modeling_gemma.py', 'models/gemma/configuration_gemma.py',
                'models/paligemma/modeling_paligemma.py', 'models/siglip/modeling_siglip.py',
                'models/siglip/check.py')})


def environment_args(native, setting):
    import yaml
    args = yaml.safe_load((native/f'task_config/{setting}.yml').read_text())
    args.update(task_name='place_phone_stand', task_config=setting, ckpt_setting='motus_release',
                policy_name='motus_openpi_direct', eval_mode=True, eval_video_log=False)
    args.pop('step_lim', None)
    embodiment = yaml.safe_load((native/'task_config/_embodiment_config.yml').read_text())
    assert args['embodiment'] == ['aloha-agilex']
    robot = embodiment['aloha-agilex']['file_path']
    args.update(left_robot_file=robot, right_robot_file=robot, dual_arm_embodied=True)
    cfg = yaml.safe_load((Path(robot)/'config.yml').read_text())
    args.update(left_embodiment_config=cfg, right_embodiment_config=cfg)
    cams = yaml.safe_load((native/'task_config/_camera_config.yml').read_text())
    cam = cams[args['camera']['head_camera_type']]
    args.update(head_camera_h=cam['h'], head_camera_w=cam['w'])
    return args


def encode(obs, instruction):
    import numpy as np
    return dict(state=np.asarray(obs['joint_action']['vector']), prompt=instruction,
        images={dest: np.transpose(obs['observation'][source]['rgb'], (2, 0, 1))
                for dest, source in (('cam_high', 'head_camera'),
                                     ('cam_left_wrist', 'left_camera'),
                                     ('cam_right_wrist', 'right_camera'))})


def panel(obs, path):
    import numpy as np
    from PIL import Image
    Image.fromarray(np.concatenate([obs['observation'][k]['rgb']
        for k in ('head_camera', 'left_camera', 'right_camera')], axis=1)).save(path)


def qualify_renderer(args, native, receipt):
    import numpy as np
    from envs.place_phone_stand import place_phone_stand
    report = dict(status='running', kind='native_renderer_qualification',
        source_commit=os.environ['TASK_SOURCE'], manifest=receipt,
        model_constructed=False, optimizer_updates=0, episodes=[], started_unix=time.time())
    save(args.output/'RESULT.json', report)
    for setting in ('demo_clean', 'demo_randomized'):
        config = environment_args(native, setting)
        task = place_phone_stand()
        task.suc = task.test_num = 0
        passed = False
        for seed in range(100000, 100004):
            assert time.time() < float(os.environ['TASK_DEADLINE']) - 90
            task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **config)
            obs = task.get_obs()
            panel(obs, args.output/f'{setting}-seed-{seed}.png')
            images = [np.asarray(obs['observation'][k]['rgb'])
                      for k in ('head_camera', 'left_camera', 'right_camera')]
            assert all(x.dtype == np.uint8 and x.ndim == 3 and x.shape[-1] == 3 for x in images)
            info = task.play_once()
            passed = bool(task.plan_success and task.check_success())
            report['episodes'].append(dict(setting=setting, seed=seed,
                expert_passed=passed, image_shapes=[list(x.shape) for x in images],
                image_ranges=[[int(x.min()), int(x.max())] for x in images]))
            task.close_env()
            save(args.output/'RESULT.json', report)
            if passed:
                break
        assert passed, f'No native expert positive control for {setting}'
        del task
    report.update(status='complete', ended_unix=time.time())
    save(args.output/'RESULT.json', report)


def run(args):
    import numpy as np
    import torch
    native, model = Path(os.environ['NATIVE_SOURCE']), Path(os.environ['NATIVE_PPO_MODEL'])
    receipt = manifest(native, model)
    # Import the complete native simulator before any model construction. In
    # CPU qualification these imports instantiate no scene or policy model.
    from envs.place_phone_stand import place_phone_stand
    from generate_episode_instructions import generate_episode_descriptions
    if args.check_only:
        cfg = configuration(model)
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        assert data.use_quantile_norm and data.norm_stats is not None
        sample = dict(state=np.zeros(14), images={'cam_high': np.zeros((3, 224, 224), np.uint8)},
                      prompt='place the phone on the stand')
        from openpi import transforms
        transformed = transforms.compose([*data.data_transforms.inputs,
            transforms.Normalize(data.norm_stats, use_quantiles=True),
            *data.model_transforms.inputs])(sample)
        assert transformed['state'].shape == (32,) and transformed['tokenized_prompt'].shape == (200,)
        for setting in ('demo_clean', 'demo_randomized'):
            assert environment_args(native, setting)['left_embodiment_config']['planner'] == 'curobo'
        save(args.output/'CPU-QUALIFICATION.json', dict(status='passed', manifest=receipt,
             transformed_state_shape=[32], token_shape=[200], model_constructed=False,
             native_simulator_imports_passed=True))
        return
    qualified = json.loads(Path(__file__).with_name('REFERENCE-CPU-QUALIFICATION.json').read_bytes())
    assert qualified['status'] == 'passed' and qualified['native_simulator_imports_passed']
    assert receipt == qualified['manifest']
    assert sys.platform == 'linux' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer
    _install_allocated_renderer(args.output/'renderers')
    if args.render_only:
        qualify_renderer(args, native, receipt)
        return
    from openpi.policies.policy_config import create_trained_policy
    policy = create_trained_policy(configuration(model), model, pytorch_device='cuda:0',
                                   sample_kwargs={'num_steps': 10})
    original = getattr(policy._model.sample_actions, '_torchdynamo_orig_callable', None)
    if original is not None:
        policy._model.sample_actions = original
        policy._sample_actions = original
    policy._model.requires_grad_(False)
    assert not policy._model.training
    versions = [(p, p._version) for p in policy._model.parameters()]
    report = dict(status='running', source_commit=os.environ['TASK_SOURCE'], manifest=receipt,
        purpose='Reconstructed direct evaluation of released frozen pi0.5; not maintainer-confirmed',
        sampler='OpenPI native ODE K10, native policy RNG, eager', action_horizon=32,
        instruction_type='unseen', native_seed_index=0, first_candidate_seed=100000,
        optimizer_updates=0, controller='unmodified native Base_Task.take_action',
        target_per_setting=args.episodes, settings={}, started_unix=time.time())
    save(args.output/'RESULT.json', report)

    def deadline():
        if time.time() > float(os.environ['TASK_DEADLINE']) - 90:
            raise TimeoutError('Evaluation deadline; preserve completed evidence')

    for setting in ('demo_clean', 'demo_randomized'):
        config = environment_args(native, setting)
        task = place_phone_stand()
        task.suc = task.test_num = 0
        entries, rejected, seed = [], [], 100000
        result = dict(episodes=entries, rejected_candidates=rejected, successes=0)
        report['settings'][setting] = result
        try:
            while len(entries) < args.episodes:
                deadline()
                assert seed < 110000, 'Exceeded candidate seed budget'
                try:
                    task.setup_demo(now_ep_num=len(entries), seed=seed, is_test=True, **config)
                    info = task.play_once()
                    feasible = bool(task.plan_success and task.check_success())
                except Exception as exc:
                    if isinstance(exc, (FileNotFoundError, ImportError, AttributeError, TypeError, KeyError)):
                        raise
                    rejected.append(dict(seed=seed, reason=type(exc).__name__, detail=str(exc)[:1000]))
                    # Repeated infrastructure errors must not silently consume
                    # the expert seed filter. Abort; retain every candidate.
                    if (len(rejected) >= 5 and type(exc).__name__ != 'UnStableError'
                            and len({(r['reason'], r.get('detail')) for r in rejected[-5:]}) == 1):
                        save(args.output/'RESULT.json', report)
                        raise RuntimeError('Repeated identical expert exception') from exc
                    task.close_env()
                    seed += 1
                    save(args.output/'RESULT.json', report)
                    continue
                task.close_env()
                if not feasible:
                    rejected.append(dict(seed=seed, reason='expert_not_successful'))
                    seed += 1
                    save(args.output/'RESULT.json', report)
                    continue
                task.setup_demo(now_ep_num=len(entries), seed=seed, is_test=True, **config)
                assert task.step_lim == 400
                descriptions = generate_episode_descriptions('place_phone_stand', [info['info']], args.episodes)
                instruction = str(np.random.choice(descriptions[0]['unseen']))
                task.set_instruction(instruction=instruction)
                episode_dir = args.output/f'{setting}-{len(entries):03d}-seed-{seed}'
                episode_dir.mkdir()
                obs = task.get_obs()
                panel(obs, episode_dir/'initial.jpg')
                native_input = encode(obs, instruction)
                np.savez_compressed(episode_dir/'initial-input.npz',
                    state=native_input['state'], prompt=np.asarray(instruction), **native_input['images'])
                row = dict(seed=seed, instruction=instruction, expert_feasible=True,
                    initial_state=np.asarray(obs['joint_action']['vector']).tolist(),
                    decisions=[], complete=False, started_unix=time.time())
                entries.append(row)
                while task.take_action_cnt < task.step_lim and not task.eval_success:
                    deadline()
                    with torch.inference_mode():
                        actions = policy.infer(encode(obs, instruction))['actions']
                    assert actions.shape == (32, 14) and np.isfinite(actions).all()
                    before = task.take_action_cnt
                    np.save(episode_dir/f'actions-{len(row["decisions"]):03d}.npy', actions)
                    # Native pi05 adapter observes after every waypoint, including
                    # targets following early success (take_action then returns).
                    for action in actions:
                        task.take_action(action)
                        obs = task.get_obs()
                    row['decisions'].append(dict(requested=32, delivered=task.take_action_cnt-before))
                    save(episode_dir/'episode.json', row)
                row.update(success=bool(task.eval_success), complete=True,
                    delivered_targets=task.take_action_cnt, step_limit=task.step_lim, ended_unix=time.time())
                assert row['success'] or row['delivered_targets'] == 400
                panel(obs, episode_dir/'final.jpg')
                save(episode_dir/'episode.json', row)
                result['successes'] += int(row['success'])
                result['completed'] = sum(e['complete'] for e in entries)
                task.close_env(clear_cache=((len(entries)+1) % config['clear_cache_freq'] == 0))
                seed += 1
                save(args.output/'RESULT.json', report)
                print(json.dumps(dict(setting=setting, completed=result['completed'],
                    successes=result['successes'], last_seed=row['seed'])), flush=True)
        finally:
            # close_env is native and idempotence is not assumed; successful
            # paths already closed. Process exit frees a partial failed scene.
            del task
    assert all(p._version == version and p.grad is None for p, version in versions)
    report.update(status='complete', frozen_parameters_unchanged=True, ended_unix=time.time())
    save(args.output/'RESULT.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, choices=(2, 100), default=2)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--render-only', action='store_true')
    options = parser.parse_args()
    assert options.output.is_dir()
    run(options)
