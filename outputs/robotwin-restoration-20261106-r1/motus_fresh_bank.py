"""Policy-free, ordered source-scene qualification. Does not launch a job.

The native expert check is left unchanged. Its result is accepted only when
the requested physical seed also survives a fresh inference-style reset.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
SCAN = ROOT/'MOTUS-FRESH-CONFIRMATION-SCAN.json'
FIELDS = {'seed', 'setup_demo_success', 'play_once_success', 'cost_time',
    'expert_physical_seed', 'initial_reset_exact', 'initial_observation_sha256'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True)+'\n')


def declaration():
    plan = json.loads(SCAN.read_bytes())
    assert plan['format'] == 'motus_fresh_confirmation_scan_v1'
    assert plan['task'] == 'place_phone_stand' and plan['required_scenes'] == 64
    assert not plan['policy_outcomes_observed'] and not plan['sampler_inventory_mutation_allowed']
    assert plan['target_training_iteration'] == 10
    train = ROOT/plan['excluded_train_file']
    consumed = ROOT/plan['consumed_readout_file']
    assert sha(train) == plan['excluded_train_sha256']
    assert sha(consumed) == plan['consumed_readout_sha256']
    old = json.loads(consumed.read_bytes())
    excluded = set(json.loads(train.read_bytes())['success_seeds'])
    excluded.update(plan['excluded_native_eval_seeds'])
    assert set(old['dev']+old['confirmation']) <= excluded
    assert len(plan['excluded_native_eval_seeds']) == 150
    candidates = list(range(plan['candidate_start'], plan['candidate_stop_exclusive']))
    assert candidates == list(range(100100242, 100101266))
    assert not set(candidates) & excluded
    return plan, candidates


def accepted(row):
    assert set(row) == FIELDS, 'Do not add policy outcomes to the qualification records'
    assert type(row['seed']) is int
    for key in ('setup_demo_success', 'play_once_success', 'initial_reset_exact'):
        assert type(row[key]) is bool
    assert isinstance(row['cost_time'], (int, float)) and 0 <= row['cost_time'] < float('inf')
    if row['play_once_success']:
        assert row['setup_demo_success']
    if row['initial_reset_exact']:
        digest = row['initial_observation_sha256']
        assert isinstance(digest, str) and len(digest) == 64
        int(digest, 16)
    else:
        assert row['initial_observation_sha256'] is None
    return (row['setup_demo_success'] and row['play_once_success']
        and row['expert_physical_seed'] == row['seed'] and row['initial_reset_exact'])


def select(records, candidates, required=64):
    """Reject missing/reordered candidates, early stopping and post-hoc selection."""
    assert len(records) <= len(candidates)
    assert [row['seed'] for row in records] == candidates[:len(records)]
    chosen = []
    for index, row in enumerate(records):
        if accepted(row):
            chosen.append(row['seed'])
        if len(chosen) == required:
            assert index == len(records)-1, 'Stop at the first64 qualified scenes'
    assert len(chosen) == required, 'No complete fresh bank; do not evaluate a partial bank'
    return chosen


def check_initial_observation(obs):
    """Qualify the actual native uint8 packet, including each wrist camera."""
    import torch
    assert torch.isfinite(obs['states']).all() and obs['states'].shape == (1, 14)
    main, wrists = obs['main_images'], obs['wrist_images']
    assert main.ndim == 4 and main.shape[0] == 1 and main.shape[-1] == 3
    assert wrists.ndim == 5 and wrists.shape[:2] == (1, 2) and wrists.shape[-1] == 3
    for image in (main[0], wrists[0, 0], wrists[0, 1]):
        assert torch.isfinite(image).all() and image.float().std().item() > 0
    assert isinstance(obs['task_descriptions'][0], str) and obs['task_descriptions'][0].strip()


def collect(contract_path):
    # Imports which can initialize a device are confined to admitted collection.
    plan, candidates = declaration()
    contract = json.loads(Path(contract_path).read_bytes())
    assert contract['status'] == 'admitted' and contract['cpu_preflight_passed']
    assert contract['closed_conservative_gpu_seconds']+contract['other_active_reserved_gpu_seconds']+contract['reserved_gpu_seconds'] <= contract['approved_cap_gpu_seconds']
    assert contract['reserved_gpu_seconds'] in (2100, 3600, 18000) and contract['GPU_count'] == 1
    if contract['approved_cap_gpu_seconds'] > 360000: assert contract['budget_extension_approved']
    assert sha(SCAN) == contract['scan_sha256']
    assert os.uname().sysname == 'Linux' and os.uname().nodename.startswith('job-')
    code = ROOT.parents[1]
    source = os.environ['TASK_SOURCE']
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == source
    for key, expected in (('REPO_PATH', plan['RLinf_commit']), ('ROBOTWIN_PATH', plan['RoboTwin_commit'])):
        assert subprocess.check_output(['git', '-C', os.environ[key], 'rev-parse', 'HEAD'], text=True).strip() == expected
        assert not subprocess.check_output(['git', '-C', os.environ[key], 'status', '--porcelain', '--untracked-files=no'])
    for name, expected in contract['source_files'].items():
        assert sha(code/name) == expected
    cpu = Path(contract['cpu_preflight_result'])
    assert sha(cpu) == contract['cpu_preflight_sha256']
    assert json.loads(cpu.read_bytes())['status'] == 'passed_fresh_bank_selection_and_native_check_seed_CPU_reference'
    native = Path(os.environ['ROBOTWIN_PATH'])/'robotwin/envs/vector_env.py'
    assert sha(native) == plan['native_vector_env_sha256']
    config = Path(contract['original_configuration'])
    assert sha(config) == contract['original_configuration_sha256']
    deadline = int(os.environ['TASK_DEADLINE'])
    assert time.time()+180 < deadline <= time.time()+contract['worker_timeout_seconds']
    output = Path(os.environ['TASK_OUT'])
    assert output.name == contract['output_identity'] and not output.exists()
    output.mkdir()

    import torch
    from omegaconf import OmegaConf
    from fluffy_bassoon.baige_gpu import mapping, pci
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import cpu_tree, tensor_hash
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer, _install_reset_retry_limit
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from robotwin.envs.vector_env import SubEnv

    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    devices = mapping(); assert len(devices) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = devices[0]['uuid']
    expected_pci = _install_allocated_renderer(output/'renderers')
    _install_reset_retry_limit(defer_scene=True)
    SubEnv._fb2_trim_after_scene_close = False
    original_cfg = json.loads(config.read_bytes())['env']['eval']
    cfg = OmegaConf.create(original_cfg)
    assert cfg.task_config.task_name == plan['task'] and cfg.max_episode_steps == 192
    # One simulator for planner qualification. No inference or sampler inventory edit.
    cfg.group_size = 1
    report = dict(status='collecting', source_commit=source, scan_sha256=sha(SCAN),
        native_vector_env_sha256=sha(native), original_configuration_sha256=sha(config),
        policy_outcomes_observed=False, model_calls=0, optimizer_updates=0,
        native_success_seed_inventory_unchanged=True, records=[], cleanup_complete=False)
    write(output/'RESULT.json', report)
    env = None; restore_backend = None
    try:
        if contract.get('expert_backend') == 'curobo':
            from motus_fresh_curobo import install
            assert cfg.task_config.planner_backend == 'mplib'
            restore_backend = install(contract, report, output)
            cfg.task_config.planner_backend = 'curobo'
        env = RoboTwinEnv(cfg, 1, seed_offset=0, total_num_processes=1, worker_info=None)
        inventory = env.success_seeds.clone()
        chosen = []
        for seed in candidates:
            assert time.time()+180 < deadline, 'Stop before the cleanup/release margin'
            sub = env.venv.envs[0]
            sub.close(clear_cache=False)
            if restore_backend is not None:
                import gc
                gc.collect(); torch.cuda.empty_cache()
            result = env.check_seeds([seed])[0]
            row = dict(seed=seed, **result, expert_physical_seed=int(getattr(sub.task, 'ep_num', -1)),
                initial_reset_exact=False, initial_observation_sha256=None)
            if result['setup_demo_success'] and result['play_once_success'] and row['expert_physical_seed'] == seed:
                actual = pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:'))
                assert actual == expected_pci
                # A setup error, identity, instruction or renderer mismatch
                # aborts the qualification and must be investigated.
                try:
                    if restore_backend is not None:
                        sub.close(clear_cache=False); gc.collect(); torch.cuda.empty_cache()
                    obs, _ = env.reset(env_seeds=[seed])
                except Exception as error:
                    raise RuntimeError(f'Initial reset failed for planner-valid seed {seed}; investigate before resuming') from error
                assert int(sub.task.ep_num) == int(sub.env_seed) == seed
                assert sub.task.get_instruction() == obs['task_descriptions'][0]
                assert sub.episode_info_list == [sub.task.get_info()]
                assert pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:')) == expected_pci
                check_initial_observation(obs)
                row.update(initial_reset_exact=True, initial_observation_sha256=tensor_hash(cpu_tree(obs)))
            assert torch.equal(inventory, env.success_seeds)
            if accepted(row): chosen.append(seed)
            report['records'].append(row)
            write(output/'RESULT.json', report)
            print('SCENE', seed, 'VALID', accepted(row), 'COUNT', len(chosen), flush=True)
            if len(chosen) == 64: break
        selected = select(report['records'], candidates)
        report.update(status='passed_policy_free_native_planner_and_exact_seed_reset', selected_seeds=selected)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if env is not None:
            env.venv.env_thread_pool.shutdown(wait=True)
            torch.cuda.synchronize()
            env.offload()
        if restore_backend is not None: restore_backend()
        report['cleanup_complete'] = True
        write(output/'RESULT.json', report)
    # Bank publication happens after complete qualification and scene cleanup.
    write(output/'BANK.json', dict(format='motus_fresh_confirmation_bank_v1', task=plan['task'],
        seeds=selected, scan_sha256=sha(SCAN), qualification_result_sha256=sha(output/'RESULT.json'),
        source_commit=source, target_training_iteration=10, policy_outcomes_observed=False,
        original_sampler_inventory_preserved=True))


def readback(directory):
    _, candidates = declaration()
    path = Path(directory)
    result = json.loads((path/'RESULT.json').read_bytes())
    bank = json.loads((path/'BANK.json').read_bytes())
    assert result['status'] == 'passed_policy_free_native_planner_and_exact_seed_reset'
    assert result['cleanup_complete'] and result['model_calls'] == result['optimizer_updates'] == 0
    assert not result['policy_outcomes_observed'] and not bank['policy_outcomes_observed']
    assert result['native_success_seed_inventory_unchanged'] and bank['original_sampler_inventory_preserved']
    plan, _ = declaration()
    assert result['native_vector_env_sha256'] == plan['native_vector_env_sha256']
    assert result['source_commit'] == bank['source_commit']
    assert bank['qualification_result_sha256'] == sha(path/'RESULT.json')
    assert result['scan_sha256'] == bank['scan_sha256'] == sha(SCAN)
    selected = select(result['records'], candidates)
    assert selected == result['selected_seeds'] == bank['seeds']
    assert bank['target_training_iteration'] == 10
    return dict(status='passed_independent_fresh_scene_bank_readback', seeds=selected,
        bank_sha256=sha(path/'BANK.json'), qualification_result_sha256=sha(path/'RESULT.json'),
        policy_outcomes_observed=False, effectiveness_not_tested=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--collect', type=Path)
    group.add_argument('--readback', type=Path)
    args = parser.parse_args()
    if args.collect: collect(args.collect)
    else: print(json.dumps(readback(args.readback), indent=2))
