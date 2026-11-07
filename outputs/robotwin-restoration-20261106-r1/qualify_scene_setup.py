"""Replay qualified B4 DEV commands to qualify opt-in deferred SAPIEN setup.

A default-setup control first qualifies this direct replay harness. Every
candidate uses a fresh process; no model, actor, SGD or new fitting data.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

ROOT = Path(__file__).resolve().parent
BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
REFERENCE = BASE/'motusdev01-signals'
PROFILES = [('default', 0)] + [('deferred', rank) for rank in range(4)]


def contract():
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha
    name = os.environ.get('TASK_CONTRACT', 'SCENESETUP01-CONTRACT.json')
    assert name in ('SCENESETUP01-CONTRACT.json', 'SCENESETUP02-CONTRACT.json', 'SCENESETUP03-CONTRACT.json')
    plan = json.loads((ROOT/name).read_bytes())
    assert plan['status'] in ('prepared', 'admitted')
    assert plan['closed_conservative_gpu_seconds']+plan['reserved_gpu_seconds'] <= 180000
    assert sha(BASE/'cpudev-readback02/RESULT.json') == plan['reference_readback_sha256']
    assert sha(REFERENCE/'FILES.json') == plan['reference_manifest_sha256']
    return plan, json.loads((REFERENCE/'FILES.json').read_bytes())


def inputs(local):
    """Copy only immutable config/commands to local storage, checking both ends."""
    import shutil
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
    plan, manifest = contract()
    selected = ['state/resolved-native-config.json'] + [
        f'state/traces/rank-{rank:02d}/{ordinal:05d}.pt.gz'
        for rank in range(4) for ordinal in range(6)]
    receipts = {}
    for name in selected:
        expected = manifest['files'][name]
        path = REFERENCE/name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        destination = local/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        assert sha(destination) == expected['sha256']
        receipts[name] = expected
    assert sum(v['bytes'] for v in receipts.values()) <= 2*2**30
    cfg = json.loads((local/selected[0]).read_bytes())
    assert cfg['env']['eval']['total_num_envs'] == 16
    assert cfg['env']['eval']['max_episode_steps'] == 192
    assert cfg['env']['eval']['auto_reset'] and cfg['env']['eval']['ignore_terminations']
    assert cfg['env']['eval']['skip_intermediate_renders']
    assert not cfg['env']['eval']['video_cfg']['save_video']
    write(local/'INPUTS.json', dict(files=receipts, environment=cfg['env']['eval'],
        reference_manifest_sha256=plan['reference_manifest_sha256']))
    return cfg


def child(profile, rank, ordinal):
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from fluffy_bassoon.baige_gpu import mapping, pci
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, sha, tensor_hash, write
    from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
    from fluffy_bassoon.rl.robotwin_execution import (
        load_accounted_controller, install_accounted_controller, collated_chunk_executions)
    from fluffy_bassoon.rl.robotwin_noise import _install_allocated_renderer, _install_reset_retry_limit
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from robotwin.envs.vector_env import SubEnv
    from dataclasses import asdict

    assert os.uname().nodename.startswith('job-') and torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    rows = mapping(); assert len(rows) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = rows[0]['uuid']
    local, output = Path(os.environ['TASK_LOCAL']), Path(os.environ['TASK_OUT'])
    directory = output/f'{ordinal:02d}-{profile}-rank-{rank:02d}'
    directory.mkdir()
    report = dict(status='starting', profile=profile, logical_reference_rank=rank,
        source_commit=os.environ['TASK_SOURCE'], optimizer_updates=0, model_calls=0, chunks=[])
    write(directory/'RESULT.json', report)
    env = None
    journal = ActionJournal(output/'action-accounting', rank=ordinal, world=5,
        attempt=output.name, maximum_actions=5120)
    started = time.monotonic()
    try:
        expected_pci = _install_allocated_renderer(directory/'renderers')
        _install_reset_retry_limit(defer_scene=profile == 'deferred')
        SubEnv._fb2_trim_after_scene_close = False
        config = json.loads((local/'state/resolved-native-config.json').read_bytes())['env']['eval']
        cfg = OmegaConf.create(config)
        references = [read_tensor(local/'state/traces'/f'rank-{rank:02d}'/f'{i:05d}.pt.gz') for i in range(6)]
        seeds = references[0]['reset_state_ids'].tolist()
        assert all(t['evaluation_data'] and t['producing_policy_iteration'] == 1
                   and t['reset_state_ids'].tolist() == seeds for t in references)
        controller = load_accounted_controller(os.environ['TASK_CONTROLLER'], os.environ['TASK_CONTROLLER_SHA'])
        slot = journal.begin(40, phase='constructor-setup-allowance')
        stamp = time.monotonic()
        env = RoboTwinEnv(cfg, 4, seed_offset=rank, total_num_processes=4, worker_info=None)
        report['constructor_seconds'] = time.monotonic()-stamp
        journal.finish(slot, 40)
        # Native EnvWorker immediately offloads evaluation after construction.
        env.offload()
        original_reset = env.reset

        def reset(instance, env_idx=None, env_seeds=None):
            indices = list(range(4)) if env_idx is None else list(env_idx)
            selected = [seeds[int(i)] for i in indices]
            instance.reset_state_ids[indices] = torch.tensor(selected, dtype=torch.long,
                device=instance.reset_state_ids.device)
            # Include possible temporary reconstruction plus real setup.
            count = len(indices)*10*(2 if not instance.venv.envs else 1)
            cursor = journal.begin(count, phase='reset-setup-allowance')
            value = original_reset(env_idx=env_idx, env_seeds=selected)
            journal.finish(cursor, count)
            for index in indices:
                sub = instance.venv.envs[index]
                assert int(sub.task.ep_num) == int(sub.env_seed) == seeds[index]
                assert sub.task.get_instruction() == value[0]['task_descriptions'][index]
                assert sub.episode_info_list == [sub.task.get_info()]
                actual = pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:'))
                assert actual == expected_pci
                install_accounted_controller(sub.task, controller, metadata=True)
            return value

        env.reset = MethodType(reset, env)
        stamp = time.monotonic()
        current, _ = env.reset()
        report['first_reset_seconds'] = time.monotonic()-stamp
        report['scene_seeds'] = seeds
        assert tensor_hash(current) == tensor_hash(references[0]['current_observation'])
        stamp = time.monotonic()
        for index, reference in enumerate(references):
            assert time.time() < int(os.environ['TASK_DEADLINE'])-60
            assert tensor_hash(current) == tensor_hash(reference['current_observation'])
            actions = np.asarray(reference['proposed_motor_actions']).copy()
            assert actions.shape == (4, 32, 14) and np.isfinite(actions).all()
            cursor = journal.begin(128, phase=f'chunk-{index}')
            actual = env.chunk_step(actions)
            execution = collated_chunk_executions(actual[4][-1], num_envs=4, requested_targets=32)
            journal.finish(cursor, sum(row.delivered_targets for row in execution))
            expected = reference['outputs']
            # Includes terminal raw packet and auto-reset/final observations.
            hashes = [tensor_hash(part) for part in actual]
            expected_hashes = [tensor_hash(part) for part in expected]
            assert hashes == expected_hashes, dict(chunk=index, actual=hashes, expected=expected_hashes)
            report['chunks'].append(dict(chunk=index, native_output_sha256=hashes,
                command_sha256=tensor_hash(actions), reference_trace_sha256=sha(
                    local/'state/traces'/f'rank-{rank:02d}'/f'{index:05d}.pt.gz'),
                measured_execution=[asdict(row) for row in execution],
                public_observation_exact=True, raw_feedback_and_autoreset_exact=True))
            current = actual[0][-1]
            write(directory/'RESULT.json', report)
            print(json.dumps(dict(profile=profile, rank=rank, chunk=index, passed=True)), flush=True)
        report['replay_seconds'] = time.monotonic()-stamp
        assert all(row['resolved'] for row in journal.rows)
        report.update(status='passed', episodes=4, charged_actions=journal.charged(),
            original_environment_config_exact=True, full_requested_chunks_unchanged=True,
            intermediate_render_profile_unchanged=True, policy_or_batch_numerics_changed=False,
            sampled_or_fitted_new_data=False, combined_training_memory_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        if env is not None:
            env.offload()
            env.venv.env_thread_pool.shutdown(wait=True)
        report['seconds'] = time.monotonic()-started
        write(directory/'RESULT.json', report)


def parent():
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import write, sha
    plan, _ = contract()
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    output = Path(os.environ['TASK_OUT'])
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'],
        optimizer_updates=0, model_calls=0, children=[], scientific_source_admission=False)
    write(output/'RESULT.json', report)
    try:
        for ordinal, (profile, rank) in enumerate(PROFILES):
            remaining = int(os.environ['TASK_DEADLINE'])-int(time.time())-75
            assert remaining >= 120, 'Keep cleanup/release margin'
            with (output/f'child-{ordinal:02d}.log').open('xb') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--profile', profile,
                    '--rank', str(rank), '--ordinal', str(ordinal)], stdout=stream, stderr=subprocess.STDOUT,
                    timeout=remaining, check=True)
            path = output/f'{ordinal:02d}-{profile}-rank-{rank:02d}'/'RESULT.json'
            value = json.loads(path.read_bytes()); assert value['status'] == 'passed'
            report['children'].append(dict(path=str(path), sha256=sha(path), profile=profile,
                rank=rank, constructor_seconds=value['constructor_seconds'],
                first_reset_seconds=value['first_reset_seconds'], replay_seconds=value['replay_seconds'],
                charged_actions=value['charged_actions']))
            write(output/'RESULT.json', report)
        assert len(report['children']) == 5
        assert sum(row['charged_actions'] for row in report['children']) <= 5120
        report.update(status='passed', default_control_episodes=4, deferred_episodes=16,
            public_observations_and_native_outputs_exact=True,
            finite_setup_guards_preserved=True, shared_native_controller=True,
            combined_training_memory_qualified=False, larger_scene_count_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error)); raise
    finally:
        write(output/'RESULT.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--stage', action='store_true')
    p.add_argument('--profile', choices=('default', 'deferred'))
    p.add_argument('--rank', type=int)
    p.add_argument('--ordinal', type=int)
    args = p.parse_args()
    if args.stage:
        inputs(Path(os.environ['TASK_LOCAL']))
    elif args.profile:
        assert (args.profile, args.rank) == PROFILES[args.ordinal]
        child(args.profile, args.rank, args.ordinal)
    else:
        parent()
