"""Fixed-command B4 lifecycle qualification in native train/eval modes.

Two collections of four episodes per profile; explicit train resets versus
native evaluation auto-reset. Every collection ends in reusable native offload.
This is environment qualification, never a model update or source corpus.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType

import qualify_scene_setup as reference


ROOT = Path(__file__).resolve().parent
PROFILES = [('eval', False), ('eval', True), ('train', False), ('train', True)]


def contract():
    plan = json.loads((ROOT/'LIFECYCLE01-CONTRACT.json').read_bytes())
    assert plan['status'] in ('prepared', 'admitted')
    assert plan['closed_conservative_gpu_seconds'] + plan['reserved_gpu_seconds'] == plan['all_worst_gpu_seconds'] <= 180000
    reference.reference_manifest(plan)
    return plan


def child(ordinal):
    import faulthandler
    faulthandler.enable()
    from dataclasses import asdict
    from omegaconf import OmegaConf
    import numpy as np
    import torch
    from fluffy_bassoon.baige_gpu import mapping, pci
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import ActionJournal, sha, tensor_hash, write
    from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
    from fluffy_bassoon.rl.robotwin_execution import (
        load_accounted_controller, install_accounted_controller, collated_chunk_executions)
    from fluffy_bassoon.rl.robotwin_noise import (
        _install_allocated_renderer, _install_reset_retry_limit,
        _install_vector_executor_recycling)
    from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv
    from robotwin.envs.vector_env import SubEnv

    mode, deferred = PROFILES[ordinal]
    plan = contract()
    assert plan['status'] == 'admitted' and os.uname().nodename.startswith('job-')
    assert torch.cuda.device_count() == 1
    torch.cuda.set_device(0)
    rows = mapping(); assert len(rows) == 1
    os.environ['LYCO_BAIGE_ISOLATED_UUID'] = rows[0]['uuid']
    local, output = Path(os.environ['TASK_LOCAL']), Path(os.environ['TASK_OUT'])
    directory = output/f'{ordinal:02d}-{mode}-{int(deferred)}'
    directory.mkdir()
    report = dict(status='starting', mode=mode, deferred_setup=deferred,
        executor_recycling=True, source_commit=os.environ['TASK_SOURCE'],
        model_calls=0, optimizer_updates=0, chunks=[], offloads=[])
    write(directory/'RESULT.json', report)
    env = None
    journal = ActionJournal(output/'action-accounting', rank=ordinal, world=4,
        attempt=output.name, maximum_actions=plan['maximum_all_in_actions'])
    try:
        expected_pci = _install_allocated_renderer(directory/'renderers')
        _install_reset_retry_limit(defer_scene=deferred)
        _install_vector_executor_recycling()
        SubEnv._fb2_trim_after_scene_close = False
        cfg = OmegaConf.create(json.loads((local/'state/resolved-native-config.json').read_bytes())['env']['eval'])
        cfg.is_eval = mode == 'eval'
        cfg.auto_reset = cfg.ignore_terminations = mode == 'eval'
        # Group1/B4 is a component fixture; no learner or training batch changes.
        assert cfg.group_size == 1 and cfg.max_episode_steps == 192
        saved = [read_tensor(local/'state/traces/rank-00'/f'{i:05d}.pt.gz') for i in range(6)]
        seeds = saved[0]['reset_state_ids'].tolist()
        assert len(seeds) == 4
        controller = load_accounted_controller(os.environ['TASK_CONTROLLER'], os.environ['TASK_CONTROLLER_SHA'])
        control = None
        if deferred:
            control = json.loads((output/f'{ordinal-1:02d}-{mode}-0/RESULT.json').read_bytes())
            assert control['status'] == 'passed' and control['mode'] == mode
        cursor = journal.begin(40, phase='constructor-allowance')
        env = RoboTwinEnv(cfg, 4, seed_offset=0, total_num_processes=4, worker_info=None)
        journal.finish(cursor, 40)
        original_reset = env.reset

        def reset(instance, env_idx=None, env_seeds=None):
            indices = list(range(4)) if env_idx is None else list(env_idx)
            selected = [seeds[int(i)] for i in indices]
            instance.reset_state_ids[indices] = torch.tensor(selected, dtype=torch.long,
                device=instance.reset_state_ids.device)
            count = len(indices)*10*(2 if not instance.venv.envs else 1)
            slot = journal.begin(count, phase='reset-allowance')
            result = original_reset(env_idx=env_idx, env_seeds=selected)
            journal.finish(slot, count)
            for index in indices:
                sub = instance.venv.envs[index]
                assert int(sub.env_seed) == int(sub.task.ep_num) == seeds[index]
                assert sub.episode_info_list == [sub.task.get_info()]
                assert sub.task.get_instruction() == result[0]['task_descriptions'][index]
                assert pci(sub.task.scene.render_system.device.pci_string.removeprefix('pci:')) == expected_pci
                install_accounted_controller(sub.task, controller, metadata=True)
            return result

        env.reset = MethodType(reset, env)

        def offload(phase):
            previous = env.venv.env_thread_pool
            threads = list(previous._threads)
            stamp = time.monotonic()
            report['cleanup_phase'] = phase
            write(directory/'RESULT.json', report)
            try:
                env.offload()
            except BaseException:
                report['native_close_failed'] = True
                raise
            assert not env.venv.envs and env.venv.env_thread_pool is not previous
            assert all(not thread.is_alive() for thread in threads)
            try:
                previous.submit(lambda: None)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Offloaded executor still accepts scene work')
            report['offloads'].append(dict(phase=phase, previous_threads=len(threads),
                all_joined=True, replacement_created=True, seconds=time.monotonic()-stamp))
            report['cleanup_phase'] = 'complete'
            write(directory/'RESULT.json', report)

        offload('native-initial-offload')
        for collection in range(2):
            current = None
            for episode in range(4):
                if current is None or mode == 'train':
                    current, _ = env.reset()
                assert tensor_hash(current) == tensor_hash(saved[0]['current_observation'])
                for index, original in enumerate(saved):
                    assert time.time() < int(os.environ['TASK_DEADLINE'])-60
                    if mode == 'eval':
                        assert tensor_hash(current) == tensor_hash(original['current_observation'])
                    actions = np.asarray(original['proposed_motor_actions']).copy()
                    assert actions.shape == (4, 32, 14) and np.isfinite(actions).all()
                    slot = journal.begin(128, phase=f'{collection}/{episode}/{index}')
                    actual = env.chunk_step(actions)
                    physical = collated_chunk_executions(actual[4][-1], num_envs=4, requested_targets=32)
                    journal.finish(slot, sum(row.delivered_targets for row in physical))
                    row = dict(collection=collection, episode=episode, chunk=index,
                        pre_observation_sha256=tensor_hash(current),
                        command_sha256=tensor_hash(actions), native_output_sha256=[tensor_hash(x) for x in actual],
                        measured_execution=[asdict(x) for x in physical],
                        reference_trace_sha256=sha(local/'state/traces/rank-00'/f'{index:05d}.pt.gz'))
                    if mode == 'eval':
                        assert row['native_output_sha256'] == [tensor_hash(x) for x in original['outputs']]
                    if control is not None:
                        assert row == control['chunks'][len(report['chunks'])]
                    report['chunks'].append(row)
                    current = actual[0][-1]
                write(directory/'RESULT.json', report)
                print(json.dumps(dict(mode=mode, deferred=deferred,
                    collection=collection, episode=episode, passed=True)), flush=True)
            offload(f'collection-{collection}')
        assert len(report['chunks']) == 48 and len(report['offloads']) == 3
        assert all(row['resolved'] for row in journal.rows)
        report.update(status='passed', episodes=32, vector_episodes=8,
            charged_actions=journal.charged(), controlled_profile_equal=True,
            retained_eval_outputs_equal=mode == 'eval', loss_masks_unchanged=True,
            combined_training_memory_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if env is not None:
            # Offload itself must provide the join barrier, not its caller.
            if env.venv.envs and not report.get('native_close_failed', False):
                env.offload()
            env.venv.env_thread_pool.shutdown(wait=True)
            report['final_executor_closed'] = True
        write(directory/'RESULT.json', report)


def parent():
    from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write
    plan = contract(); assert plan['status'] == 'admitted'
    output = Path(os.environ['TASK_OUT'])
    report = dict(status='starting', source_commit=os.environ['TASK_SOURCE'],
        model_calls=0, optimizer_updates=0, children=[])
    write(output/'RESULT.json', report)
    try:
        for ordinal, (mode, deferred) in enumerate(PROFILES):
            remaining = int(os.environ['TASK_DEADLINE'])-int(time.time())-75
            assert remaining >= 120
            with (output/f'child-{ordinal:02d}.log').open('xb') as stream:
                subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child', str(ordinal)],
                    stdout=stream, stderr=subprocess.STDOUT, timeout=remaining, check=True)
            path = output/f'{ordinal:02d}-{mode}-{int(deferred)}'/'RESULT.json'
            value = json.loads(path.read_bytes())
            assert value['status'] == 'passed' and value['final_executor_closed']
            report['children'].append(dict(path=str(path), sha256=sha(path),
                mode=mode, deferred_setup=deferred, episodes=value['episodes'],
                charged_actions=value['charged_actions']))
            write(output/'RESULT.json', report)
        assert sum(x['charged_actions'] for x in report['children']) <= plan['maximum_all_in_actions']
        report.update(status='passed', episodes=128, vector_episodes=32,
            vector_chunks=192, recurrent_offloads=12, modes=['train', 'eval'],
            source_corpus_admitted=False, combined_training_memory_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'RESULT.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', action='store_true')
    parser.add_argument('--child', type=int, choices=range(4))
    args = parser.parse_args()
    if args.stage:
        reference.inputs(Path(os.environ['TASK_LOCAL']), plan=contract())
    elif args.child is not None:
        child(args.child)
    else:
        parent()
