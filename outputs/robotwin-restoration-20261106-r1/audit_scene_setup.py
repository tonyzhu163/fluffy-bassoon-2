"""Independent CPU cold readback of the fixed B4 scene-setup qualification."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.rl.robotwin_execution import collated_chunk_executions


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def audit():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    root, output = BASE/'scenesetup05-signals', BASE/'scenesetup05-readback01'
    output.mkdir()
    report = dict(status='checking', gpu_seconds=0, model_calls=0, optimizer_updates=0,
        audit_source_commit=os.environ['TASK_AUDIT_SOURCE'],
        runtime_source_commit=os.environ['TASK_SOURCE'])
    try:
        code = Path(os.environ['TASK_CODE'])
        assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
        assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
        plan = json.loads((code/'outputs/robotwin-restoration-20261106-r1/SCENESETUP05-CONTRACT.json').read_bytes())
        complete = json.loads((root/'COMPLETE.json').read_bytes())
        result = json.loads((root/'RESULT.json').read_bytes())
        assert complete['status'] == 'complete_deferred_setup_replay' and result['status'] == 'passed'
        assert complete['result_sha256'] == sha(root/'RESULT.json')
        assert json.loads((root/'EXIT.json').read_bytes())['exit_code'] == 0
        manifest = json.loads((root/'FILES.json').read_bytes())
        for name, expected in manifest['files'].items():
            path = root/name
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
        reference = BASE/'motusdev01-signals'
        assert sha(reference/'FILES.json') == plan['reference_manifest_sha256']
        assert sha(BASE/'cpudev-readback02/RESULT.json') == plan['reference_readback_sha256']
        references = json.loads((reference/'FILES.json').read_bytes())['files']
        cache, summaries, charged, chunks = {}, [], 0, 0
        profiles = [('default', 0)] + [('deferred', rank) for rank in range(4)]
        assert len(result['children']) == len(profiles)
        for ordinal, (profile, rank) in enumerate(profiles):
            path = root/f'{ordinal:02d}-{profile}-rank-{rank:02d}'/'RESULT.json'
            child = json.loads(path.read_bytes())
            assert child['status'] == 'passed' and child['cleanup_phase'] == 'complete'
            assert child['executor_joined_before_scene_offload']
            assert child['source_commit'] == report['runtime_source_commit']
            assert child['profile'] == profile and child['logical_reference_rank'] == rank
            assert child['model_calls'] == child['optimizer_updates'] == 0
            assert len(child['chunks']) == 6 and child['episodes'] == 4
            binding = result['children'][ordinal]
            assert Path(binding['path']) == path and binding['sha256'] == sha(path)
            for index, recorded in enumerate(child['chunks']):
                key = rank, index
                name = f'state/traces/rank-{rank:02d}/{index:05d}.pt.gz'
                expected = references[name]
                if key not in cache:
                    source = reference/name
                    assert source.stat().st_size == expected['bytes'] and sha(source) == expected['sha256']
                    cache[key] = read_tensor(source)
                original = cache[key]
                assert original['evaluation_data'] and original['producing_policy_iteration'] == 1
                assert child['scene_seeds'] == original['reset_state_ids'].tolist()
                assert recorded['chunk'] == index and recorded['reference_trace_sha256'] == expected['sha256']
                assert recorded['command_sha256'] == tensor_hash(np.asarray(original['proposed_motor_actions']).copy())
                assert recorded['native_output_sha256'] == [tensor_hash(part) for part in original['outputs']]
                physical = collated_chunk_executions(original['outputs'][4][-1], num_envs=4, requested_targets=32)
                assert recorded['measured_execution'] == [asdict(row) for row in physical]
                chunks += 1
            journals = list((root/'action-accounting'/f'rank-{ordinal:02d}').glob('*.json'))
            assert len(journals) == 1
            journal = json.loads(journals[0].read_bytes())
            assert all(call['resolved'] for call in journal['calls'])
            amount = sum(call['executed_or_conservative'] for call in journal['calls'])
            assert amount == child['charged_actions'] == binding['charged_actions']
            assert amount <= journal['rank_action_ceiling']
            charged += amount
            summaries.append({key: child[key] for key in ('profile', 'logical_reference_rank',
                'constructor_seconds', 'first_reset_seconds', 'replay_seconds', 'charged_actions')})
        resources = json.loads((root/'resources.json').read_bytes())
        assert resources['sampled_reserve_passed']
        assert all(row['minimum_free_bytes'] >= max(8*2**30, .08*row['total_bytes'])
                   for row in resources['gpus'].values())
        assert chunks == 30 and len(cache) == 24 and charged <= plan['maximum_all_in_actions']
        assert not torch.cuda.is_initialized()
        report.update(status='passed', cold_json_files=len(manifest['files']),
            complete_sha256=sha(root/'COMPLETE.json'), result_sha256=sha(root/'RESULT.json'),
            manifest_sha256=sha(root/'FILES.json'), independent_reference_traces=24,
            replay_chunks=30, default_control_episodes=4, deferred_episodes=16,
            all5_native_parts_exact=True, physical_execution_exact=True,
            all5_children_closed=True, charged_actions=charged, sampled_reserve_passed=True,
            profiles=summaries, resources=resources, cuda_initialized=False,
            combined_training_memory_qualified=False, larger_scene_count_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    audit()
