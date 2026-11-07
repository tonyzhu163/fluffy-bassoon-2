"""Independent Linux CPU readback of the bounded native offload fixture."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from fluffy_bassoon.baige_gpu import pci
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.rl.robotwin_execution import ChunkExecution, collated_chunk_executions


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
PROFILES = [('eval', False), ('eval', True), ('train', False), ('train', True)]


def audit():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = Path(os.environ['TASK_CODE'])
    root, output = BASE/'lifecycle01-signals', BASE/'lifecycle01-readback01'
    output.mkdir()
    report = dict(status='checking', gpu_seconds=0, model_calls=0, optimizer_updates=0,
        runtime_source_commit=os.environ['TASK_SOURCE'],
        audit_source_commit=os.environ['TASK_AUDIT_SOURCE'])
    try:
        assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == report['runtime_source_commit']
        assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
        plan = json.loads((code/'outputs/robotwin-restoration-20261106-r1/LIFECYCLE01-CONTRACT.json').read_bytes())
        result = json.loads((root/'RESULT.json').read_bytes())
        complete = json.loads((root/'COMPLETE.json').read_bytes())
        assert complete['status'] == 'complete_native_lifecycle'
        assert complete['result_sha256'] == sha(root/'RESULT.json')
        assert result['status'] == 'passed' and len(result['children']) == 4
        assert json.loads((root/'EXIT.json').read_bytes())['exit_code'] == 0
        manifest = json.loads((root/'FILES.json').read_bytes())['files']
        payloads = {}
        for name, expected in manifest.items():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            path = root/name
            assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
            payloads[name] = json.loads(path.read_bytes())
        reference = BASE/'motusdev01-signals'
        assert sha(reference/'FILES.json') == plan['reference_manifest_sha256']
        assert sha(BASE/'cpudev-readback02/RESULT.json') == plan['reference_readback_sha256']
        original_manifest = json.loads((reference/'FILES.json').read_bytes())['files']
        original = []
        for index in range(6):
            name = f'state/traces/rank-00/{index:05d}.pt.gz'
            path = reference/name
            assert path.stat().st_size == original_manifest[name]['bytes']
            assert sha(path) == original_manifest[name]['sha256']
            row = read_tensor(path)
            assert row['evaluation_data'] and row['producing_policy_iteration'] == 1
            original.append(row)
        probe = payloads['gpu-probe.json']
        assert probe['status'] == 'passed' and probe['hostname'] == 'job-si1w13odluqx-master-0'
        assert len(probe['devices']) == 1
        device = probe['devices'][0]
        expected_pci = pci(device['pci_bus_id'])
        charged, chunks, offloads, profiles = 0, 0, 0, []
        for ordinal, (mode, deferred) in enumerate(PROFILES):
            relative = f'{ordinal:02d}-{mode}-{int(deferred)}'
            path = root/relative/'RESULT.json'
            child = payloads[relative+'/RESULT.json']
            binding = result['children'][ordinal]
            assert Path(binding['path']) == path and binding['sha256'] == sha(path)
            assert child['status'] == 'passed' and child['final_executor_closed']
            assert child['cleanup_phase'] == 'complete' and not child.get('native_close_failed', False)
            assert child['source_commit'] == report['runtime_source_commit']
            assert child['mode'] == mode and child['deferred_setup'] == deferred
            assert child['executor_recycling'] and child['model_calls'] == child['optimizer_updates'] == 0
            assert child['episodes'] == 16 and len(child['chunks']) == 24
            assert [row['phase'] for row in child['offloads']] == ['native-initial-offload', 'collection-0', 'collection-1']
            assert all(row['all_joined'] and row['replacement_created'] for row in child['offloads'])
            setup = payloads[relative+'/setup-profiles/rank-00.json']
            assert setup == dict(defer_scene_setup=deferred, recycle_scene_executor=True,
                shared_native_training_and_evaluation=True)
            worker = payloads[relative+'/worker-gpus/env-00.json']
            assert pci(worker['cuda_pci']) == expected_pci and worker['scheduler_uuid'] == device['uuid']
            renderers = [value for name, value in payloads.items() if name.startswith(relative+'/renderers/')]
            assert renderers and all(pci(row['actual_render_pci']) == pci(row['expected_cuda_pci']) == expected_pci for row in renderers)
            for number, recorded in enumerate(child['chunks']):
                collection, episode, index = number//12, number//6 % 2, number % 6
                assert (recorded['collection'], recorded['episode'], recorded['chunk']) == (collection, episode, index)
                name = f'state/traces/rank-00/{index:05d}.pt.gz'
                assert recorded['reference_trace_sha256'] == original_manifest[name]['sha256']
                assert recorded['command_sha256'] == tensor_hash(np.asarray(original[index]['proposed_motor_actions']).copy())
                measured = [ChunkExecution(**row) for row in recorded['measured_execution']]
                assert len(measured) == 4
                for row in measured:
                    assert row.requested_targets == 32
                    assert row.completed_targets + row.partial_targets == row.delivered_targets <= 32
                    assert row.partial_targets in (0, 1) and row.physics_steps >= 0 and row.timestep_seconds > 0
                    assert (row.physics_steps == 0) == (row.delivered_targets == 0)
                if mode == 'eval':
                    assert recorded['pre_observation_sha256'] == tensor_hash(original[index]['current_observation'])
                    assert recorded['native_output_sha256'] == [tensor_hash(part) for part in original[index]['outputs']]
                    expected = collated_chunk_executions(original[index]['outputs'][4][-1], num_envs=4, requested_targets=32)
                    assert recorded['measured_execution'] == [asdict(row) for row in expected]
                if deferred:
                    control = payloads[f'{ordinal-1:02d}-{mode}-0/RESULT.json']
                    assert recorded == control['chunks'][number]
                chunks += 1
            journals = [value for name, value in payloads.items() if name.startswith(f'action-accounting/rank-{ordinal:02d}/')]
            assert len(journals) == 1
            journal = journals[0]
            assert all(row['resolved'] for row in journal['calls'])
            amount = sum(row['executed_or_conservative'] for row in journal['calls'])
            assert amount == child['charged_actions'] == binding['charged_actions'] <= journal['rank_action_ceiling']
            charged += amount
            offloads += len(child['offloads'])
            profiles.append(dict(mode=mode, deferred_setup=deferred, episodes=16,
                replay_chunks=24, native_offloads=3, charged_actions=amount,
                renderer_records=len(renderers), all_joined=True, final_executor_closed=True))
        resources = payloads['resources.json']
        assert resources['sampled_reserve_passed'] and resources['samples'] > 0
        assert resources['host_limit_bytes'] >= 128*2**30
        assert set(resources['gpus']) == {device['uuid']}
        assert all(row['minimum_free_bytes'] >= max(8*2**30, .08*row['total_bytes']) for row in resources['gpus'].values())
        assert chunks == plan['vector_chunks'] == 96 and offloads == 12
        assert charged <= plan['maximum_all_in_actions']
        assert not torch.cuda.is_initialized()
        report.update(status='passed', cold_json_files=len(manifest), independent_reference_traces=6,
            complete_sha256=sha(root/'COMPLETE.json'), result_sha256=sha(root/'RESULT.json'),
            manifest_sha256=sha(root/'FILES.json'), episodes=64, vector_chunks=chunks,
            recurrent_offloads=offloads, charged_actions=charged, profiles=profiles,
            eval_reference_parts_exact=True, default_deferred_outputs_exact=True,
            allocated_renderer_binding_exact=True, resources=resources, cuda_initialized=False,
            combined_training_memory_qualified=False, larger_scene_count_qualified=False,
            long_uninterrupted_reset_stress_qualified=False)
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write(output/'RESULT.json', report)
        print(json.dumps(report), flush=True)


if __name__ == '__main__':
    audit()
