"""Cold CPU comparison of every owned optimizer entry across native DCP save."""
import copy
import json
import os
from pathlib import Path

import torch

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def no_devices(value):
    if isinstance(value, dict): return {k:no_devices(v) for k,v in value.items() if k != 'device'}
    if isinstance(value, list): return [no_devices(v) for v in value]
    return value


def compare(before, after):
    assert before['rank'] == after['rank'] and before['optimizer_steps'] == after['optimizer_steps'] == 15
    assert no_devices(before['groups']) == no_devices(after['groups'])
    assert before['parameters'].keys() == after['parameters'].keys()
    empty_mapping_names = []
    owned = 0
    for name, a0 in before['parameters'].items():
        a, b = copy.deepcopy(a0), copy.deepcopy(after['parameters'][name])
        assert a['registered_model_parameter'] and a['requires_grad']
        assert b['registered_model_parameter'] and b['requires_grad']
        if a['local_nonempty_parameter']:
            assert a['parameter_numel'] > 0
            assert no_devices(a) == no_devices(b), name
            assert set(a['state']) == {'step','exp_avg','exp_avg_sq'}
            assert all(t['finite'] for t in a['state'].values())
            owned += 1
        else:
            for row in (a,b):
                assert row['parameter_shape'] == [0] and row['parameter_numel'] == 0
                state = row.pop('state')
                present = row.pop('state_present')
                if state:
                    assert present and set(state) == {'step','exp_avg','exp_avg_sq'}
                    assert state['step']['values'] == [0.] and state['step']['finite']
                    for key in ('exp_avg','exp_avg_sq'):
                        assert state[key]['shape'] == [0] and state[key]['numel'] == 0
                        assert state[key]['finite'] and state[key]['values'] == []
                elif present and row is b:
                    empty_mapping_names.append(name)
            assert a == b, name
    return dict(owned_parameters_exact=owned, empty_mapping_names=empty_mapping_names)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    source = BASE/'motuscheckpointaudit01-signals'
    gpu = json.loads((source/'RESULT.json').read_bytes())
    assert gpu['status'] == 'passed_measured_checkpoint_representation_roundtrip'
    assert gpu['optimizer_updates'] == gpu['simulator_actions'] == 0
    assert json.loads((source/'EXIT.json').read_bytes())['exit_code'] == 0
    assert json.loads((source/'resources.json').read_bytes())['sampled_reserve_passed']
    rows = sorted(gpu['actors'],key=lambda x:x['rank'])
    assert [r['rank'] for r in rows] == list(range(8))
    receipts, hashes = [], {}
    for row in rows:
        rank = row['rank']
        data = {}
        for kind in ('before','after-save','after-restore'):
            path = source/f'{kind}-rank-{rank:02d}.json'
            data[kind] = json.loads(path.read_bytes()); hashes[path.name] = sha(path)
        saved = compare(data['before'],data['after-save'])
        restored = compare(data['before'],data['after-restore'])
        assert saved['empty_mapping_names'] and sorted(saved['empty_mapping_names']) == sorted(row['normalized_names'])
        assert sorted(restored['empty_mapping_names']) == sorted(row['after_restore_empty_names'])
        assert row['original_guard_failure'] in data['after-save']['parameters']
        assert data['after-save']['parameters'][row['original_guard_failure']]['state'] == {}
        assert row['owned_optimizer_state_exact'] and row['policy_exact'] and row['rng_restored_exactly']
        receipts.append(dict(rank=rank, after_save=saved, after_restore=restored))
    output = BASE/'motuscheckpointaudit01-readback01'; output.mkdir()
    write(output/'RESULT.json',dict(status='passed_independent_checkpoint_representation_readback',
        source_commit=os.environ['TASK_SOURCE'],gpu_result_sha256=sha(source/'RESULT.json'),
        all24_raw_layout_hashes=hashes, ranks=receipts, gpu_seconds=0,
        all_owned_optimizer_entries_exact=True, empty_unowned_mapping_reproduced_all8=True,
        production_rule_changed=False, learning_gain_claim=False))


if __name__ == '__main__': main()
