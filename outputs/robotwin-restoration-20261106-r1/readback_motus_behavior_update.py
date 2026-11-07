"""Independent CPU cold readback of the actual B32 update and learner archive."""
import copy
import json
import os
from pathlib import Path
import time

import torch

from optimizer_semantics import semantic_digest, without_devices
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, write

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def canonical(layout):
    layout = copy.deepcopy(layout)
    names = []
    for name, row in layout['parameters'].items():
        if row['state_present'] and row['state'] == {}:
            assert row['registered_model_parameter'] and row['requires_grad']
            assert not row['local_nonempty_parameter']
            assert row['parameter_shape'] == [0] and row['parameter_numel'] == 0
            row['state_present'] = False
            names.append(name)
    return semantic_digest(layout), sorted(names)


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    source = BASE/'motusbehaviorupdate01-signals'
    gpu = json.loads((source/'RESULT.json').read_bytes())
    assert gpu['status'] == 'passed_actual_B32_update_checkpoint_roundtrip'
    assert gpu['source_commit'] == '4568c24da4f24df19d964d3d50bc429ea44e9ba2'
    assert gpu['optimizer_updates'] == 15 and gpu['simulator_actions'] == 0
    assert not gpu['production_rule_changed'] and not gpu['learning_gain_claim']
    assert json.loads((source/'EXIT.json').read_bytes())['exit_code'] == 0
    assert json.loads((source/'resources.json').read_bytes())['sampled_reserve_passed']
    hashes, rows = {}, []
    for rank, row in enumerate(sorted(gpu['actors'],key=lambda r:r['rank'])):
        assert row['rank'] == rank and row['optimizer_steps'] == 30
        assert row['micro_batch'] == 32 and row['accumulation'] == 8
        assert row['owned_optimizer_state_exact'] and row['policy_exact_after_restore']
        assert row['frozen_parameters_exact'] and row['original_behavior_match_required']
        for key in ('actor/original_behavior_logprob_max_abs',
                    'actor/original_behavior_ratio_outside_clip_fraction',
                    'actor/rollout_train_logprob_gap'):
            assert row['native_metrics'][key] == 0, (rank,key)
        layouts = []
        for kind in ('after-update','after-save','after-restore'):
            path = source/f'{kind}-rank-{rank:02d}.json'
            value = json.loads(path.read_bytes());hashes[path.name]=sha(path)
            assert value['rank'] == rank and value['optimizer_steps'] == 30
            layouts.append(value)
        baseline = layouts[0]
        digest, names = canonical(baseline)
        owned = 0
        for name, entry in baseline['parameters'].items():
            if entry['local_nonempty_parameter']:
                owned += 1
                assert entry['parameter_numel'] > 0 and entry['state_present']
                assert set(entry['state']) == {'step','exp_avg','exp_avg_sq'}
                assert all(v['finite'] for v in entry['state'].values())
                for other in layouts[1:]:
                    assert without_devices(entry) == without_devices(other['parameters'][name]), (rank,name)
        for other in layouts[1:]:
            assert canonical(other)[0] == digest, rank
        assert names == sorted(row['normalized_empty_unowned_names'])
        original_error = None
        try: semantic_digest(baseline)
        except AssertionError as error: original_error = str(error)
        assert original_error == row['original_guard_failure']
        first = source/f'first-replay/rank-{rank:02d}.json'
        replay = json.loads(first.read_bytes());hashes[str(first.relative_to(source))] = sha(first)
        assert replay['passed'] and replay['logprob_max_abs'] == 0
        assert replay['micro_batch'] == 32 and replay['optimizer_steps'] == 15
        memory = source/f'actor-memory/iteration-0002-rank-{rank:02d}.json'
        measured = json.loads(memory.read_bytes());hashes[str(memory.relative_to(source))] = sha(memory)
        assert measured['micro_batch'] == 32 and measured['accumulation'] == 8
        rows.append(dict(rank=rank,owned_parameters_exact=owned,
            normalized_empty_unowned_count=len(names),original_guard_failure=original_error,
            update_seconds=measured['seconds'],peak_allocated_bytes=measured['peak_allocated_bytes']))
    assert len(rows) == 8
    checkpoint = Path(gpu['retained_learner_checkpoint'])
    assert sha(checkpoint/'MANIFEST.json') == gpu['checkpoint_manifest_sha256']
    manifest = json.loads((checkpoint/'MANIFEST.json').read_bytes())
    assert manifest['scope'] == 'learner_only_retained_batch_qualification'
    assert sorted(manifest['files']) == sorted(str(p.relative_to(checkpoint))
        for p in checkpoint.rglob('*') if p.is_file() and p.name != 'MANIFEST.json')
    started = time.monotonic()
    for name, receipt in manifest['files'].items():
        path = checkpoint/name
        assert path.stat().st_size == receipt['bytes'] and sha(path) == receipt['sha256'], name
    output = BASE/'motusbehaviorupdate01-readback01';output.mkdir()
    write(output/'RESULT.json',dict(status='passed_independent_actual_B32_update_checkpoint_readback',
        source_commit=os.environ['TASK_SOURCE'],gpu_result_sha256=sha(source/'RESULT.json'),
        all24_raw_layout_and16_guard_memory_hashes=hashes,actors=rows,
        checkpoint_manifest_sha256=sha(checkpoint/'MANIFEST.json'),checkpoint_files=len(manifest['files']),
        checkpoint_bytes=sum(r['bytes'] for r in manifest['files'].values()),
        full_checkpoint_cold_readback=True,readback_seconds=time.monotonic()-started,
        all_owned_Adam_bytes_exact=True,all8_original_behavior_likelihoods_exact=True,
        gpu_seconds=0,production_rule_changed=False,fresh_online_collection=False,learning_gain_claim=False))


if __name__ == '__main__': main()
