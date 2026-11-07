"""Cold CPU readback of fixed DEV traces and the paired original baseline."""
import gzip
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from rlinf.data.schema.embodied_types import EnvOutput
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.rl.robotwin_execution import collated_chunk_executions

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def read(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def panel(root, policy, seeds, *, first_rng=None):
    report = dict(trace_files=0, sampling_files=0, proposed_targets=0,
                  delivered_targets=0, partial_targets=0, physics_steps=0,
                  total_physics_duration_seconds=0., zero_duration_nonzero_feedback=0,
                  file_sha256={}, cases=[], first_rng={})
    for rank in range(4):
        previous = None
        final = None
        positive = [False]*4
        for ordinal in range(6):
            trace_path = root/'traces'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz'
            sampling_path = root/'sampling'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz'
            trace, sampling = read(trace_path), read(sampling_path)
            for path in (trace_path, sampling_path):
                report['file_sha256'][str(path)] = sha(path)
            assert trace['evaluation_data'] and sampling['evaluation_data']
            assert trace['producing_policy_iteration'] == sampling['producing_policy_iteration'] == policy
            assert trace['rank'] == sampling['rank'] == rank
            assert trace['chunk_index'] == ordinal
            assert trace['reset_state_ids'].tolist() == seeds[rank*4:rank*4+4]
            actions = torch.as_tensor(trace['proposed_motor_actions'])
            assert actions.shape == (4, 32, 14) and torch.isfinite(actions).all()
            assert tensor_hash(actions) == sampling['actions_sha256']
            public = EnvOutput(obs=trace['current_observation']).to_dict()['obs']
            assert tensor_hash(public) == sampling['public_observation_sha256']
            if previous is not None:
                assert tensor_hash(trace['current_observation']) == tensor_hash(previous)
            if ordinal == 0:
                rng = sampling['rng_before']
                report['first_rng'][rank] = rng
                if first_rng is not None:
                    old = first_rng[rank]
                    assert torch.equal(rng['torch'], old['torch'])
                    assert len(rng['cuda']) == len(old['cuda']) == 1
                    assert torch.equal(rng['cuda'][0], old['cuda'][0])
            outputs = trace['outputs']
            previous = outputs[0][-1]
            info = outputs[4][-1]
            packet = info.get('final_info', info)
            rows = collated_chunk_executions(info, num_envs=4, requested_targets=32)
            measured = trace['physical_execution']
            masks = torch.tensor([row.valid_target_mask for row in rows])
            assert trace['passive_physical_accounting'] and trace['measured_valid_mask']
            assert torch.equal(masks, measured['valid_target_mask'])
            assert int(masks.sum()) == trace['executed_or_conservative']
            for index, row in enumerate(rows):
                raw = packet['fb2_native_feedback'][index]
                for key in ('reward', 'terminated', 'truncated'):
                    assert np.asarray(raw[key]).shape == (1,)
                    assert np.isfinite(np.asarray(raw[key])).all()
                reward = float(np.asarray(raw['reward']).item())
                positive[index] |= reward > 0
                report['zero_duration_nonzero_feedback'] += int(row.physics_steps == 0 and reward != 0)
                report['delivered_targets'] += row.delivered_targets
                report['partial_targets'] += row.partial_targets
                report['physics_steps'] += row.physics_steps
                report['total_physics_duration_seconds'] += row.duration_seconds
                assert measured['physics_steps'][index] == row.physics_steps
                assert measured['duration_seconds'][index] == row.duration_seconds
            if ordinal < 5:
                assert not outputs[2].any() and not outputs[3].any()
            else:
                assert outputs[3][:, -1].all()
                assert 'final_info' in info and 'final_observation' in info
                assert info['_final_info'].all()
                final = packet['episode']
                assert torch.as_tensor(final['episode_len']).eq(192).all()
            report['trace_files'] += 1
            report['sampling_files'] += 1
            report['proposed_targets'] += actions.shape[0]*actions.shape[1]
        print('PANEL_RANK_READ', policy, rank, flush=True)
        for index, seed in enumerate(seeds[rank*4:rank*4+4]):
            report['cases'].append(dict(seed=seed, rank=rank, batch_row=index,
                success_once=bool(torch.as_tensor(final['success_once'])[index]),
                success_at_end=bool(torch.as_tensor(final['success_at_end'])[index]),
                native_return=float(torch.as_tensor(final['return'])[index]),
                raw_positive_reward=positive[index]))
    report['successes'] = sum(row['success_once'] for row in report['cases'])
    report['raw_positive_reward_cases'] = sum(row['raw_positive_reward'] for row in report['cases'])
    assert len(report['cases']) == 16 and report['trace_files'] == report['sampling_files'] == 24
    assert report['proposed_targets'] == 3072
    return report


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_initialized()
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    assert not destination.exists()
    destination.mkdir()
    root = BASE/'motusdev01-signals'
    complete = json.loads((root/'COMPLETE.json').read_bytes())
    assert complete['status'] == 'complete_native_source_dev'
    manifest = json.loads((root/'FILES.json').read_bytes())
    for name, expected in manifest['files'].items():
        path = root/name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256']
    result = json.loads((root/'state/RESULT.json').read_bytes())
    assert result['status'] == 'complete' and result['optimizer_updates_during_eval'] == 0
    assert all(row['frozen_parameters_unchanged'] for row in result['frozen_rollouts'])
    assert all(row['passed'] and row['first_sft_action_max_abs_error'] == 0
               for row in result['baseline_calibration'])
    seeds = json.loads((source/'MOTUS-DEV-CASES.json').read_bytes())['seeds']
    baseline = panel(BASE/'motusppo04/state', 0, seeds)
    updated = panel(root/'state', 1, seeds, first_rng=baseline['first_rng'])
    baseline.pop('first_rng'); updated.pop('first_rng')
    assert baseline['successes'] == 4
    assert updated['successes']/16 == result['metrics']['success_once']
    paired = [dict(seed=a['seed'], before=a['success_once'], after=b['success_once'])
              for a, b in zip(baseline['cases'], updated['cases'], strict=True)]
    report = dict(status='passed', source_commit=os.environ['TASK_SOURCE'],
        manifest_sha256=sha(root/'FILES.json'), result_sha256=sha(root/'state/RESULT.json'),
        complete_sha256=sha(root/'COMPLETE.json'), all_new_state_files_cold_read=True,
        baseline=baseline, updated=updated, paired=paired,
        original_batch_and_first_torch_cuda_rng_equal=True,
        raw_pre_conversion_feedback_retained=True, terminal_autoreset_packet_verified=True,
        optimizer_updates_during_eval=0, held_out_transfer_evaluation=False,
        scientific_source_admission=False, learning_improvement_claim=False,
        cuda_initialized=torch.cuda.is_initialized())
    write(destination/'RESULT.json', report)
    print(json.dumps({key:report[key] for key in ('status','source_commit','result_sha256')}), flush=True)
    print('SUCCESSES', baseline['successes'], updated['successes'], 'PAIRED', paired, flush=True)


if __name__ == '__main__':
    main()
