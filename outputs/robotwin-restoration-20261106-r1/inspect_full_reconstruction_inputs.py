"""Stream all four canonical ranks into native-B64 reconstruction packets.

Student case partitions are fixed before features, statistics, windows or fits.
Every replica/issued command remains present; physical no-ops are only audited.
"""
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from rlinf.data.schema.embodied_types import EnvOutput

from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor
from fluffy_bassoon.rl.robotwin_execution import collated_chunk_executions
from inspect_reconstruction_inputs import BASE, ROOT, read_record, available_memory


def stage_rank(rank, binding, schedule, split, output, report):
    assert available_memory() >= 24*2**30
    canonical_path = Path(binding['path'])
    assert canonical_path.stat().st_size == binding['bytes'] and sha(canonical_path) == binding['sha256']
    print('READ_CANONICAL_BEGIN', rank, flush=True)
    canonical = read_tensor(canonical_path)
    assert canonical['rank'] == rank and canonical['producing_policy_iteration'] == 0
    assert canonical['collection_iteration'] == 1 and canonical['optimizer_steps_before'] == 0
    assert canonical['dropped_fields'] == []
    batch = canonical['rollout_batch']
    assert tuple(batch['forward_inputs']['chains'].shape) == (6, 256, 6, 32, 32)
    print('READ_CANONICAL_COMPLETE', rank, flush=True)
    state = BASE/'motusppo04/state'
    samplings = {}
    for path in sorted((state/f'sampling/rank-{rank:02d}').glob('*.pt.gz')):
        value = read_record(path)
        if value['mode'] != 'train':
            continue
        assert not value['evaluation_data'] and value['rank'] == rank
        key = value['public_observation_sha256'], value['actions_sha256']
        samplings.setdefault(key, []).append(path)
    del value
    count = 0
    for path in sorted((state/f'traces/rank-{rank:02d}').glob('*.pt.gz')):
        trace = read_record(path)
        if trace['evaluation_data']:
            continue
        round_index, index = divmod(count, 6)
        assert trace['rank'] == rank and trace['producing_policy_iteration'] == 0
        assert (trace['round_index'], trace['chunk_index']) == (round_index, index)
        groups = sorted((row for row in schedule['groups'] if row['rank'] == rank and
            row['policy_iteration'] == 0 and row['round_index'] == round_index), key=lambda row: row['local_group'])
        seeds = [row['scene_seed'] for row in groups for _ in range(8)]
        partitions = [split['partition_by_local_group'][str(row['local_group'])] for row in groups for _ in range(8)]
        assert len(seeds) == 64 and trace['reset_state_ids'].tolist() == seeds
        actions = torch.as_tensor(trace['proposed_motor_actions'])
        assert tuple(actions.shape) == (64, 32, 14) and torch.isfinite(actions).all()
        public = EnvOutput(obs=trace['current_observation']).to_dict()['obs']
        matches = samplings[(tensor_hash(public), tensor_hash(trace['proposed_motor_actions']))]
        assert len(matches) == 1, 'Ambiguous native sampler binding'
        sampling_path = matches[0]
        sampling = read_record(sampling_path)
        assert sampling['producing_policy_iteration'] == 0 and not sampling['evaluation_data']
        fields = sampling['forward_sampling_fields']
        chains = fields['chains']
        assert tuple(chains.shape) == (64, 6, 32, 32) and chains.dtype == torch.float32
        section = slice(round_index*64, (round_index+1)*64)
        assert torch.equal(chains, batch['forward_inputs']['chains'][index, section])
        assert torch.equal(fields['denoise_inds'], batch['forward_inputs']['denoise_inds'][index, section])
        assert torch.equal(fields['action'].reshape(64, 32, 14), actions)
        assert torch.equal(fields['model_action'].reshape(64, 32, 32), chains[:, -1])
        physics = collated_chunk_executions(trace['outputs'][4][-1], num_envs=64, requested_targets=32)
        info = trace['outputs'][4][-1]
        info = info.get('final_info', info)
        raw = info['fb2_native_feedback']
        for row, feedback in zip(physics, raw, strict=True):
            reward = float(np.asarray(feedback['reward']).item())
            assert np.isfinite(reward)
            report['measured_zero_duration_calls'] += int(row.physics_steps == 0)
            report['nonzero_feedback_noops'] += int(row.physics_steps == 0 and reward != 0)
        packet = dict(format='motus_native_b64_reconstruction_input_v1', chunk=index,
            rank=rank, round_index=round_index, policy_iteration=0, evaluation_data=False,
            current_observation=trace['current_observation'], original_gaussian=chains[:, 0].clone(),
            teacher_actions=actions, teacher_model_action=fields['model_action'],
            recorded_forward_action=fields['action'], processed_model_state=sampling['processed_model_state'],
            physical_execution=trace['physical_execution'], native_feedback=raw,
            native_terminated=trace['outputs'][2], native_truncated=trace['outputs'][3],
            reset_state_ids=trace['reset_state_ids'], episode_serials=trace['episode_serials'],
            student_partition_per_row=partitions, split_sha256=sha(ROOT/'STUDENT-CASE-SPLIT.json'),
            trace=dict(path=str(path), sha256=sha(path)),
            sampling=dict(path=str(sampling_path), sha256=sha(sampling_path)),
            canonical=dict(path=str(canonical_path), sha256=binding['sha256'],
                time_index=index, lane_slice=[round_index*64, (round_index+1)*64]),
            current_observation_sha256=tensor_hash(trace['current_observation']),
            public_sampler_observation_sha256=tensor_hash(public), teacher_command_sha256=tensor_hash(actions))
        ordinal = len(report['packets'])
        destination = output/f'packet-{ordinal:03d}.pt.gz'
        write_tensor(destination, packet)
        row = dict(ordinal=ordinal, rank=rank, round_index=round_index, chunk=index,
            packet=destination.name, bytes=destination.stat().st_size, sha256=sha(destination),
            sampling_prediction_ordinal=sampling['prediction_ordinal'],
            full_teacher_command_and_final_chain_exact=True, canonical_gaussian_and_denoising_indices_exact=True)
        report['packets'].append(row)
        assert sum(row['bytes'] for row in report['packets']) <= 16*2**30
        write(output/'RESULT.json', report)
        print(json.dumps(row), flush=True)
        count += 1
        del packet, trace, sampling, fields, chains, actions, public, physics, info, raw
    assert count == 24
    del canonical, batch
    gc.collect()


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output = BASE/'cpureconstruction03'
    output.mkdir()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'], gpu_seconds=0,
        native_motus_or_simulator_constructed=False, source_corpus_admitted=False,
        fixed_selection=dict(ranks=[0, 1, 2, 3], rounds=[0, 1, 2, 3], batch=64, chunks=6, policy_iteration=0),
        packets=[], measured_zero_duration_calls=0, nonzero_feedback_noops=0, started_unix=time.time())
    try:
        assert any(os.path.ismount(path) for path in (BASE, Path('/mnt/bos/adgfyl'), Path('/mnt/bos')))
        probe = output/'write-probe'
        with probe.open('xb') as stream:
            stream.write(b'native-b64-reconstruction-inputs');stream.flush();os.fsync(stream.fileno())
        assert probe.read_bytes() == b'native-b64-reconstruction-inputs'
        probe.unlink()
        previous = json.loads((ROOT/'MOTUSREPLAY01-CONTRACT.json').read_bytes())
        audit_path = BASE/'collection-audit02/RESULT.json'
        assert sha(audit_path) == previous['collection_audit_sha256']
        audited = json.loads(audit_path.read_bytes())
        assert audited['status'] == 'learner_replay_inputs_complete' and len(audited['ranks']) == 4
        schedule = json.loads((ROOT/'MOTUS-TRAIN-CASES.json').read_bytes())
        split = json.loads((ROOT/'STUDENT-CASE-SPLIT.json').read_bytes())
        assert split['case_manifest_sha256'] == sha(ROOT/'MOTUS-TRAIN-CASES.json')
        assert split['partition_by_local_group'] == {str(i):'TRAIN' if i < 6 else 'DEV' if i == 6 else 'TEST' for i in range(8)}
        report.update(collection_audit_sha256=sha(audit_path), split_sha256=sha(ROOT/'STUDENT-CASE-SPLIT.json'))
        for rank, binding in enumerate(audited['ranks']):
            assert binding['rank'] == rank
            stage_rank(rank, binding, schedule, split, output, report)
        assert len(report['packets']) == 96 and not torch.cuda.is_initialized()
        report.update(status='inputs_bound', complete_native_b64_collection=True,
            original_gaussians_and_denoising_indices_match_canonical=True,
            issued_command_rows=6144, episode_occurrences=1024, unique_source_scenes=128,
            source_corpus_admitted=False, cuda_initialized=False,
            required_next='Qualify original frozen nativeB64/row ODE reconstruction and teacher decode; audit physical episode/no-op semantics before student data admission.')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.time()-report['started_unix']
        write(output/'RESULT.json', report)
        print(json.dumps({key:value for key,value in report.items() if key != 'packets'}), flush=True)


if __name__ == '__main__':
    main()
