"""CPU-only binding of one complete native B64 round before base reconstruction.

This stages fixed rank0/round0, including every replica and every issued chunk.
It does not admit student data, fit statistics, load a motor or act in a scene.
"""
import gzip
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
from audit_motus_collection import available_memory


BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
ROOT = Path(__file__).resolve().parent


def read_record(path):
    with gzip.open(path, 'rb') as stream:
        return torch.load(stream, map_location='cpu', weights_only=False)


def main():
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized() and available_memory() >= 24*2**30
    code = ROOT.parents[1]
    assert subprocess.check_output(['git', '-C', str(code), 'rev-parse', 'HEAD'], text=True).strip() == os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git', '-C', str(code), 'status', '--porcelain', '--untracked-files=no'])
    output = BASE/'cpureconstruction01'
    output.mkdir()
    report = dict(status='checking', source_commit=os.environ['TASK_SOURCE'], gpu_seconds=0,
        native_motus_or_simulator_constructed=False, source_corpus_admitted=False,
        fixed_selection=dict(rank=0, round=0, batch=64, chunks=6, policy_iteration=0),
        packets=[], sampling_headers=[], started_unix=time.time())
    try:
        previous = json.loads((ROOT/'MOTUSREPLAY01-CONTRACT.json').read_bytes())
        audit_path = BASE/'collection-audit02/RESULT.json'
        assert sha(audit_path) == previous['collection_audit_sha256']
        audited = json.loads(audit_path.read_bytes())
        assert audited['status'] == 'learner_replay_inputs_complete'
        binding = audited['ranks'][0]
        canonical_path = Path(binding['path'])
        assert canonical_path.stat().st_size == binding['bytes'] and sha(canonical_path) == binding['sha256']
        canonical = read_tensor(canonical_path)
        assert canonical['rank'] == canonical['producing_policy_iteration'] == 0
        assert canonical['collection_iteration'] == 1 and canonical['optimizer_steps_before'] == 0
        assert canonical['dropped_fields'] == []
        batch = canonical['rollout_batch']
        assert tuple(batch['forward_inputs']['chains'].shape) == (6, 256, 6, 32, 32)
        report.update(canonical_path=str(canonical_path), canonical_sha256=binding['sha256'],
            canonical_layout=canonical['layout'], collection_audit_sha256=sha(audit_path))
        state = BASE/'motusppo04/state'
        samplings = {}
        for path in sorted((state/'sampling/rank-00').glob('*.pt.gz')):
            value = read_record(path)
            header = {key: value[key] for key in ('mode', 'rank', 'prediction_ordinal',
                'producing_policy_iteration', 'evaluation_data', 'terminal_value_only')}
            report['sampling_headers'].append(header)
            if value['mode'] != 'train':
                continue
            assert not value['evaluation_data'] and value['rank'] == 0
            key = value['public_observation_sha256'], value['actions_sha256']
            # Preserve all candidates: a repeated no-op may share observations.
            samplings.setdefault(key, []).append((path, header))
        selected = []
        for path in sorted((state/'traces/rank-00').glob('*.pt.gz')):
            trace = read_record(path)
            if trace['evaluation_data'] or trace['round_index'] != 0:
                continue
            assert trace['rank'] == trace['producing_policy_iteration'] == 0
            selected.append((path, trace))
        selected.sort(key=lambda row: row[1]['chunk_index'])
        assert len(selected) == 6
        schedule = json.loads((ROOT/'MOTUS-TRAIN-CASES.json').read_bytes())
        groups = sorted((row for row in schedule['groups'] if row['rank'] == 0 and
            row['policy_iteration'] == row['round_index'] == 0), key=lambda row: row['local_group'])
        seeds = [row['scene_seed'] for row in groups for _ in range(8)]
        assert len(seeds) == 64
        noops = nonzero_noops = 0
        for index, (path, trace) in enumerate(selected):
            assert trace['chunk_index'] == index and trace['reset_state_ids'].tolist() == seeds
            actions = torch.as_tensor(trace['proposed_motor_actions'])
            assert tuple(actions.shape) == (64, 32, 14) and torch.isfinite(actions).all()
            public = EnvOutput(obs=trace['current_observation']).to_dict()['obs']
            matches = samplings[(tensor_hash(public), tensor_hash(trace['proposed_motor_actions']))]
            assert len(matches) == 1, 'Ambiguous behavior-sampler binding'
            sampling_path, header = matches[0]
            sampling = read_record(sampling_path)
            fields = sampling['forward_sampling_fields']
            chains = fields['chains']
            assert tuple(chains.shape) == (64, 6, 32, 32) and chains.dtype == torch.float32
            assert torch.equal(chains, batch['forward_inputs']['chains'][index, :64])
            assert torch.equal(fields['denoise_inds'], batch['forward_inputs']['denoise_inds'][index, :64])
            assert not sampling['evaluation_data'] and sampling['producing_policy_iteration'] == 0
            physics = collated_chunk_executions(trace['outputs'][4][-1], num_envs=64, requested_targets=32)
            info = trace['outputs'][4][-1]
            info = info.get('final_info', info)
            raw_feedback = info['fb2_native_feedback']
            for row, feedback in zip(physics, raw_feedback, strict=True):
                reward = float(np.asarray(feedback['reward']).item())
                assert np.isfinite(reward)
                noops += int(row.physics_steps == 0)
                nonzero_noops += int(row.physics_steps == 0 and reward != 0)
            packet = dict(format='motus_native_b64_reconstruction_input_v1', chunk=index,
                rank=0, round_index=0, policy_iteration=0, evaluation_data=False,
                current_observation=trace['current_observation'], original_gaussian=chains[:, 0].clone(),
                teacher_actions=actions, teacher_model_action=fields.get('model_action'),
                recorded_forward_action=fields.get('action'), processed_model_state=sampling['processed_model_state'],
                physical_execution=trace['physical_execution'], native_feedback=raw_feedback,
                native_terminated=trace['outputs'][2], native_truncated=trace['outputs'][3],
                reset_state_ids=trace['reset_state_ids'], episode_serials=trace['episode_serials'],
                trace=dict(path=str(path), sha256=sha(path)),
                sampling=dict(path=str(sampling_path), sha256=sha(sampling_path)),
                canonical=dict(path=str(canonical_path), sha256=binding['sha256'],
                    time_index=index, lane_slice=[0, 64]),
                current_observation_sha256=tensor_hash(trace['current_observation']),
                public_sampler_observation_sha256=tensor_hash(public),
                teacher_command_sha256=tensor_hash(actions))
            destination = output/f'chunk-{index:02d}.pt.gz'
            write_tensor(destination, packet)
            assert sum(p.stat().st_size for p in output.glob('*.pt.gz')) <= 3*2**30
            row = dict(chunk=index, packet=destination.name, bytes=destination.stat().st_size,
                sha256=sha(destination), sampling=header,
                fields={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                    for key,value in fields.items() if isinstance(value, torch.Tensor)},
                teacher_model_action_present=fields.get('model_action') is not None,
                processed_state_shape=list(sampling['processed_model_state'].shape))
            report['packets'].append(row)
            write(output/'RESULT.json', report)
            print(json.dumps(row), flush=True)
        assert not torch.cuda.is_initialized()
        report.update(status='inputs_bound', complete_native_b64_round=True,
            original_gaussians_and_denoising_indices_match_canonical=True,
            measured_zero_duration_calls=noops, nonzero_feedback_noops=nonzero_noops,
            source_corpus_admitted=False, cuda_initialized=False,
            required_next='Qualify original frozen Motus nativeB64/row ODE reconstruction, teacher decode and public features before dataset admission.')
    except BaseException as error:
        report.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report['seconds'] = time.time()-report['started_unix']
        write(output/'RESULT.json', report)
        print(json.dumps({key:value for key,value in report.items() if key not in ('packets','sampling_headers')}), flush=True)


if __name__ == '__main__':
    main()
