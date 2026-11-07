"""Independent CPU trace, actual-noise and paired-outcome reduction."""
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from rlinf.data.schema.embodied_types import EnvOutput
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor
from fluffy_bassoon.rl.robotwin_execution import collated_chunk_executions
import motus_teacher_entry
from recovery_signature import digest

BASE = Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')
ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('identity')
    args = parser.parse_args()
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized()
    identity = args.identity
    assert identity in [f'motus-source-{m}-{b}01' for m in ('ppo','grpo') for b in ('dev','confirmation')]
    state = BASE/identity/'state';signals = BASE/(identity+'-signals')
    assert json.loads((signals/'EXIT.json').read_bytes())['exit_code'] == 0
    complete = json.loads((signals/'COMPLETE.json').read_bytes())
    result = json.loads((state/'RESULT.json').read_bytes())
    assert result['status'] == 'complete_readonly_full_restore_paired_source_readout'
    assert result['episodes'] == 128 and result['optimizer_updates'] == 0
    assert result['full24worker_restores_each_policy'] and result['actor_Adam_scheduler_scaler_RNG_unchanged']
    assert json.loads((signals/'resources.json').read_bytes())['sampled_reserve_passed']
    assert complete['result_sha256'] == sha(state/'RESULT.json')
    manifest = json.loads((signals/'FILES.json').read_bytes())
    assert sorted(manifest['files']) == sorted(str(p.relative_to(state)) for p in state.rglob('*') if p.is_file())
    for name,expected in manifest['files'].items():
        path = state/name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    cases = json.loads((ROOT/'MOTUS-SOURCE-READOUT-CASES.json').read_bytes())[result['bank']]
    noises,initial,episodes = {},{},{}
    physical = dict(delivered_targets=0,physics_steps=0,duration_seconds=0.)
    for policy_ordinal,step in enumerate((0,2)):
        for panel in range(4):
            panel_rows = []
            for rank in range(8):
                previous = final = None
                positive = [False,False]
                noise_report = json.loads((state/'actual-eval-noise'/f'policy-{step:04d}-panel-{panel:02d}-rank-{rank:02d}.json').read_bytes())
                assert noise_report['frozen_parameters_unchanged'] and noise_report['native_ODE_sampler_unchanged']
                assert len(noise_report['samples']) == 6 and noise_report['optimizer_updates'] == 0
                seeds = cases[panel*16+rank*2:panel*16+rank*2+2]
                for chunk in range(6):
                    ordinal = policy_ordinal*24+panel*6+chunk
                    trace = read_tensor(state/'traces'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz')
                    sampling = read_tensor(state/'sampling'/f'rank-{rank:02d}'/f'{ordinal:05d}.pt.gz')
                    assert trace['evaluation_data'] and sampling['evaluation_data']
                    assert trace['phase'] == f'readout-policy-{step:04d}-panel-{panel:02d}'
                    assert trace['rank'] == sampling['rank'] == rank
                    assert trace['producing_policy_iteration'] == sampling['producing_policy_iteration'] == step
                    assert trace['chunk_index'] == chunk and trace['reset_state_ids'].tolist() == seeds
                    actions = torch.as_tensor(trace['proposed_motor_actions'])
                    assert actions.shape == (2,32,14) and torch.isfinite(actions).all()
                    assert tensor_hash(actions) == sampling['actions_sha256']
                    public = EnvOutput(obs=trace['current_observation']).to_dict()['obs']
                    assert tensor_hash(public) == sampling['public_observation_sha256']
                    if previous is not None:assert tensor_hash(trace['current_observation']) == tensor_hash(previous)
                    if chunk == 0:
                        key = panel,rank
                        if step == 0:initial[key] = tensor_hash(trace['current_observation'])
                        else:assert initial[key] == tensor_hash(trace['current_observation'])
                    row = noise_report['samples'][chunk]
                    assert row['prediction_ordinal'] == sampling['prediction_ordinal'] == ordinal
                    assert row['extra_random_draws'] == 0
                    path = Path(row['raw_tensor']['path'])
                    assert sha(path) == row['raw_tensor']['sha256']
                    noise = read_tensor(path)['initial_gaussian']
                    assert list(noise.shape) == row['shape'] == [2,32,32]
                    assert digest(noise) == row['initial_gaussian_sha256']
                    assert [digest(v) for v in noise] == row['per_row_initial_gaussian_sha256']
                    key = panel,rank,chunk
                    if step == 0:noises[key] = noise
                    else:assert torch.equal(noises[key],noise)
                    if step == 0 and panel == 0 and result['bank'] == 'dev':assert row['source_SFT_actions_pixels_RNG_exact']
                    outputs = trace['outputs'];previous = outputs[0][-1]
                    info = outputs[4][-1];packet = info.get('final_info',info)
                    measured = trace['physical_execution']
                    execution = collated_chunk_executions(info,num_envs=2,requested_targets=32)
                    assert trace['passive_physical_accounting'] and trace['measured_valid_mask']
                    mask = torch.tensor([v.valid_target_mask for v in execution])
                    assert torch.equal(mask,measured['valid_target_mask'])
                    assert int(mask.sum()) == trace['executed_or_conservative']
                    for i,value in enumerate(execution):
                        raw = packet['fb2_native_feedback'][i]
                        for k in ('reward','terminated','truncated'):assert np.asarray(raw[k]).shape == (1,) and np.isfinite(np.asarray(raw[k])).all()
                        positive[i] |= float(np.asarray(raw['reward']).item()) > 0
                        assert measured['physics_steps'][i] == value.physics_steps
                        assert measured['duration_seconds'][i] == value.duration_seconds
                        physical['physics_steps'] += value.physics_steps
                        physical['delivered_targets'] += value.delivered_targets
                        physical['duration_seconds'] += value.duration_seconds
                    if chunk < 5:assert not outputs[2].any() and not outputs[3].any()
                    else:
                        assert outputs[3][:,-1].all() and info['_final_info'].all()
                        assert 'final_observation' in info
                        final = packet['episode']
                        assert torch.as_tensor(final['episode_len']).eq(192).all()
                for i,seed in enumerate(seeds):
                    row = dict(seed=seed,policy_iteration=step,rank=rank,batch_row=i,panel=panel,
                        success=bool(torch.as_tensor(final['success_once'])[i]),
                        success_at_end=bool(torch.as_tensor(final['success_at_end'])[i]),
                        native_return=float(torch.as_tensor(final['return'])[i]),raw_positive_reward=positive[i])
                    episodes[step,seed] = row;panel_rows.append(row)
            native = next(v for v in result['panels'] if v['step'] == step and v['panel'] == panel)
            assert sum(v['success'] for v in panel_rows)/16 == native['metrics']['success_once']
            print(identity,step,panel,'independent traces/noise/outcomes passed',flush=True)
    paired = [dict(seed=s,before=episodes[0,s]['success'],after=episodes[2,s]['success']) for s in cases]
    gains = sum(not v['before'] and v['after'] for v in paired)
    losses = sum(v['before'] and not v['after'] for v in paired)
    discordant = gains+losses
    pvalue = min(1.,2*sum(math.comb(discordant,k) for k in range(min(gains,losses)+1))/2**discordant) if discordant else 1.
    gain = (gains-losses)/64
    output = BASE/(identity+'-readback01');output.mkdir()
    write(output/'RESULT.json',dict(status='passed_independent_paired_source_trace_noise_readback',
        source_commit=os.environ['TASK_SOURCE'],algorithm=result['algorithm'],bank=result['bank'],
        GPU_result_sha256=sha(state/'RESULT.json'),all_state_files_cold_hashed=True,
        file_manifest_sha256=sha(signals/'FILES.json'),files=len(manifest['files']),
        actual_noise_all384_batch_draws_recorded=True,all192_paired_batch_draws_tensor_exact=True,
        all32_initial_B2_scene_observations_exact=True,native_batch=2,episode_horizon=192,
        episodes=128,baseline_successes=sum(v['before'] for v in paired),updated_successes=sum(v['after'] for v in paired),
        gained_cases=gains,lost_cases=losses,paired_gain=gain,exact_two_sided_paired_p=pvalue,paired=paired,
        practical_and_statistical_threshold_passed=(gain>=.1 and pvalue<.05),
        effective_source_confirmation_passed=(result['bank']=='confirmation' and gain>=.1 and pvalue<.05),
        one_training_seed_only=True,held_out_transfer_claim=False,optimizer_updates=0,gpu_seconds=0,physical=physical))
    print('PAIRED_RESULT',gains,losses,'gain',gain,'p',pvalue,flush=True)


if __name__ == '__main__':main()
