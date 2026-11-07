"""Cold restore completed offline fits and diagnose teacher-forced source DEV.

No TEST tensors, simulator, action dispatch, new SGD or checkpoint selection.
These diagnostics do not estimate closed-loop robot success or adaptation.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from fit_decoded_student import BASE, physical_data
from fluffy_bassoon.meta_rl.chunk_training import ChunkTrainer, load_chunk_policy
from fluffy_bassoon.meta_rl.chunk_value import chunk_batch
from fluffy_bassoon.meta_rl.history_value import importance_weights
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash, write
from fluffy_bassoon.meta_rl.warmstart import checked_load


def main(args):
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    code = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip()
    assert source==os.environ['TASK_SOURCE']
    assert not subprocess.check_output(['git','-C',str(code),'status','--porcelain','--untracked-files=no'])
    root = BASE/args.fit_name
    parent = json.loads((root/'RESULT.json').read_bytes())
    assert parent['status']=='completed_offline_fits'
    assert set(parent['children'])=={'dit','ic_iql','static_behavior_cloning'}
    assert all(row['exit_code']==0 and not Path('/proc',str(row['pid'])).exists()
               for row in parent['children'].values())
    output = BASE/args.output_name
    assert root.resolve().parent==output.resolve().parent==BASE.resolve()
    output.mkdir()
    started = time.monotonic()
    report = dict(status='checking',source_commit=source,fit_source=parent['source_commit'],
        fit_report_sha256=sha(root/'RESULT.json'),gpu_seconds=0,optimizer_updates=0,
        test_tensor_loaded=False,teacher_forced_source_dev_only=True,
        robot_success_or_online_adaptation_qualified=False,methods={})
    write(output/'RESULT.json',report)
    try:
        _,_,data,data_contract = physical_data()
        for method in parent['children']:
            path = root/method
            result = json.loads((path/'RESULT.json').read_bytes())
            assert result['status']=='complete_offline_student_fitting'
            assert result['source_commit']==parent['source_commit']
            assert result['contract']['data']==data_contract
            assert result['updates']==dict(value=0 if method=='static_behavior_cloning' else 1000,actor=1000)
            assert not result['cuda_initialized'] and not result['test_tensor_loaded']
            last = result['checkpoints'][-1]
            saved = checked_load(path/last['file'],last['sha256'],weights_only=False)
            identities = {name:tensor_hash(saved[name]) for name in result['final_checkpoint_identity']}
            assert identities==result['final_checkpoint_identity']
            sampler = tensor_hash(saved['rng']['sampler'])
            final_actor = tensor_hash({k:v for k,v in saved['learner'].items() if k.startswith('actor.')})
            del saved;gc.collect()
            initial = result['checkpoints'][0]
            assert initial['update']==0
            saved = checked_load(path/initial['file'],initial['sha256'],weights_only=False)
            assert tensor_hash({k:v for k,v in saved['learner'].items() if k.startswith('actor.')})!=final_actor
            del saved;gc.collect()
            trainer = ChunkTrainer.restore(path/last['file'],last['sha256'],
                expected_contract=result['contract'],device='cpu')
            assert trainer.phase=='complete' and trainer.updates==result['updates']
            trainer.learner.eval().requires_grad_(False)
            policy,_ = load_chunk_policy(path/'policy.pt',result['policy_sha256'],
                                        expected_contract=result['contract'],device='cpu')
            assert all(torch.equal(value,policy.state_dict()[name])
                       for name,value in trainer.learner.actor.state_dict().items())
            mode = trainer.fit_config.mode
            with torch.no_grad():
                batch = chunk_batch(data['DEV'],policy,mode=mode,discount=trainer.objective_config.discount)
                valid = batch['valid'];reset = trainer.learner._sequence_resets(batch,mode)
                nll = -policy.log_prob_sequence(batch['tokens'],batch['actions'],reset=reset,valid=valid)
                embedding = policy.encode_sequence(batch['tokens'],reset=reset,valid=valid)
                predicted = policy.actor(embedding).tanh().reshape_as(batch['actions'])*policy.residual_radius
                scaled = predicted/policy.residual_radius
                row = dict(result_sha256=sha(path/'RESULT.json'),checkpoint_sha256=last['sha256'],
                    policy_sha256=result['policy_sha256'],full_resume_restored=True,frozen_export_exact=True,
                    actor_changed_from_initial=True,updates=result['updates'],valid_positions=result['valid_positions'],
                    actor_sampler_sha256=sampler,source_dev_physical_decisions=int(valid.sum()),
                    teacher_forced_history='complete32-episode source DEV streams; four scenes per stream',
                    actor_mode=mode,unweighted_full_command_nll=float(nll[valid].mean()),
                    normalized_residual_mae=float(((predicted-batch['actions'])/policy.residual_radius).abs()[valid].mean()),
                    zero_residual_normalized_mae=float((batch['actions']/policy.residual_radius).abs()[valid].mean()),
                    predicted_normalized_magnitude=float(scaled.abs()[valid].mean()),
                    predicted_coordinate_saturation_fraction=float((scaled.abs()[valid]>.95).float().mean()))
                if trainer.fit_config.value_updates:
                    q = trainer.learner._critic_sequence(trainer.learner.q,batch,mode)
                    v = trainer.learner._critic_sequence(trainer.learner.v,batch,mode)
                    weights = importance_weights(q-v,trainer.objective_config)[valid]
                    row.update(q_min=float(q[valid].min()),q_max=float(q[valid].max()),
                        behavior_mc_return_rmse=float((q-batch['returns'])[valid].square().mean().sqrt()),
                        weight_mean=float(weights.mean()),weight_max=float(weights.max()),
                        weight_cap_fraction=float((weights>=trainer.objective_config.maximum_weight).float().mean()),
                        weight_effective_sample_size=float(weights.sum().square()/weights.square().sum()),
                        effective_inverse_temperature=1/trainer.objective_config.temperature)
            assert row['source_dev_physical_decisions']==744
            report['methods'][method] = row
            write(output/'RESULT.json',report)
            print(json.dumps(dict(method=method,**row),allow_nan=False),flush=True)
            del trainer,policy,batch,embedding,predicted,nll
            gc.collect()
        rows = list(report['methods'].values())
        assert len({row['actor_sampler_sha256'] for row in rows})==1
        assert len({row['valid_positions']['actor'] for row in rows})==1
        assert not torch.cuda.is_initialized()
        report.update(status='passed',actor_window_streams_match=True,all_complete_recoveries_passed=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        report['seconds'] = time.monotonic()-started
        write(output/'RESULT.json',report)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fit-name',default='decodedfit02')
    parser.add_argument('--output-name',default='decodedfit02-readback01')
    main(parser.parse_args())
