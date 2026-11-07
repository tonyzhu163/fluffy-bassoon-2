"""Cold-restore the matched history-imitation fit and diagnose source DEV only."""
import gc
import json
import os
import sys

import torch

from fit_decoded_student import BASE, physical_data, write
from fluffy_bassoon.meta_rl.chunk_training import ChunkTrainer, load_chunk_policy
from fluffy_bassoon.meta_rl.chunk_value import chunk_batch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha, tensor_hash
from fluffy_bassoon.meta_rl.warmstart import checked_load


def main():
    assert sys.platform=='linux' and os.environ['CUDA_VISIBLE_DEVICES']==''
    root=BASE/'decodedhistorybc01';out=BASE/'decodedhistorybc01-readback01';out.mkdir()
    report=dict(status='checking',source_commit=os.environ['TASK_SOURCE'],gpu_seconds=0,
        optimizer_updates=0,test_tensor_loaded=False,chronological_ad_qualified=False,
        robot_success_or_adaptation_qualified=False)
    write(out/'RESULT.json',report)
    try:
        result=json.loads((root/'RESULT.json').read_bytes())
        assert result['status']=='complete_offline_student_fitting'
        assert result['updates']==dict(value=0,actor=1000)
        _,_,data,contract=physical_data(history_imitation=True)
        assert contract==result['contract']['data']
        boundary=result['checkpoints'][-1]
        saved=checked_load(root/boundary['file'],boundary['sha256'],weights_only=False)
        assert {key:tensor_hash(saved[key]) for key in result['final_checkpoint_identity']}==result['final_checkpoint_identity']
        sampler=tensor_hash(saved['rng']['sampler'])
        final_actor=tensor_hash({k:v for k,v in saved['learner'].items() if k.startswith('actor.')})
        del saved;gc.collect()
        initial=result['lineage_initial_checkpoint']
        saved=checked_load(initial['path'],initial['sha256'],weights_only=False)
        assert final_actor!=tensor_hash({k:v for k,v in saved['learner'].items() if k.startswith('actor.')})
        del saved;gc.collect()
        trainer=ChunkTrainer.restore(root/boundary['file'],boundary['sha256'],expected_contract=result['contract'],device='cpu')
        assert trainer.phase=='complete' and trainer.updates==result['updates']
        policy,_=load_chunk_policy(root/'policy.pt',result['policy_sha256'],expected_contract=result['contract'],device='cpu')
        assert all(torch.equal(value,policy.state_dict()[name]) for name,value in trainer.learner.actor.state_dict().items())
        matched=json.loads((BASE/'decodedfit02-readback01/RESULT.json').read_bytes())
        assert matched['status']=='passed'
        for other in matched['methods'].values():
            assert sampler==other['actor_sampler_sha256']
            assert result['valid_positions']['actor']==other['valid_positions']['actor']
        with torch.no_grad():
            batch=chunk_batch(data['DEV'],policy,mode='full');valid=batch['valid']
            reset=trainer.learner._sequence_resets(batch,'full')
            nll=-policy.log_prob_sequence(batch['tokens'],batch['actions'],reset=reset,valid=valid)
            embedding=policy.encode_sequence(batch['tokens'],reset=reset,valid=valid)
            predicted=policy.actor(embedding).tanh().reshape_as(batch['actions'])*policy.residual_radius
            assert int(valid.sum())==744
            metrics=dict(source_dev_decisions=744,unweighted_full_command_nll=float(nll[valid].mean()),
                normalized_residual_mae=float(((predicted-batch['actions'])/policy.residual_radius).abs()[valid].mean()),
                zero_residual_normalized_mae=float((batch['actions']/policy.residual_radius).abs()[valid].mean()))
        assert not torch.cuda.is_initialized()
        report.update(status='passed',fit_source=result['source_commit'],fit_result_sha256=sha(root/'RESULT.json'),
            checkpoint_sha256=boundary['sha256'],policy_sha256=result['policy_sha256'],
            updates=result['updates'],full_learner_optimizer_sampler_rng_restored=True,frozen_export_exact=True,
            actor_changed_from_initial=True,actor_window_stream_matches_all3_previous_methods=True,
            actor_valid_positions=result['valid_positions']['actor'],teacher_forced_dev_only=True,**metrics)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:write(out/'RESULT.json',report)


if __name__=='__main__':main()
