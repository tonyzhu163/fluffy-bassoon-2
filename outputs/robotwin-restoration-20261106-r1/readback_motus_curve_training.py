"""Independent Linux CPU audit of original or resumed native GRPO collections."""
import argparse
import json
import os
from pathlib import Path

import torch

import motus_teacher_entry
from motus_learner_reference import audit
from optimizer_semantics import semantic_digest
from profile_checkpoint import verify
from fluffy_bassoon.meta_rl.rlinf_native_ppo import sha,write
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor

ROOT=Path(__file__).resolve().parent
BASE=Path('/mnt/bos/adgfyl/strong-plant-archives/robotwin-restoration-20261106-r1')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--contract',default='MOTUS-CURVE-GRPO-TRAIN01-CONTRACT.json')
    parser.add_argument('--training-source',required=True)
    parser.add_argument('--output-identity',required=True)
    args=parser.parse_args()
    assert Path(args.contract).name==args.contract and args.contract.endswith('-CONTRACT.json')
    assert __import__('re').fullmatch('[a-z][a-z0-9-]+',args.output_identity)
    assert os.uname().sysname=='Linux' and os.environ['CUDA_VISIBLE_DEVICES']=='' and not torch.cuda.is_initialized()
    plan=json.loads((ROOT/args.contract).read_bytes());assert plan['algorithm']=='grpo'
    start=plan.get('start_iteration',0);stop=plan.get('stop_iteration',plan['iterations'])
    identity=plan.get('output_identity','motusbehaviorgrpo01')
    root=BASE/identity;state=root/'state';signals=BASE/(identity+'-signals')
    checked={}
    def load(path):
        value=json.loads(path.read_bytes());checked[str(path)]=sha(path);return value
    assert load(signals/'EXIT.json')['exit_code']==0
    complete=load(signals/'COMPLETE.json')
    result=load(state/'RESULT.json');cfg=load(state/'resolved-native-config.json')
    assert result['status']=='completed_admitted_prefix' and result['completed_iterations']==stop
    assert result['intended_iterations']==plan.get('intended_iterations',plan['iterations'])
    assert result['native_adam_updates_per_rank']==15*stop
    assert complete['result_sha256']==sha(state/'RESULT.json')
    assert cfg['algorithm']['adv_type']=='grpo' and cfg['algorithm']['group_size']==8
    assert cfg['actor']['micro_batch_size']==32 and cfg['native_pilot']['require_behavior_likelihood_match']
    assert not cfg['actor']['model']['add_value_head'] and not cfg['algorithm']['filter_rewards']
    assert load(signals/'resources.json')['sampled_reserve_passed']
    signatures={};checkpoints=[]
    for step in range(start,stop+1):
        imported=step==start and start>0
        if imported:
            checkpoint=Path(plan['resume_checkpoint']['path'])
            expected=plan['resume_checkpoint']['manifest_sha256']
        else:
            receipt=load(state/'checkpoint-receipts'/f'step-{step:04d}.json')
            assert receipt['completed'] and receipt['global_step']==step
            assert receipt['full_cold_readback'] and receipt['full_local_stageback']
            checkpoint=Path(receipt['cold']);expected=receipt['manifest_sha256']
        assert sha(checkpoint/'MANIFEST.json')==expected
        manifest=load(checkpoint/'MANIFEST.json');assert manifest['global_step']==step
        if not imported:assert manifest['source_commit']==args.training_source
        verify(checkpoint,manifest)
        for rank in range(8):
            layout=load(checkpoint/'pilot'/f'motus-optimizer-layout-{rank:02d}.json')
            signature=load(checkpoint/'pilot'/f'motus-actor-signature-{rank:02d}.json')
            assert layout['optimizer_steps']==signature['optimizer_steps']==15*step
            assert semantic_digest(layout)==signature['optimizer_semantic_sha256']
            sampler=load(checkpoint/'pilot'/f'motus-case-schedule-env-{rank:02d}.json')
            assert sampler['next_reset_ordinal']==4*step
            required=(cfg['native_pilot']['prefix_case_manifest_sha256'] if imported and start==2
                else cfg['native_pilot']['case_manifest_sha256'])
            assert sampler['manifest_sha256']==required
            assert all((checkpoint/'pilot'/f'{role}-{rank:02d}.pt').is_file() for role in ('actor','rollout','env'))
            signatures[step,rank]=signature
        checkpoints.append(dict(step=step,manifest_sha256=expected,path=str(checkpoint),files=len(manifest['files'])))
        print('CURVE_COLD_BOUNDARY',step,'passed',flush=True)
    if start:
        restored=load(state/'operational-restore.json')
        assert restored['global_step']==start and len(restored['workers'])==24
        manifest=load(Path(plan['resume_checkpoint']['path'])/'MANIFEST.json')
        assert {(r['role'],r['rank']) for r in restored['workers']}=={(role,rank) for role in ('actor','rollout','env') for rank in range(8)}
        for row in restored['workers']:
            assert row['restored'] and row['snapshot_sha256']==manifest['files'][f"pilot/{row['role']}-{row['rank']:02d}.pt"]['sha256']
        for rank in range(8):
            actual=load(state/'recovery-signatures'/f'actor-{rank:02d}.json')
            assert actual['owned_state_exact'] and actual['actual']['optimizer_steps']==15*start
        assert load(state/'resume-baseline-reference.json')['all48_predictions_exact']
        if start==2:
            for rank in range(8):
                migration=load(state/'case-migration'/f'env-{rank:02d}.json')
                assert migration['exact_two_iteration_prefix_preserved'] and migration['next_reset_ordinal']==8
                assert migration['from_manifest_sha256']==cfg['native_pilot']['prefix_case_manifest_sha256']
                assert migration['to_manifest_sha256']==cfg['native_pilot']['case_manifest_sha256']
    references=[]
    for iteration in range(start+1,stop+1):
        for rank in range(8):
            receipt=load(state/'training-rollouts'/f'iteration-{iteration:04d}-policy-{iteration-1:04d}-rank-{rank:02d}.json')
            assert receipt['completed'] and not receipt['evaluation_data']
            assert receipt['collection_iteration']==iteration and receipt['producing_policy_iteration']==iteration-1
            assert receipt['optimizer_steps_before']==15*(iteration-1)
            archive=Path(receipt['path']);assert archive.stat().st_size==receipt['bytes'] and sha(archive)==receipt['sha256']
            saved=read_tensor(archive)
            assert saved['producing_policy_iteration']==iteration-1 and saved['optimizer_steps_before']==15*(iteration-1)
            independent=audit(saved['rollout_batch'],'grpo')
            assert independent['passed'] and independent==load(state/'learner-reference'/f'iteration-{iteration:04d}-rank-{rank:02d}.json')
            references.append(dict(iteration=iteration,rank=rank,**independent));del saved
            frozen=load(state/'actor-invariants'/f'rank-{rank:02d}-step-{iteration-1}-after.json')
            assert frozen['unchanged'] and frozen['before']==frozen['after']
            memory=load(state/'actor-memory'/f'iteration-{iteration:04d}-rank-{rank:02d}.json')
            assert memory['micro_batch']==32 and memory['accumulation']==8 and memory['global_batch']==2048
            if start:
                behavior=load(state/'behavior-reference'/f'policy-{iteration-1:04d}-rank-{rank:02d}.json')
                assert behavior['original_behavior_exact'] and behavior['optimizer_steps_before']==15*(iteration-1)
        assert any(signatures[iteration,rank]['trainable_parameters_sha256']!=signatures[iteration-1,rank]['trainable_parameters_sha256'] for rank in range(8))
        print('CURVE_FRESH_COLLECTION',iteration,'all8 NumPy references passed',flush=True)
    coverage=[]
    for ordinal in range(4*start,4*stop):
        for rank in range(8):
            row=load(state/'group-resets'/f'rank-{rank:02d}-reset-{ordinal:04d}.json')
            assert row['passed'] and row['raw_public_reference_verified'] and row['reset_ordinal']==ordinal and row['rank']==rank
            assert row['manifest_sha256']==cfg['native_pilot']['case_manifest_sha256'] and len(row['groups'])==4
            assert all(g['passed'] and g['replica_count']==8 for g in row['groups'])
            coverage.extend(dict(iteration=ordinal//4,seed=g['scene_seed']) for g in row['groups'])
    for rank in range(8):
        first=load(state/'first-replay'/f'rank-{rank:02d}.json')
        assert first['passed'] and first['logprob_max_abs']==0 and first['optimizer_steps']==15*start
    for step in range(start,stop+1):
        dev=load(state/'dev'/f'iteration-{step:04d}.json')
        assert dev['completed'] and dev['episodes']==16 and dev['optimizer_updates_during_eval']==0
    out=BASE/args.output_identity;out.mkdir()
    write(out/'RESULT.json',dict(status='passed_independent_native_GRPO_fresh_collection_curve_readback',
        source_commit=os.environ['TASK_SOURCE'],training_source_commit=args.training_source,GPU_seconds=0,
        start_iteration=start,stop_iteration=stop,new_collections=stop-start,new_episodes=1024*(stop-start),
        final_Adam_updates_per_rank=15*stop,full_checkpoints=checkpoints,independent_whole_batch_references=references,
        all_group_resets_verified=True,unique_scenes_by_collection={str(step):len({r['seed'] for r in coverage if r['iteration']==step}) for step in range(start,stop)},
        full24_worker_restore_and_48_baseline_predictions_verified=bool(start),
        owned_Adam_descriptors_and_frozen_parameters_verified=True,larger_paired_evaluation_required=True,
        learning_gain_claim=False,checked_json_hashes=checked))
    print('CURVE_READBACK_PASSED',str(out/'RESULT.json'),sha(out/'RESULT.json'),flush=True)


if __name__=='__main__':main()
