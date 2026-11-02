"""Audit retained rollout records on allocated Linux; no models or new actions."""
import json
from pathlib import Path
import numpy as np
import torch
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha, digest, write_new
from fluffy_bassoon.meta_rl.rlinf_robocasa_ppo import case_bank

root=Path('/mowenjie/fluffy-bassoon-2/outputs')
out=root/'robocasa-original-pi05-eval50-20261102-r1'
bases=dict(S=root/'robocasa-simulator-ppo-eval50-20261102-r1',
           F=root/'robocasa-opendrawer-flora-resume-20261102-r1')
profiles={k:json.loads((p/'policy/profile.json').read_text()) for k,p in bases.items()}
shared=['initializer','motor_seed','residual_seed','update_seed','ppo','evaluation_motor_seeds']
assert all(profiles['S'][key]==profiles['F'][key] for key in shared)
assert profiles['S']['training_cases'][:384]==profiles['F']['training_cases'][:384]
def stable_meta(value):
    # Native object-category membership is serialized from an unordered set.
    # Preserve every value and every other list's order, including placements.
    if isinstance(value,dict):
        return {k:sorted(v) if k=='groups_containing_sampled_obj' else stable_meta(v) for k,v in value.items()}
    if isinstance(value,list):return [stable_meta(v) for v in value]
    return value
panels={}
for phase,task,motor_seed,chunks in [('ID','OpenDrawer',910100,15),('OOD','CloseFridge',1010100,18)]:
    actual={};metadata={}
    for arm,base in bases.items():
        records={}
        paths=sorted((base/f'policy/{phase}-006/videos').glob('parallel-*/rank-*/result.pt'))
        assert len(paths)==8
        for p in paths:
            result=torch.load(p,map_location='cpu',weights_only=False)
            offset=0
            for case in result['cases']:
                rows=result['records'][offset:offset+chunks];offset+=chunks
                assert len(rows)==chunks and rows[0]['episode_start'] and rows[-1]['done']
                records[case['slot']]=dict(case=case,seeds=[r['noise_seed'] for r in rows],
                    features=rows[0]['features'],base_chunk=rows[0]['base_chunk'])
            assert offset==len(result['records'])
        assert len(records)==50 and [records[i]['case'] for i in range(50)]==case_bank(task,phase,count=50)
        actual[arm]=records
        metadata[arm]={v['case']['slot']:digest(stable_meta(v['native_ep_meta'])) for p in
            (base/f'monitoring/{phase}-006').glob('*.json') for v in [json.loads(p.read_text())]}
    generator=np.random.default_rng(motor_seed)
    expected=[[int(generator.integers(0,2**63-1)) for _ in range(chunks)] for _ in range(50)]
    assert [actual['S'][i]['seeds'] for i in range(50)]==expected
    assert [actual['F'][i]['seeds'] for i in range(50)]==expected
    assert metadata['S']==metadata['F']
    features=[np.max(np.abs(actual['S'][i]['features']-actual['F'][i]['features'])) for i in range(50)]
    base_diff=[np.max(np.abs(actual['S'][i]['base_chunk']-actual['F'][i]['base_chunk'])) for i in range(50)]
    panels[phase]=dict(task=task,cases=case_bank(task,phase,count=50),motor_seed=motor_seed,
        chunks_per_episode=chunks,per_case_noise_seeds=expected,actual_S_F_noise_seeds_equal=True,
        actual_S_F_native_ep_meta_equal_ignoring_unordered_group_names=True,
        metadata_normalization='sort only groups_containing_sampled_obj membership names; preserve all other arrays/values',
        first_features_bitwise_equal=all(float(v)==0 for v in features),
        first_base_chunks_bitwise_equal=all(float(v)==0 for v in base_diff),
        max_initial_feature_abs_difference=float(max(features)),max_initial_base_chunk_abs_difference=float(max(base_diff)),
        canonical_case_bank_sha256=digest(case_bank(task,phase,count=50)),noise_seed_schedule_sha256=digest(expected))
value=dict(completed=True,existing_S_F_actual_seed_schedules_equal=True,
    original_bound_to_same_schedules=True,original_actual_result_verification='pending after execution',
    shared_training_and_evaluation_profile_keys=shared,motor_seed=710100,residual_seed=42,update_seed=43,
    first384_training_cases_equal=True,lanes_per_GPU=1,deterministic_residual_mean=True,panels=panels,
    source_profiles={arm:dict(path=str(base/'policy/profile.json'),sha256=sha(base/'policy/profile.json')) for arm,base in bases.items()})
write_new(out/'SEED-AUDIT.json',value)
print(json.dumps({k:v for k,v in value.items() if k!='panels'}))
print(json.dumps({k:{x:v for x,v in p.items() if x not in ('cases','per_case_noise_seeds')} for k,p in panels.items()}))
