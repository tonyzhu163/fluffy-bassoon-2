"""Focused synthetic reset precision, aborted-slot and original-clock repair."""
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import vla_source40_teacher_assessment as a
from fluffy_bassoon.meta_rl import vla_source40_teacher_repair as r


def initial():
    physical=np.arange(92,dtype=np.float64)
    obs=dict(image=np.full((4,4,3),64,np.uint8),wrist=np.full((4,4,3),60,np.uint8),state=np.zeros(8,np.float32))
    return dict(initial_simulator_state=physical,initial_simulator_state_sha256=hashlib.sha256(physical.tobytes()).hexdigest(),
        initial_state_vector_sha256='same-nominal',initial_observation_sha256='raw-reference',observation=obs)


def reference():return dict(initial=initial(),features=np.zeros(4104,np.float32),instruction='original instruction')


def test_opt_in_existing_one_lsb_preserves_raw_pixels_and_audits_feature_difference(tmp_path):
    ref=reference();current=deepcopy(ref['initial']);current['observation']['wrist'][1,2,0]+=1
    current['initial_observation_sha256']='distinct-raw';feature=np.ones(4104,np.float32)
    before=deepcopy(current);path=tmp_path/'audit.pkl.gz'
    a.check_pair(ref,current,feature,ref['instruction'],camera_max_lsb=1,audit_output=path)
    with gzip.open(path,'rb') as stream:audit=pickle.load(stream)
    assert audit['allowed'] and not audit['feature_vectors_equal'] and not audit['feature_equality_required']
    assert audit['feature_maximum_absolute_delta']==1. and audit['comparison']['cameras']['wrist']['maximum_absolute_lsb']==1
    assert not audit['pixels_modified']
    np.testing.assert_array_equal(current['observation']['wrist'],before['observation']['wrist'])
    np.testing.assert_array_equal(audit['computed_features'],feature)


@pytest.mark.parametrize('fault',['two_lsb','physics','proprio','nominal','instruction','nonfinite_feature','wrong_feature_ABI','camera_dtype'])
def test_source_precision_still_refuses_real_identity_or_ABI_change_with_saved_evidence(tmp_path,fault):
    ref=reference();current=deepcopy(ref['initial']);feature=np.zeros(4104,np.float32);instruction=ref['instruction']
    if fault=='two_lsb':current['observation']['wrist'][0,0,0]+=2
    elif fault=='physics':current['initial_simulator_state'][0]+=1
    elif fault=='proprio':current['observation']['state'][0]+=1
    elif fault=='nominal':current['initial_state_vector_sha256']='different'
    elif fault=='instruction':instruction='different instruction'
    elif fault=='nonfinite_feature':feature[0]=np.nan
    elif fault=='wrong_feature_ABI':feature=np.zeros(4103,np.float32)
    elif fault=='camera_dtype':current['observation']['wrist']=current['observation']['wrist'].astype(np.float32)
    path=tmp_path/'actual-feature-audit.pkl.gz'
    with pytest.raises(ValueError,match='audit preserved'):
        a.check_pair(ref,current,feature,instruction,camera_max_lsb=1,audit_output=path)
    with gzip.open(path,'rb') as stream:audit=pickle.load(stream)
    assert not audit['allowed'] and audit['computed_features'].shape==feature.shape
    assert audit['instruction']==instruction and 'comparison' in audit


def table():
    return dict(actors=[dict(source_ordinal=i,task_key=f'{a.SUITES[i%4]}:{i//4:02d}') for i in range(40)])


def progress():
    return dict(contract='original-contract',active=161,cursor=161,receipts=[dict(path=f'original/{i}',sha256=str(i)) for i in range(161)],
        costs=deepcopy(r.CLOSED_COSTS),qualification_complete=True,case_key='libero_spatial:05/state-40',
        references={str(i):dict(path=f'old-reference-{i}',sha256=str(i)) for i in range(41)},
        global_rng=dict(numpy=np.arange(625,dtype=np.uint32),python=('state',3)),motor_rng=dict(key_data=np.array([3,4],np.uint32)),
        actor_rng=None,case_fork={'global':{'original':np.arange(32,dtype=np.uint8)},'motor':np.array([8,9],np.uint32)})


def test_exact_metadata_migration_preserves_every_non_disposition_field_and_null_scores_failed_role():
    old=progress();before=r.numerical_digest(old);new=r.migrate_progress(old,table=table(),old_contract='original-contract',new_contract='new-contract')
    assert old['cursor']==old['active']==161 and old['contract']=='original-contract'
    assert new['cursor']==162 and new['active'] is None and len(new['receipts'])==161
    assert r.numerical_digest(new)==before and new['case_fork'] is old['case_fork'] and new['global_rng'] is old['global_rng']
    assert new['costs']==dict(all_in_actions=26096,physical_episodes=162,uncertain_actions=0)
    aborted=new['aborted'][0]
    assert aborted['ordinal']==161 and aborted['success'] is aborted['performance_score'] is None
    assert aborted['status']=='ABORTED_RESET_GUARD' and aborted['completed'] is False
    assert aborted['primitive_steps']==0 and aborted['settling_steps']==10


@pytest.mark.parametrize('fault',['active','cursor','prefix','old_cost','qualification','references','already_migrated'])
def test_abort_migration_refuses_other_or_incomplete_progress(fault):
    old=progress()
    if fault=='active':old['active']=162
    elif fault=='cursor':old['cursor']=160
    elif fault=='prefix':old['receipts'].pop()
    elif fault=='old_cost':old['costs']['all_in_actions']+=1
    elif fault=='qualification':old['qualification_complete']=False
    elif fault=='references':old['references'].pop('0')
    elif fault=='already_migrated':old['migration']={'already':True}
    with pytest.raises(ValueError):r.migrate_progress(old,table=table(),old_contract='original-contract',new_contract='new-contract')


def admitted():
    packet=Path(__file__).resolve().parents[1]/'outputs/libero40-teacher-assessment-repair-code-20261101-r1'
    clock=json.loads((packet/'ORIGINAL-CLOSED-CLOCK.json').read_text());old=clock['attempts'][0]
    current=dict(path=str(r.ROOT/'clock/attempts/NEW-CHILD'),status='started',started_unix=1793507000.,slice_seconds=9360,reserved_seconds=9420)
    clock['attempts'].append(current)
    ticket=dict(source_commit='f'*40,clock_contract=deepcopy(clock['contract']),clock_root=str(r.ROOT/'clock'),
        run_id='libero40-teacher-assessment-20261101-r1',repair_profile=dict(sha256=r.PROFILE_SHA))
    return ticket,clock


def test_new_execution_pin_uses_original_header_clock_origin_and_1340_charge():
    ticket,clock=admitted();expected=r.repair_admission(ticket,clock,source='f'*40,output=r.CHILD/'data',profile=r.expected_profile())
    assert expected['source_commit']==r.SOURCE and clock['attempts'][0]['charged_seconds']==1340
    assert sum(x.get('charged_seconds',x['reserved_seconds']) for x in clock['attempts'])==10760
    assert clock['worker_seconds']==10800


@pytest.mark.parametrize('fault',['fresh_root','rewritten_source_header','lost_old_charge','new_origin','oversize_slice'])
def test_resource_migration_never_resets_or_rewrites_original_clock(fault):
    ticket,clock=admitted()
    if fault=='fresh_root':ticket['clock_root']=str(r.ROOT/'fresh-clock')
    elif fault=='rewritten_source_header':clock['contract']['source_commit']='f'*40
    elif fault=='lost_old_charge':clock['attempts'][0]['charged_seconds']=0
    elif fault=='new_origin':clock['attempts'][0]['started_unix']+=1
    elif fault=='oversize_slice':clock['attempts'][1]['slice_seconds']+=1
    with pytest.raises(ValueError):r.repair_admission(ticket,clock,source='f'*40,output=r.CHILD/'data',profile=r.expected_profile())


def test_reporting_keeps_all480_slots_and_aborted_role_is_not_a_completed_receipt(tmp_path):
    old=progress();new=r.migrate_progress(old,table=table(),old_contract='original-contract',new_contract='new-contract')
    a.write_results(tmp_path,table(),[],aborted=new['aborted'])
    import csv
    rows=list(csv.DictReader((tmp_path/'case-results.csv').open()))
    assert len(rows)==480 and sum(row['status'].startswith('OMITTED') for row in rows)==160
    aborted=[row for row in rows if row['status']=='ABORTED_RESET_GUARD']
    assert len(aborted)==1 and aborted[0]['success']=='' and aborted[0]['all_in_actions']=='10'
    summary=json.loads((tmp_path/'results-summary.json').read_text())
    assert summary['by_role'][a.ROLES[1]]['aborted']==1 and summary['by_role'][a.ROLES[1]]['complete']==0


@pytest.mark.parametrize('fault',['another_skipped_role','fake_scored_abort'])
def test_recovery_cannot_hide_extra_skipped_roles_or_score_the_failed_reset(fault):
    new=r.migrate_progress(progress(),table=table(),old_contract='original-contract',new_contract='new-contract')
    if fault=='another_skipped_role':new['aborted'].append(deepcopy(new['aborted'][0]));new['cursor']+=1
    else:new['aborted'][0]['completed']=True;new['aborted'][0]['success']=True
    with pytest.raises(ValueError,match='single diagnosed'):r.validate_disposition(new,table())
