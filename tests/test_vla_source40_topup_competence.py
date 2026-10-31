"""Allocated metadata regression tests for explicit retained bridge producers."""
import copy
from pathlib import Path

import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup_competence as competence


def bank(tmp_path,monkeypatch,*,retained=True):
    source=competence.topup;outputs=tmp_path/'outputs';outputs.mkdir()
    plan=dict(mode='production',source_commit='a'*40,plan_sha256='f'*64,slots=[])
    ledger=dict(historical_extra_actions=sum(source.EXTRAS.values()),leaves={})
    retained_keys={'libero_spatial:00','libero_object:00'} if retained else set()
    rows={}
    for ordinal,key in enumerate(competence.ALL_TASKS):
        slot=key+'/train';root=outputs/f'producer-{ordinal}';exp=root/'experiment';exp.mkdir(parents=True)
        (root/'launcher.lock').touch();(root/'exit-code.txt').write_text('0\n')
        source.write(root/'schedule.json',dict(schedule_sha256='d'*64))
        motor=dict(receipt_sha256=source.PI05_RECEIPT,openpi_commit='2'*40,flow_steps=10,
            predicted_horizon=10,action_dim=32,weights_frozen=True)
        recipe=dict(warmup_chunks=500,batch_size=256,utd=20,gamma_per_primitive=.999)
        original=tmp_path/'original'/f'goal-{ordinal}';(original/'experiment').mkdir(parents=True)
        base=dict(task=dict(task_key=key),partition='train',cases=[dict(initial_state=i%40) for i in range(512)],
            history_id=key,source_recipe=recipe,learner_seed=ordinal,horizon=400,chunk=10,
            feature_contract=dict(checkpoint_receipt_sha256=source.PI05_RECEIPT,observation_dim=4104),
            noise_contract=competence.NOISE)
        source.write(original/'experiment/contract.json',base)
        parent=dict(root=str(original),experiment=str(original/'experiment'),actions=8000,source_updates=5000,
            contract_sha256=source.sha(original/'experiment/contract.json'))
        pin='b'*40 if key in retained_keys else plan['source_commit']
        phase='e'*64 if key in retained_keys else plan['plan_sha256']
        continuation=dict(format=source.FORMAT,plan_sha256=phase,parent=parent)
        contract=dict(base,source_commit=pin,source_only=True,source_data_from_targets=False,
            maximum_total_actions=21800,motor=motor,schedule_sha256='d'*64,continuation=continuation)
        source.write(exp/'contract.json',contract)
        (exp/'source-recovery.pkl').write_bytes(b'full replay optimizer sampler and RNG recovery '+key.encode())
        checkpoint=exp/'case-00009-source.pkl';checkpoint.write_bytes(b'exact published complete-case actor '+key.encode())
        recovery_sha=source.sha(exp/'source-recovery.pkl');checkpoint_sha=source.sha(checkpoint)
        actual=dict(continuation,new_optimizer_updates=200,new_complete_episodes=4,
            motor_parameters_unchanged=True,all_new_behavior_postwarmup=True)
        index=dict(completed=True,collection_qualified=True,episode_in_case=0,slice_completed=True,
            source_only=True,partition='train',task_key=key,production_quota_exhausted=True,
            total_action_budget_charge=21000,source_updates=5200,chunks=600,case_index=10,
            episode_receipts=[{}]*40,source_contract_sha256=source.content_hash(contract),schedule_sha256='d'*64,
            continuation=actual,latest_recovery_sha256=recovery_sha,
            publications=[dict(status='paired_ad_dpt',through_case=9,source_checkpoint=checkpoint.name,
                source_checkpoint_sha256=checkpoint_sha,episodes=40,transitions=600)])
        source.write(exp/'index.json',index)
        source.write(exp/'candidate.json',dict(full_checkpoint=checkpoint.name,sha256=checkpoint_sha,
            full_latest_recovery_sha256=recovery_sha,cumulative_stream_actions=21000,teacher_updates=5200,plan_sha256=phase))
        source.write(exp/'migration.json',dict(format=source.FORMAT,plan_sha256=phase,parent=parent,numerical_state_sha256='7'*64))
        (root/'attempts/model-1').mkdir(parents=True)
        source.write(root/'attempts/model-1/parameters-after.json',dict(before=dict(sha256='c'*64),after=dict(sha256='c'*64),unchanged=True))
        source.write(root/'worker-time.json',dict(worker_seconds=1800 if key in retained_keys else 2100,
            contract=dict(source_commit=pin,plan_sha256=phase,slot=slot,run_id=root.name),
            attempts=[dict(status='finished',exit_code=0,charged_seconds=1200)]))
        leaf=dict(root=str(root),maximum_total_actions=21800,source_commit=pin,plan_sha256=phase,
            parent_root=parent['root'],inherited_action_charge=parent['actions'])
        ledger['leaves'][slot]=leaf
        if key in retained_keys:
            declared=dict(slot=slot,root=str(root),experiment=str(exp),actions=21000,source_updates=5200,
                chunks=600,next_case=10,episodes=40,source_commit=pin,contract_sha256=source.sha(exp/'contract.json'),
                index_sha256=source.sha(exp/'index.json'),recovery_sha256=recovery_sha,schedule_sha256=source.sha(root/'schedule.json'))
        else:declared=parent
        rows[slot]=dict(slot=slot,launchable=key not in retained_keys,maximum_total_actions=21800,parent=declared)
    for slot in source.expected_slots():
        if slot.endswith('/validation'):
            rows[slot]=dict(slot=slot,launchable=False,maximum_total_actions=10000,parent={})
            ledger['leaves'][slot]=dict(root='/retained/validation/'+slot,maximum_total_actions=10000)
    plan['slots']=[rows[slot] for slot in source.expected_slots()]
    plan_path,ledger_path=tmp_path/'production.json',tmp_path/'canonical.json'
    source.write(plan_path,plan);source.write(ledger_path,ledger)
    monkeypatch.setattr(source,'load_plan',lambda path:plan)
    return plan,ledger,plan_path,ledger_path


def test_explicit_retained_bridges_plus38_fresh_are_admitted_and_old_default_stays_strict(tmp_path,monkeypatch):
    plan,ledger,path,canonical=bank(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='explicit retained-parent'):
        competence.prepare(path,canonical,tmp_path/'strict',run_id='final',source_commit='a'*40)
    alias=competence.prepare(path,canonical,tmp_path/'explicit',run_id='final',source_commit='a'*40,allow_retained_parents=True)
    manifest=competence.topup.read(alias['producer_manifest'])
    assert manifest['all40_admitted']==40 and manifest['retained_parent_count']==2
    assert alias['nominal_queries']==48 and alias['maximum_total_actions']==20000
    reservations=competence.topup.read(alias['source_ledger'])['reservations']
    assert reservations['libero_spatial:00/train']['source_commit']=='b'*40
    assert reservations['libero_spatial:01/train']['source_commit']=='a'*40
    alias_path=tmp_path/'explicit/alias.json'
    assert competence.load_alias(alias_path,competence.topup.sha(alias_path),run_id='final',source_commit='a'*40)==alias


@pytest.mark.parametrize('fault',['stale_parent_hash','mismatched_leaf','non_topup','unlisted_old_pin'])
def test_retained_admission_does_not_accept_stale_or_unlisted_original_actors(tmp_path,monkeypatch,fault):
    plan,ledger,_,_=bank(tmp_path,monkeypatch)
    key='libero_spatial:00';slot=key+'/train';row=next(row for row in plan['slots'] if row['slot']==slot)
    if fault=='stale_parent_hash':row['parent']['recovery_sha256']='0'*64
    if fault=='mismatched_leaf':ledger['leaves'][slot]['root']=ledger['leaves']['libero_spatial:01/train']['root']
    if fault=='non_topup':
        root=Path(ledger['leaves'][slot]['root']);contract=competence.topup.read(root/'experiment/contract.json')
        contract.pop('continuation');competence.topup.write(root/'experiment/contract.json',contract)
        row['parent']['contract_sha256']=competence.topup.sha(root/'experiment/contract.json')
    if fault=='unlisted_old_pin':row['launchable']=True
    with pytest.raises(ValueError):competence.producer(plan,ledger,key,allow_retained_parents=True)


def test_fresh_only_default_still_accepts_exact_latest_producers(tmp_path,monkeypatch):
    _,_,path,canonical=bank(tmp_path,monkeypatch,retained=False)
    alias=competence.prepare(path,canonical,tmp_path/'fresh',run_id='final',source_commit='a'*40)
    manifest=competence.topup.read(alias['producer_manifest'])
    assert alias['producer_admission']=='latest_production_only' and manifest['retained_parent_count']==0
