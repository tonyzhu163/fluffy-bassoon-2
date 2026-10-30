"""Allocated metadata tests: macro overhead, immutable resume and child limits."""
import copy
import json
from pathlib import Path

import pytest

from fluffy_bassoon.meta_rl import vla_source40_topup_worker as macro


def prepared(tmp_path,monkeypatch):
    outputs=tmp_path/'outputs';root=outputs/'campaign/worker-0'
    plan=dict(mode='production',source_commit='a'*40,plan_sha256='f'*64,maximum_total_actions=1000000,
        slots=[dict(slot=f'{suite}:{n:02d}/train',launchable=True,maximum_total_actions=21800,
            parent=dict(root=f'/original/{suite}-{n}',source_updates=5000)) for n in range(10)
            for suite in ('libero_spatial','libero_object','libero_goal','libero_10')])
    plan_path,budget_path,ledger=tmp_path/'plan.json',tmp_path/'budget.json',tmp_path/'ledger.json'
    macro.source.write(plan_path,plan)
    budget=dict(format=macro.FORMAT,funding_status='root_reconciled',maximum_new_gpu_seconds=86400,
        workers=8,worker_execution_seconds=10500,maximum_worker_charge_seconds=10560,
        startup_allowance_seconds_per_attempt=60,maximum_final_competence_charge_seconds=1400,
        plan_sha256=plan['plan_sha256'],source_commit=plan['source_commit'],campaign_prefix='campaign',
        campaign_root=str(root.parent),historical_qualification_pool='original32h_counted_once_not_new_GPU_time')
    macro.source.write(budget_path,budget)
    monkeypatch.setattr(macro.source,'load_plan',lambda path:plan)
    args=(root,plan_path,ledger,budget_path,'campaign',0,'a'*40)
    return args,plan,budget


def child_clock(row,contract,*,charged=560,closed=True):
    root=Path(row['root']);root.mkdir(parents=True,exist_ok=True)
    expected=dict(format=macro.source.FORMAT,plan_sha256=contract['plan_sha256'],slot=row['slot'],
        source_commit=contract['source_commit'],run_id=root.name,
        purpose='Source-only SAC top-up; no resource approval granted by this receipt')
    clock=dict(worker_seconds=2100,contract=expected,attempts=[dict(status='finished' if closed else 'started',
        reserved_seconds=2100,charged_seconds=charged,exit_code=0)])
    macro.source.write(root/'worker-time.json',clock)
    (root/'exit-code.txt').write_text('0\n')
    return root


def fulfilled(row,contract):
    root=child_clock(row,contract);exp=root/'experiment';exp.mkdir()
    source=dict(source_commit=contract['source_commit'],maximum_total_actions=21800,
        source_only=True,source_data_from_targets=False,task=dict(task_key=row['slot'].split('/')[0]),
        partition='train',cases=[{}]*512)
    macro.source.write(exp/'contract.json',source)
    recovery=exp/'source-recovery.pkl';recovery.write_bytes(b'complete learner replay optimizers sampler and all RNG bytes')
    digest=macro.source.sha(recovery)
    index=dict(completed=True,collection_qualified=True,episode_in_case=0,slice_completed=True,source_only=True,
        continuation=dict(parent=row['parent'],plan_sha256=contract['plan_sha256'],new_optimizer_updates=200,motor_parameters_unchanged=True),
        production_quota_exhausted=True,total_action_budget_charge=21000,source_updates=5200,case_index=25,
        latest_recovery_sha256=digest,source_contract_sha256=macro.source.content_hash(source))
    macro.source.write(exp/'index.json',index)
    macro.source.write(exp/'candidate.json',dict(plan_sha256=contract['plan_sha256'],full_latest_recovery_sha256=digest,teacher_updates=5200))
    macro.source.write(contract['canonical_ledger'],dict(leaves={row['slot']:dict(root=str(root),maximum_total_actions=21800)}))
    return root,index


def test_macro_cumulative_idle_failed_attempts_and_immutable_assignment_on_resume(tmp_path,monkeypatch):
    args,_,_=prepared(tmp_path,monkeypatch)
    attempt,deadline,seconds=macro.begin(*args,now=1000)
    assert seconds==10500 and deadline==11500
    assignment=(args[0]/'assignment.json').read_bytes();text=(args[0]/'assigned.txt').read_bytes()
    #300s spent in setup/children/idle, plus the one actual startup allowance.
    first=macro.finish(args[0],attempt,1,now=1300)
    assert first['cumulative_macro_charged_seconds']==360
    assert macro.remaining(args[0])['maximum_new_scheduler_seconds']==10140
    old_outcome=(attempt/'macro-outcome.json').read_bytes()
    resumed,end,left=macro.begin(*args,resume=True,now=2000)
    assert resumed!=attempt and left==10140 and end==12140
    assert (args[0]/'assignment.json').read_bytes()==assignment and (args[0]/'assigned.txt').read_bytes()==text
    assert (attempt/'macro-outcome.json').read_bytes()==old_outcome
    assert first['fulfilled_streams']==0 and first['assigned_streams']==5 and not first['all40_claim']


def test_partial_child_uses_remaining_original2100_not_fresh2040(tmp_path,monkeypatch):
    args,_,_=prepared(tmp_path,monkeypatch);attempt,_,_=macro.begin(*args,now=1000)
    contract=macro.source.read(args[0]/'worker-time.json')['contract'];row=macro.source.read(args[0]/'assignment.json')['slots'][0]
    child_clock(row,contract,charged=560)
    decision,_=macro.selection(args[0],attempt,row['slot'],now=1400)
    assert decision['decision']=='launch' and decision['resume'] and decision['slice_seconds']==1480
    assert decision['slice_seconds']+60+560==2100
    child_clock(row,contract,charged=2030)
    blocked,_=macro.selection(args[0],attempt,row['slot'],now=1500)
    assert blocked['decision']=='blocked' and not blocked['fulfilled']


def test_skip_requires_terminal_fulfilled_quota_and_exact_recovery(tmp_path,monkeypatch):
    args,_,_=prepared(tmp_path,monkeypatch);attempt,_,_=macro.begin(*args,now=1000)
    contract=macro.source.read(args[0]/'worker-time.json')['contract'];row=macro.source.read(args[0]/'assignment.json')['slots'][0]
    root,index=fulfilled(row,contract)
    decision,_=macro.selection(args[0],attempt,row['slot'],now=1100)
    assert decision['decision']=='fulfilled' and decision['slice_seconds']==0
    index.update(production_quota_exhausted=False,total_action_budget_charge=12000)
    macro.source.write(root/'experiment/index.json',index)
    assert not macro.child_state(row,contract)['fulfilled']  # completed=True alone is insufficient.
    index.update(production_quota_exhausted=True,total_action_budget_charge=21000)
    macro.source.write(root/'experiment/index.json',index)
    (root/'experiment/source-recovery.pkl').write_bytes(b'changed complete state')
    assert not macro.child_state(row,contract)['fulfilled']


def test_active_child_or_changed_child_ceiling_cannot_be_restarted(tmp_path,monkeypatch):
    args,_,_=prepared(tmp_path,monkeypatch);attempt,_,_=macro.begin(*args,now=1000)
    contract=macro.source.read(args[0]/'worker-time.json')['contract'];row=macro.source.read(args[0]/'assignment.json')['slots'][0]
    root=child_clock(row,contract,closed=False)
    with pytest.raises(ValueError,match='Unclosed child'):
        macro.selection(args[0],attempt,row['slot'],now=1100)
    value=macro.source.read(root/'worker-time.json');value['worker_seconds']=1000
    macro.source.write(root/'worker-time.json',value)
    with pytest.raises(ValueError,match='cumulative2100'):
        macro.selection(args[0],attempt,row['slot'],now=1100)


def test_macro_contract_cannot_reset_failed_attempts_and_unfunded_budget_rejected(tmp_path,monkeypatch):
    args,plan,budget=prepared(tmp_path,monkeypatch)
    budget['funding_status']='UNFUNDED_TEMPLATE';macro.source.write(args[3],budget)
    with pytest.raises(ValueError,match='root-funded'):
        macro.begin(*args,now=1000)
    budget['funding_status']='root_reconciled';macro.source.write(args[3],budget)
    attempt,_,_=macro.begin(*args,now=1000)
    with pytest.raises(ValueError,match='Unclosed macro'):
        macro.begin(*args,resume=True,now=1200)
    macro.finish(args[0],attempt,1,now=1400)
    wrong=list(args);wrong[-1]='b'*40
    with pytest.raises(ValueError,match='source pin'):
        macro.begin(*wrong,resume=True,now=1500)
    assert 8*(macro.ACTIVE_SECONDS+macro.ALLOWANCE)+macro.FINAL_SECONDS==85880<86400
    assert 40*macro.CHILD_SECONDS+1000+macro.FINAL_SECONDS==86400
