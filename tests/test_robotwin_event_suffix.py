"""Changed-path suffix tests; execute only on allocated Linux."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fluffy_bassoon.meta_rl import robotwin_event_suffix as module
from fluffy_bassoon.meta_rl.robotwin_event_admission import build_event_plan
from fluffy_bassoon.meta_rl.robotwin_event_replay import EventEpisodeControl
from fluffy_bassoon.meta_rl.robotwin_event_schedule import build_schedule
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.meta_rl.robotwin_reactive_capacity import validate_capacity_locked


def bank():
    demos=dict(episodes=[dict(id=f'phone-{i:02d}',split='train' if i<8 else 'validation',
        reset_seed=i+1,instruction=f'original instruction {i}',canonical_index=i) for i in range(12)])
    original={str(i):dict(spec=dict(task='place_phone_stand',case=i,stage='development',reset_seed=i),
        complete=True,reset_info=dict(actual_seed=i,observation_sha256='a'*64,features_sha256='b'*64,instruction=['original']),
        reset_identity=dict(actual_seed=i,observation_sha256='a'*64,features_sha256='b'*64)) for i in range(8)}
    return demos,dict(plan=build_event_plan(original,demos))


def test_suffix_preserves_original_specs_split_and_continues_every_complete_failure():
    demos,prior=bank();specs=module.fixed_suffix(prior,demos)
    assert [s['teacher_id'] for s in specs]==module.SUFFIX
    assert [prior['plan']['whole_split'][s['teacher_id']] for s in specs]==['train']*3+['validation']*4
    calls=[]
    def episode(spec,*,qualify):
        calls.append((deepcopy(spec),qualify))
        return dict(complete=True,success=False,primitive_steps=400,spec=spec)
    module.run_fixed_suffix(specs,episode)
    assert [s for s,_ in calls]==specs and [q for _,q in calls]==[True]+[False]*6
    assert prior['plan']['teacher_replays'][5:]==specs  # no spec mutation


@pytest.mark.parametrize('mutation',['04retry','filter','instruction','seed','motor','split'])
def test_changed_fixed_bank_is_rejected(mutation):
    demos,prior=bank()
    if mutation=='04retry':prior['plan']['teacher_replays'][5]=deepcopy(prior['plan']['teacher_replays'][4])
    if mutation=='filter':prior['plan']['teacher_replays'].pop()
    if mutation=='instruction':prior['plan']['teacher_replays'][5]['paired_instruction']=['replacement']
    if mutation=='seed':prior['plan']['teacher_replays'][5]['reset_seed']+=1
    if mutation=='motor':prior['plan']['teacher_replays'][5]['motor_seed']+=1
    if mutation=='split':prior['plan']['whole_split']['phone-08']='train'
    with pytest.raises(ValueError):module.fixed_suffix(prior,demos)


def test_runtime_integrity_error_stops_without_retrying_or_skipping_pending_case():
    demos,prior=bank();specs=module.fixed_suffix(prior,demos);calls=[]
    def episode(spec,*,qualify):
        calls.append(spec['teacher_id'])
        if spec['teacher_id']=='phone-06':raise RuntimeError('unresolved actual native intent')
    with pytest.raises(RuntimeError):module.run_fixed_suffix(specs,episode)
    assert calls==['phone-05','phone-06']


@pytest.mark.parametrize('key,value',[
    ('reserved_gpu_seconds',1801),('prior_a1_gpu_seconds',2036),('capacity_action_bound',3200),
    ('driver_deadline_unix',1900),('run_id','old'),('scheduler_start_unix',float('nan'))])
def test_finite_allin_residual_rejects_budget_resets(key,value):
    ticket=dict(mode=module.FORMAT,run_id='new',prior_a1_gpu_seconds=2334,prior_phase_gpu_seconds=2334,
        reserved_gpu_seconds=1800,capacity_action_bound=2800,scheduler_start_unix=100,driver_deadline_unix=1840)
    module.validate_allocation(ticket,now=110,run_id='new')
    with pytest.raises(ValueError):module.validate_allocation(dict(ticket,**{key:value}),now=110,run_id='new')


def closed_snapshot():
    targets=np.zeros((4,14),np.float32)
    return targets,dict(mode='action',elapsed_steps=400,motor_rng=np.random.default_rng(42).bit_generator.state,
        source_schedule=build_schedule(targets),source_cursor=dict(chunk_index=13,physical_slots=400,source_cursor=4,closed=True),
        teacher_reference=dict(path='/original04',sha256='a'*64),last_native_rng_audit=dict(retained='original native audit'))


def test_exact_closed_control_restore_uses_existing_controller_and_keeps_all_private_fields():
    targets,snapshot=closed_snapshot()
    class Head:
        def config(self):return dict(synthetic=True)
    env=SimpleNamespace(active=False,elapsed_steps=0,contract={},backend=SimpleNamespace(
        model=SimpleNamespace(device='cpu')))
    writer=SimpleNamespace(register_runtime=lambda _:None)
    control=EventEpisodeControl(env,{'kv':Head(),'action':Head()},writer,{})
    module.restore_closed_control(control,snapshot,targets)
    assert control.snapshot()==snapshot
    wrong=deepcopy(snapshot);wrong['source_schedule']['reference_frames']+=1
    with pytest.raises(ValueError):module.restore_closed_control(control,wrong,targets)


def test_recovery_requires_full_closed04_and_global_rng_not_fresh_replacement():
    _,snapshot=closed_snapshot();prior=dict(closed_control_recoveries=[snapshot])
    saved=dict(state=prior,fits={},fit_recovery_refs={},control=deepcopy(snapshot),
               global_rng={k:'retained' for k in ('python','numpy','torch','cuda')})
    module.validate_recovery(saved,prior)
    for key,value in [('control',dict(snapshot,elapsed_steps=399)),('fits',dict(kv='forbidden')),
                      ('global_rng',dict(saved['global_rng'],cuda=None)),('state',{})]:
        with pytest.raises(ValueError):module.validate_recovery(dict(saved,**{key:value}),prior)


def test_original_canonical_ledger_lock_keeps_costs_and_only_one_fresh_run(tmp_path):
    kwargs=dict(campaign_id='original',total_actions=300000,source_caps={},phase_caps={'a1':30000},contract={})
    root=tmp_path/'ledger'
    with CampaignBudget(root,run_id='ended',source_commit='a'*40,**kwargs) as budget:
        budget.reserve('ended:04:0',400,phase='a1',method='event_aligned_teacher',task='place_phone_stand',episode='04',chunk=0,policy_version='original04')
        budget.resolve('ended:04:0',400);prior=dict(actions=budget.snapshot())
    with CampaignBudget(root,run_id='new',source_commit='b'*40,**kwargs) as budget:
        validate_capacity_locked(budget,prior)
        assert budget.snapshot()['actual_actions']==400
        altered=deepcopy(prior);altered['actions']['charged_actions']=0
        with pytest.raises(ValueError):validate_capacity_locked(budget,altered)


def test_manifest_retains_all12_and_never_licenses_fit_on_easier_suffix():
    path=Path(__file__).parents[1]/'outputs/archive/robotwin-event-fixed-suffix-20261101-r1/MANIFEST.json'
    manifest=json.loads(path.read_text())
    assert len(manifest['original_teacher_ids'])==12 and len(manifest['suffix_specs'])==7
    assert list(manifest['whole_split'].values()).count('validation')==4
    assert manifest['cache_fit_source_student_admission'] is False
    assert manifest['expected_outcomes']['prediction'].startswith('uncertain: fixed04')


def migration_fixture(tmp_path,monkeypatch):
    demos,prior=bank()
    records=[]
    for spec,success,steps in zip([prior['plan']['legacy_control'],*prior['plan']['teacher_replays'][:5]],
                                   [False,True,True,True,True,False],[400,193,193,193,193,400]):
        records.append(dict(spec=dict(spec,teacher_initial_state=[0.]*14),complete=True,success=success,primitive_steps=steps))
    root=tmp_path/'phase/ledger';events=root/'events';events.mkdir(parents=True)
    for i in range(1331):(events/f'{i:010d}.json').write_text('{}')
    header={'bound':'original header'}
    (events/'0000000000.json').write_text(json.dumps(dict(header=header)))
    (events/'0000001330.json').write_text(json.dumps(dict(sha256=module.TIP)))
    ledger=dict(root=str(root),contract={'original':True})
    recovery=tmp_path/'closed04.pt';recovery.write_bytes(b'x'*223589)
    old=dict(actual_actions=20171,charged_actions=20171,pending_actions=0,uncertain_actions=0,
        unresolved_decisions={},journal_events=1331,journal_sha256=module.TIP,runs={'ended':module.PRODUCER},header=header)
    prior.update(source_commit=module.PRODUCER,status='blocked_failure',stage='teacher_executability',
        failure=deepcopy(module.FAILURE),pending=None,records=records,closed_control_recoveries=[{}]*6,
        action_ledger=ledger,actions=old)
    proof=dict(format=module.FORMAT,source_commit_after='b'*40,run_id='fresh',
        result=dict(path='/saved/result.json',sha256=module.RESULT_SHA),
        recovery=dict(path=str(recovery),sha256=module.RECOVERY_SHA),
        terminal=dict(path='/saved/terminal.json',sha256=module.TERMINAL_SHA),
        allocation_closeout=dict(path='/saved/closeout.json',sha256=module.CLOSEOUT_SHA))
    terminal=dict(job_id='job-2rkimbvypowd',source_commit=module.PRODUCER,scheduler_status='Failed',exit_code=1,
        cumulative_a1_gpu_seconds=2334,charged_gpu_seconds=298,unrun_aligned_teacher_suffix=module.SUFFIX,
        cache_created=False,all_twelve_teacher_gate_passed=False,
        full_recovery=dict(proof['recovery'],bytes=223589,opaque_no_deserialization=True))
    closeout=dict(source_commit=module.PRODUCER,charged_gpu_seconds=298,exit_code=1,error=None,
        attempt=dict(status='finished',slice_overrun=False))
    bindings={'/saved/result.json':prior,'/saved/terminal.json':terminal,'/saved/closeout.json':closeout}
    monkeypatch.setattr(module,'bound_json',lambda ref:bindings[ref['path']])
    monkeypatch.setattr(module,'file_sha',lambda path:module.RECOVERY_SHA)
    allocation=dict(action_ledger=deepcopy(ledger),phase_output_root=str(root.parent))
    return proof,prior,terminal,allocation,recovery


def test_migration_accepts_only_bound_failed04_closed_tip_and_complete_recovery(tmp_path,monkeypatch):
    proof,prior,terminal,ticket,_=migration_fixture(tmp_path,monkeypatch)
    module.validate_migration(proof,prior,terminal,ticket,source='b'*40,run_id='fresh')


@pytest.mark.parametrize('mutation',['same_total_foreign_root','tip','events','pending','used_run',
    'changed04','recovery_bytes','old_result','cost_refund','open_scheduler'])
def test_migration_blocks_changed_identity_history_costs_or_unclosed_work(tmp_path,monkeypatch,mutation):
    proof,prior,terminal,ticket,recovery=migration_fixture(tmp_path,monkeypatch)
    if mutation=='same_total_foreign_root':ticket['action_ledger']['root']=str(tmp_path)
    if mutation=='tip':(Path(ticket['action_ledger']['root'])/'events/0000001330.json').write_text(json.dumps(dict(sha256='f'*64)))
    if mutation=='events':(Path(ticket['action_ledger']['root'])/'events/0000001331.json').write_text('{}')
    if mutation=='pending':prior['pending']={'id':'actual pending'}
    if mutation=='used_run':prior['actions']['runs']['fresh']='b'*40
    if mutation=='changed04':prior['records'][-1]['success']=True
    if mutation=='recovery_bytes':recovery.write_bytes(b'short')
    if mutation=='old_result':proof['result']['sha256']='0'*64
    if mutation=='cost_refund':terminal['charged_gpu_seconds']=0
    if mutation=='open_scheduler':terminal['scheduler_status']='Running'
    with pytest.raises(ValueError):module.validate_migration(proof,prior,terminal,ticket,source='b'*40,run_id='fresh')
