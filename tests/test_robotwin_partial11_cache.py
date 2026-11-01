"""Wrapper-only pure CPU tests; allocated Linux only, no model/simulator."""
from copy import deepcopy
from pathlib import Path
import socket

import pytest

from fluffy_bassoon.meta_rl import robotwin_partial11_cache as module


def clock_fixture(tmp_path):
    pair=tmp_path/'clock/pair';current=tmp_path/'clock/cache'
    rows=[dict(path=str(tmp_path/str(i)),status='finished',charged_seconds=v,reserved_seconds=v)
          for i,v in enumerate([632,1138,266,298,318])]
    rows.append(dict(path=str(pair),status='started',reserved_seconds=600))
    rows.append(dict(path=str(current),status='started',reserved_seconds=1200,slice_seconds=1140,
                     scheduler_allowance_seconds=60,started_unix=100,lineage=module.CACHE_LINEAGE,lineage_seconds=1200))
    clock=dict(contract=deepcopy(module.CLOCK_CONTRACT),worker_seconds=28800,attempts=rows)
    root=tmp_path/'cache-root';out=root/'fresh'
    ticket=dict(mode=module.FORMAT,source_commit='a'*40,run_id='fresh',hostname=socket.gethostname(),gpu_count=1,
        output=str(out),cache_root=str(root),prior_cache_gpu_seconds=0,maximum_robot_actions=0,
        scope_sha256=module.SCOPE_SHA,resource_attempt=str(current),pair_resource_attempt=str(pair),
        reserved_gpu_seconds=1200,scheduler_start_unix=100,driver_deadline_unix=1120)
    return ticket,clock,root,out


def admit(ticket,clock,root,out,prior=0):
    return module.validate_clock(ticket,clock,now=110,source='a'*40,run_id='fresh',output=out,cache_root=root,
                                 prior_cache_seconds=prior)


def test_one_shared_clock_reservation_retains_pair_hold_even_after_small_actual_closure(tmp_path):
    ticket,clock,root,out=clock_fixture(tmp_path)
    admit(ticket,clock,root,out)
    clock['attempts'][-2].update(status='finished',charged_seconds=100)
    admit(ticket,clock,root,out)  # Still retains full600 pair hold; no projected release is spent.
    clock['attempts'].append(dict(path='/other',status='started',reserved_seconds=2800))
    with pytest.raises(ValueError):admit(ticket,clock,root,out)


@pytest.mark.parametrize('fault',['new_clock','extra_gpu_time','reset_clock','prior_cost_reset','robot_actions','wrong_pair','wrong_output'])
def test_allin_budget_and_identity_cannot_reset_or_borrow(tmp_path,fault):
    ticket,clock,root,out=clock_fixture(tmp_path)
    if fault=='new_clock':clock['contract']['phase_id']='replacement'
    if fault=='extra_gpu_time':ticket['reserved_gpu_seconds']=1201
    if fault=='reset_clock':ticket['driver_deadline_unix']=1300
    if fault=='prior_cost_reset':ticket['prior_cache_gpu_seconds']=1
    if fault=='robot_actions':ticket['maximum_robot_actions']=1
    if fault=='wrong_pair':ticket['pair_resource_attempt']=ticket['resource_attempt']
    if fault=='wrong_output':ticket['output']='/other'
    with pytest.raises(ValueError):admit(ticket,clock,root,out)


def test_resume_charges_actual_parent_closure_once_and_reduces_new_slice(tmp_path):
    cursor=dict(resource_attempt='/old',source_commit='a'*40,prior_cache_gpu_seconds=0)
    closeout=dict(source_commit='a'*40,charged_gpu_seconds=150,
        attempt=dict(path='/old',status='finished',charged_seconds=150,slice_overrun=False))
    cost=module.prior_cost(cursor,closeout);assert cost==150
    ticket,clock,root,out=clock_fixture(tmp_path);ticket['prior_cache_gpu_seconds']=cost
    ticket['reserved_gpu_seconds']=1050;ticket['driver_deadline_unix']=970
    clock['attempts'].insert(-1,dict(path='/old',status='finished',charged_seconds=150,reserved_seconds=660,
        lineage=module.CACHE_LINEAGE,lineage_seconds=1200))
    clock['attempts'][-1].update(reserved_seconds=1050,slice_seconds=990)
    admit(ticket,clock,root,out,prior=cost)
    ticket['reserved_gpu_seconds']=1200
    with pytest.raises(ValueError):admit(ticket,clock,root,out,prior=cost)
    closeout['attempt']['status']='started'
    with pytest.raises(ValueError):module.prior_cost(cursor,closeout)


def rows(identity):
    return [dict(context=dict(teacher_id=identity),anchor=i,native_actual_actions=32 if i<6 else 1) for i in range(7)]


def bank():return dict(entries=[dict(id=i) for i in module.IDS])


def test_case_cursor_commits_only_complete_cases_and_preserves_interrupted_case():
    state=dict(complete_cases=[],records=[],active_case=None,status='running');snapshots=[];calls=[]
    def persist():snapshots.append(deepcopy(state))
    def one(entry):
        calls.append(entry['id'])
        if entry['id']==module.IDS[1]:raise RuntimeError('pure cache interruption; no physical step')
        return rows(entry['id'])
    with pytest.raises(RuntimeError):module.run_cases(bank(),state,one,persist,should_start=lambda:True)
    assert state['complete_cases']==module.IDS[:1] and len(state['records'])==7
    assert state['active_case']==module.IDS[1]
    resumed=deepcopy(state)
    module.run_cases(bank(),resumed,lambda e:rows(e['id']),lambda:None,should_start=lambda:True)
    assert resumed['complete_cases']==module.IDS and len(resumed['records'])==77
    assert resumed['status']=='complete_partial11' and 'phone-04' not in resumed['complete_cases']


def test_boundary_stop_makes_no_new_case_and_no_false_complete_status():
    state=dict(complete_cases=[],records=[],active_case=None,status='running')
    module.run_cases(bank(),state,lambda _:pytest.fail('Boundary must not start a cache case'),lambda:None,should_start=lambda:False)
    assert state['status']=='partial_boundary_stop' and state['complete_cases']==[] and state['records']==[]


@pytest.mark.parametrize('fault',['missing_sample','duplicate_anchor','wrong_case','wrong_cost'])
def test_incomplete_case_cannot_advance_resumable_cursor(fault):
    state=dict(complete_cases=[],records=[],active_case=module.IDS[0]);value=rows(module.IDS[0])
    if fault=='missing_sample':value.pop()
    if fault=='duplicate_anchor':value[-1]['anchor']=0
    if fault=='wrong_case':value[-1]['context']['teacher_id']='phone-04'
    if fault=='wrong_cost':value[-1]['native_actual_actions']=2
    with pytest.raises(ValueError):module.commit_case(state,module.IDS[0],value)
    assert state['complete_cases']==[] and state['records']==[]


def test_profile_is_always_partial11_not_twelve_fit_or_new_robot_work():
    import json
    profile_path=Path(__file__).parents[1]/'outputs/archive/robotwin-partial11-cache-20261101-r1/PROFILE.json'
    value=json.loads(profile_path.read_text());module.validate_profile(value)
    for key,replacement in [('partial_only',False),('fit_eligible',True),('maximum_robot_actions',1),
                            ('maximum_aggregate_gpu_seconds',1201),('case_ids',module.IDS+['phone-04'])]:
        with pytest.raises(ValueError):module.validate_profile(dict(value,**{key:replacement}))


def test_prior_cost_receipt_must_be_an_exact_row_in_original_shared_clock(tmp_path):
    _,clock,_,_=clock_fixture(tmp_path)
    closeout=dict(attempt=deepcopy(clock['attempts'][0]))
    module.validate_prior_in_clock(clock,closeout)
    closeout['attempt']['charged_seconds']=0
    with pytest.raises(ValueError):module.validate_prior_in_clock(clock,closeout)


@pytest.mark.parametrize('fault',['profile','context','payload','outside_root','nonprefix'])
def test_resumable_case_cursor_rechecks_closed_payload_identity(tmp_path,fault):
    root=tmp_path/'cache';root.mkdir()
    identity=module.IDS[0]
    entry=dict(id=identity,split='train',producer_result=dict(sha256='a'*64),producer_commit='a'*40,
        profile=dict(kind='old'),record=dict(spec=dict(id='old-record'),episode_manifest_ref=dict(sha256='b'*64)),
        original_source=dict(sha256='c'*64))
    bank_value=dict(entries=[entry]);context=dict(teacher_id=identity,split=entry['split'],producer_result=entry['producer_result'],
        producer_commit=entry['producer_commit'],profile=entry['profile'],record_id='old-record',
        episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
    value=rows(identity)
    for i,row in enumerate(value):
        path=root/f'{i}.pt';path.write_bytes(b'closed cached payload')
        row.update(path=str(path),sha256=module.file_sha(path),context=deepcopy(context))
    state=dict(format=module.FORMAT,source_commit='a'*40,profile_sha256='b'*64,bank_sha256=module.bank_api.digest(bank_value),
        complete_cases=[identity],records=value,status='partial_boundary_stop',new_robot_actions=0,fit_eligible=False)
    module.validate_cursor(state,bank_value,source='a'*40,profile_sha='b'*64,cache_root=root)
    if fault=='profile':state['profile_sha256']='c'*64
    if fault=='context':state['records'][0]['context']['record_id']='different'
    if fault=='payload':Path(state['records'][0]['path']).write_bytes(b'changed')
    if fault=='outside_root':
        path=tmp_path/'foreign.pt';path.write_bytes(b'closed cached payload');state['records'][0]['path']=str(path)
    if fault=='nonprefix':state['complete_cases']=[module.IDS[1]]
    with pytest.raises(ValueError):module.validate_cursor(state,bank_value,source='a'*40,profile_sha='b'*64,cache_root=root)


def test_fixed_cache_layout_and_tensor_byte_refusal_happen_before_serialization():
    import torch
    small=tuple((torch.zeros(1,1,1,256),torch.zeros(1,1,1,256)) for _ in range(18))
    module.validate_cache_buffers(small)
    with pytest.raises(ValueError):module.validate_cache_buffers(small[:-1])
    # meta tensors test the exact byte arithmetic without allocating payloads
    # or constructing a real model/cache/GPU context in this CPU gate.
    huge=tuple((torch.empty(1,4096,1,256,device='meta'),torch.empty(1,4096,1,256,device='meta')) for _ in range(18))
    with pytest.raises(ValueError,match='storage bound'):module.validate_cache_buffers(huge)


def test_stale_A_cursor_cannot_omit_later_finished_B_cost(tmp_path):
    ticket,clock,root,out=clock_fixture(tmp_path)
    old_a=dict(path='/A',status='finished',charged_seconds=200,reserved_seconds=660,
        lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    later_b=dict(old_a,path='/B')
    clock['attempts'][-1].update(reserved_seconds=1000,slice_seconds=940)
    clock['attempts'][-1]['started_unix']=100
    clock['attempts'][5:5]=[old_a,later_b]
    ticket.update(prior_cache_gpu_seconds=200,reserved_gpu_seconds=1000,driver_deadline_unix=920)
    with pytest.raises(ValueError,match='cache-lineage'):
        admit(ticket,clock,root,out,prior=200)  # Actual200+200+1000 exceeds dedicated1200.
    # Even a smaller fitting reservation cannot resume a stale ancestry that
    # underreports B; the exact previous-cost field must reconcile ALL rows.
    clock['attempts'][-1].update(reserved_seconds=660,slice_seconds=600)
    ticket.update(reserved_gpu_seconds=660,driver_deadline_unix=580)
    with pytest.raises(ValueError,match='omits later'):
        admit(ticket,clock,root,out,prior=200)


def test_outstanding_cache_sibling_reservation_is_not_omitted(tmp_path):
    ticket,clock,root,out=clock_fixture(tmp_path)
    clock['attempts'].insert(-1,dict(path='/sibling',status='started',reserved_seconds=660,
        lineage=module.CACHE_LINEAGE,lineage_seconds=1200))
    with pytest.raises(ValueError,match='sibling'):
        admit(ticket,clock,root,out)


def test_existing_clock_atomically_enforces_serial_lineage_subcap_without_new_clock(tmp_path):
    import json
    from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
    root=tmp_path/'original-clock';contract=dict(original='same')
    a=begin(root,contract,slice_seconds=600,worker_seconds=28800,now=0,
        lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    with pytest.raises(RuntimeError,match='outstanding'):
        begin(root,contract,slice_seconds=600,worker_seconds=28800,resume=True,now=1,
            lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    finish(root,a,0,now=140)  # Actual200 including original60 startup allowance.
    b=begin(root,contract,slice_seconds=600,worker_seconds=28800,resume=True,now=200,
        lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    finish(root,b,0,now=340)  # Another200 actual.
    with pytest.raises(RuntimeError,match='lineage aggregate'):
        begin(root,contract,slice_seconds=940,worker_seconds=28800,resume=True,now=400,
            lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    with pytest.raises(ValueError,match='subcap'):
        begin(root,contract,slice_seconds=600,worker_seconds=28800,resume=True,now=401,
            lineage=module.CACHE_LINEAGE,lineage_seconds=2400)
    ledger=json.loads((root/'worker-time.json').read_text())
    assert len(ledger['attempts'])==2 and sum(r['charged_seconds'] for r in ledger['attempts'])==400


def test_original_clock_default_api_and_rows_stay_unchanged(tmp_path):
    import json
    from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin,finish
    root=tmp_path/'clock'
    attempt=begin(root,{'original':True},slice_seconds=600,worker_seconds=28800,now=0)
    row=finish(root,attempt,0,now=140)
    assert row['charged_seconds']==200 and 'lineage' not in row and 'lineage_seconds' not in row
    assert json.loads((root/'worker-time.json').read_text())['contract']=={'original':True}


def test_recovery_publication_interruption_keeps_previous_cursor_and_rng_generation(tmp_path,monkeypatch):
    import json
    import torch
    output=tmp_path/'fresh-child';output.mkdir()
    state=dict(complete_cases=['phone-00'],records=[],status='partial_boundary_stop')
    old_rng=dict(python='old-python',numpy='old-numpy',torch=torch.tensor([1,2],dtype=torch.uint8),cuda=None)
    module.publish_cursor(output,state,old_rng)
    previous=json.loads((output/'cursor.json').read_text());old_ref=previous['recovery'];old_bytes=Path(old_ref['path']).read_bytes()
    real_publish=module._atomic_json
    def crash_before_cursor(path,value,**kwargs):
        if Path(path).name=='cursor.json':raise OSError('interrupted after durable generation, before cursor publication')
        return real_publish(path,value,**kwargs)
    monkeypatch.setattr(module,'_atomic_json',crash_before_cursor)
    state['complete_cases'].append('phone-01')
    with pytest.raises(OSError):module.publish_cursor(output,state,dict(old_rng,python='new-python'))
    assert json.loads((output/'cursor.json').read_text())==previous
    assert Path(old_ref['path']).read_bytes()==old_bytes and module.file_sha(old_ref['path'])==old_ref['sha256']
    recovered=torch.load(old_ref['path'],weights_only=False)
    assert recovered['state']=={k:v for k,v in previous.items() if k!='recovery'}
    assert recovered['global_rng']['python']=='old-python' and torch.equal(recovered['global_rng']['torch'],old_rng['torch'])
    assert len(list((output/'recoveries').glob('*.pt')))==2  # New unreferenced generation retained, old never overwritten.


def test_qualified_short_slice_optin_composes_with_lineage_without_changing_default(tmp_path):
    import json
    from fluffy_bassoon.meta_rl.vla_source40_ppo_worker import begin
    root=tmp_path/'clock'
    with pytest.raises(ValueError):begin(root,{'old':True},slice_seconds=120,worker_seconds=28800,now=0)
    attempt=begin(root,{'old':True},slice_seconds=120,worker_seconds=28800,now=0,
        allow_short_slice=True,lineage=module.CACHE_LINEAGE,lineage_seconds=1200)
    row=json.loads((attempt/'attempt.json').read_text())
    assert row['slice_seconds']==120 and row['reserved_seconds']==180 and row['lineage_seconds']==1200
