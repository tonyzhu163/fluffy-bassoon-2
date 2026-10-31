"""Allocated CPU only: one closed case, full RNG restore, immutable failures."""
from copy import deepcopy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_cache04_recovery as recovery
from fluffy_bassoon.rl.robotwin_corpus import _atomic_json
from fluffy_bassoon.meta_rl.robotwin_teacher_bank import digest


def fixture(tmp_path):
    motor=np.random.default_rng(77);initial=deepcopy(motor.bit_generator.state)
    motor.standard_normal((32,32));final=deepcopy(motor.bit_generator.state)
    binding=dict(source_commit='a'*40,profile_sha256='p'*64,bank_ref={'path':'bound-bank','sha256':'b'*64},
        bank_sha256=digest({'diagnostic':True}),partial11={'cursor':'paid-cursor','checkpoint':'paid-checkpoint','cache_receipts':['paid11']},
        motor_anchors=dict(initial=initial,final=final),record_count=2,native_actions=3,
        cache_root=str(tmp_path),runtime_blob_sha256={'qualified-publisher':recovery.PUBLISHER_SHA},
        recovery_contract=recovery.FORMAT,clock_root='original-clock',clock_contract={'ceiling':600})
    return binding


def make_case(binding,seen=None):
    def cache(destination):
        destination.mkdir();rows=[]
        for i,n in enumerate([1,2]):
            path=destination/f'sample-{i}.pt';path.write_bytes(b'preserved-case-payload'+bytes([i]))
            rows.append(dict(path=str(path),sha256=recovery.file_sha(path),anchor=i,
                             context={'teacher_id':'phone-04'},native_actual_actions=n))
        raw=dict(bank={'diagnostic':True},records=rows,fit_eligible=False)
        _atomic_json(destination/'partial-cache.json',raw)
        if seen is not None:seen.append(str(destination))
        return raw
    return cache


def ref(path):return dict(path=str(path),sha256=recovery.file_sha(path))


def test_initial_cursor_exists_before_encoding_and_whole_case_commits(tmp_path):
    binding=fixture(tmp_path);out=tmp_path/'first';inner=make_case(binding)
    def cache(destination):
        state,rng=recovery.read_snapshot(ref(out/'cursor.json'),deadline=1e20)
        assert state['boundary']=='before_phone04' and state['records']==[]
        assert rng['motor_rng']==binding['motor_anchors']['initial'] and rng['private_rng']==recovery.ROLE
        return inner(destination)
    result=recovery.run_boundary(out,binding,'attempt1',{'current':'qualification'},cache,guard=lambda:None,deadline=1e20)
    state,rng=recovery.read_snapshot(result['cache04_closed_case']['cursor'],deadline=1e20)
    assert state['boundary']=='after_phone04' and state['complete_cases']==['phone-04']
    assert len(state['records'])==2 and rng['motor_rng']==binding['motor_anchors']['final']
    assert (out/'boundary-before-phone04.json').is_file() and (out/'boundary-after-phone04.json').is_file()
    assert len(list((out/'recoveries').glob('*.pt')))==2


def test_interrupted_payload_is_immutable_and_rng_restores_after_fresh_init(tmp_path):
    binding=fixture(tmp_path);first=tmp_path/'first'
    def interrupt(destination):
        destination.mkdir();(destination/'partial.pt').write_bytes(b'keep this orphan')
        random.random();np.random.random();torch.rand(2)
        raise RuntimeError('diagnosed synthetic interruption')
    with pytest.raises(RuntimeError):
        recovery.run_boundary(first,binding,'old',{},interrupt,guard=lambda:None,deadline=1e20)
    old=ref(first/'cursor.json');state,saved=recovery.read_snapshot(old,deadline=1e20)
    assert state['boundary']=='before_phone04' and state['records']==[]
    assert (first/'payload'/'partial.pt').read_bytes()==b'keep this orphan'
    motor=np.random.default_rng(0);recovery.restore_snapshot(saved,motor)
    expected=(random.random(),float(np.random.random()),torch.rand(2),motor.standard_normal(2))
    random.random();np.random.random();torch.rand(9)  # Fresh-model initialization may advance globals.
    private=np.random.default_rng(0);recovery.restore_snapshot(saved,private)
    assert np.array_equal(private.standard_normal(2),expected[3])
    state['cursor_ref']=old
    def resumed(destination):
        actual=(random.random(),float(np.random.random()),torch.rand(2))
        assert actual[0]==expected[0] and actual[1]==expected[1] and torch.equal(actual[2],expected[2])
        return make_case(binding)(destination)
    result=recovery.run_boundary(tmp_path/'fresh',binding,'new',{},resumed,guard=lambda:None,deadline=1e20,
                                 resume_state=state,resume_rng=saved)
    assert len(result['records'])==2 and recovery.file_sha(old['path'])==old['sha256']
    assert (first/'payload'/'partial.pt').read_bytes()==b'keep this orphan'


def test_failed_completed_pointer_publication_keeps_initial_and_orphan_snapshot(tmp_path,monkeypatch):
    binding=fixture(tmp_path);out=tmp_path/'first'
    from fluffy_bassoon.meta_rl import robotwin_partial11_cache as publisher
    actual=publisher._atomic_json;calls=0
    def fail_second(path,value,**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:raise OSError('cursor publication window')
        return actual(path,value,**kwargs)
    monkeypatch.setattr(publisher,'_atomic_json',fail_second)
    with pytest.raises(OSError):
        recovery.run_boundary(out,binding,'attempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    state,_=recovery.read_snapshot(ref(out/'cursor.json'),deadline=1e20)
    assert state['boundary']=='before_phone04' and state['records']==[]
    assert len(list((out/'recoveries').glob('*.pt')))==2
    assert (out/'payload'/'sample-0.pt').is_file() and not (out/'partial-cache.json').exists()


def resume_fixture(tmp_path):
    binding=fixture(tmp_path);out=tmp_path/'old'
    def interrupt(destination):raise RuntimeError('stop before first encoded input')
    with pytest.raises(RuntimeError):recovery.run_boundary(out,binding,'oldattempt',{},interrupt,guard=lambda:None,deadline=1e20)
    cursor=ref(out/'cursor.json')
    row=dict(path='oldattempt',status='finished',charged_seconds=70,reserved_seconds=300,slice_overrun=False,exit_code=1)
    close=tmp_path/'close.json';_atomic_json(close,dict(attempt=row,source_commit=binding['source_commit'],charged_gpu_seconds=70,exit_code=1,error=None))
    diagnosis=tmp_path/'diagnosis.json';_atomic_json(diagnosis,dict(cursor_sha256=cursor['sha256'],reason='actual immutable diagnosed interruption'))
    return binding,dict(cursor=cursor,closeout=ref(close),diagnosis=ref(diagnosis)),dict(attempts=[row])


@pytest.mark.parametrize('fault',['profile','bank','partial11','checkpoint','unclosed','fresh_child','owner_cleanup_error','contradictory_exit'])
def test_resume_binds_input_checkpoint_cost_and_fresh_identity(tmp_path,fault):
    binding,resume,clock=resume_fixture(tmp_path);output=tmp_path/'fresh'
    if fault=='profile':binding['profile_sha256']='changed'
    if fault=='bank':binding['bank_sha256']='changed'
    if fault=='partial11':binding['partial11']['checkpoint']='changed'
    if fault=='checkpoint':
        state=json.loads(Path(resume['cursor']['path']).read_text());Path(state['recovery']['path']).write_bytes(b'changed')
    if fault=='unclosed':clock['attempts'][0]['status']='started'
    if fault in ('owner_cleanup_error','contradictory_exit'):
        path=Path(resume['closeout']['path']);value=json.loads(path.read_text())
        if fault=='owner_cleanup_error':value['error']='owned cleanup failed'
        else:value['exit_code']=0
        path.unlink();_atomic_json(path,value);resume['closeout']=ref(path)
    if fault=='fresh_child':output=Path(resume['cursor']['path']).parent
    with pytest.raises(ValueError):recovery.resume_initial(resume,binding,clock,output,deadline=1e20)


def test_exact_initial_resume_accepts_same_closed_cost_without_clock_mutation(tmp_path):
    binding,resume,clock=resume_fixture(tmp_path);old=deepcopy(clock)
    state,rng=recovery.resume_initial(resume,binding,clock,tmp_path/'fresh',deadline=1e20)
    assert state['boundary']=='before_phone04' and rng['private_rng']==recovery.ROLE and clock==old


def test_completed_boundary_is_cpu_reconcilable_and_cannot_gpu_resume(tmp_path):
    binding=fixture(tmp_path);out=tmp_path/'done'
    raw=recovery.run_boundary(out,binding,'oldattempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    boundary=raw['cache04_closed_case']['cursor']
    assert recovery.committed_receipt(boundary,deadline=1e20)==raw
    with pytest.raises(ValueError,match='Completed cache04 lineage'):
        recovery.resume_initial(dict(cursor=boundary),binding,{},tmp_path/'fresh',deadline=1e20)
    with pytest.raises(ValueError,match='Completed cache04 lineage'):
        recovery.resume_initial(dict(cursor=ref(out/'boundary-before-phone04.json')),binding,{},tmp_path/'fresh',deadline=1e20)
    Path(raw['records'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError):recovery.committed_receipt(boundary,deadline=1e20)


@pytest.mark.parametrize('fault',[None,'cursor_records','checkpoint','active_writer','contradictory_exit','owner_cleanup_error'])
def test_paid11_binding_is_hash_bound_complete_and_never_reencodes_them(tmp_path,fault):
    ids=[f'phone-{i:02d}' for i in range(12) if i!=4];entries=[];rows=[]
    for identity in ids:
        entry=dict(id=identity,split='train' if int(identity[-2:])<8 else 'validation',
            producer_result={'path':'original','sha256':'a'*64},producer_commit='a'*40,profile={'original':True},
            record=dict(spec={'id':'teacher-'+identity},episode_manifest_ref={'path':'manifest','sha256':'b'*64}),
            original_source={'path':'original-source','sha256':'c'*64})
        entries.append(entry)
        context=dict(teacher_id=identity,split=entry['split'],producer_result=entry['producer_result'],producer_commit=entry['producer_commit'],
            profile=entry['profile'],record_id=entry['record']['spec']['id'],episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
        rows.extend(dict(path='/must-not-open-paid-tensors/'+identity+str(i),sha256='e'*64,context=context,anchor=i,
                         native_actual_actions=32 if i<6 else 1) for i in range(7))
    checkpoint=tmp_path/'paid-recovery.pt';checkpoint.write_bytes(b'hash-only paid checkpoint, never deserialize')
    cursor=dict(format='robotwin_partial11_model_cache_v1',complete_cases=ids,records=deepcopy(rows),fit_eligible=False,
        status='complete_partial11',cleanup_error=None,active_case=None,new_robot_actions=0,full12_eligible=False,
        frozen_after='frozen',resource_attempt='paid-attempt',source_commit='a'*40,recovery=ref(checkpoint))
    if fault=='cursor_records':cursor['records'][0]['anchor']=1
    if fault=='active_writer':cursor.pop('cleanup_error')
    c=tmp_path/'paid-cursor.json';_atomic_json(c,cursor)
    receipt=tmp_path/'paid-receipt.json';_atomic_json(receipt,dict(format='robotwin_fixed12_cache_receipt_v1',fit_eligible=False,records=rows))
    close=tmp_path/'paid-close.json';_atomic_json(close,dict(source_commit='a'*40,exit_code=0,charged_gpu_seconds=10,
        attempt=dict(path='paid-attempt',status='finished',slice_overrun=False,charged_seconds=10,exit_code=0)))
    request=dict(partial11_cursor=ref(c),cache_receipts=[ref(receipt)],partial11_closeout=ref(close))
    if fault=='checkpoint':checkpoint.write_bytes(b'changed')
    if fault in ('contradictory_exit','owner_cleanup_error'):
        value=json.loads(close.read_text())
        if fault=='contradictory_exit':value['exit_code']=1
        else:value['error']='owner settlement/cleanup failure'
        close.unlink();_atomic_json(close,value);request['partial11_closeout']=ref(close)
    if fault is not None:
        with pytest.raises(ValueError):recovery.partial11_binding(request,dict(entries=entries,frozen_parameter_sha256='frozen'),deadline=1e20)
    else:
        result=recovery.partial11_binding(request,dict(entries=entries,frozen_parameter_sha256='frozen'),deadline=1e20)
        assert result['cursor']==ref(c) and result['checkpoint']==ref(checkpoint) and len(result['cache_receipts'])==1


def test_payload_directory_durability_precedes_completed_cursor(tmp_path,monkeypatch):
    binding=fixture(tmp_path);out=tmp_path/'case';events=[]
    real_sync=recovery._fsync_directory;real_publish=recovery.publish_cursor
    def sync(path):events.append(('sync',str(path)));real_sync(path)
    def publish(output,state,rng):
        if state['boundary']=='after_phone04':
            assert ('sync',str(out/'payload')) in events and ('sync',str(out)) in events
            events.append(('completed_cursor',str(output)))
        real_publish(output,state,rng)
    monkeypatch.setattr(recovery,'_fsync_directory',sync);monkeypatch.setattr(recovery,'publish_cursor',publish)
    recovery.run_boundary(out,binding,'attempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    assert events.index(('sync',str(out/'payload')))<events.index(('completed_cursor',str(out)))


@pytest.mark.parametrize('fault',[None,'reset_limit','overflow','sibling'])
def test_cache04_keeps_same_aggregate600_and_all_old_attempts(fault):
    rows=[dict(path='old',lineage='fixed12-command-diagnostic/cache04',lineage_seconds=600,status='finished',charged_seconds=70,reserved_seconds=300),
          dict(path='fresh',lineage='fixed12-command-diagnostic/cache04',lineage_seconds=600,status='started',reserved_seconds=500)]
    if fault=='reset_limit':rows[1]['lineage_seconds']=1200
    if fault=='overflow':rows[0]['charged_seconds']=101
    if fault=='sibling':rows[0]['status']='started'
    if fault is None:recovery.validate_cache_budget({'attempts':rows},'fresh')
    else:
        with pytest.raises(ValueError):recovery.validate_cache_budget({'attempts':rows},'fresh')



def run_data_payload(monkeypatch,tmp_path,operation,request):
    """Execute the production heredoc, including audited request and output paths."""
    script=Path(__file__).resolve().parents[1]/'outputs/robotwin-command-diagnostic-20261101-r1/DATA-PAYLOAD.sh'
    source=script.read_text().split("python - <<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    path=tmp_path/'data-request.json';_atomic_json(path,request)
    output=tmp_path/'data-output.json'
    for key,value in dict(DATA_OPERATION=operation,DATA_REQUEST=str(path),DATA_REQUEST_SHA=recovery.file_sha(path),
                          DATA_OUTPUT=str(output),OWNER_DATA_DEADLINE='1e20').items():monkeypatch.setenv(key,value)
    exec(compile(source,str(script),"exec"),{'__name__':'__main__'})
    return json.loads(output.read_text())


def test_actual_data_payload_reconciles_completed04_on_cpu(tmp_path,monkeypatch):
    binding=fixture(tmp_path);out=tmp_path/'done'
    expected=recovery.run_boundary(out,binding,'attempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    assert run_data_payload(monkeypatch,tmp_path,'reconcile-cache04',
        dict(cache04_cursor=expected['cache04_closed_case']['cursor']))==expected


def test_actual_data_payload_resolves_audited_readout_inputs(tmp_path,monkeypatch):
    from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_bank as bank
    b=tmp_path/'bank.json';p=tmp_path/'proposal.json'
    _atomic_json(b,dict(exact='bank'));_atomic_json(p,dict(exact='proposal'));seen=[]
    def resolve(actual_bank,actual_proposal,*,deadline):
        seen.append((actual_bank,actual_proposal,deadline));return dict(resolved=True,admitted=False)
    monkeypatch.setattr(bank,'readout_bindings',resolve)
    assert run_data_payload(monkeypatch,tmp_path,'readout-bindings',dict(bank=ref(b),proposal=ref(p)))==dict(resolved=True,admitted=False)
    assert seen==[({'exact':'bank'},{'exact':'proposal'},1e20)]


def test_actual_data_payload_refuses_unknown_operation(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='Only assemble'):
        run_data_payload(monkeypatch,tmp_path,'unreviewed-operation',{})
    assert not (tmp_path/'data-output.json').exists()


@pytest.mark.parametrize('fault',[None,'raw04','initial_boundary','changed_receipt'])
def test_public_diagnostic_sealer_requires_committed04_before_core(tmp_path,monkeypatch,fault):
    from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_bank as bank
    binding=fixture(tmp_path);out=tmp_path/'done'
    receipt=recovery.run_boundary(out,binding,'attempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    if fault=='raw04':receipt.pop('cache04_closed_case')
    if fault=='initial_boundary':receipt['cache04_closed_case']['cursor']=ref(out/'boundary-before-phone04.json')
    if fault=='changed_receipt':receipt['records'][0]['native_actual_actions']+=1
    path=tmp_path/'receipt.json';_atomic_json(path,receipt);called=[]
    # Isolate the added public proof gate; the existing bulk core has its own
    # qualified suite. No physical tensor/source admission is invented here.
    result=dict(core_sealer_reached=True)
    def core_seal(actual_bank,actual_refs,*,deadline,_api):
        assert actual_bank=={'exact_bank':True} and actual_refs==[ref(path)] and deadline==1e20 and _api is bank
        called.append(True);return result
    checked=[]
    monkeypatch.setattr(bank.core,'seal_full_cache',core_seal)
    monkeypatch.setattr(bank,'validate_complete_cache',lambda value:checked.append(value))
    if fault is None:
        assert bank.seal_full_cache({'exact_bank':True},[ref(path)],deadline=1e20) is result
        assert called==[True] and checked==[result]
    else:
        with pytest.raises(ValueError):bank.seal_full_cache({'exact_bank':True},[ref(path)],deadline=1e20)
        assert called==[] and checked==[]



def test_completed_sibling_forbids_restarting_older_failed_child(tmp_path):
    binding,resume,clock=resume_fixture(tmp_path)
    state,rng=recovery.resume_initial(resume,binding,clock,tmp_path/'second',deadline=1e20)
    state['cursor_ref']=deepcopy(resume['cursor'])
    recovery.run_boundary(tmp_path/'second',binding,'secondattempt',{},make_case(binding),guard=lambda:None,
                          deadline=1e20,resume_state=state,resume_rng=rng)
    old=Path(resume['cursor']['path']).read_bytes()
    with pytest.raises(ValueError,match='Completed cache04 lineage'):
        recovery.resume_initial(resume,binding,clock,tmp_path/'third',deadline=1e20)
    with pytest.raises(ValueError,match='Completed cache04 lineage'):
        recovery.validate_lineage_head(tmp_path,resume,deadline=1e20)
    assert Path(resume['cursor']['path']).read_bytes()==old and not (tmp_path/'third').exists()


def test_unpublished_complete_orphan_is_not_cpu_reconcilable(tmp_path):
    binding=fixture(tmp_path);out=tmp_path/'done'
    recovery.run_boundary(out,binding,'attempt',{},make_case(binding),guard=lambda:None,deadline=1e20)
    complete=ref(out/'boundary-after-phone04.json')
    # Reconstruct the failed pointer-publication state: a saved completed
    # checkpoint exists, but the actual public cursor still names the initial.
    initial=json.loads((out/'boundary-before-phone04.json').read_text())
    _atomic_json(out/'cursor.json',initial,replace=True)
    with pytest.raises(ValueError,match='actually published'):
        recovery.committed_receipt(complete,deadline=1e20)
