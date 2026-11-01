"""Eight focused allocated-Linux CPU cases; synthetic episodes, no admission."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from fluffy_bassoon.meta_rl import robotwin_train_capacity8 as entry

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Allocated Linux CPU only')
ROOT=Path(__file__).resolve().parents[1]/'outputs/archive/robotwin-train-capacity8-driver-20261101-r1'


def documents():
    return [json.loads((ROOT/name).read_text()) for name in ('PROTOCOL.json','GRANT.json','AUTHORITY.json')]


def fixture_data():
    protocol,_,_=documents();static=json.loads((ROOT/'BINDINGS.json').read_text())
    descriptor=deepcopy(static['frozen_frontend']);inventory=dict(episodes=[]);producers={};manifests={}
    for i in range(4):
        ident=f'phone-{i:02d}';row=static['canonical_records'][ident]
        inventory['episodes'].append(deepcopy(row));outcome=descriptor['fit_provenance']['outcome_evidence'][ident]
        reset=dict(actual_seed=row['reset_seed'],observation_sha256=f'{i+1:064x}',features_sha256=f'{i+5:064x}')
        original=dict(spec=dict(id=outcome['record_id'],reset_seed=row['reset_seed'],paired_instruction=[row['selected_instruction']]),
            complete=True,success=True,primitive_steps=outcome['primitive_steps'],chunks=outcome['chunks'],
            episode_manifest_ref=deepcopy(outcome['episode_manifest_ref']),reset_identity=deepcopy(reset),
            reset_info=dict(reset,instruction=[row['selected_instruction']]))
        producers[ident]=dict(records=[original]);manifests[ident]=dict(pending_decision=None,
            completed_chunks=outcome['chunks'],executed_primitive_steps=outcome['primitive_steps'])
    return protocol,inventory,descriptor,producers,manifests


def initial():
    return dict(status='running',next_cell=0,records=[],pending=None,control_active=False,source_eligible=False,new_optimizer_updates=0)


def test_exact_parent_first8_scope_and_no_old_credit_expansion(tmp_path):
    protocol,grant,authority=documents();entry.validate_protocol(protocol,grant,authority)
    for document,key,value in [(protocol,'episodes',23),(protocol,'maximum_controller_actions',9200),
            (grant,'maximum_GPU_seconds',2400),(grant,'maximum_output_bytes',8<<30),
            (authority,'only_first8_of_unfunded23_template',False)]:
        old=document[key];document[key]=value
        with pytest.raises(ValueError):entry.validate_protocol(protocol,grant,authority)
        document[key]=old
    assert grant['old_budgets_untouched']['old_A1_actions_remaining']==78
    assert authority['no_automatic_source_or_student_admission'] is True
    request={key:dict(path=str(tmp_path/grant[key]['path']),sha256=grant[key]['sha256']) for key in ('authority','protocol')}
    unchanged=deepcopy(grant);entry.validate_parent_links(grant,request)
    assert grant==unchanged  # The actual published relative-path grant is never rewritten.
    request['protocol']['path']=str(tmp_path/'wrong.json')
    with pytest.raises(ValueError,match='suffix'):entry.validate_parent_links(grant,request)


def test_actual_canonical_train_records_instructions_seeds_and_failed04_preserved():
    data=fixture_data();specs=entry.build_cells(*data)
    assert len(specs)==8 and sum(s['horizon'] for s in specs)==3200
    assert [s['role'] for s in specs]==list(entry.MODES)*4
    assert [s['reset_seed'] for s in specs]==[i for i in range(4) for _ in range(2)]
    assert len({s['paired_instruction'][0] for s in specs})==4
    for s in specs:
        assert s['motor_seed']==20261301+s['case'] and s['coefficients']==[0.]*4
        assert s['new_optimizer_updates']==0
        assert s['actual_acting_checkpoint_iteration']==(0 if s['arm']=='native' else 5000)
    assert data[2]['fit_provenance']['outcome_evidence']['phone-04']['success'] is False
    assert data[2]['fit_provenance']['source_eligible'] is False


def test_rejects_dev_substitution_instruction_manifest_outcome_or_split_changes():
    for fault in ('seed','instruction','manifest','reset','outcome','split','failed04','head'):
        data=list(fixture_data());p,inv,d,producers,manifests=data;r=producers['phone-00']['records'][0]
        if fault=='seed':inv['episodes'][0]['reset_seed']=19
        elif fault=='instruction':r['spec']['paired_instruction']=['wrong DEV instruction']
        elif fault=='manifest':manifests['phone-00']['pending_decision']='uncertain'
        elif fault=='reset':r['reset_identity']['features_sha256']='0'*64
        elif fault=='outcome':r['success']=False
        elif fault=='split':d['fit_provenance']['profile']['whole_split']['phone-00']='validation'
        elif fault=='failed04':d['fit_provenance']['outcome_evidence']['phone-04']['success']=True
        else:d['checkpoint']['sha256']='f'*64
        with pytest.raises(ValueError):entry.build_cells(*data)


def test_all8_failures_retained_four_domain_forks_stop_and_partial_refusal():
    specs=entry.build_cells(*fixture_data());state=initial();forks={};restored=[];saved=[]
    def capture():return {k:len(state['records']) for k in ('python','numpy','torch','cuda')}
    def episode(spec,*,qualify):
        assert qualify is (len(state['records'])==0)
        return dict(spec=spec,complete=True,success=False,primitive_steps=400)
    entry.run_cells(state,specs,forks,episode,capture=capture,restore=restored.append,admit=lambda:True,persist=saved.append)
    assert state['status']=='complete' and len(state['records'])==8 and all(not r['success'] for r in state['records'])
    assert len(forks)==4 and all(restored[i]==restored[i+1] for i in (0,2,4,6))
    assert sum(r['primitive_steps'] for r in state['records'])==3200 and state['source_eligible'] is False
    entry.validate_prefix(state,specs)
    state=initial();entry.run_cells(state,specs,{},episode,capture=capture,restore=lambda _:None,admit=lambda:False,persist=lambda _:None)
    assert state['status']=='closed_boundary_stop' and state['next_cell']==0 and state['pending'] is None
    state['pending']=specs[0]
    with pytest.raises(ValueError):entry.validate_prefix(state,specs)


def test_real_common_loop_canonical_decision_metadata_has_checkpoint_and_zero_new_updates(tmp_path):
    import numpy as np
    from fluffy_bassoon.meta_rl.robotwin_steering_screen import run_complete_episode
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    spec=entry.build_cells(*fixture_data())[1];observed=[]
    class Control:
        native_horizon=32;elapsed_steps=0;active=False
        env=type('Env',(),{'corpus_writer':None})()
        def episode_limit_for_task(self,_):return 400
        def set_recording_context(self,_):pass
        def set_coefficients(self,arm,values):self.coefficients=np.asarray(values)
        def reset(self,task,seed,**kwargs):
            self.active=True;self.last_reset_info=dict(spec['expected_reset'],reset_policy_actions=0,instruction=spec['paired_instruction'])
            return np.zeros(2062,np.float32)
        def record_decision_metadata(self,value):observed.append(deepcopy(value))
        def step(self,values):
            self.active=False;self.elapsed_steps=17
            return np.zeros(2062,np.float32),0.,True,False,dict(primitive_steps=17,success=False)
        def snapshot(self):return dict(active=False)
    with CampaignBudget(tmp_path/'budget',campaign_id=entry.PHASE,total_actions=3200,source_caps={},
            phase_caps={entry.ACTION_PHASE:3200},contract={},run_id='synthetic',source_commit='a'*40) as budget:
        result=run_complete_episode(Control(),spec,budget,run_id='synthetic',phase=entry.ACTION_PHASE,controller=entry.FrozenZeroController(spec))
        assert result['complete'] and result['primitive_steps']==17 and budget.snapshot()['actual_actions']==17
    assert observed[0]['actual_acting_checkpoint_iteration']==5000
    assert observed[0]['new_optimizer_updates']==observed[0]['optimizer_updates']==0
    assert observed[0]['steering_partition']=='train' and observed[0]['behavior_version']==entry.CHECKPOINT


def test_fresh_generic3200_includes_both_controls_and_crashed_intent_never_refunded(tmp_path):
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    args=dict(campaign_id=entry.PHASE,total_actions=3200,source_caps={},phase_caps={entry.ACTION_PHASE:3200},
        contract=dict(prior_closed_actions=29922,source_eligible=False),run_id='synthetic',source_commit='a'*40)
    with CampaignBudget(tmp_path,**args) as budget:
        for i in range(7):
            r=budget.reserve(str(i),400,phase=entry.ACTION_PHASE,method=entry.MODES[i%2],task='place_phone_stand',episode=str(i),chunk=0,policy_version='synthetic')
            assert r.execute;budget.resolve(str(i),400)
        r=budget.reserve('crash',32,phase=entry.ACTION_PHASE,method=entry.MODES[1],task='place_phone_stand',episode='7',chunk=0,policy_version='synthetic')
        assert r.execute and budget.snapshot()['charged_actions']==2832
    with CampaignBudget(tmp_path,**args) as budget:
        assert budget.snapshot()['uncertain_actions']==32
        assert not budget.reserve('crash',32,phase=entry.ACTION_PHASE,method=entry.MODES[1],task='place_phone_stand',episode='7',chunk=0,policy_version='synthetic').execute
        assert budget.remaining(phase=entry.ACTION_PHASE,method=entry.MODES[0])==368
        assert budget.header['total_actions']==3200 and budget.snapshot()['actual_actions']==2800


def test_real_publisher_on_empty_prepared_output_retains_closed_and_partial_state(tmp_path):
    import torch
    from fluffy_bassoon.meta_rl.robotwin_command_diagnostic_readout import prepare_output,publish_recovery
    output=tmp_path/'new';prepare_output(output);state=initial()
    payload=dict(state=deepcopy(state),global_rng={k:'synthetic' for k in ('python','numpy','torch','cuda')},
        case_rngs={},control=dict(active=False,motor_rng={'synthetic':True}),optimizer=None)
    publish_recovery(output,state,payload,boundary=True)
    ref=state['closed_recovery_ref'];assert entry.sha(ref['path'])==ref['sha256']
    loaded=torch.load(ref['path'],weights_only=False);assert loaded==payload
    state['pending']={'id':'partial'};publish_recovery(output,state,dict(payload,state=deepcopy(state)),boundary=False)
    assert state['closed_recovery_ref']==ref and json.loads((output/'result.json').read_text())['pending']=={'id':'partial'}
    assert entry.sha(ref['path'])==ref['sha256']
    old=dict(actual_actions=29922,charged_actions=29922)
    origin=dict(state=dict(status='complete_diagnostic_readout',next_cell=15,records=[dict(complete=True) for _ in range(15)],
        actions=old,pending=None,control_active=False),control=dict(active=False,motor_rng={'synthetic':True}),
        optimizer=None,global_rng={k:'synthetic' for k in ('python','numpy','torch','cuda')})
    entry.validate_origin(origin,old)
    for key,value in [('status','complete'),('pending',{'partial':True}),('control_active',True)]:
        changed=deepcopy(origin);changed['state'][key]=value
        with pytest.raises(ValueError,match='complete_diagnostic_readout'):entry.validate_origin(changed,old)


def test_unbound_template_refuses_before_tensor_imports_and_no_automatic_resume(tmp_path,monkeypatch):
    import builtins
    request=json.loads((ROOT/'REQUEST-TEMPLATE.json').read_text());p=tmp_path/'request.json';p.write_text(json.dumps(request))
    t=tmp_path/'ticket.json';t.write_text(json.dumps(dict(admitted=False)));original=builtins.__import__
    def no_runtime(name,*args,**kwargs):
        if name in ('numpy','torch'):raise AssertionError('No tensor imports before authority')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',no_runtime)
    argv=['--request',str(p),'--request-sha256',entry.sha(p),'--ticket',str(t),'--ticket-sha256',entry.sha(t)]
    for name in ('gpu-probe','model-path','norm-stats','robotwin-root','rlinf-root','output'):argv+=['--'+name,str(tmp_path/name)]
    with pytest.raises(ValueError,match='UNBOUND'):entry.main(argv)
    assert not (tmp_path/'output').exists()
    request.update(status='owner_bound_request',resume={'forbidden':'automatic retry'})
    with pytest.raises(ValueError,match='automatic replay'):
        entry.validate_ticket(request,{}, {},source='a'*40,request_sha='b'*64,hostname='fake',output=tmp_path/'output',now=1.)


def clock_case(tmp_path):
    request=dict(format=entry.FORMAT,status='owner_bound_request',source_eligible=False,resume=None)
    ticket=dict(admitted=True,source_commit='a'*40,phase_id=entry.PHASE,request_sha256='b'*64,
        hostname='test',gpu_count=1,maximum_gpu_seconds=1200,maximum_output_bytes=4<<30,automatic_retry=False,
        scope='first8_TRAIN_controls_only',output=str(tmp_path/'experiment'),aggregate_output_root=str(tmp_path),
        clock_contract={'phase':entry.PHASE},resource_attempt='second',cleanup_seconds=60,
        driver_deadline_unix=900,minimum_episode_admission_seconds=100,continuation_admitted=True,
        diagnosis_receipt=dict(path='/actual/diagnosis.json',sha256='c'*64),resume=None)
    clock=dict(contract=ticket['clock_contract'],worker_seconds=1200,attempts=[
        dict(path='first',status='finished',lineage=entry.PHASE,lineage_seconds=1200,started_unix=100,
            ended_unix=200,charged_seconds=130,reserved_seconds=150,slice_seconds=100),
        dict(path='second',status='started',lineage=entry.PHASE,lineage_seconds=1200,started_unix=300,
            reserved_seconds=1000,slice_seconds=900)])
    args=dict(source='a'*40,request_sha='b'*64,hostname='test',output=tmp_path/'experiment',now=400)
    return request,ticket,clock,args


def test_explicit_diagnosed_charged_prior_attempts_share1200_without_overlap_or_refill(tmp_path):
    request,ticket,clock,args=clock_case(tmp_path);entry.validate_ticket(request,ticket,clock,**args)
    for fault in ('overcap','overlap','two_started','missing_owner','changed_cap'):
        r,t,c,a=deepcopy((request,ticket,clock,args))
        if fault=='overcap':c['attempts'][0]['charged_seconds']=201
        elif fault=='overlap':c['attempts'][0]['ended_unix']=301
        elif fault=='two_started':c['attempts'][0]['status']='started'
        elif fault=='missing_owner':t['continuation_admitted']=False
        else:c['worker_seconds']=2400
        with pytest.raises(ValueError):entry.validate_ticket(r,t,c,**a)


def resume_fixture(tmp_path):
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    specs=entry.build_cells(*fixture_data());args=dict(campaign_id=entry.PHASE,total_actions=3200,source_caps={},
        phase_caps={entry.ACTION_PHASE:3200},contract={'same':'panel'},run_id='first',source_commit='a'*40)
    root=tmp_path/'budget'
    with CampaignBudget(root,**args) as budget:
        for i in range(2):
            r=budget.reserve(str(i),400,phase=entry.ACTION_PHASE,method=specs[i]['method'],task='place_phone_stand',
                episode=specs[i]['id'],chunk=0,policy_version=specs[i]['policy_version'])
            assert r.execute;budget.resolve(str(i),400)
        actions=budget.snapshot()
    ledger=dict(root=str(root),**{k:args[k] for k in ('campaign_id','total_actions','source_caps','phase_caps','contract')})
    contract=dict(source='same',head=entry.CHECKPOINT,protocol='same')
    state=initial();state.update(format=entry.FORMAT,next_cell=2,status='closed_boundary_stop',panel_contract=contract,
        action_ledger=ledger,actions=actions,output=str(tmp_path/'experiment'),qualification={'status':'passed'},
        closed_controls=[dict(path=f'/closed/{i}.pt',sha256='d'*64) for i in range(2)],
        records=[dict(spec=specs[i],complete=True,success=False,primitive_steps=400) for i in range(2)])
    domains={k:'saved' for k in ('python','numpy','torch','cuda')}
    saved=dict(state=deepcopy(state),control=dict(active=False,motor_rng={'saved':True},mapping={'same':'frozen'}),
        global_rng=domains,case_rngs={'phone-00':deepcopy(domains)},optimizer=None,last_closed_control={'saved':True})
    return saved,state,specs,contract,ledger,args


def test_explicit_closed_prefix_reopens_same_counter_and_resumes_remaining_cells_only(tmp_path):
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    from fluffy_bassoon.meta_rl.robotwin_reactive_capacity import validate_capacity_locked
    saved,head,specs,contract,ledger,args=resume_fixture(tmp_path)
    entry.validate_resume(saved,head,specs,contract=contract,ledger=ledger,output=tmp_path/'experiment')
    entry.verify_resume_journal(head,ledger,'second')
    with CampaignBudget(ledger['root'],**dict(args,run_id='second')) as budget:
        validate_capacity_locked(budget,head)
        assert budget.snapshot()['actual_actions']==800 and budget.remaining(phase=entry.ACTION_PHASE,method='readout')==2400
    state=deepcopy(saved['state']);forks=deepcopy(saved['case_rngs']);seen=[]
    def episode(spec,*,qualify):
        assert not qualify;seen.append(spec['id']);return dict(spec=spec,complete=True,success=False,primitive_steps=400)
    entry.run_cells(state,specs,forks,episode,capture=lambda:{k:'new' for k in ('python','numpy','torch','cuda')},
        restore=lambda _:None,admit=lambda:True,persist=lambda _:None)
    assert seen==[s['id'] for s in specs[2:]] and len(state['records'])==8
    assert sum(r['primitive_steps'] for r in state['records'])==3200 and len(forks)==4


def test_closed_prefix_refuses_pending_uncertain_head_or_panel_drift(tmp_path):
    values=resume_fixture(tmp_path)
    for fault in ('pending','uncertain','head_drift','source_drift','case_rng','active','charge_drift'):
        saved,head,specs,contract,ledger,_=deepcopy(values)
        if fault=='pending':saved['state']['pending']=specs[2]
        elif fault=='uncertain':head['actions']['uncertain_actions']=32
        elif fault=='head_drift':saved['state']['records'][0]['spec']['policy_version']='changed'
        elif fault=='source_drift':contract['source']='changed'
        elif fault=='case_rng':saved['case_rngs']={}
        elif fault=='active':saved['control']['active']=True
        else:saved['state']['actions']['actual_actions']=0
        with pytest.raises(ValueError):entry.validate_resume(saved,head,specs,contract=contract,ledger=ledger,output=tmp_path/'experiment')
