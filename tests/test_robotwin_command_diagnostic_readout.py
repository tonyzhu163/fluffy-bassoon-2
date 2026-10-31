"""Prepared allocated-Linux CPU cases; synthetic, never physical qualification."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import random
import socket
import sys

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import robotwin_command_diagnostic_readout as readout
from fluffy_bassoon.meta_rl.checkpoint import rng_state, restore_rng
from fluffy_bassoon.rl.pi05_residual import ReactiveKVResidual
from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual
from test_robotwin_steering import interface

pytestmark=pytest.mark.skipif(sys.platform!='linux',reason='Allocated Linux CPU only')
ROOT=Path(__file__).resolve().parents[1]/'outputs/robotwin-command-readout-driver-20261101-r1'


def paired_data():
    proposal=json.loads((ROOT/'READOUT-PROPOSAL-V2.json').read_text())
    demos=dict(episodes=[dict(id=f'phone-{i:02d}',instruction=f'canonical instruction {i}',
        reset_seed=readout.CASES.get(i,i)) for i in range(12)])
    resolved={}
    for i,seed in readout.CASES.items():
        identity=dict(actual_seed=seed,observation_sha256=str(i)*64,features_sha256=str(seed)*64)
        resolved[f'phone-{i:02d}']=dict(binding=deepcopy(proposal['canonical_bindings'][f'phone-{i:02d}']),
            reset_seed=seed,paired_instruction=[demos['episodes'][i]['instruction']],reset_info=identity,
            reset_identity=identity,episode_manifest_ref=dict(path=f'original-{i}.json',sha256='a'*64),
            original_task_success=i!=4)
    bindings=dict(format='robotwin_command_diagnostic_canonical_readout_bindings_v2',proposal=deepcopy(proposal),
        diagnostic_purpose=readout.DIAGNOSTIC,source_eligible=False,admitted=False,resolved=resolved)
    return proposal,bindings,demos


def test_exact_canonical_records_instruction_private_seed_and_retained_failed04():
    proposal,bindings,demos=paired_data();specs=readout.cells(proposal,bindings,demos)
    assert len(specs)==15 and sum(s['horizon'] for s in specs)==6000
    assert [s['case'] for s in specs]==[i for i in readout.CASES for _ in range(3)]
    assert [s['reset_seed'] for s in specs]==[v for v in readout.CASES.values() for _ in range(3)]
    for s in specs:
        assert s['motor_seed']==20261301+s['case'] and s['paired_instruction']==[demos['episodes'][s['case']]['instruction']]
        assert s['coefficients']==[0.]*4 and s['arm']==readout.MODES[s['condition']]
    assert bindings['resolved']['phone-04']['original_task_success'] is False


@pytest.mark.parametrize('fault',['ordinal_seed','private_seed','order','filter04','instruction','reset_hash','successful04','old_proposal'])
def test_rejects_old_or_filtered_or_unpaired_schedule(fault):
    p,b,d=paired_data()
    if fault=='ordinal_seed':p['cells'][3]['reset_seed']=8
    elif fault=='private_seed':p['cells'][3]['private_motor_seed']=20261301+11
    elif fault=='order':p['cells'][0],p['cells'][1]=p['cells'][1],p['cells'][0]
    elif fault=='filter04':p['cells']=p['cells'][3:]
    elif fault=='instruction':b['resolved']['phone-08']['paired_instruction']=['changed']
    elif fault=='reset_hash':b['resolved']['phone-08']['reset_info']=dict(b['resolved']['phone-08']['reset_info'],features_sha256='changed')
    elif fault=='successful04':b['resolved']['phone-04']['original_task_success']=True
    elif fault=='old_proposal':p['format']='robotwin_fixed12_command_diagnostic_readout_proposal_v1'
    b['proposal']=deepcopy(p)
    with pytest.raises(ValueError):readout.cells(p,b,d)


@pytest.fixture
def endpoint_pair(tmp_path,interface):
    value=dict(deepcopy(interface),head_dim=256);results={}
    profile=json.loads((ROOT/'FIT-PROFILE.json').read_text())
    for kind in ('action','kv'):
        if kind=='action':
            head=WideActionResidual(joint_lower=value['joint_lower'],joint_upper=value['joint_upper'],
                motor_sha256=value['motor_sha256'],normalization_sha256=value['normalization_sha256'],
                joint_convention_sha256=value['joint_convention_sha256'],base_interface_sha256='f'*64)
        else:head=ReactiveKVResidual(public_dim=2062,kv_heads=1,head_dim=256,rank=4,layers=(-1,),scale=.1)
        with torch.no_grad():next(reversed(list(head.parameters()))).fill_(.01)
        outcomes={f'phone-{i:02d}':dict(success=i!=4,primitive_steps=400,chunks=13,record_id=f'event-teacher-phone-{i:02d}',
            producer_commit='d'*40,producer_result=dict(path=f'/original/{i}.json',sha256='a'*64),
            episode_manifest_ref=dict(path=f'original-{i}.json',sha256='a'*64)) for i in range(12)}
        contract=dict(source_commit='d'*40,profile=deepcopy(profile),complete_cache_sha256='e'*64,
            diagnostic_purpose=readout.DIAGNOSTIC,source_eligible=False,outcome_evidence=outcomes,
            command_geometry=dict(synthetic_test_only=True))
        path=tmp_path/f'{kind}-005000.pt'
        torch.save(dict(kind=kind,iteration=5000,contract=contract,config=head.config(),policy=head.state_dict()),path)
        endpoint=dict(path=str(path),sha256=readout.file_sha(path))
        results[kind]=dict(complete=True,kind=kind,iteration=5000,contract=contract,head_configuration=head.config(),
            primary_endpoint=endpoint,milestones={str(i):dict(policy=deepcopy(endpoint)) for i in (0,100,500,2000,5000)})
    return results,value


def assert_rng_equal(a,b):
    assert a['python']==b['python']
    assert a['numpy'][0]==b['numpy'][0] and a['numpy'][2:]==b['numpy'][2:]
    np.testing.assert_array_equal(a['numpy'][1],b['numpy'][1]);assert torch.equal(a['torch'],b['torch'])
    assert a['cuda'] is None and b['cuda'] is None


def test_actual_endpoint_abi_uses_published_loader_without_A1_and_preserves_rng(endpoint_pair):
    results,interface=endpoint_pair;before=rng_state()
    heads,bindings=readout.validate_endpoints(results,interface,'f'*64,device='cpu')
    assert_rng_equal(before,rng_state())
    for kind,head in heads.items():
        assert not head.training and all(not p.requires_grad for p in head.parameters())
        assert bindings[kind]['fit_provenance']==results[kind]['contract']
        assert readout.state_sha(head.state_dict())==bindings[kind]['head_state_sha256']
        assert 'physical_basis_sha256' not in bindings[kind] and 'A1' not in bindings[kind]
    assert bindings['action']['fit_provenance']['outcome_evidence']['phone-04']['success'] is False


@pytest.mark.parametrize('fault',['not_complete','selected_not_final','iteration','profile_updates','purpose','source_eligible','split','failed04','cache','source_files','config','state'])
def test_actual_endpoint_rejects_changed_final_abi_or_provenance(endpoint_pair,fault):
    results,interface=endpoint_pair;row=results['action'];path=Path(row['primary_endpoint']['path'])
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if fault=='not_complete':row['complete']=False
    elif fault=='selected_not_final':row['milestones']['5000']['policy']['sha256']='0'*64
    elif fault=='iteration':row['iteration']=2000
    elif fault=='profile_updates':saved['contract']['profile']['updates']=4999
    elif fault=='purpose':saved['contract']['diagnostic_purpose']='source_eligible'
    elif fault=='source_eligible':saved['contract']['source_eligible']=True
    elif fault=='split':saved['contract']['profile']['whole_split']['phone-04']='validation'
    elif fault=='failed04':saved['contract']['outcome_evidence']['phone-04']['success']=True
    elif fault=='cache':saved['contract']['complete_cache_sha256']='0'*64
    elif fault=='source_files':saved['config']['source_bindings']['motor_sha256']='0'*64
    elif fault=='config':saved['config']['width']=32
    elif fault=='state':saved['policy'].pop(next(iter(saved['policy'])))
    if fault not in ('not_complete','selected_not_final','iteration'):
        row['contract']=deepcopy(saved['contract']);row['head_configuration']=deepcopy(saved['config'])
        torch.save(saved,path);row['primary_endpoint']['sha256']=readout.file_sha(path)
        row['milestones']['5000']['policy']=deepcopy(row['primary_endpoint'])
    before=rng_state()
    with pytest.raises((ValueError,RuntimeError)):readout.validate_endpoints(results,interface,'f'*64,device='cpu')
    assert_rng_equal(before,rng_state())


class Env:
    active=False;elapsed_steps=0;native_horizon=32
    contract=dict(synthetic_test_only=True)
    def __init__(self,root):
        self.root=root
        self.backend=SimpleNamespace(model=SimpleNamespace(device=torch.device('cpu')),
            kv_adapter=None,action_residual=None,post_decode_residual=None,post_command_transform=None,
            env=SimpleNamespace(venv=SimpleNamespace(envs=[SimpleNamespace(task=SimpleNamespace(robot=SimpleNamespace()))])),
            _check_frozen=lambda:None)
        self.noises=[]
    def record_decision_metadata(self,value):self.metadata=value
    def step(self,noise):
        self.noises.append(noise.copy());self.elapsed_steps+=32
        relative=f'episodes/synthetic/chunks/{len(self.noises):06d}.json'
        path=self.root/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('{}')
        return np.zeros(2062,np.float32),0.,False,False,dict(primitive_steps=32,issued_primitive_steps=32,corpus_chunk=relative)


def control_fixture(endpoint_pair):
    results,interface=endpoint_pair;heads,bindings=readout.validate_endpoints(results,interface,'f'*64,device='cpu')
    root=Path(results['action']['primary_endpoint']['path']).parent/'corpus'
    env=Env(root);writer=SimpleNamespace(root=root,register_runtime=lambda value:None)
    return readout.ReadoutControl(env,heads,writer,interface,bindings)


@pytest.mark.parametrize('arm',['native','action','kv'])
def test_pure_full_head_routing_and_existing_private_full1024_noise(endpoint_pair,arm):
    control=control_fixture(endpoint_pair);control.motor_rng=np.random.default_rng(20261309)
    control.set_coefficients(arm,[0.]*4)
    assert control.env.backend.kv_adapter is (control.heads['kv'] if arm=='kv' else None)
    assert control.env.backend.action_residual is (control.heads['action'] if arm=='action' else None)
    assert control.step([0.]*4)[-1]['primitive_steps']==32
    assert control.last_audit['native_plan_observation']['available'] is False
    assert control.last_audit['physical_after_audit']['actors']['phone']['available'] is False
    expected=np.random.default_rng(20261309).standard_normal((32,32)).astype(np.float32)
    np.testing.assert_array_equal(control.env.noises[0],expected)
    assert control.env.metadata['capacity_motor_rng_before']!=control.env.metadata['capacity_motor_rng_after']
    assert control.snapshot()['readonly'] and control.snapshot()['frozen_endpoints']==control.references
    with pytest.raises(ValueError):control.set_coefficients(arm,[.01,0,0,0])


@pytest.mark.parametrize('fault',['post_decode','calibration','gradient','train','weight'])
def test_pure_head_rejects_physical4d_calibration_or_changed_weights(endpoint_pair,fault):
    control=control_fixture(endpoint_pair)
    if fault=='post_decode':control.env.backend.post_decode_residual=object()
    elif fault=='calibration':control.env.backend.post_command_transform=object()
    elif fault=='gradient':next(control.heads['kv'].parameters()).requires_grad_(True)
    elif fault=='train':control.heads['action'].train()
    elif fault=='weight':
        with torch.no_grad():next(control.heads['action'].parameters()).flatten()[0].add_(.1)
    with pytest.raises(RuntimeError):control.assert_frozen()


def test_first_native_observational_qualification_restores_final_heads_and_rng(endpoint_pair,tmp_path,monkeypatch):
    control=control_fixture(endpoint_pair);control.motor_rng=np.random.default_rng(20261305)
    heads=control.heads;before=rng_state();private=deepcopy(control.motor_rng.bit_generator.state)
    monkeypatch.setattr(readout,'make_heads',lambda *args:dict(kv=deepcopy(heads['kv']),action=deepcopy(heads['action'])))
    def no_action_qualification(current,spec,output):
        assert current.heads is not heads and current.heads['action'].config()==heads['action'].config()
        assert current.env.noises==[]
        random.random();np.random.rand();torch.rand(1)
        return dict(status='decoded_and_gradient_pass_episode_pending')
    monkeypatch.setattr(readout,'qualify_real_motor',no_action_qualification)
    proof=readout.qualify_first_native(control,dict(arm='native'),tmp_path,control.interface)
    assert control.heads is heads and control.env.noises==[] and proof['extra_episodes']==proof['extra_robot_actions']==0
    assert control.motor_rng.bit_generator.state==private;assert_rng_equal(before,rng_state());control.assert_frozen()


def fresh_state(contract=None):
    return dict(contract=contract or {},status='running',next_cell=0,records=[],closed_controls=[],pending=None,control_active=False)


def test_all15_failed_outcomes_are_retained_with_same_per_case_global_forks_and_single_qualification():
    specs=readout.cells(*paired_data());state=fresh_state();forks={};seen=[];durable=[]
    def episode(spec,qualify):
        seen.append((spec['record_id'],(random.random(),float(np.random.rand()),float(torch.rand(()))),qualify))
        state['closed_controls'].append(dict(private_motor_seed=spec['motor_seed']))
        return dict(spec=spec,complete=True,success=False,primitive_steps=400)
    readout.run_cells(state,specs,forks,episode,admit=lambda:True,persist=lambda boundary:durable.append((boundary,state['next_cell'],deepcopy(state['pending']))))
    assert state['status']=='complete_diagnostic_readout' and len(state['records'])==15
    assert sum(r['primitive_steps'] for r in state['records'])==6000 and not any(r['success'] for r in state['records'])
    assert sum(row[2] for row in seen)==1 and seen[0][2]
    for i in range(0,15,3):assert seen[i][1]==seen[i+1][1]==seen[i+2][1]
    assert len(forks)==5 and len([r for r in durable if not r[0]])==15


def test_closed_mid_case_rollover_preserves_original_case_fork_without_repeating_first_episode():
    specs=readout.cells(*paired_data());state=fresh_state();forks={};seen=[]
    def episode(spec,qualify):
        seen.append((spec['id'],float(np.random.rand()),qualify));state['closed_controls'].append({'closed':True})
        return dict(spec=spec,complete=True,success=False,primitive_steps=400)
    readout.run_cells(state,specs,forks,episode,admit=lambda:state['next_cell']<1,persist=lambda boundary:None)
    assert state['next_cell']==1 and state['status']=='closed_boundary_stop' and state['pending'] is None
    readout.validate_cursor(state,specs,{})
    state['status']='running'
    readout.run_cells(state,specs,forks,episode,admit=lambda:True,persist=lambda boundary:None)
    assert len(seen)==15 and seen[0][1]==seen[1][1]==seen[2][1] and sum(row[2] for row in seen)==1


@pytest.mark.parametrize('fault',['pending','active','prefix','closed_control','complete15','failed_status'])
def test_resume_rejects_partial_unresolved_changed_or_complete_prefix(fault):
    specs=readout.cells(*paired_data());state=fresh_state()
    state.update(next_cell=1,records=[dict(spec=deepcopy(specs[0]),complete=True)],closed_controls=[{}])
    if fault=='pending':state['pending']=specs[1]
    elif fault=='active':state['control_active']=True
    elif fault=='prefix':state['records'][0]['spec']['motor_seed']+=1
    elif fault=='closed_control':state['closed_controls']=[]
    elif fault=='complete15':state['next_cell']=15
    elif fault=='failed_status':state['status']='blocked_failure'
    with pytest.raises(ValueError):readout.validate_cursor(state,specs,{})


def test_pending_cell_failure_never_retries_or_advances_cursor():
    specs=readout.cells(*paired_data());state=fresh_state();calls=[]
    def episode(spec,qualify):calls.append(spec['id']);raise RuntimeError('actual chunk failed')
    with pytest.raises(RuntimeError):readout.run_cells(state,specs,{},episode,admit=lambda:True,persist=lambda boundary:None)
    assert calls==[specs[0]['id']] and state['next_cell']==0 and state['pending']==specs[0]
    with pytest.raises(ValueError):readout.validate_cursor(state,specs,{})


def test_common_loop_natural400_and_full_episode_admission_use_existing_canonical_ledger(tmp_path):
    class Control:
        native_horizon=32;active=False;elapsed_steps=0;resets=0
        env=SimpleNamespace(corpus_writer=None)
        def episode_limit_for_task(self,task):return 400
        def set_recording_context(self,value):pass
        def set_coefficients(self,arm,values):self.coefficients=np.array(values)
        def reset(self,task,seed,**kw):
            self.resets+=1;self.active=True;self.elapsed_steps=0
            self.last_reset_info=dict(reset_policy_actions=0,actual_seed=seed,observation_sha256='o',features_sha256='f',instruction=kw['paired_instruction'])
            return np.zeros(2062,np.float32)
        def record_decision_metadata(self,value):pass
        def step(self,values):
            actual=min(32,400-self.elapsed_steps);self.elapsed_steps+=actual;self.active=self.elapsed_steps<400
            return np.zeros(2062,np.float32),0.,False,not self.active,dict(primitive_steps=actual,success=False)
        def snapshot(self):return dict(active=self.active,elapsed_steps=self.elapsed_steps)
    specs=readout.cells(*paired_data());control=Control()
    with readout.CampaignBudget(tmp_path/'ledger',campaign_id='synthetic',total_actions=1200,source_caps={},
            phase_caps={'a1':1200},contract={'synthetic_test_only':True},run_id='synthetic',source_commit='a'*40) as budget:
        for spec in specs[:3]:
            spec=dict(spec,expected_reset=dict(actual_seed=spec['reset_seed'],observation_sha256='o',features_sha256='f'))
            record=readout.run_complete_episode(control,spec,budget,run_id='synthetic')
            assert record['complete'] and record['primitive_steps']==400 and record['chunks']==13
        assert budget.snapshot()['actual_actions']==1200 and budget.snapshot()['unresolved_decisions']=={}
        with pytest.raises(RuntimeError,match='full native-episode admission'):
            readout.run_complete_episode(control,specs[3],budget,run_id='synthetic')
        assert control.resets==3


def ticket_data(tmp_path):
    root=str(tmp_path);attempt=root+'/attempt-current';contract=dict(original_clock=True)
    row=dict(path=attempt,lineage='fixed12-command-diagnostic/readout',lineage_seconds=1800,
        started_unix=100.,slice_seconds=1140,reserved_seconds=1200,status='started')
    clock=dict(contract=contract,worker_seconds=10000,attempts=[dict(path='old-fit',lineage='fixed12-command-diagnostic/fit-action',
        reserved_seconds=2000,charged_seconds=800,status='finished'),row])
    plan=dict(admitted=True,source_eligible=False,diagnostic_purpose=readout.DIAGNOSTIC,clock_root=root,clock_contract=contract,
        stage_gpu_limits=dict(readout=1800),readout_proposal_sha256=readout.PROPOSAL_SHA,
        maximum_readout_controller_actions=6000,prior_cost_receipts=[dict(path='old-cost',sha256='a'*64)],storage_admission={'path':'storage','sha256':'b'*64})
    ticket=dict(format=readout.FORMAT,admitted=True,source_commit='c'*40,request_sha256='d'*64,hostname=socket.gethostname(),
        gpu_count=1,maximum_robot_actions=6000,output=str(tmp_path/'output'),clock_contract=contract,clock_root=root,
        aggregate_output_root=str(tmp_path),maximum_output_bytes=1<<30,
        resource_attempt=attempt,driver_deadline_unix=1180.,cleanup_seconds=60,minimum_episode_admission_seconds=120)
    return ticket,clock,plan


def test_actual_measured_clock_can_exceed_900_estimate_without_new_allocator(tmp_path):
    ticket,clock,plan=ticket_data(tmp_path)
    readout.validate_ticket(ticket,clock,plan,source='c'*40,request_sha='d'*64,output=tmp_path/'output',now=150)
    assert clock['attempts'][-1]['reserved_seconds']>900


@pytest.mark.parametrize('fault',['cap_reset','over_spent','over_reserved','over_deadline','other_live','unknown_lineage','wrong_host','wrong_proposal','wrong_clock','zero_actions_fit_ticket'])
def test_actual_readout_admission_rejects_old_or_overlapping_or_reset_scope(tmp_path,fault):
    ticket,clock,plan=ticket_data(tmp_path)
    if fault=='cap_reset':clock['attempts'][-1]['lineage_seconds']=1900
    elif fault=='over_spent':clock['attempts'].append(dict(path='old-readout',lineage='fixed12-command-diagnostic/readout',lineage_seconds=1800,charged_seconds=700,reserved_seconds=700,status='finished'))
    elif fault=='over_reserved':clock['worker_seconds']=1500
    elif fault=='over_deadline':ticket['driver_deadline_unix']=1240.
    elif fault=='other_live':clock['attempts'].append(dict(clock['attempts'][-1],path='other'))
    elif fault=='unknown_lineage':clock['attempts'].append(dict(path='unknown',lineage='fixed12-command-diagnostic/readout-v2',reserved_seconds=1,status='finished'))
    elif fault=='wrong_host':ticket['hostname']='another-job'
    elif fault=='wrong_proposal':plan['readout_proposal_sha256']='0'*64
    elif fault=='wrong_clock':plan['clock_contract']={}
    elif fault=='zero_actions_fit_ticket':ticket['maximum_robot_actions']=0
    with pytest.raises(ValueError):readout.validate_ticket(ticket,clock,plan,source='c'*40,request_sha='d'*64,output=tmp_path/'output',now=150)


def test_four_domain_case_fork_resume_requires_original_prefix():
    specs=readout.cells(*paired_data());fork=dict(python=(),numpy=(),torch=(),cuda=[])
    readout.validate_case_forks({'phone-04':fork},specs,2)
    for forks in ({},{'phone-04':dict(fork,cuda=None)},{'phone-04':fork,'phone-08':fork}):
        with pytest.raises(ValueError):readout.validate_case_forks(forks,specs,2)


def test_actual_initial_actor_measured_and_drive_identity_is_required():
    identity={key:dict(item=dict(available=True,value={'actual':1})) for key in ('actors','articulations','grippers')}
    assert readout.initial_physical_identity(identity)==identity
    identity['grippers']['item']['available']=False
    with pytest.raises(ValueError):readout.initial_physical_identity(identity)


def test_fresh_output_and_recoveries_ancestry_is_durable_before_any_reset(tmp_path,monkeypatch):
    from fluffy_bassoon.rl import robotwin_corpus as corpus
    original=corpus._fsync_directory;seen=[]
    def fsync(path):seen.append(Path(path));original(path)
    monkeypatch.setattr(corpus,'_fsync_directory',fsync);monkeypatch.setattr(readout,'_fsync_directory',fsync)
    output=tmp_path/'new-parent'/'runtime';readout.prepare_output(output)
    assert (output/'recoveries').is_dir()
    assert seen.index(tmp_path)<seen.index(output.parent)<seen.index(output)
    with pytest.raises(FileExistsError):readout.prepare_output(output)


@pytest.mark.parametrize('interrupt',[False,True])
def test_checkpoint_child_directory_is_durable_before_publishing_physical_cursor(tmp_path,monkeypatch,interrupt):
    output=tmp_path/'runtime';readout.prepare_output(output)
    state=dict(next_cell=1,closed_recovery_ref={'path':'previous-safe.pt','sha256':'a'*64})
    previous=deepcopy(state);readout._atomic_json(output/'result.json',state)
    events=[];save,json_write,fsync=readout.atomic_save,readout._atomic_json,readout._fsync_directory
    def checkpoint(path,payload):
        value=save(path,payload);assert Path(path).is_file();events.append('checkpoint_renamed');return value
    def directory(path):
        assert Path(path)==output/'recoveries' and events==['checkpoint_renamed']
        if interrupt:raise OSError('synthetic directory-sync interruption')
        fsync(path);events.append('child_directory_durable')
    def publish(path,value,**kw):
        assert events==['checkpoint_renamed','child_directory_durable']
        ref=value['closed_recovery_ref'];assert readout.file_sha(ref['path'])==ref['sha256']
        events.append('cursor_published');return json_write(path,value,**kw)
    monkeypatch.setattr(readout,'atomic_save',checkpoint);monkeypatch.setattr(readout,'_fsync_directory',directory)
    monkeypatch.setattr(readout,'_atomic_json',publish)
    if interrupt:
        with pytest.raises(OSError):readout.publish_recovery(output,state,dict(state=deepcopy(state)),boundary=True)
        assert state==previous and json.loads((output/'result.json').read_text())==previous
        assert events==['checkpoint_renamed']  # Unpublished orphan is safe; old cursor stays authoritative.
    else:
        readout.publish_recovery(output,state,dict(state=deepcopy(state)),boundary=True)
        assert events==['checkpoint_renamed','child_directory_durable','cursor_published']
