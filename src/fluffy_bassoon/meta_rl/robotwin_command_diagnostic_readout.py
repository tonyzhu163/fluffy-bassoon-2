"""Fifteen complete pure-head diagnostic episodes; no allocator, learner or4D mapping."""
import argparse
from copy import deepcopy
import inspect
import json
import math
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import numpy as np
import torch

from fluffy_bassoon.baige_gpu import verify_selected_cuda,selected_device
from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter,_atomic_json,_ensure_directory,_fsync_directory
from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv,load_benchmark_episode_limits,_RoboTwinBackend
from fluffy_bassoon.rl.robotwin_fitted_physical import load_frontend,state_sha,content_sha,DIAGNOSTIC
from fluffy_bassoon.rl.robotwin_wide_action import WideActionResidual
from fluffy_bassoon.rl.robotwin_steering import validate_interface
from .checkpoint import rng_state,restore_rng
from .warmstart import atomic_save
from .robotwin_replay_diagnosis import Audit
from .robotwin_demo_capacity import file_sha,validate_demo_manifest
from .robotwin_reactive_fit import ReactiveEpisodeControl,make_heads
from .robotwin_reactive_capacity import CONFIG,validate_capacity_locked
from .robotwin_event_replay import physical_audit,EventEpisodeControl
from .robotwin_prelift_probe import observe_native_plans
from .robotwin_command_readout_resources import validate_lineages
from .robotwin_steering_screen import run_complete_episode,qualify_real_motor,verify_source_conventions

FORMAT='robotwin_command_diagnostic_physical_readout_v1'
PROPOSAL_SHA='4fcfbb4a4a5d941713c517e1589e55f12104578cb188a66293e234704ef21c55'
LOADER_SHA='7d9206feff92b0dddedd4f2d3e266586b87029625e40b164af9e387a45ce22e9'
FIT_PROFILE_SHA='6c1641b348c1ae6a254341b04fec070cdfcfe3ac34b60946d55ffdb12daeb04c'
CASES={4:4,8:11,9:14,10:16,11:18}
MODES={'native':'native','wide-final5000':'action','KV-final5000':'kv'}
FROZEN='721a7ed12c70d36e08882b02d8afee709286f6f27e72482570d7ac7451c0ae41'


def cells(proposal,bindings,demos):
    """Consume the source owner's v2 canonical receipt, never the old ordinal seeds."""
    if (proposal.get('format')!='robotwin_fixed12_command_diagnostic_readout_proposal_v2'
            or proposal.get('diagnostic_purpose')!=DIAGNOSTIC or proposal.get('source_eligible') is not False
            or proposal['case_order']!=list(CASES) or proposal['condition_order']!=list(MODES)
            or proposal['maximum_complete_episodes']!=15 or proposal['maximum_new_actions']!=6000
            or proposal['native_episode_limit']!=400 or proposal['action_horizon']!=32
            or bindings.get('format')!='robotwin_command_diagnostic_canonical_readout_bindings_v2'
            or bindings.get('proposal')!=proposal or bindings.get('source_eligible') is not False
            or bindings.get('diagnostic_purpose')!=DIAGNOSTIC or bindings.get('admitted') is not False
            or set(bindings['resolved'])!={f'phone-{i:02d}' for i in CASES} or len(proposal['cells'])!=15):
        raise ValueError('Exact canonical15/readout-v2 diagnostic data receipt required; no source admission')
    result=[]
    for item,(record,condition) in zip(proposal['cells'],((i,c) for i in CASES for c in MODES)):
        ident=f'phone-{record:02d}';source=bindings['resolved'][ident];row=demos['episodes'][record]
        expected=dict(id=f'diagnostic-case{record:02d}-{condition}',case=record,record_id=ident,canonical_binding=ident,
            reset_seed=CASES[record],private_motor_seed=20261301+record,condition=condition,
            checkpoint_iteration=None if condition=='native' else 5000)
        if (item!=expected or row['id']!=ident or row['reset_seed']!=CASES[record]
                or source['binding']!=proposal['canonical_bindings'][ident] or source['reset_seed']!=CASES[record]
                or source['paired_instruction']!=[row['instruction']]
                or source['reset_identity']!={k:source['reset_info'][k] for k in ('actual_seed','observation_sha256','features_sha256')}
                or source['reset_identity']['actual_seed']!=CASES[record]
                or source['original_task_success'] is not (record!=4)):
            raise ValueError('Canonical record, instruction/reset, original failed04 or endpoint changed')
        result.append(dict(item,task='place_phone_stand',arm=MODES[condition],method='command_readout_'+MODES[condition],
            coefficients=[0.]*4,stage='readonly_command_diagnostic',horizon=400,motor_seed=item['private_motor_seed'],
            paired_instruction=source['paired_instruction'],expected_reset=source['reset_identity']))
    return result


def validate_endpoints(results,interface,interface_sha,*,device):
    """Use the published loader verbatim; no frontend+A1/physical4D proof is needed."""
    heads={};bindings={};before=rng_state()
    try:
        if set(results)!={'action','kv'}:raise ValueError('Both actual final heads required')
        for kind,result in results.items():
            contract=result['contract'];endpoint=result['primary_endpoint'];path=Path(endpoint['path'])
            if not path.is_absolute() or path.is_symlink() or file_sha(path)!=endpoint['sha256']:raise ValueError('Actual primary endpoint bytes required')
            saved=torch.load(path,map_location='cpu',weights_only=False)
            if (result.get('complete') is not True or result['kind']!=kind or result['iteration']!=5000
                    or saved.get('kind')!=kind or saved.get('iteration')!=5000 or saved['contract']!=contract
                    or saved['config']!=result['head_configuration']
                    or contract.get('diagnostic_purpose')!=DIAGNOSTIC or contract.get('source_eligible') is not False
                    or contract['profile']['format']!='robotwin_fixed12_command_diagnostic_curve_v1'
                    or content_sha(contract['profile'])!=FIT_PROFILE_SHA
                    or set(result['milestones'])!={'0','100','500','2000','5000'}
                    or any(result['milestones']['5000']['policy'][k]!=endpoint[k]
                           for k in ('path','sha256'))):
                raise ValueError('Actual complete diagnostic5000 primary endpoint only; no VAL selection')
            # Descriptor only: existing ABI/digest/loader, no new policy serialization.
            binding=dict(kind=kind,iteration=5000,checkpoint=deepcopy(endpoint),fit_provenance=deepcopy(contract),
                fit_contract_sha256=content_sha(contract),fit_profile_sha256=content_sha(contract['profile']),
                fit_profile_format=contract['profile']['format'],fit_complete_cache_sha256=contract['complete_cache_sha256'],
                fit_source_commit=contract['source_commit'],head_config_sha256=content_sha(saved['config']),
                head_state_sha256=state_sha(saved['policy']),base_interface_sha256=interface_sha,
                **{k:interface[k] for k in ('motor_sha256','normalization_sha256','joint_convention_sha256')})
            head=load_frontend(binding,device=device)
            if head.config()!=result['head_configuration']:raise ValueError('Fit/loaded source class configuration differs')
            heads[kind]=head;bindings[kind]=binding
        if bindings['action']['fit_complete_cache_sha256']!=bindings['kv']['fit_complete_cache_sha256']:
            raise ValueError('Both heads must fit the same actual complete command-diagnostic cache')
    finally:restore_rng(before)
    return heads,bindings


class ReadoutControl(EventEpisodeControl):
    """Pure full-head hooks on the existing controller, never post-decode4D."""
    def __init__(self,env,heads,writer,interface,bindings):
        super().__init__(env,heads,writer,interface);self.bindings=deepcopy(bindings)
        self.references={k:dict(v['checkpoint'],iteration=5000,head_state_sha256=v['head_state_sha256']) for k,v in bindings.items()}
    def assert_frozen(self):
        backend=self.env.backend
        if getattr(backend,'post_decode_residual',None) is not None or getattr(backend,'post_command_transform',None) is not None:
            raise RuntimeError('Pure-head diagnostic cannot install composed physical4D/calibration hooks')
        for kind,head in self.heads.items():
            if head.training or any(p.requires_grad or p.grad is not None for p in head.parameters()) or state_sha(head.state_dict())!=self.bindings[kind]['head_state_sha256']:
                raise RuntimeError('Readonly endpoint changed')
        backend._check_frozen()
    def step(self,value):
        self.assert_frozen()
        with observe_native_plans(self.env.backend.env.venv.envs[0].task) as plans:
            result=super().step(value)
        self.last_audit['native_plan_observation']=plans
        for key in ('cursor_before','cursor_after'):self.last_audit.pop(key,None)  # No teacher/source cursor in this panel.
        self.assert_frozen();return result
    def snapshot(self):
        return dict(super().snapshot(),active=self.active,frozen_endpoints=deepcopy(self.references),readonly=True)


def initial_physical_identity(value):
    identity={k:deepcopy(value[k]) for k in ('actors','articulations','grippers')}
    if any(item.get('available') is not True for group in identity.values() for item in group.values()):
        raise ValueError('Actual initial phone/stand/measured/drive/gripper identity must be available')
    return identity


def qualify_first_native(control,spec,output,interface):
    """Existing no-action observation/cache checks share the FIRST native episode."""
    old=control.heads;before=rng_state();private=deepcopy(control.motor_rng.bit_generator.state)
    try:
        zero=make_heads(control.env.backend,interface,CONFIG)
        zero['action']=WideActionResidual.from_config(old['action'].config()).to(control.env.backend.model.device).eval()
        control.heads=zero;control.kv,control.action=zero['kv'],zero['action']
        proof=qualify_real_motor(control,spec,output)
        proof.update(zero_wide_configuration=zero['action'].config(),extra_robot_actions=0,extra_episodes=0)
        return proof
    finally:
        control.heads=old;control.kv,control.action=old['kv'],old['action'];control.set_coefficients('native',[0.]*4)
        restore_rng(before)
        if control.motor_rng.bit_generator.state!=private:raise RuntimeError('Qualification advanced private acting RNG')


def validate_cursor(state,specs,contract):
    n=state['next_cell']
    if (state['contract']!=contract or type(n) is not int or not 0<=n<15 or len(state['records'])!=n
            or state.get('pending') is not None or state.get('control_active',False)
            or state['status'] not in ('running','closed_boundary_stop') or len(state['closed_controls'])!=n
            or any(not r['complete'] or r['spec']!=specs[i] for i,r in enumerate(state['records']))):
        raise ValueError('Resume only the exact closed prefix; never replay partial/unresolved physics')


def validate_case_forks(forks,specs,n):
    if set(forks)!={spec['record_id'] for spec in specs[:n]} or any(
            set(value)!={'python','numpy','torch','cuda'} or value['cuda'] is None for value in forks.values()):
        raise ValueError('Closed readout prefix requires its exact saved four-domain case forks')


def run_cells(state,specs,forks,episode,*,admit,persist,capture=rng_state,restore=restore_rng):
    """One common complete-episode call per cell; failures are not filtered/retried."""
    for i in range(state['next_cell'],15):
        if not admit():state['status']='closed_boundary_stop';persist(True);return
        spec=specs[i];key=spec['record_id']
        if key not in forks:forks[key]=deepcopy(capture())
        restore(deepcopy(forks[key]));state['pending']=deepcopy(spec);persist(False)
        record=episode(deepcopy(spec),qualify=(i==0))
        if record.get('complete') is not True:raise RuntimeError('A cell did not close naturally')
        state['records'].append(record);state.update(next_cell=i+1,pending=None,control_active=False);persist(True)
    state['status']='complete_diagnostic_readout';persist(True)


def validate_ticket(ticket,clock,plan,*,source,request_sha,output,now,_audit=None):
    lineage='fixed12-command-diagnostic/readout';rows=[r for r in clock['attempts'] if r.get('lineage')==lineage]
    current=[r for r in rows if r['path']==ticket['resource_attempt']]
    limit=plan['stage_gpu_limits']['readout'];end=ticket['driver_deadline_unix'];cleanup=ticket['cleanup_seconds']
    if (ticket.get('format')!=FORMAT or ticket.get('admitted') is not True or ticket['source_commit']!=source
            or ticket['request_sha256']!=request_sha or ticket['hostname']!=socket.gethostname() or ticket['gpu_count']!=1
            or ticket['maximum_robot_actions']!=6000 or Path(ticket['output']).resolve()!=Path(output).resolve()
            or not Path(output).resolve().is_relative_to(Path(ticket['aggregate_output_root']).resolve())
            or type(ticket['maximum_output_bytes']) is not int or ticket['maximum_output_bytes']<1
            or ticket['clock_contract']!=clock['contract'] or ticket['clock_root']!=plan['clock_root']
            or plan['clock_contract']!=clock['contract'] or plan.get('source_eligible') is not False
            or not plan.get('prior_cost_receipts') or not plan.get('storage_admission')
            or plan.get('admitted') is not True or plan.get('diagnostic_purpose')!=DIAGNOSTIC
            or plan.get('readout_proposal_sha256')!=PROPOSAL_SHA or plan.get('maximum_readout_controller_actions')!=6000
            or type(limit) is not int or limit<1 or len(current)!=1 or current[0]['status']!='started'
            or any(r.get('lineage_seconds')!=limit for r in rows)
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in rows)>limit
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in clock['attempts'])>clock['worker_seconds']
            or any(r['status']=='started' and r is not current[0] for r in rows)
            or type(cleanup) is not int or cleanup<60 or not math.isfinite(end)
            or not current[0]['started_unix']<=now<end<=current[0]['started_unix']+current[0]['slice_seconds']-cleanup
            or not 120<=current[0]['reserved_seconds']<8*3600
            or not 0<ticket['minimum_episode_admission_seconds']<current[0]['slice_seconds']-cleanup):
        raise ValueError('Actual measured owner readout stage/ticket/clock required;900 is only an estimate')
    validate_lineages(ticket,clock,plan,audit=_audit)


def validate_closed_head(head,ticket,run_id):
    """Read-only before opening the existing lock; no foreign intent recovery."""
    old=head['actions'];ledger=head['action_ledger'];root=Path(ledger['root']).resolve(strict=True);count=old['journal_events']
    header=dict(schema=CampaignBudget.schema,**{k:ledger[k] for k in ('campaign_id','total_actions','source_caps','phase_caps','contract')})
    if (ledger!=ticket['action_ledger'] or root.parent!=Path(ticket['phase_output_root']).resolve(strict=True)
            or old['header']!=header or type(count) is not int or count<1
            or ledger['total_actions']!=300000 or ledger['phase_caps']['a1']!=30000
            or old['actual_actions']!=old['charged_actions'] or old['actual_actions']<23922
            or old['pending_actions'] or old['uncertain_actions'] or old['unresolved_decisions'] or head.get('pending') is not None
            or run_id in old['runs'] or sorted(p.name for p in (root/'events').glob('*.json'))!=[f'{i:010d}.json' for i in range(count)]
            or json.loads((root/'events/0000000000.json').read_text())['header']!=old['header']
            or json.loads((root/'events'/f'{count-1:010d}.json').read_text())['sha256']!=old['journal_sha256']):
        raise ValueError('Actual closed current canonical root/header/head required before fresh registration')


def prepare_output(output):
    """Reuse the corpus directory durability helpers before any physical reset."""
    output=Path(output);_ensure_directory(output.parent)
    output.mkdir(exist_ok=False);_fsync_directory(output.parent)
    (output/'recoveries').mkdir();_fsync_directory(output)


def publish_recovery(output,state,payload,*,boundary):
    """Existing checkpoint/JSON APIs, with child-directory fsync before the ref."""
    output=Path(output);ref=output/'recoveries'/f'cell-{state["next_cell"]:02d}-{uuid.uuid4().hex}.pt'
    sha=atomic_save(ref,payload);_fsync_directory(ref.parent)
    if boundary:state['closed_recovery_ref']=dict(path=str(ref),sha256=sha)
    _atomic_json(output/'result.json',state,replace=True)


def checked_runtime_source(obj,root,expected):
    """Only the approved native-plan contextmanager may use its wrapped source."""
    target=inspect.unwrap(obj) if obj is observe_native_plans else obj
    path=Path(inspect.getfile(target)).resolve();rel=str(path.relative_to(root))
    if expected.get(rel)!=file_sha(path):raise ValueError('Loaded readout/runtime blob differs: '+rel)
    return path


def main(argv=None):
    if sys.platform!='linux':raise RuntimeError('Readout computation requires the actual allocated Linux job')
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('request','ticket','gpu-probe','model-path','norm-stats','robotwin-root','rlinf-root','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--request-sha256',required=True);p.add_argument('--ticket-sha256',required=True);a=p.parse_args(argv)
    audit=Audit(time.time()+60);request=audit.json(dict(path=str(a.request),sha256=a.request_sha256));ticket=audit.json(dict(path=str(a.ticket),sha256=a.ticket_sha256))
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();root=Path(subprocess.check_output(['git','rev-parse','--show-toplevel'],text=True).strip()).resolve()
    if (request.get('format')!=FORMAT or request.get('diagnostic_purpose')!=DIAGNOSTIC
            or request.get('source_eligible') is not False or request.get('maximum_controller_actions')!=6000):raise ValueError('Explicit readonly diagnostic request required')
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Clean pinned readout source required')
    clock=json.loads((Path(ticket['clock_root'])/'worker-time.json').read_text());plan=audit.json(ticket['resource_plan_receipt'])
    validate_ticket(ticket,clock,plan,source=source,request_sha=a.request_sha256,output=a.output,now=time.time(),_audit=audit);audit=Audit(ticket['driver_deadline_unix'])
    audit.json(ticket['authority_receipt'])
    for ref in plan['prior_cost_receipts']:audit.json(ref)
    audit.json(plan['storage_admission'])
    for obj in (main,load_frontend,WideActionResidual,_RoboTwinBackend,ReactiveEpisodeControl,run_complete_episode,qualify_real_motor,
                physical_audit,observe_native_plans,validate_lineages,validate_capacity_locked,Audit,selected_device,rng_state,atomic_save,CampaignBudget,RobotwinCorpusWriter):
        checked_runtime_source(obj,root,request['runtime_blob_sha256'])
    for rel,digest in request['runtime_blob_sha256'].items():
        if Path(rel).is_absolute() or '..' in Path(rel).parts:raise ValueError('Runtime blob must be inside the deployed source')
        audit.bound(root/rel,digest)
    if file_sha(Path(inspect.getfile(load_frontend)))!=LOADER_SHA:raise ValueError('Exact published bbb200 loader required')
    proposal=audit.json(request['proposal']);canonical=audit.json(request['canonical_bindings']);demos=validate_demo_manifest(audit.json(request['demos']))
    if request['proposal']['sha256']!=PROPOSAL_SHA:raise ValueError('Only explicit pre-outcome readout-v2 amendment')
    specs=cells(proposal,canonical,demos);interface=validate_interface(audit.json(request['interface']))
    for identity,resolved in canonical['resolved'].items():
        if any(resolved['binding']['demos_ref'][k]!=request['demos'][k] for k in ('path','sha256')):
            raise ValueError('Actual demos differ from the original canonical record binding')
        producer=audit.json(resolved['binding']['producer_result']);matches=[r for r in producer['records'] if r['spec']['id']==resolved['binding']['record_id']]
        if (len(matches)!=1 or matches[0]['spec']['paired_instruction']!=resolved['paired_instruction']
                or any(matches[0][key]!=resolved[key] for key in ('reset_info','reset_identity','episode_manifest_ref'))):
            raise ValueError('Canonical DATA receipt differs from actual original named record')
    verify_source_conventions(audit.json(request['joint_conventions']),interface,dict(robotwin=a.robotwin_root,rlinf=a.rlinf_root))
    if file_sha(a.model_path/'model.safetensors')!=interface['motor_sha256'] or file_sha(a.norm_stats)!=interface['normalization_sha256']:raise ValueError('Exact motor/normalization files required')
    limits=load_benchmark_episode_limits(a.robotwin_root,['place_phone_stand']);probe=audit.json(dict(path=str(a.gpu_probe),sha256=file_sha(a.gpu_probe)))
    if limits['place_phone_stand']!=400 or probe['status']!='passed' or probe['hostname']!=socket.gethostname() or len(probe['devices'])!=1:raise ValueError('Actual native400/oneGPU probe required')
    if verify_selected_cuda(torch)!=selected_device()['uuid'] or selected_device()['cuda_uuid']!=probe['devices'][0]['cuda_uuid']:raise ValueError('Actual CUDA/render selection differs from probe')
    head=audit.json(request['current_head']);run_id=ticket['run_id'];validate_closed_head(head,ticket,run_id)
    terminal=audit.json(ticket['predecessor_terminal_receipt'])
    if terminal['canonical_snapshot']!=head['actions'] or terminal['result']['sha256']!=request['current_head']['sha256'] or terminal['exit_code']!=0:raise ValueError('Current head requires actual terminal release/cost receipt')
    contract=dict(format=FORMAT,logical_panel_id=request['logical_panel_id'],proposal=request['proposal'],canonical_bindings=request['canonical_bindings'],
        demos=request['demos'],interface=request['interface'],joint_conventions=request['joint_conventions'],fit_results=request['fit_results'],runtime_blob_sha256=request['runtime_blob_sha256'],source_commit=source)
    saved=None;forks={};state=dict(format=FORMAT,contract=contract,status='running',next_cell=0,records=[],pending=None,control_active=False,
        actions=deepcopy(head['actions']),action_ledger=deepcopy(head['action_ledger']),physical_resets={},closed_controls=[],post_rng_refs=[],
        native_reset_audit_refs=[],native_chunk_audit_refs=[],optional_audit_unavailable=[],
        readonly=True,diagnostic_purpose=DIAGNOSTIC,source_eligible=False,fit_updates=0,new_source_actions=0,locked32_or_population_claim=False)
    origin=request.get('resume') or request['origin_recovery']
    if any(terminal['full_recovery'][k]!=origin[k] for k in ('path','sha256')):
        raise ValueError('Recovery must be bound by the CURRENT terminal release receipt')
    path=audit.bound(origin['path'],origin['sha256']);saved=torch.load(path,map_location='cpu',weights_only=False)
    if request.get('resume'):
        validate_cursor(saved['state'],specs,contract)
        if (saved['state']['actions']!=head['actions'] or saved['control']['active']
                or saved.get('last_closed_control')!=(saved['state']['closed_controls'][-1] if saved['state']['records'] else saved['state']['previous_control'])):
            raise ValueError('Resume only CURRENT closed eval head/private state')
        state=deepcopy(saved['state']);state['status']='running';forks=deepcopy(saved['case_rngs'])
        validate_case_forks(forks,specs,state['next_cell'])
    elif saved['state']!=head or saved.get('control',{}).get('source_cursor',{}).get('closed') is not True:
        raise ValueError('Initial readout fork requires exact current closed physical recovery')
    if set(saved['global_rng'])!={'python','numpy','torch','cuda'} or saved['global_rng']['cuda'] is None:raise ValueError('All four existing global RNG domains required')
    state.setdefault('previous_control',deepcopy(saved.get('last_closed_control') or saved['control']))
    prepare_output(a.output);env=None;control=None
    def persist(boundary=False):
        state['control_active']=False if control is None else control.active
        last=state['closed_controls'][-1] if state['closed_controls'] else state['previous_control']
        publish_recovery(a.output,state,dict(state=deepcopy(state),global_rng=rng_state(),case_rngs=deepcopy(forks),
            control=None if control is None else control.snapshot(),last_closed_control=deepcopy(last),optimizer=None),boundary=boundary)
    def guard():
        if time.time()+ticket['cleanup_seconds']>=ticket['driver_deadline_unix']:raise TimeoutError('Actual owner deadline; preserve partial/no automatic replay')
        if sum(f.stat().st_size for f in Path(ticket['aggregate_output_root']).rglob('*') if f.is_file())>ticket['maximum_output_bytes']:raise RuntimeError('Actual aggregate output cap')
    try:
        ledger=head['action_ledger']
        with CampaignBudget(ledger['root'],campaign_id=ledger['campaign_id'],total_actions=ledger['total_actions'],source_caps=ledger['source_caps'],phase_caps=ledger['phase_caps'],contract=ledger['contract'],run_id=run_id,source_commit=source) as budget:
            try:
                validate_capacity_locked(budget,head)
                if budget.remaining(phase='a1',method='command_readout_native')<400*(15-state['next_cell']):raise ValueError('Reserve every remaining complete native episode before reset')
                guard();env=RobotwinNoiseEnv(model_path=a.model_path,norm_stats=a.norm_stats,robotwin_root=a.robotwin_root,rlinf_root=a.rlinf_root,output_dir=a.output/'runtime',action_horizon=32,num_steps=5,task_episode_limits=limits)
                results={k:audit.json(v) for k,v in request['fit_results'].items()};heads,bindings=validate_endpoints(results,interface,request['interface']['sha256'],device=env.backend.model.device)
                if 'frozen_endpoints' in state and state['frozen_endpoints']!=bindings:raise ValueError('Resume endpoints/provenance changed')
                state['frozen_endpoints']=bindings
                for binding in bindings.values():
                    for identity,resolved in canonical['resolved'].items():
                        outcome=binding['fit_provenance']['outcome_evidence'][identity]
                        if outcome['episode_manifest_ref']!=resolved['episode_manifest_ref'] or outcome['success']!=resolved['original_task_success']:
                            raise ValueError('Readout canonical cases differ from fit diagnostic provenance')
                writer=RobotwinCorpusWriter(a.output/'corpus',dict(role=FORMAT,source_commit=source,contract=contract,readonly=True))
                control=ReadoutControl(env,heads,writer,interface,bindings);control.assert_frozen()
                if env.frozen_fingerprint()!=FROZEN:raise ValueError('Frozen motor differs')
                restore_rng(saved['global_rng']);state['previous_recovery']=origin;state['actions']=budget.snapshot();persist(True)
                def episode(spec,*,qualify):
                    guard();key=spec['record_id']
                    def optional_audit(kind,publish):
                        try:state['native_'+kind+'_audit_refs'].append(publish(a.output/'native-audits'))
                        except (OSError,ValueError,TypeError) as error:
                            state['optional_audit_unavailable'].append(dict(cell=spec['id'],kind=kind,reason=type(error).__name__))
                    def before(current,actual):
                        physical=initial_physical_identity(current.reset_physical_audit)
                        if key in state['physical_resets'] and state['physical_resets'][key]!=physical:raise ValueError('Matched initial actor/measured/drive identity differs')
                        state['physical_resets'][key]=physical
                        optional_audit('reset',current.publish_reset_audit)
                        if qualify:state['qualification']=qualify_first_native(current,actual,a.output,interface)
                        current.assert_frozen()
                    def chunk(_):
                        guard();state['actions']=budget.snapshot()
                        optional_audit('chunk',control.publish_chunk_audit)
                        if sum(r['primitive_steps'] for r in state['records'])+control.elapsed_steps>6000:raise RuntimeError('Panel controller-action ceiling')
                        persist(False)
                    result=run_complete_episode(control,spec,budget,run_id=run_id,phase='a1',before_step=before,on_chunk=chunk)
                    if qualify:
                        state['qualification'].update(status='passed',actual_actions=result['primitive_steps'],complete_real_episode=spec['id'])
                        _atomic_json(a.output/'gpu-qualification.json',state['qualification'],replace=True)
                    state['closed_controls'].append(result['control_recovery']);result={k:v for k,v in result.items() if k!='control_recovery'}
                    post=a.output/(spec['id']+'-post-rng.pt');sha=atomic_save(post,dict(global_rng=rng_state(),control=control.snapshot()))
                    state['post_rng_refs'].append(dict(path=str(post),sha256=sha));state['actions']=budget.snapshot();return result
                run_cells(state,specs,forks,episode,admit=lambda:time.time()+ticket['minimum_episode_admission_seconds']+ticket['cleanup_seconds']<ticket['driver_deadline_unix'],persist=persist)
                control.assert_frozen();state['frozen_after']=env.frozen_fingerprint()
                if state['frozen_after']!=FROZEN:raise RuntimeError('Readonly motor changed')
            finally:state['actions']=budget.snapshot()
    except BaseException as error:
        state.update(status='blocked_failure',failure=dict(type=type(error).__name__,message=str(error)));raise
    finally:
        try:
            persist(False)
        finally:
            try:
                if env is not None:env.close()
            except BaseException as error:
                state.update(status='blocked_failure',cleanup_error=dict(type=type(error).__name__,message=str(error)));raise
            finally:_atomic_json(a.output/'result.json',state,replace=True)


if __name__=='__main__':main()
