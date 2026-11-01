"""Fixed ALL40 existing-TRAIN actor assessment; no learner updates or admission.

Uses qualified source SAC, OpenPI, simulator, rollout, capture and action journal.
Only the preselected final actors are available in this frozen asset inventory.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import socket
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from .vla_history import file_sha256
from .vla_libero import write_json
from .vla_source40_schedule import content_hash

FORMAT='libero40_existing_train_teacher_assessment_v1'
INVENTORY_SHA='6fc0c22a70b5f30d7dd59738dbfae1ced8fb81c6a6e28de36d0b0fb158a3a4a6'
TABLE_SHA='49ebaaf9227d7ca6315ee8f4eff7c5aebfff880551b584b7618d3dd7fe6e4894'
ASSET_SHA='3c32f791a555c3d30639ecd62f576351c8282f5e4921a451cc098b8ef5ec36f0'
GRANT_SHA='5e2384e028a023fbac2f497b1664ddfb1bd2ed4a5376e964aff6c45fea1d3436'
PROTOCOL_SHA='900eab7abdd4b9287c10148b756eb7b3ef53864d1a9fcaeae97e2273050f1fc2'
AUTHORITY_SHA='27ed35a2a1546355da2fbee0773493902f378d673b94c191232445304d1a4e0c'
MOTOR_RECEIPT='ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
MOTOR_PARAMS='c565dd3f0c4005be728cf7597698cba937c4011cd5bd813700a89b789f9d69e9'
SUITES=('libero_spatial','libero_object','libero_goal','libero_10')
ROLES=('raw_native_full_H_Gaussian','exact_source_tied_Gaussian_warmup',
       'preselected_final_TRAIN_SAC_mean','preselected_final_TRAIN_SAC_sample')
OMITTED=('original_initialized_SAC_mean_if_retained','original_initialized_SAC_sample_if_retained')
NOISE='32D tied across native H10; warmup unbounded N(0,I), then SAC clipped[-1,1]; no rescaling'
MAX_ACTIONS=196800
SCHEDULE_ACTIONS=131200
MAX_BYTES=32<<30
RUNTIME_FILES=('vla_source40_teacher_assessment.py','vla_source40_evaluate.py','vla_libero.py',
    'vla_source40_sac_diagnostic.py','vla_source40_subset_evaluate.py','vla_source40_memory_replication.py',
    'vla_source40_corpus.py','vla_source40_multitask_ppo.py','vla_source40_schedule.py',
    'vla_history.py','vla_features.py','vla_motor_replay.py','vla_source40_ppo_worker.py')


def read_ref(ref):
    path=Path(ref['path'])
    if path.is_symlink() or file_sha256(path)!=ref['sha256']:raise ValueError('Immutable assessment metadata checksum differs')
    return json.loads(path.read_text())


def worker_contract(source,run):
    if not re.fullmatch('[0-9a-f]{40}',source) or not re.fullmatch('[A-Za-z0-9-]+',run):
        raise ValueError('Exact source pin and independent assessment run identity required')
    return dict(format=FORMAT,source_commit=source,run_id=run,maximum_worker_seconds=10800,
        grant_sha256=GRANT_SHA,authority_sha256=AUTHORITY_SHA,protocol_sha256=PROTOCOL_SHA,
        asset_binding_sha256=ASSET_SHA,actor_table_sha256=TABLE_SHA,inventory_sha256=INVENTORY_SHA,
        roles=list(ROLES),omitted_roles=list(OMITTED),states=[40,41],scene_seed=123,
        planned_episodes=320,schedule_action_bound=SCHEDULE_ACTIONS,original_grant_actions=MAX_ACTIONS,
        new_output_bytes=MAX_BYTES,optimizer_updates=0,source_pretraining_admitted=False)


def validate_admission(ticket,clock,probe,*,source,output,hostname,now):
    """Pure check of the execution owner's existing begin/finish clock and proof."""
    expected=worker_contract(source,ticket['run_id'])
    rows=[r for r in clock['attempts'] if r['path']==ticket['resource_attempt']]
    charges=[r.get('charged_seconds',r['reserved_seconds']) for r in clock['attempts']]
    allocations=ticket['allocation']
    refs=ticket['refs']
    expected_refs={'grant':GRANT_SHA,'authority':AUTHORITY_SHA,'protocol':PROTOCOL_SHA,
        'asset_binding':ASSET_SHA,'actor_table':TABLE_SHA,'inventory':INVENTORY_SHA}
    if (ticket.get('format')!=FORMAT or ticket.get('admitted') is not True or ticket['source_commit']!=source
            or ticket['hostname']!=hostname or not ticket.get('job_id') or ticket['clock_contract']!=expected
            or Path(ticket['output']).resolve()!=Path(output).resolve()
            or clock.get('worker_seconds')!=10800 or clock.get('contract')!=expected
            or len(rows)!=1 or rows[0]['status']!='started'
            or [r for r in clock['attempts'] if r['status']!='finished']!=rows
            or not all(type(v) is int and v>=0 for v in charges) or sum(charges)>10800
            or not all(math.isfinite(float(v)) for v in (now,ticket['driver_deadline_unix'],rows[0]['started_unix']))
            or not rows[0]['started_unix']<=now<ticket['driver_deadline_unix']<=rows[0]['started_unix']+rows[0]['slice_seconds']-120
            or rows[0]['reserved_seconds']!=rows[0]['slice_seconds']+60
            or not 600<=rows[0]['slice_seconds']<=10740
            or allocations!={'GPU':1,'CPU':16,'memory_GiB':96,'shared_memory_GiB':8,'automatic_retry':False,'exclusive':True}
            or {k:v['sha256'] for k,v in refs.items()}!=expected_refs
            or probe.get('status')!='passed' or probe.get('hostname')!=hostname or len(probe.get('devices',[]))!=1
            or probe['devices'][0]['uuid']!=ticket['gpu_uuid']):
        raise ValueError('Actual standalone assessment allocation, immutable inputs and inclusive clock required')
    return expected


def validate_assets(table,metadata):
    if (table.get('format')!='all40_existing_terminal_train_actor_metadata_table_v1'
            or table.get('inventory_sha256')!=INVENTORY_SHA or table.get('all40_preserved') is not True
            or table.get('initial_actor_candidates_proven')!=0 or len(table['actors'])!=40
            or metadata.get('unresolved_competing_leaves')!=[] or len(metadata['all40'])!=40
            or metadata.get('new_assessment_outcomes_seen') is not False):
        raise ValueError('Exact frozen all40/no-initial prospective inventory required')
    roots=set();histories=set()
    for ordinal,(row,original) in enumerate(zip(table['actors'],metadata['all40'])):
        key=f'{SUITES[ordinal%4]}:{ordinal//4:02d}';leaf=original['canonical_terminal_train'];fixed=leaf['fixed_contract']
        if (row['source_ordinal']!=ordinal or original['source_ordinal']!=ordinal or row['task_key']!=key
                or original['task_key']!=key or row['suite']!=SUITES[ordinal%4] or row['task_id']!=ordinal//4
                or row['retained_original_initial_actor'] is not None or leaf['pre_SGD_candidates']!=[]
                or row['terminal_train_root']!=leaf['root'] or row['index']!=leaf['index'] or row['contract']!=leaf['contract']
                or row['checkpoint']!=dict(path=leaf['final_publication_checkpoint']['path'],
                    published_sha256=leaf['final_publication_checkpoint']['published_sha256'],
                    bytes=leaf['final_publication_checkpoint']['bytes'],payload_not_rehashed_or_loaded=True)
                or row['recorded_updates']!=leaf['source_updates'] or row['episodes']!=leaf['episodes']
                or row['chunks']!=leaf['chunks'] or row['latest_complete_case']!=leaf['final_publication']['through_case']
                or not row['BDDL']['matches'] or row['BDDL']['actual_sha256']!=row['BDDL']['published_sha256']
                or row['BDDL']!=original['BDDL'] or original['canonical_equals_production_plan'] is not True
                or leaf['terminal_flags']['completed'] is not True or not leaf['final_publication_checkpoint']['exists']
                or leaf['final_publication']['episodes']!=row['episodes'] or leaf['final_publication']['transitions']!=row['chunks']
                or fixed['motor']['receipt_sha256']!=MOTOR_RECEIPT or (fixed['horizon'],fixed['chunk'])!=(400,10)
                or fixed['noise_contract']!=NOISE or fixed['feature_contract']['observation_dim']!=4104
                or fixed['source_recipe']['warmup_chunks']!=500 or row['recorded_updates']<1
                or row['terminal_train_root'] in roots or fixed['history_id'] in histories):
            raise ValueError('Actor/ordinal/canonical TRAIN/input/noise/public-feature binding differs')
        roots.add(row['terminal_train_root']);histories.add(fixed['history_id'])
    # Each current leaf charge already includes its ancestors. Never add them again.
    return dict(metadata_cumulative_TRAIN_actions=sum(r['canonical_terminal_train']['actions'] for r in metadata['all40']),
        source_cost_groups=[dict(history_id=r['canonical_terminal_train']['fixed_contract']['history_id'],
            current_root=r['canonical_terminal_train']['root'],cumulative_actions=r['canonical_terminal_train']['actions'],
            ancestors_not_added=r['ancestry']) for r in metadata['all40']],
        canonical_ledger_ref=metadata['canonical_ledger_ref'],prior_costs_charged_again=False,
        prior_GPU_cost='Not summed or inferred from historical caps',scope='Bound metadata, not a fresh raw-journal replay')


def panel(table,*,include_omitted=False):
    rows=[]
    for actor in table['actors']:
        for state in (40,41):
            for role in ('raw_native_full_H_Gaussian','exact_source_tied_Gaussian_warmup',OMITTED[0],ROLES[2],OMITTED[1],ROLES[3]):
                if role in OMITTED and not include_omitted:continue
                rows.append(dict(source_ordinal=actor['source_ordinal'],task_key=actor['task_key'],state=state,scene_seed=123,
                    role=role,seed=20261101+1000*actor['source_ordinal']+state,
                    status='OMITTED_UNPROVEN_INITIAL_IN_BOUNDED_INDEX_SCOPE' if role in OMITTED else 'PLANNED'))
    return rows


def _leaf(path):return dict(path=str(path),sha256=file_sha256(path))


def bind_payloads(table,metadata,*,deadline):
    """Allocated GPU-phase startup only, BEFORE any simulator/outcome; no inference."""
    from flax import serialization
    from .vla_libero import structure_digest
    result=[]
    for row,original in zip(table['actors'],metadata['all40']):
        if time.time()+120>=deadline:raise TimeoutError('Inclusive startup deadline; no assessment episode started')
        contract,index=read_ref(row['contract']),read_ref(row['index']);leaf=original['canonical_terminal_train']
        publication=max((p for p in index['publications'] if p.get('status')=='paired_ad_dpt'),key=lambda p:p['through_case'])
        checkpoint=Path(row['checkpoint']['path']);root=(Path(row['terminal_train_root'])/'experiment').resolve()
        if (checkpoint.is_symlink() or not checkpoint.resolve().is_relative_to(root)
                or checkpoint.stat().st_size!=row['checkpoint']['bytes']
                or file_sha256(checkpoint)!=row['checkpoint']['published_sha256']
                or contract['partition']!='train' or contract['source_only'] is not True or contract['source_data_from_targets'] is not False
                or contract['task']['task_key']!=row['task_key'] or contract['task']['bddl_sha256']!=row['BDDL']['actual_sha256']
                or contract['motor']['receipt_sha256']!=MOTOR_RECEIPT or contract['noise_contract']!=NOISE
                or content_hash(contract)!=index['source_contract_sha256'] or not index['completed']
                or publication!=leaf['final_publication'] or index['source_updates']!=row['recorded_updates']):
            raise ValueError('Actual unchanged terminal TRAIN publication/contract differs')
        with checkpoint.open('rb') as stream:saved=pickle.load(stream)
        if (saved['contract']!=contract or saved['boundary']!='complete physical episode'
                or saved['updates']!=row['recorded_updates'] or len(saved['replay'])!=row['chunks']
                or saved['progress']['case_index']!=row['latest_complete_case'] or saved['progress']['episode_in_case']!=4
                or len(saved['progress']['episode_receipts'])!=row['episodes']):
            raise ValueError('Actual final TRAIN full recovery/counters differ')
        agent=serialization.msgpack_restore(saved['agent'])
        result.append(dict(task_key=row['task_key'],checkpoint=dict(path=str(checkpoint),sha256=row['checkpoint']['published_sha256']),contract=row['contract'],
            updates=saved['updates'],replay_rows=len(saved['replay']),pure_actor_sha256=structure_digest(agent['actor']['params']),
            full_learning_state_sha256=structure_digest(agent),replay_sha256=hashlib.sha256(pickle.dumps(saved['replay'],protocol=5)).hexdigest(),
            source_rng_sha256=hashlib.sha256(pickle.dumps(dict(
                source_state={k:saved[k] for k in ('agent_rng','sampler','numpy_rng','python_rng')},
                source_motor_rng=saved['progress'].get('motor_rng')),protocol=5)).hexdigest(),
            original_source_motor_rng_present=saved['progress'].get('motor_rng') is not None))
        del saved,agent
    return dict(format=FORMAT,actors=result,initial_roles_omitted=160,planned_episodes=320,
        source_assessment_only=True,source_pretraining_admitted=False)


class Actors:
    def __init__(self,dsrl_root,table,binding):
        from .vla_libero import OfficialSAC
        self.learner=OfficialSAC(dsrl_root,0);self.table=table;self.binding=binding;self.active=None
    def activate(self,row):
        from flax import serialization
        from .vla_libero import structure_digest
        if self.active==row['task_key']:return
        proof=self.binding['actors'][row['source_ordinal']];contract=read_ref(row['contract'])
        self.learner.restore(proof['checkpoint']['path'],np.random.default_rng(0),contract)
        actual=structure_digest(serialization.to_state_dict(self.learner.agent._save_dict))
        if (actual!=proof['full_learning_state_sha256'] or self.learner.updates!=proof['updates']
                or self.fingerprint()['replay']!=proof['replay_sha256']):raise ValueError('Loaded full source state differs from fixed actor binding')
        self.active=row['task_key']
    def controller(self,role):
        from .vla_source40_sac_diagnostic import FrozenTeacher
        if role not in ROLES[2:]:return None
        controller=FrozenTeacher(self.learner,deterministic=role==ROLES[2],output=None)
        actual_seed=controller.set_episode_seed
        def seed(value):actual_seed(value);controller.assessment_rng_before=controller.snapshot()
        controller.set_episode_seed=seed
        return controller
    def fingerprint(self):
        from .vla_source40_subset_evaluate import FrozenSAC
        return dict(full=FrozenSAC(self.learner).frozen_signature(),updates=self.learner.updates,replay_rows=len(self.learner.replay),
            replay=hashlib.sha256(pickle.dumps(self.learner.replay,protocol=5)).hexdigest())


def action_charge(output):
    from .vla_source40_evaluate import CostLedger
    total,episodes,uncertain=0,0,0
    for path in Path(output).glob('episode-*/physical-journal'):
        costs=CostLedger(path,max_primitive_steps=MAX_ACTIONS,max_settling_steps=MAX_ACTIONS).costs()
        for value in costs.values():
            episodes+=value['reset_attempts']
            total+=sum(value[k] for k in ('primitive_steps','settling_steps','uncertain_primitive_steps_ceiling','uncertain_settling_steps_ceiling'))
            uncertain+=value['uncertain_primitive_steps_ceiling']+value['uncertain_settling_steps_ceiling']
    return dict(all_in_actions=total,physical_episodes=episodes,uncertain_actions=uncertain)


def check_pair(reference,initial,features,instruction):
    from .vla_source40_memory_replication import same_public
    keys=('initial_observation_sha256','initial_state_vector_sha256','initial_simulator_state_sha256')
    if (any(not isinstance(initial.get(k),str) or initial[k]!=reference['initial'][k] for k in keys)
            or instruction!=reference['instruction'] or not same_public(reference['initial']['observation'],initial['observation'])
            or not np.array_equal(reference['features'],features)):
        raise ValueError('Actual same-case reset/raw/public inputs differ BEFORE steering; preserve partial result')


def run_panel(table,motor,actors,output,*,server_factory,deadline,qualify,forks,provenance):
    from .vla_source40_corpus import restore_motor_rng,save_motor_rng
    from .vla_source40_evaluate import CostLedger,_save_recovery,rollout_episode
    from .vla_source40_memory_replication import CaptureMotor,raw_record
    from .vla_source40_sac_diagnostic import QuerySimulator,SourceCase
    output=Path(output);output.mkdir(parents=True,exist_ok=True);schedule=panel(table)
    recovery=output/'assessment-recovery.pkl'
    contract=content_hash(dict(table=table,format=FORMAT,roles=ROLES,states=[40,41],provenance=provenance))
    if recovery.exists():
        with recovery.open('rb') as stream:progress=pickle.load(stream)
        if progress['contract']!=contract or progress['active'] is not None:raise ValueError('Resume only exact closed episodes; no replay of an interrupted physical trial')
        if action_charge(output)!=progress['costs']:raise ValueError('Saved closed cursor/action journal differs')
        if (not 0<=progress['cursor']<=len(schedule) or len(progress['receipts'])!=progress['cursor']
                or any(file_sha256(r['path'])!=r['sha256'] for r in progress['receipts'])
                or any(file_sha256(r['path'])!=r['sha256'] for r in progress['references'].values())):
            raise ValueError('Closed evaluation receipt/cursor changed')
        forks.restore_global(progress['global_rng']);motor.policy._rng=restore_motor_rng(progress['motor_rng'])
    else:
        if any(output.glob('episode-*')) or any(output.glob('case-reference-*')):
            raise ValueError('Existing physical outputs without their recovery must remain preserved')
        progress=dict(contract=contract,cursor=0,active=None,receipts=[],references={},case_key=None,case_fork=None,
            global_rng=forks.capture_global(),motor_rng=save_motor_rng(motor.policy._rng),actor_rng=None,
            qualification_complete=False,costs=action_charge(output))
        _save_recovery(recovery,progress)
    write_results(output,table,progress['receipts'])
    server=None;task_key=None;stop=None
    try:
        for ordinal in range(progress['cursor'],len(schedule)):
            spec=schedule[ordinal];row=table['actors'][spec['source_ordinal']];costs=action_charge(output)
            used=sum(p.stat().st_size for p in output.parent.rglob('*') if p.is_file())
            if costs['uncertain_actions'] or costs['all_in_actions']+410>SCHEDULE_ACTIONS or costs['physical_episodes']>=320:
                stop='whole_episode_action_admission';break
            if time.time()+120>=deadline:stop='whole_episode_time_admission';break
            if used+(128<<20)>MAX_BYTES:stop='whole_episode_storage_admission';break
            if task_key!=row['task_key']:
                if server is not None:server.close()
                actors.activate(row)
                task=dict(task_key=row['task_key'],suite=row['suite'],task_id=row['task_id'],bddl_sha256=row['BDDL']['actual_sha256'])
                server=server_factory(task,ordinal);task_key=row['task_key']
            case_key=f"{row['task_key']}/state-{spec['state']}"
            if progress['case_key']!=case_key:
                progress['case_key']=case_key;progress['case_fork']=forks.new_case(spec['seed'])
            forks.restore_global(progress['case_fork']['global']);motor.policy._rng=restore_motor_rng(progress['case_fork']['motor'])
            progress['active']=ordinal;_save_recovery(recovery,progress)
            episode=output/f'episode-{ordinal:03d}';episode.mkdir(exist_ok=False);raw=episode/'raw-rpc';raw.mkdir()
            captured_motor=CaptureMotor(motor,raw);captured_sim=QuerySimulator(server,captured_motor,raw)
            controller=actors.controller(spec['role']);before=actors.fingerprint();before_global=forks.capture_global()
            def on_reset(observation):
                initial=captured_sim.initial_reset;feature=np.asarray(motor.feature_encoder(observation,server.metadata['instruction']))
                if feature.shape!=(4104,) or not np.isfinite(feature).all():raise ValueError('Exact public4104 feature ABI required')
                if case_key in progress['references']:
                    ref=progress['references'][case_key]
                    if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Initial case reference changed')
                    with gzip.open(ref['path'],'rb') as stream:reference=pickle.load(stream)
                    check_pair(reference,initial,feature,server.metadata['instruction'])
                else:
                    if spec['role']!=ROLES[0]:raise ValueError('No paired native reference may be manufactured by a later role')
                    path=output/f'case-reference-{spec["source_ordinal"]:02d}-{spec["state"]}.pkl.gz'
                    raw_record(path,dict(initial=initial,features=feature,instruction=server.metadata['instruction']))
                    progress['references'][case_key]=_leaf(path)
                if not progress['qualification_complete']:
                    g=forks.capture_global();m=save_motor_rng(motor.policy._rng)
                    qualify(observation,server.metadata['instruction'],episode)
                    forks.restore_global(g);motor.policy._rng=restore_motor_rng(m)
            ledger=CostLedger(episode/'physical-journal',max_primitive_steps=400,max_settling_steps=10,max_total_actions=410)
            method='native' if spec['role']==ROLES[0] else 'tied_gaussian' if spec['role']==ROLES[1] else 'frozen_source_sac'
            case=SourceCase('source40',case_key,row['suite'],row['task_id'],spec['state'],123)
            rows,receipt,noise_after=rollout_episode(captured_sim,captured_motor,controller,case=case,method=method,
                role='query',ordinal=0,episode_id=f'{case_key}/{spec["role"]}',ledger=ledger,
                reset_motor_rng=forks.seed_motor,qualify=on_reset,deadline=time.monotonic()+max(0.,deadline-time.time()),episode_seed=spec['seed'])
            if actors.fingerprint()!=before:raise RuntimeError('Frozen source learning state/replay/update counters changed')
            raw_record(episode/'trace.pkl.gz',dict(rows=rows,receipt=receipt,spec=spec,source_assessment_only=True,source_pretraining_admitted=False))
            raw_record(episode/'rng-context.pkl.gz',dict(case_fork=progress['case_fork'],global_before=before_global,
                global_after=forks.capture_global(),motor_after=save_motor_rng(motor.policy._rng),noise_sampler_after=noise_after,
                actor_before=None if controller is None else controller.assessment_rng_before,
                actor_after=None if controller is None else controller.snapshot(),source_learning_state=before))
            write_json(episode/'receipt.json',dict(receipt,spec=spec,source_actor=row['checkpoint'],source_updates=row['recorded_updates'],
                frozen_learning_state=before,source_assembly_admitted=False,optimizer_updates=0,trace=_leaf(episode/'trace.pkl.gz'),RNG=_leaf(episode/'rng-context.pkl.gz')))
            progress['receipts'].append(_leaf(episode/'receipt.json'))
            progress.update(cursor=ordinal+1,active=None,qualification_complete=True,costs=action_charge(output),
                global_rng=forks.capture_global(),motor_rng=save_motor_rng(motor.policy._rng),actor_rng=None if controller is None else controller.snapshot())
            _save_recovery(recovery,progress)
            write_json(output/'progress.json',dict(cursor=progress['cursor'],planned_episodes=320,costs=progress['costs'],recovery=_leaf(recovery)))
            write_results(output,table,progress['receipts'])
    finally:
        if server is not None:server.close()
    result=dict(format=FORMAT,completed=progress['cursor']==320,cursor=progress['cursor'],receipts=progress['receipts'],
        omitted=[r for r in panel(table,include_omitted=True) if r['status']!='PLANNED'],stop_reason=stop,costs=action_charge(output),
        recovery=_leaf(recovery),provenance=provenance,optimizer_updates=0,source_assessment_only=True,source_pretraining_admitted=False)
    write_json(output/'index.json',result);return result


def write_results(output,table,receipts):
    """All prespecified cells remain visible, including unrun and absent initials."""
    measured={}
    for ref in receipts:
        r=read_ref(ref);s=r['spec'];key=(s['task_key'],s['state'],s['role'])
        if key in measured or not r.get('completed') or r.get('optimizer_updates')!=0:
            raise ValueError('Duplicate/incomplete/updated assessment result')
        measured[key]=r
    rows=[]
    for spec in panel(table,include_omitted=True):
        r=measured.get((spec['task_key'],spec['state'],spec['role']))
        rows.append(dict(spec,status='COMPLETE' if r else spec['status'],
            success=None if r is None else r['success'],
            primitive_steps=None if r is None else r['primitive_steps'],
            settling_steps=None if r is None else r['settling_steps'],
            all_in_actions=None if r is None else r['primitive_steps']+r['settling_steps'],
            elapsed_seconds=None if r is None else r['elapsed_seconds']))
    path=Path(output)/'case-results.csv';temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temporary.replace(path)
    summary={role:dict(planned=80,complete=sum(r['role']==role and r['status']=='COMPLETE' for r in rows),
        successes=sum(r['role']==role and r['status']=='COMPLETE' and bool(r['success']) for r in rows)) for role in ROLES}
    write_json(Path(output)/'results-summary.json',dict(by_role=summary,omitted_initial_slots=160,
        all40_retained=True,source_pretraining_admitted=False,interpretation='Two fixed cases per goal; descriptive assessment, no competence certificate or measured initial-to-final curve'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('output','ticket','gpu-probe','actor-table','metadata-inventory','openpi-root','checkpoint','dsrl-root','simulator-python'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('ticket-sha256','expected-source','simulator-pythonpath'):p.add_argument('--'+name,required=True)
    p.add_argument('--port',type=int,default=8783);a=p.parse_args(argv);source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.expected_source or subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],text=True).strip():raise ValueError('Clean exact execution pin required')
    ticket=read_ref(dict(path=str(a.ticket),sha256=a.ticket_sha256));clock=json.loads((Path(ticket['clock_root'])/'worker-time.json').read_text())
    if (a.gpu_probe.resolve()!=Path(ticket['gpu_probe']['path']).resolve()
            or file_sha256(a.gpu_probe)!=ticket['gpu_probe']['sha256']):
        raise ValueError('Actual own GPU proof file differs from admission')
    probe=json.loads(a.gpu_probe.read_text())
    contract=validate_admission(ticket,clock,probe,source=source,output=a.output,hostname=socket.gethostname(),now=time.time())
    for ref in ticket['refs'].values():read_ref(ref)
    runtime={name:file_sha256(Path(__file__).parent/name) for name in RUNTIME_FILES}
    if runtime!=ticket['runtime_files']:raise ValueError('Actual imported runtime file hashes differ')
    table=read_ref(dict(path=str(a.actor_table),sha256=TABLE_SHA));metadata=read_ref(dict(path=str(a.metadata_inventory),sha256=INVENTORY_SHA))
    provenance=validate_assets(table,metadata);a.output.mkdir(parents=True,exist_ok=True);deadline=ticket['driver_deadline_unix']
    if a.output.is_symlink() or not Path(ticket['clock_root']).resolve().is_relative_to(a.output.parent.resolve()):
        raise ValueError('Independent assessment clock and outputs must share the declared phase directory')
    used=sum(path.stat().st_size for path in a.output.parent.rglob('*') if path.is_file())
    if os.statvfs(a.output).f_bavail*os.statvfs(a.output).f_frsize<max(0,MAX_BYTES-used):
        raise ValueError('Insufficient capacity for the bounded phase output')
    binding=bind_payloads(table,metadata,deadline=deadline);binding_path=a.output/'loaded-actor-bindings.json'
    if binding_path.exists():
        if json.loads(binding_path.read_text())!=binding:raise ValueError('Frozen actor availability/identity changed on assessment resume')
    else:write_json(binding_path,binding)
    write_json(a.output/'source-provenance.json',provenance)
    from .vla_libero import FrozenOpenPI
    from .vla_source40_corpus import restore_motor_rng,save_motor_rng
    from .vla_source40_memory_replication import global_rng,restore_global_rng
    from .vla_source40_multitask_ppo import OwnedSimulator
    from .vla_motor_replay import parameter_signature
    import jax
    import torch
    a.model_config='pi05_libero';a.feature_interface='vla-prefix';motor=FrozenOpenPI(a)
    before=parameter_signature(motor.policy._model)
    if motor.metadata['receipt_sha256']!=MOTOR_RECEIPT or before['sha256']!=MOTOR_PARAMS:raise ValueError('Exact qualified frozen OpenPI40 motor required')
    actors=Actors(a.dsrl_root,table,binding)
    def seed_motor(seed):motor.policy._rng=jax.random.key(seed)
    def new_case(seed):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        seed_motor(seed);return {'global':global_rng(),'motor':save_motor_rng(motor.policy._rng),'assessment_seed':seed}
    def qualify(observation,instruction,episode):
        zeros=np.zeros((10,32),np.float32);first=motor.infer(observation,instruction,zeros);second=motor.infer(observation,instruction,zeros.copy())
        delta=float(np.max(np.abs(first-second)))
        if not np.isfinite(delta) or delta>1e-5:raise RuntimeError('Native same-noise qualification failed')
        write_json(episode/'qualification.json',dict(passed=True,same_noise_delta=delta,
            same_noise_VLA_calls=2,feature=motor.feature_encoder.qualify(observation,instruction),
            extra_physical_episodes=0,extra_physical_actions=0))
    forks=SimpleNamespace(new_case=new_case,capture_global=global_rng,restore_global=restore_global_rng,seed_motor=seed_motor)
    try:
        return run_panel(table,motor,actors,a.output,server_factory=lambda task,n:OwnedSimulator(task,python=a.simulator_python,
            pythonpath=a.simulator_pythonpath,output=a.output/f'server-{n}-{time.time_ns()}',port=a.port),deadline=deadline,qualify=qualify,forks=forks,
            provenance=dict(worker_contract=contract,runtime_files=runtime,loaded_actors=_leaf(binding_path)))
    finally:
        after=parameter_signature(motor.policy._model);write_json(a.output/'motor-final.json',dict(before=before,after=after,unchanged=before==after))
        if before!=after:raise RuntimeError('Frozen motor parameters changed')


if __name__=='__main__':main()
