"""Launch artifact: supervise the qualified driver's durable handoffs only.

No learner or simulator implementation is duplicated here. All Torch reads
below are CPU metadata/hash validation inside the same allocated job.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback

LINEAGE='db0184089fa88a5f96c153f9b80029bb91b77def'
PARENT_SHA='1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0'
CAPS={'ppo':1226400,'sac':1194400,'ppo_reset':1168800}
CALIBRATION_SHA='821995144fe49d5503e07ac05b1932ba729ce19d64e101740e06b0fa11ac3f61'
ROOT=Path('/mowenjie/fluffy-bassoon-2')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def atom(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def bind(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))


def read_bound(ref):
    path=Path(ref['path']).resolve()
    if sha(path)!=ref['sha256']:raise ValueError(f'Bound artifact changed: {path}')
    return json.loads(path.read_text())


def ledger_view(root,contract,worker):
    """Read without registering a run, recovering intents or editing a header."""
    root=Path(root)
    with (root/'campaign.lock').open('r') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        value=json.loads((root/'snapshot.json').read_text())
        header=value['header']
        if (header['campaign_id']!=worker['campaign_id'] or canonical(header['contract'])!=canonical(contract)
                or header['total_actions']!=worker['maximum_actions']
                or header['source_caps']!={worker['method']:1100000}
                or value.get('pending_actions')!=0 or value.get('uncertain_actions')!=0
                or value.get('unresolved_decisions')):
            raise ValueError('Existing ledger identity, cap, contract or action-intent state differs')
        count=value['journal_events'];tail=root/'events'/f'{count-1:010d}.json'
        event=json.loads(tail.read_text());digest=event.pop('sha256')
        if (digest!=value['journal_sha256'] or hashlib.sha256(canonical(event).encode()).hexdigest()!=digest
                or (root/'events'/f'{count:010d}.json').exists()):
            raise ValueError('Live ledger snapshot is stale or its journal head changed')
        return value


def check_handoff(ref,args,worker,*,expected_root=None):
    """Use the sole implementation owner's hash/prefix validation."""
    import torch
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation
    manifest=read_bound(ref);checkpoint_ref=manifest['checkpoint'];checkpoint_path=Path(checkpoint_ref['path']).resolve()
    if sha(checkpoint_path)!=checkpoint_ref['sha256']:raise ValueError('Resume checkpoint SHA differs')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    if expected_root is not None and checkpoint_path!=Path(expected_root)/'coordinator-recovery.pt':
        raise ValueError('Child continuation refers to another output checkpoint')
    continuation=load_trial_continuation(ref['path'],ref['sha256'],checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,execution_commit=args.source_commit,lineage_commit=LINEAGE)
    if manifest['method']!=args.method:raise ValueError('Another method cannot provide this continuation')
    contract=checkpoint['contract'];state=checkpoint['states'][args.method]
    view=ledger_view(worker['ledger_root'],contract,worker)
    if view['journal_sha256']!=continuation.receipt['ledger_journal_sha256']:
        raise ValueError('Physical ledger advanced after its bound handoff')
    return manifest,checkpoint,state,view


def ancestors(ref):
    """Follow hash-bound execution migrations, retaining every source corpus."""
    roots=[];seen=set()
    while True:
        identity=(ref['path'],ref['sha256'])
        if identity in seen:raise ValueError('Continuation ancestry cycle')
        seen.add(identity);manifest=read_bound(ref);root=Path(manifest['corpus_root']).resolve()
        if str(root) not in roots:roots.append(str(root))
        migration_path=root.parent/'execution-migration.json'
        if not migration_path.is_file():break
        migration=json.loads(migration_path.read_text())
        if (migration['lineage_source_commit']!=LINEAGE
                or migration['manifest_sha256']!=read_bound(manifest['corpus_contract']).get('resume_trial_manifest_sha256')):
            raise ValueError('Corpus execution migration lost its original handoff binding')
        ref=dict(path=migration['manifest_path'],sha256=migration['manifest_sha256'])
    return list(reversed(roots))


def inventory_roots(roots,old=None):
    """Index reset/case provenance only; never copy RGB/replay/model data."""
    value=old or dict(corpus_roots=[],occurrences=[],unique_cases={},source_episode_keys={})
    for root_arg in roots:
        if root_arg in value['corpus_roots']:continue
        root=Path(root_arg)
        for reset_path in sorted((root/'episodes').glob('*/reset.json')):
            reset=json.loads(reset_path.read_text());context=reset.get('context',{});info=reset.get('reset_info',{})
            manifest_path=reset_path.parent/'manifest.json'
            episode=json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
            case=dict(task=info.get('task'),requested_seed=info.get('requested_seed'),actual_seed=info.get('actual_seed'))
            case_id=hashlib.sha256(canonical(case).encode()).hexdigest()
            occurrence=dict(case_id=case_id,**case,corpus_root=root_arg,reset=bind(reset_path),
                phase=context.get('phase'),method=context.get('method'),trial_id=context.get('trial_id'),
                initial_observation_sha256=reset.get('initial_observation',{}).get('sha256'),
                completed_chunks=episode.get('completed_chunks',0),
                executed_primitive_steps=episode.get('executed_primitive_steps',0),
                provenance_only=True)
            if episode.get('pending_decision'):raise ValueError('A composed corpus contains an unresolved decision')
            if manifest_path.is_file():occurrence['manifest']=bind(manifest_path)
            if occurrence['phase']=='source' and occurrence['completed_chunks']:
                key=canonical([occurrence['method'],occurrence['trial_id'],case['task'],case['requested_seed']])
                if key in value['source_episode_keys']:raise ValueError('Source episode appears to have executed twice across slices')
                value['source_episode_keys'][key]=occurrence['reset']
            value['occurrences'].append(occurrence)
            value['unique_cases'].setdefault(case_id,case)
        value['corpus_roots'].append(root_arg)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-commit','run-id','method','resume-trial-manifest-sha256'):parser.add_argument('--'+name,required=True)
    for name in ('parent-manifest','resume-trial-manifest','code-root','output','robotwin-root','rlinf-root','model-path','norm-stats'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,required=True)
    args=parser.parse_args();started=time.monotonic();deadline=started+args.wall_seconds
    out=args.output.resolve();code=args.code_root.resolve();host=socket.gethostname()
    if sys.platform!='linux' or not host.startswith('job-') or args.method not in CAPS:
        raise ValueError('Require the actual allocated Linux source worker')
    if not re.fullmatch(r'[0-9a-f]{40}',args.source_commit):raise ValueError('Require exact executor pin')
    parent_ref=bind(args.parent_manifest)
    if parent_ref['sha256']!=PARENT_SHA:raise ValueError('Original immutable parent SHA differs')
    parent=json.loads(args.parent_manifest.read_text());worker=parent['workers'][args.method]
    if (parent['source_commit']!=LINEAGE or worker['maximum_actions']!=CAPS[args.method]
            or worker['source_safety_actions']!=1100000):raise ValueError('Parent lineage or child reservation changed')
    horizon=args.robotwin_root/'task_config/_eval_step_limit.yml'
    if sha(horizon)!=parent['benchmark_horizon_config_sha256']:raise ValueError('Native benchmark horizon file changed')
    current_ref=dict(path=str(args.resume_trial_manifest.resolve()),sha256=args.resume_trial_manifest_sha256)
    initial,checkpoint,state,view=check_handoff(current_ref,args,worker)
    initial_contract=checkpoint['contract'];initial_header=view['header']
    calibration=ROOT/'outputs/robotwin-staged-20261030-r2/optimizer-calibration'
    if sha(calibration/'result.json')!=CALIBRATION_SHA:raise ValueError('Qualified scalar LR receipt changed')
    qualified=json.loads((calibration/'admission.json').read_text())
    if (not qualified['qualified'] or qualified['learning_rate']!=1e-6
            or qualified['calibration_sha256']!=CALIBRATION_SHA):raise ValueError('Expected qualified source PPO learning rate')
    target=out/'optimizer-calibration';target.mkdir()
    for name in ('result.json','admission.json'):shutil.copy2(calibration/name,target/name)
    admission=dict(format='robotwin_execution_continuation_admission_v1',parent=parent_ref,
        lineage_source_commit=LINEAGE,execution_source_commit=args.source_commit,method=args.method,
        initial_manifest=current_ref,initial_checkpoint=initial['checkpoint'],
        scientific_contract_sha256=hashlib.sha256(canonical(initial_contract).encode()).hexdigest(),
        original_ledger_header_sha256=hashlib.sha256(canonical(initial_header).encode()).hexdigest(),
        existing_ledger_root=worker['ledger_root'],campaign_id=worker['campaign_id'],
        unchanged_action_cap=worker['maximum_actions'],unchanged_source_safety_cap=1100000,
        calibration_result_sha256=CALIBRATION_SHA,trained_weights_restored=True,
        diagnostic_weights_transferred=False,no_prior_receipt_imports=True,
        migration='original parent/DB scientific contract to explicitly identified new executor only')
    atom(out/'parent-admission.json',admission)
    inventory=inventory_roots(ancestors(current_ref));atom(out/'case-provenance.json',inventory)
    atom(out/'corpus-roots.json',dict(corpus_roots=inventory['corpus_roots'],selection='lineage provenance only; no new data copied'))
    probe_command=[sys.executable,str(code/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--output',str(out/'gpu-probe.json'),'--purpose','training','--baige-container']
    subprocess.run(probe_command,cwd=code,check=True,timeout=90)
    os.environ['LYCO_BAIGE_GPU_MAP']=str(out/'gpu-probe.json')
    children=[];status='allocation_wall_budget_stop';failure=None;cleanup_faults=0
    try:
        for number in range(1,1000):
            remaining=deadline-time.monotonic()
            if remaining<4200:break
            directory=out/'slices'/f'{number:04d}';directory.mkdir(parents=True)
            experiment=directory/'experiment'
            current=read_bound(current_ref);resume=current['checkpoint']['path']
            command=[sys.executable,str(code/'scripts/diagnostics/robotwin_noise_compare.py'),
                '--source-commit',args.source_commit,'--lineage-source-commit',LINEAGE,
                '--model-path',str(args.model_path),'--norm-stats',str(args.norm_stats),
                '--robotwin-root',str(args.robotwin_root),'--rlinf-root',str(args.rlinf_root),
                '--output',str(experiment),'--seed','20261030','--source-steps','1100000',
                '--source-iterations','100','--source-eval-iterations','10','25','50','100',
                '--benchmark-episode-limits','--ppo-trials-per-update','6','--ppo-epochs','5',
                '--ppo-minibatch-trials','16','--learning-rate','1e-6','--ppo-max-update-kl','.1',
                '--train-episodes','4','--eval-episodes','8','--methods',args.method,
                '--native-horizon','32','--episode-limit','600','--eval-tasks','adjust_bottle','move_stapler_pad',
                '--source-validation-episodes','4','--native-episodes',str(worker['native_episodes']),
                '--task-config-name','demo_clean','--sac-warmup','500','--sac-batch-size','256','--sac-utd','20',
                '--campaign-root',worker['ledger_root'],'--campaign-id',worker['campaign_id'],
                '--campaign-max-actions',str(worker['maximum_actions']),'--record-corpus','--final-observation-only',
                '--capture-qualification',str(ROOT/'outputs/robotwin-capture-qual-20261029-r1/qualification/qualification.json'),
                '--wall-seconds','3600','--boundary-reserve-seconds','300',
                '--resume',resume,'--resume-trial-manifest',current_ref['path'],
                '--resume-trial-manifest-sha256',current_ref['sha256'],
                '--max-source-trials-per-process','6','--recycle-at-round-boundary']
            atom(directory/'launch.json',dict(command=command,input_handoff=current_ref,
                execution_source_commit=args.source_commit,lineage_source_commit=LINEAGE,
                calibration_result_sha256=CALIBRATION_SHA,started_at_utc=datetime.now(timezone.utc).isoformat()))
            child_started=time.monotonic()
            with (directory/'console.log').open('w') as stream:
                process=subprocess.Popen(command,cwd=code,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                timed_out=False
                try:raw_exit=process.wait(timeout=3900)
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                    try:raw_exit=process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);raw_exit=process.wait(timeout=10)
            normalized=raw_exit if raw_exit>=0 else 128-raw_exit
            row=dict(slice=number,output=str(experiment),raw_exit_code=raw_exit,exit_code=normalized,
                timed_out=timed_out,elapsed_seconds=time.monotonic()-child_started,input_handoff=current_ref)
            children.append(row);atom(directory/'exit.json',row)
            if timed_out:raise RuntimeError('Owned child exceeded its bounded lifetime; no automatic replay')
            summary=json.loads((experiment/'summary.json').read_text())
            if canonical(summary['contract'])!=canonical(initial_contract):raise ValueError('Child scientific lineage contract changed')
            frozen=json.loads((experiment/'frozen-motor.json').read_text())
            if not frozen.get('unchanged') or frozen.get('before')!=frozen.get('after'):
                raise ValueError('Frozen motor changed or receipt is incomplete')
            migration=json.loads((experiment/'execution-migration.json').read_text())
            expected_run=f'{directory.name}:experiment:'+hashlib.sha256(str(experiment.resolve()).encode()).hexdigest()[:16]
            if (migration['execution_run_id']!=expected_run or migration['execution_source_commit']!=args.source_commit
                    or migration['lineage_source_commit']!=LINEAGE):raise ValueError('Execution identity/pin migration receipt differs')
            if summary['status']=='complete':
                import torch
                final=torch.load(experiment/'coordinator-recovery.pt',map_location='cpu',weights_only=False)
                saved=final['states'][args.method];view=ledger_view(worker['ledger_root'],initial_contract,worker)
                if (normalized!=0 or final['contract']!=initial_contract
                        or saved['completed_source_iterations']!=100 or saved['completed_source_trials']!=600
                        or saved['schedule']['trials']!=600 or saved.get('pending_rollouts')
                        or (args.method!='sac' and saved['learner']['updates']!=100)
                        or any(summary['stages'].get(f'iteration:{n}',{}).get('status')!='complete' for n in (10,25,50,100))
                        or any(f'stage:iteration:{n}:complete' not in final['completed_units'] for n in (10,25,50,100))):
                    raise ValueError('Completion requires normal exit,100rounds,empty pending data and all final evaluations')
                row.update(status='complete',checkpoint=bind(experiment/'coordinator-recovery.pt'),source=summary['source'])
                inventory=inventory_roots([str(experiment/'corpus')],inventory)
                status='complete';break
            if summary['status']!='wall_budget_stop' or normalized not in (0,134,139):
                raise RuntimeError(f'Unexpected child status/exit: {summary["status"]}/{normalized}')
            next_ref=summary.get('continuation_manifest')
            if not next_ref:raise ValueError('A wallstop without a bound next manifest cannot continue')
            manifest,next_checkpoint,next_state,view=check_handoff(next_ref,args,worker,expected_root=experiment)
            if view['header']!=initial_header:raise ValueError('Original child ledger header changed')
            inventory=inventory_roots([manifest['corpus_root']],inventory)
            at_round=(next_state['completed_source_trials']%6==0 and not next_state.get('pending_rollouts'))
            row.update(status='verified_wall_budget_stop',next_handoff=next_ref,
                checkpoint=manifest['checkpoint'],round_boundary=at_round,source=summary['source'],
                native_cleanup_fault=normalized in (134,139),ledger_journal_sha256=view['journal_sha256'])
            if normalized in (134,139):
                if not at_round:raise ValueError('Native cleanup fault lacks a complete round/empty-pending handoff')
                cleanup_faults+=1
            before=(state['completed_source_trials'],state['completed_source_iterations'],len(checkpoint.get('completed_units',[])))
            after=(next_state['completed_source_trials'],next_state['completed_source_iterations'],len(next_checkpoint.get('completed_units',[])))
            current_ref=next_ref;checkpoint=next_checkpoint;state=next_state
            atom(out/'cursor.json',dict(next_manifest=current_ref,checkpoint=manifest['checkpoint'],source=summary['source']))
            atom(out/'case-provenance.json',inventory)
            atom(out/'corpus-roots.json',dict(corpus_roots=inventory['corpus_roots'],selection='lineage provenance only; no new data copied'))
            atom(out/'children.json',children)
            print(json.dumps(dict(stage='verified_handoff',slice=number,source=summary['source'],
                native_cleanup_fault=row['native_cleanup_fault'],next_manifest=current_ref)),flush=True)
            if not at_round or before==after:break
        else:raise RuntimeError('Unexpected supervisor slice count; halt rather than loop indefinitely')
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));status='halted_requires_audit';traceback.print_exc()
    finally:
        atom(out/'case-provenance.json',inventory)
        atom(out/'corpus-roots.json',dict(corpus_roots=inventory['corpus_roots'],selection='lineage provenance only; no new data copied'))
        atom(out/'children.json',children)
        atom(out/'supervisor-summary.json',dict(status=status,method=args.method,
            execution_source_commit=args.source_commit,lineage_source_commit=LINEAGE,
            original_parent=parent_ref,existing_campaign_id=worker['campaign_id'],
            existing_ledger_root=worker['ledger_root'],unchanged_action_cap=worker['maximum_actions'],
            elapsed_seconds=time.monotonic()-started,completed_child_processes=len(children),
            verified_native_cleanup_faults=cleanup_faults,failure=failure,
            next_manifest=None if status=='complete' else current_ref,
            corpus_roots=inventory['corpus_roots'],unique_cases=len(inventory['unique_cases']),
            reset_occurrences=len(inventory['occurrences']),no_source_data_copied=True,
            no_weights_reinitialized=True,no_prior_receipt_imports=True,
            completion_rule='normalcomplete100rounds plus all10/25/50/100 evaluations; source_budget_match alone is insufficient'))
    return 3 if failure else 0


if __name__=='__main__':raise SystemExit(main())
