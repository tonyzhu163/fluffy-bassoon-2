"""Metadata-only production macro clocks; child clocks are overlapping bounds.

Eight10500s execution ceilings plus a60s startup allowance each and1400s final
competence total85880s NEW GPU time, below24h. Historical qualifications stay
in their original pool. This module schedules no resources or model work.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from . import vla_source40_topup as source
from .vla_source40_ppo_worker import begin as clock_begin, finish as clock_finish


FORMAT='source40_sac_production_macro_budget_v1'
ACTIVE_SECONDS,ALLOWANCE,WORKER_SECONDS=10500,60,10560
CHILD_SECONDS,FINAL_SECONDS,NEW_SECONDS=2100,1400,86400


def immutable(path,value,*,text=False):
    path=Path(path)
    data=value if text else json.dumps(value,indent=2,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=data:raise ValueError('Immutable assignment/outcome bytes differ')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def charge(clock):
    return sum(row.get('charged_seconds',row['reserved_seconds']) for row in clock['attempts'])


def remaining(root):
    """Read before rescheduling; no fresh10500s allocation on a partial clock."""
    path=Path(root)/'worker-time.json'
    clock=source.read(path) if path.exists() else None
    used=charge(clock) if clock else 0
    seconds=max(0,WORKER_SECONDS-used-ALLOWANCE)
    closed=clock is None or all(row['status']=='finished' for row in clock['attempts'])
    return dict(cumulative_macro_charge_seconds=used,maximum_new_scheduler_seconds=seconds,
        launchable=closed and seconds>=600,unclosed_attempt=not closed,
        accounting='One original10560s ceiling; prior attempts and next60s startup allowance retained')


def train_admission(plan):
    rows=[row for row in plan['slots'] if row['slot'].endswith('/train')]
    if len(rows)!=40 or [row['slot'] for row in rows]!=source.expected_slots()[:40]:
        raise ValueError('Production packing requires all40 admitted train streams in their fixed source order')
    retained=[row for row in rows if not row['launchable']]
    proofs=[]
    for row in retained:
        headroom=row['maximum_total_actions']-row['parent']['actions']
        if not 0<=headroom<1650:
            raise ValueError('Retained source leaves must lack quota for qualification plus another complete K4')
        proofs.append(dict(slot=row['slot'],parent=row['parent'],maximum_total_actions=row['maximum_total_actions'],
            remaining_action_headroom=headroom,reason='Quota-based preservation of the exact terminal parent; no cold relaunch'))
    return dict(all40_admitted_train_count=40,all40_train_rows_sha256=source.content_hash(rows),
        launchable_train_count=40-len(retained),retained_only_train_count=len(retained),
        retained_only_source_leaves=proofs)


def assigned(plan,prefix,worker,outputs):
    train_admission(plan)
    rows=[row for row in plan['slots'] if row['slot'].endswith('/train') and row['launchable']]
    result=[]
    names=dict(libero_spatial='s',libero_object='o',libero_goal='g',libero_10='l')
    for row in rows[worker::8]:
        key,partition=row['slot'].split('/');suite,number=key.split(':')
        run=f'{prefix}-{names[suite]}{int(number):02d}-{partition}'
        result.append(dict(slot=row['slot'],root=str(Path(outputs)/run),run_id=run,
            maximum_total_actions=row['maximum_total_actions'],parent=row['parent']))
    return result


def funding(path,plan,prefix,worker,root):
    value=source.read(path)
    if (value.get('format')!=FORMAT or value.get('funding_status')!='root_reconciled'
            or value.get('maximum_new_gpu_seconds')!=NEW_SECONDS or value.get('workers')!=8
            or value.get('worker_execution_seconds')!=ACTIVE_SECONDS
            or value.get('maximum_worker_charge_seconds')!=WORKER_SECONDS
            or value.get('startup_allowance_seconds_per_attempt')!=ALLOWANCE
            or value.get('maximum_final_competence_charge_seconds')!=FINAL_SECONDS
            or value.get('plan_sha256')!=plan['plan_sha256'] or value.get('source_commit')!=plan['source_commit']
            or value.get('campaign_prefix')!=prefix or type(worker) is not int or not 0<=worker<8
            or Path(root).resolve()!=Path(value['campaign_root']).resolve()/f'worker-{worker}'
            or value.get('historical_qualification_pool')!='original32h_counted_once_not_new_GPU_time'
            or plan.get('mode')!='production' or not prefix or any(not(c.isalnum() or c=='-') for c in prefix)):
        raise ValueError('Require exact root-funded24h production macro budget, source, prefix and worker identity')
    return value


def begin(root,plan_path,ledger,budget_path,prefix,worker,source_commit,*,resume=False,now=None):
    root=Path(root).resolve();plan=source.load_plan(plan_path)
    funding(budget_path,plan,prefix,worker,root)
    if source_commit!=plan['source_commit']:raise ValueError('Macro source pin differs')
    slots=assigned(plan,prefix,worker,root.parent.parent)
    assignment=dict(plan_sha256=plan['plan_sha256'],source_commit=source_commit,
        canonical_ledger=str(Path(ledger).resolve()),slots=slots,source_admission=train_admission(plan))
    root.mkdir(parents=True,exist_ok=True)
    immutable(root/'assignment.json',assignment)
    immutable(root/'assigned.txt',''.join(row['slot']+'\n' for row in slots),text=True)
    contract=dict(format=FORMAT,source_commit=source_commit,run_id=root.name,campaign_prefix=prefix,
        budget_path=str(Path(budget_path).resolve()),budget_sha256=source.sha(budget_path),
        plan_path=str(Path(plan_path).resolve()),plan_sha256=plan['plan_sha256'],
        assignment_sha256=source.sha(root/'assignment.json'),canonical_ledger=str(Path(ledger).resolve()),
        maximum_total_actions=plan['maximum_total_actions'],macro_clock_covers='Entire allocated worker setup, children, gaps, idle and teardown',
        child_accounting='Overlapping child bounds, not added to macroGPU charge',worker=worker)
    clock_path=root/'worker-time.json'
    previous=source.read(clock_path) if clock_path.exists() else None
    if previous and any(row['status']!='finished' for row in previous['attempts']):
        raise ValueError('Unclosed macro attempt remains fully charged; explicit terminal reconciliation required')
    used=charge(previous) if previous else 0
    slice_seconds=WORKER_SECONDS-used-ALLOWANCE
    attempt=clock_begin(root,contract,slice_seconds=slice_seconds,worker_seconds=WORKER_SECONDS,resume=resume,now=now)
    started=source.read(attempt/'attempt.json')['started_unix']
    return attempt,int(started+slice_seconds),slice_seconds


def child_state(row,contract,*,strict=False):
    root=Path(row['root']);path=root/'worker-time.json'
    empty=dict(slot=row['slot'],root=str(root),fulfilled=False,status='unattempted',charged_child_seconds=0)
    if not path.exists():
        if strict and root.exists() and any(root.iterdir()):
            raise ValueError('Child output exists without an original clock; preserve and diagnose')
        return empty
    clock=source.read(path);expected=dict(format=source.FORMAT,plan_sha256=contract['plan_sha256'],
        slot=row['slot'],source_commit=contract['source_commit'],run_id=root.name,
        purpose='Source-only SAC top-up; no resource approval granted by this receipt')
    if clock.get('worker_seconds')!=CHILD_SECONDS or clock.get('contract')!=expected:
        raise ValueError('Existing child has a different source/plan or cumulative2100s ceiling')
    closed=bool(clock['attempts']) and all(attempt['status']=='finished' for attempt in clock['attempts'])
    if strict and not closed:raise ValueError('Unclosed child attempt remains charged; reconcile before resume')
    lock_path=root/'launcher.lock'
    if strict and lock_path.exists():
        with lock_path.open('r') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as error:raise ValueError('Child simulator/collector is still active') from error
    result=dict(empty,status='unfulfilled',charged_child_seconds=charge(clock),worker_time_sha256=source.sha(path))
    index_path=root/'experiment/index.json';exit_path=root/'exit-code.txt'
    if not index_path.exists():return result
    index=source.read(index_path);continuation=index.get('continuation',{})
    result.update(index_committed_actions=index.get('total_action_budget_charge'),
        committed_case_cursor=index.get('case_index'),episode_in_case=index.get('episode_in_case'),
        committed_source_updates=index.get('source_updates'),index_sha256=source.sha(index_path))
    if (not closed or not exit_path.exists() or exit_path.read_text().strip()!='0'
            or clock['attempts'][-1].get('exit_code')!=0 or index.get('completed') is not True
            or index.get('collection_qualified') is not True or index.get('episode_in_case')!=0
            or index.get('slice_completed') is not True or index.get('source_only') is not True
            or continuation.get('parent')!=row['parent'] or continuation.get('plan_sha256')!=contract['plan_sha256']
            or continuation.get('new_optimizer_updates',0)<=0 or continuation.get('motor_parameters_unchanged') is not True):
        return result
    source_contract=source.read(root/'experiment/contract.json')
    quota_done=index.get('production_quota_exhausted') is True and index['total_action_budget_charge']+1640>row['maximum_total_actions']
    bank_done=index.get('case_index')==len(source_contract.get('cases',[]))==512
    recovery=root/'experiment/source-recovery.pkl';candidate=root/'experiment/candidate.json'
    if (not (quota_done or bank_done) or source_contract.get('source_commit')!=contract['source_commit']
            or source_contract.get('maximum_total_actions')!=row['maximum_total_actions']
            or source_contract.get('source_only') is not True or source_contract.get('source_data_from_targets') is not False
            or source_contract.get('task',{}).get('task_key')!=row['slot'].split('/')[0]
            or source_contract.get('partition')!=row['slot'].split('/')[1]
            or index['total_action_budget_charge']>row['maximum_total_actions']
            or index.get('source_contract_sha256')!=source.content_hash(source_contract)
            or not recovery.is_file() or source.sha(recovery)!=index['latest_recovery_sha256'] or not candidate.is_file()):return result
    final=source.read(candidate)
    if (final.get('plan_sha256')!=contract['plan_sha256'] or final.get('full_latest_recovery_sha256')!=index['latest_recovery_sha256']
            or final.get('teacher_updates')!=index['source_updates']):return result
    ledger=source.read(contract['canonical_ledger']);leaf=ledger['leaves'][row['slot']]
    if Path(leaf['root']).resolve()!=root.resolve() or leaf['maximum_total_actions']!=row['maximum_total_actions']:
        raise ValueError('Canonical source leaf no longer owns this terminal child')
    result.update(fulfilled=True,status='fulfilled',recovery_sha256=index['latest_recovery_sha256'])
    return result


def selection(root,attempt,slot,*,now=None):
    root,attempt=Path(root),Path(attempt);clock=source.read(root/'worker-time.json');contract=clock['contract']
    assignment=source.read(root/'assignment.json')
    if source.sha(root/'assignment.json')!=contract['assignment_sha256']:raise ValueError('Assigned source stream order changed')
    active=clock['attempts'][-1]
    if active['path']!=str(attempt) or active['status']!='started':raise ValueError('No owned active macro attempt')
    row=next(value for value in assignment['slots'] if value['slot']==slot)
    state=child_state(row,contract,strict=True)
    current=time.time() if now is None else now
    left=math.floor(active['started_unix']+active['slice_seconds']-current)
    decision=dict(state,decision='fulfilled' if state['fulfilled'] else 'blocked',reason='terminal_quota_proof' if state['fulfilled'] else 'insufficient_cumulative_time',
        run_id=row['run_id'],resume=state['status']!='unattempted',macro_seconds_left=left,
        child_seconds_left=CHILD_SECONDS-state['charged_child_seconds'],slice_seconds=0,timeout_seconds=0)
    if not state['fulfilled']:
        slice_seconds=min(CHILD_SECONDS-state['charged_child_seconds']-ALLOWANCE,left-90)
        if slice_seconds>=600:
            decision.update(decision='launch',reason='exact_remaining_child_and_macro_time',slice_seconds=slice_seconds,
                timeout_seconds=min(left-30,slice_seconds+20))
    path=attempt/f'decision-{time.time_ns()}.json';immutable(path,decision)
    return decision,path


def outcome(root,attempt,slot,exit_code):
    root,attempt=Path(root),Path(attempt);clock=source.read(root/'worker-time.json')
    row=next(row for row in source.read(root/'assignment.json')['slots'] if row['slot']==slot)
    value=dict(exit_code=int(exit_code),**child_state(row,clock['contract']))
    immutable(attempt/f'child-outcome-{time.time_ns()}.json',value)
    return value


def finish(root,attempt,exit_code,*,now=None):
    root,attempt=Path(root),Path(attempt)
    row=clock_finish(root,attempt,exit_code,now=now);clock=source.read(root/'worker-time.json')
    assignment=source.read(root/'assignment.json')
    states=[child_state(value,clock['contract']) for value in assignment['slots']]
    value=dict(format=FORMAT,macro_attempt=row,cumulative_macro_charged_seconds=charge(clock),
        maximum_macro_charged_seconds=WORKER_SECONDS,within_macro_ceiling=charge(clock)<=WORKER_SECONDS,
        child_accounting='Overlapping controls; NEVER sum child charges with macro charge',
        children=states,fulfilled_streams=sum(value['fulfilled'] for value in states),assigned_streams=len(states),
        source_admission=assignment['source_admission'],
        all40_claim=False,coverage='Assignments/admission are not completed source quota or512-case-bank exhaustion')
    immutable(attempt/'macro-outcome.json',value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='operation',required=True)
    start=sub.add_parser('begin')
    for name in ('root','plan','ledger','budget'):start.add_argument('--'+name,type=Path,required=True)
    for name in ('prefix','source-commit'):start.add_argument('--'+name,required=True)
    start.add_argument('--worker',type=int,required=True);start.add_argument('--started-unix',type=float,required=True)
    start.add_argument('--resume',action='store_true')
    available=sub.add_parser('remaining');available.add_argument('--root',type=Path,required=True)
    for op in ('select','outcome','finish'):
        p=sub.add_parser(op)
        for name in ('root','attempt'):p.add_argument('--'+name,type=Path,required=True)
        if op!='finish':p.add_argument('--slot',required=True)
        if op!='select':p.add_argument('--exit-code',type=int,required=True)
    args=parser.parse_args(argv)
    if args.operation=='remaining':print(json.dumps(remaining(args.root)))
    elif args.operation=='begin':
        attempt,deadline,seconds=begin(args.root,args.plan,args.ledger,args.budget,args.prefix,args.worker,args.source_commit,
            resume=args.resume,now=args.started_unix)
        print(f'{attempt}\t{deadline}\t{seconds}')
    elif args.operation=='select':
        value,path=selection(args.root,args.attempt,args.slot)
        print(f"{value['decision']}\t{value['run_id']}\t{value['slice_seconds']}\t{int(value['resume'])}\t{value['timeout_seconds']}\t{path}")
    elif args.operation=='outcome':
        value=outcome(args.root,args.attempt,args.slot,args.exit_code);print(json.dumps(value))
        if args.exit_code==0 and not value['fulfilled']:raise SystemExit(3)
    else:print(json.dumps(finish(args.root,args.attempt,args.exit_code)))


if __name__=='__main__':main()
