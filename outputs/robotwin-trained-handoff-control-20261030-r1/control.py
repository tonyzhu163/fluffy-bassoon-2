"""Owner-scoped pause-and-audit control; never signals a source learner.

The CLI is production-specific. The process primitives are separately exercised
with finite fake owned processes in an allocated Linux CPU test job.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass,replace
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import zipfile

SUPERVISOR_SHA='492efd191e0dd1d45433d290a92c6bcd766593bfd8783e97321e7705909413bf'
EXECUTOR='e825821a7c2231952bcf1042162f3470bd8ecdea'
LINEAGE='db0184089fa88a5f96c153f9b80029bb91b77def'
PARENT_SHA='1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0'

class ControlError(RuntimeError):pass
class ChildWaitTimeout(ControlError):pass


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):value.update(block)
    return value.hexdigest()


def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)


def bound(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


@dataclass(frozen=True)
class Owner:
    hostname:str
    pid:int
    start_ticks:int
    supervisor_script:Path
    supervisor_sha256:str
    python:Path
    run_id:str
    method:str
    source_commit:str
    code_root:Path
    output:Path
    check_git:bool=True


def proc(pid):
    root=Path('/proc')/str(pid)
    try:
        text=(root/'stat').read_text();fields=text.rsplit(')',1)[1].split()
        argv=(root/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
        uid=int(next(x for x in (root/'status').read_text().splitlines() if x.startswith('Uid:')).split()[1])
        return dict(pid=pid,ppid=int(fields[1]),state=fields[0],start_ticks=int(fields[19]),
            wait_status=int(fields[49]) if len(fields)>49 else None,uid=uid,argv=argv)
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None


def pids():
    return [int(path.name) for path in Path('/proc').iterdir() if path.name.isdigit()]


def option(argv,name,*,many=False):
    matches=[i for i,value in enumerate(argv) if value==name]
    if len(matches)!=1:raise ControlError(f'Require exactly one {name} in owned command')
    i=matches[0]+1;values=[]
    while i<len(argv) and not argv[i].startswith('--'):
        values.append(argv[i]);i+=1
    if (not values) or (not many and len(values)!=1):raise ControlError(f'Invalid owned command option {name}')
    return values if many else values[0]


def parent_command_matches(value,owner):
    argv=value['argv']
    if (value['uid']!=os.getuid() or len(argv)<2
            or Path(argv[0]).resolve()!=owner.python.resolve()
            or Path(argv[1]).resolve()!=owner.supervisor_script.resolve()):return False
    try:
        expected={'--run-id':owner.run_id,'--method':owner.method,'--source-commit':owner.source_commit,
            '--code-root':str(owner.code_root.resolve()),'--output':str(owner.output.resolve())}
        return all(option(argv,key)==wanted for key,wanted in expected.items())
    except ControlError:return False


def check_parent(owner,*,allow_stopped=True,unique=True,full=True):
    if sys.platform!='linux' or socket.gethostname()!=owner.hostname:raise ControlError('Linux hostname ownership mismatch')
    if owner.pid<=1 or owner.start_ticks<1:raise ControlError('Invalid explicit supervisor PID/start time')
    if full and sha(owner.supervisor_script)!=owner.supervisor_sha256:raise ControlError('Supervisor script changed from its pinned artifact')
    if full and owner.check_git:
        actual=subprocess.check_output(['git','rev-parse','HEAD'],cwd=owner.code_root,text=True).strip()
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=owner.code_root,text=True).strip()
        if actual!=owner.source_commit or dirty:raise ControlError('Owned executor worktree is not the exact clean pin')
    current=proc(owner.pid)
    if (current is None or current['start_ticks']!=owner.start_ticks
            or current['state']=='Z' or not parent_command_matches(current,owner)):
        raise ControlError('Supervisor PID/start time/command ownership mismatch')
    if not allow_stopped and current['state'] in ('T','t'):raise ControlError('Supervisor was already stopped by another owner action')
    if full and unique:
        matching=[value for pid in pids() if (value:=proc(pid)) is not None and value['state']!='Z' and parent_command_matches(value,owner)]
        if len(matching)!=1 or matching[0]['pid']!=owner.pid:raise ControlError('Owned supervisor command is not unique')
    return current


def checked_signal(owner,fd,number,signals):
    if number not in (signal.SIGSTOP,signal.SIGCONT):raise ControlError('Controller may only stop/continue its CPU supervisor')
    current=check_parent(owner,allow_stopped=number!=signal.SIGSTOP)
    signal.pidfd_send_signal(fd,number,None,0)
    signals.append(dict(signal=signal.Signals(number).name,pid=owner.pid,start_ticks=current['start_ticks']))


def source_child_matches(value,owner):
    argv=value['argv'];driver=owner.code_root/'scripts/diagnostics/robotwin_noise_compare.py'
    if (value['ppid']!=owner.pid or value['uid']!=os.getuid() or len(argv)<2
            or Path(argv[0]).resolve()!=owner.python.resolve() or Path(argv[1]).resolve()!=driver.resolve()):return False
    try:
        output=Path(option(argv,'--output')).resolve()
        relative=output.relative_to(owner.output.resolve()/'slices')
        return (len(relative.parts)==2 and re.fullmatch(r'\d{4}',relative.parts[0]) is not None
            and relative.parts[1]=='experiment' and option(argv,'--source-commit')==owner.source_commit
            and option(argv,'--lineage-source-commit')==LINEAGE
            and option(argv,'--methods',many=True)==[owner.method]
            and option(argv,'--max-source-trials-per-process')=='6'
            and argv.count('--recycle-at-round-boundary')==1)
    except (ControlError,ValueError):return False


def direct_children(owner):
    return [value for pid in pids() if (value:=proc(pid)) is not None and value['ppid']==owner.pid]


def pause_and_drain(owner,*,wait_seconds,poll_seconds=1.,rollback_on_timeout=False,audit=None):
    """Only the supervisor receives signals; finite direct children keep running."""
    if not hasattr(os,'pidfd_open') or not hasattr(signal,'pidfd_send_signal'):
        raise ControlError('Linux pidfd signalling is required; no unsafe PID fallback')
    check_parent(owner,allow_stopped=False)
    known={}
    for value in direct_children(owner):
        if value['state']=='Z':raise ControlError('Unidentified exited child before pause; audit its owner first')
        if not source_child_matches(value,owner):raise ControlError('Unexpected live direct child; no pause signal sent')
        known[(value['pid'],value['start_ticks'])]=dict(value)
    fd=os.pidfd_open(owner.pid,0);signals=[];paused=False
    receipt=dict(schema='robotwin_owner_handoff_control_v1',hostname=owner.hostname,
        supervisor_pid=owner.pid,supervisor_start_ticks=owner.start_ticks,run_id=owner.run_id,
        method=owner.method,source_commit=owner.source_commit,output=str(owner.output.resolve()),
        started_at_utc=datetime.now(timezone.utc).isoformat(),signals=signals,
        permission_to_start_new_writer=False,required_next_action='Root audits then stops the exact old allocation in UI')
    try:
        checked_signal(owner,fd,signal.SIGSTOP,signals);paused=True
        stop_deadline=time.monotonic()+2.
        while check_parent(owner,full=False,unique=False)['state'] not in ('T','t'):
            if time.monotonic()>=stop_deadline:raise ControlError('Supervisor did not acknowledge SIGSTOP')
            time.sleep(.01)
        deadline=time.monotonic()+wait_seconds;observations=[]
        while True:
            parent=check_parent(owner,full=False,unique=False)
            if parent['state'] not in ('T','t'):raise ControlError('Supervisor did not remain stopped')
            live=[]
            for value in direct_children(owner):
                key=(value['pid'],value['start_ticks'])
                if value['state']=='Z':
                    if key not in known:raise ControlError('Unknown zombie child; cannot prove its command identity')
                    known[key]['exit_wait_status']=value['wait_status'];continue
                if not source_child_matches(value,owner):raise ControlError('An unexpected direct child exists after pause')
                known.setdefault(key,dict(value));live.append(value)
                if value['state'] in ('T','t'):raise ControlError('Source child is externally stopped; controller will not signal it')
            observations.append(dict(live=[(x['pid'],x['start_ticks'],x['state']) for x in live]))
            if len(live)>1:raise ControlError('More than one live owned source writer')
            if not live:break
            if time.monotonic()>=deadline:raise ChildWaitTimeout('Finite source child did not drain within the owner wait bound')
            time.sleep(poll_seconds)
        # A vanished child may have been reaped just before SIGSTOP. Its exact
        # exit must then be established from its fresh slice exit.json in audit.
        receipt.update(owned_children=list(known.values()),parent_state=check_parent(owner)['state'],
            source_children_inactive=True,wait_observation_count=len(observations))
        if audit is not None:receipt['handoff']=audit(owner,list(known.values()))
        final=check_parent(owner)
        if final['state'] not in ('T','t') or any(value['state']!='Z' for value in direct_children(owner)):
            raise ControlError('Supervisor resumed or a writer appeared during handoff audit')
        receipt.update(status='paused_child_complete_handoff_verified',parent_left_stopped=True,parent_state=final['state'])
        return receipt
    except Exception as error:
        receipt.update(status='halted_parent_left_stopped' if paused else 'rejected_no_signal',
            failure=dict(type=type(error).__name__,message=str(error)),parent_left_stopped=paused)
        if isinstance(error,ChildWaitTimeout) and rollback_on_timeout and paused:
            try:
                checked_signal(owner,fd,signal.SIGCONT,signals)
                receipt.update(status='timeout_rolled_back',parent_left_stopped=False,rollback_verified=True)
            except Exception as rollback_error:
                receipt['rollback_error']=str(rollback_error)
        error.receipt=receipt
        raise
    finally:os.close(fd)


def bound_json(ref):
    if sha(ref['path'])!=ref['sha256']:raise ControlError('A bound handoff artifact changed')
    return json.loads(Path(ref['path']).read_text())


def ledger_snapshot(root):
    """Read-only and deliberately no flock/Budget constructor."""
    root=Path(root);snapshot=root/'snapshot.json';raw=snapshot.read_bytes();value=json.loads(raw)
    if value.get('pending_actions')!=0 or value.get('uncertain_actions')!=0 or value.get('unresolved_decisions'):
        raise ControlError('Ledger has pending/uncertain real actions')
    n=value['journal_events'];tail=root/'events'/f'{n-1:010d}.json';event=json.loads(tail.read_text());digest=event.pop('sha256')
    if (digest!=value['journal_sha256'] or hashlib.sha256(canonical(event).encode()).hexdigest()!=digest
            or (root/'events'/f'{n:010d}.json').exists()):raise ControlError('Ledger head and snapshot differ')
    return value,hashlib.sha256(raw).hexdigest()


def corpus_chain(ref):
    roots=[];seen=set()
    while True:
        key=(str(Path(ref['path']).resolve()),ref['sha256'])
        if key in seen:raise ControlError('Corpus ancestry cycle')
        seen.add(key);manifest=bound_json(ref);root=Path(manifest['corpus_root']).resolve()
        contract_ref=manifest['corpus_contract'];contract=bound_json(contract_ref)
        if (Path(contract_ref['path']).resolve()!=root/'contract.json'
                or contract.get('source_commit')!=LINEAGE):
            raise ControlError('Bound corpus contract path/lineage differs')
        if str(root) not in roots:roots.append(str(root))
        migration=root.parent/'execution-migration.json'
        resume_sha=contract.get('resume_trial_manifest_sha256')
        if not migration.exists():
            if resume_sha:raise ControlError('Bound resumed corpus lost its execution migration')
            break
        metadata=json.loads(migration.read_text())
        if (not resume_sha or metadata.get('lineage_source_commit')!=LINEAGE
                or metadata.get('manifest_sha256')!=resume_sha
                or metadata.get('execution_source_commit')!=contract.get('execution_source_commit')):
            raise ControlError('Corpus ancestry is not bound by its current contract')
        ref=dict(path=metadata['manifest_path'],sha256=metadata['manifest_sha256'])
    return list(reversed(roots))


def scoped_file(root,path):
    candidate=(root/path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ControlError('A referenced raw artifact is missing or outside its corpus')
    return candidate


def referenced_artifact(root,ref,cache,index_digest):
    """Check reference, ZIP members and numeric array headers, not full blob SHA.

    RGB payloads are not decompressed/hashed. Only native valid-action masks are
    loaded separately below to independently verify executed-action durations.
    """
    import math
    import numpy as np
    path=scoped_file(root,ref['path']);identity=canonical(ref)
    if str(path) in cache:
        if cache[str(path)]!=identity:raise ControlError('Conflicting raw artifact references')
        return path
    if (type(ref.get('bytes')) is not int or ref['bytes']!=path.stat().st_size
            or re.fullmatch(r'[0-9a-f]{64}',ref.get('sha256','')) is None):
        raise ControlError('Referenced artifact size/hash metadata differs')
    arrays=ref.get('arrays')
    if not isinstance(arrays,dict) or not arrays:raise ControlError('Missing typed raw array metadata')
    with zipfile.ZipFile(path) as archive:
        members=archive.namelist()
        if len(members)!=len(arrays) or set(members)!={name+'.npy' for name in arrays}:
            raise ControlError('Raw NPZ members differ from bound array metadata')
        for name,expected in arrays.items():
            with archive.open(name+'.npy') as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ControlError('Unsupported raw array header version')
                if (dtype.hasobject or dtype.kind not in 'buifc' or str(dtype)!=expected.get('dtype')
                        or list(shape)!=expected.get('shape')
                        or re.fullmatch(r'[0-9a-f]{64}',expected.get('sha256','')) is None
                        or archive.getinfo(name+'.npy').file_size!=stream.tell()+math.prod(shape)*dtype.itemsize):
                    raise ControlError('Raw array shape/dtype/size metadata differs')
    if 'content_sha256' in ref:
        sidecar=path.with_suffix('.json')
        if not sidecar.is_file() or canonical(json.loads(sidecar.read_text()))!=identity:
            raise ControlError('Referenced frame sidecar is missing or differs')
        index_digest.update(str(sidecar).encode()+sha(sidecar).encode())
    cache[str(path)]=identity;index_digest.update(str(path).encode()+identity.encode())
    return path


def check_encoded_arrays(value,ref):
    if isinstance(value,dict):
        if 'numeric_array' in value and (set(value)!={'numeric_array'} or value['numeric_array'] not in ref['arrays']):
            raise ControlError('Encoded metadata references a missing typed raw array')
        for child in value.values():check_encoded_arrays(child,ref)
    elif isinstance(value,list):
        for child in value:check_encoded_arrays(child,ref)


def raw_action_counts(roots):
    import numpy as np
    counts={};files=hashlib.sha256();episodes=0;chunks=0;cache={};source_keys=set()
    for root_arg in roots:
        root=Path(root_arg).resolve()
        if not (root/'episodes').is_dir():raise ControlError('A composed corpus episodes directory is missing')
        reset_paths=set((root/'episodes').glob('*/reset.json'))
        manifest_paths=set((root/'episodes').glob('*/manifest.json'))
        if {path.parent for path in reset_paths}!={path.parent for path in manifest_paths}:
            raise ControlError('Raw reset/episode manifest inventory differs')
        for manifest_path in sorted(manifest_paths):
            manifest=json.loads(manifest_path.read_text());reset_path=manifest_path.parent/'reset.json'
            reset=json.loads(reset_path.read_text());ctx=reset['context'];amount=manifest['executed_primitive_steps']
            refs=manifest['chunks'];episode_id=manifest_path.parent.name
            if manifest.get('pending_decision'):raise ControlError('Composed corpus has an unresolved physical decision')
            if (type(amount) is not int or amount<0 or manifest.get('reset')!='reset.json'
                    or manifest.get('episode_id')!=episode_id or reset.get('episode_id')!=episode_id
                    or len(refs)!=manifest.get('completed_chunks')):
                raise ControlError('Invalid raw episode identity/count')
            if amount and ctx['phase']=='source':
                info=reset['reset_info']
                key=canonical([ctx['method'],ctx['trial_id'],info['task'],info['requested_seed']])
                if key in source_keys:raise ControlError('Source episode executed twice across composed corpus roots')
                source_keys.add(key)
            previous=reset['initial_observation'];referenced_artifact(root,previous,cache,files)
            total=0;listed=[]
            for index,ref in enumerate(refs,1):
                path=scoped_file(root,str(manifest_path.parent.relative_to(root)/ref['path']))
                if path.parent!=manifest_path.parent/'chunks' or path.name!=f'{index:06d}.json' or sha(path)!=ref['sha256']:
                    raise ControlError('Raw decision index/order/hash differs')
                listed.append(path);record=json.loads(path.read_text());native=record['native_result'];learning=record['learning_result']
                if (record.get('status')!='complete' or record.get('episode_id')!=episode_id
                        or record.get('chunk_index')!=index or record.get('previous_chunk_index')!=(index-1 if index>1 else None)
                        or any(record['context'].get(key)!=ctx.get(key) for key in ('phase','method','trial_id'))
                        or canonical(record['observation'])!=canonical(previous)):
                    raise ControlError('Raw decision identity/closed status/frame chain differs')
                for frame in (record['observation'],record['next_observation']):referenced_artifact(root,frame,cache,files)
                decision_ref=record['decision_arrays'];native_ref=native['arrays']
                referenced_artifact(root,decision_ref,cache,files)
                native_path=referenced_artifact(root,native_ref,cache,files)
                check_encoded_arrays(record['decision_metadata'],decision_ref);check_encoded_arrays(native.get('info',{}),native_ref)
                issued=record['issued_primitive_steps'];executed=native['executed_primitive_steps']
                if (type(issued) is not int or not 0<issued<=32 or type(executed) is not int or not 0<=executed<=issued
                        or decision_ref['arrays'].get('issued_motor_joint_targets',{}).get('shape',[None])[0]!=issued):
                    raise ControlError('Raw issued/executed action durations differ')
                with np.load(native_path,allow_pickle=False) as archive:mask=archive['valid_action_mask']
                mask_sha=hashlib.sha256(str(mask.dtype).encode()+str(mask.shape).encode()+mask.tobytes()).hexdigest()
                if (not np.isin(mask,(0,1)).all() or int(mask.sum())!=executed
                        or mask_sha!=native_ref['arrays']['valid_action_mask']['sha256']):
                    raise ControlError('Native valid-action mask does not bind executed duration')
                total+=executed;closed=bool(learning['terminated'] or learning['truncated'])
                if learning['episode_primitive_steps']!=total or closed!=(index==len(refs)):
                    raise ControlError('Raw cumulative duration/episode closure differs')
                previous=record['next_observation'];files.update(str(path).encode()+ref['sha256'].encode());chunks+=1
            if set(listed)!={path.resolve() for path in (manifest_path.parent/'chunks').glob('*.json')}:
                raise ControlError('Raw decision inventory contains unlisted chunks')
            if total!=amount:raise ControlError('Raw episode action sum differs from every listed decision')
            key=f'{ctx["phase"]}/{ctx["method"]}';counts[key]=counts.get(key,0)+amount
            files.update(str(manifest_path).encode()+sha(manifest_path).encode()+sha(reset_path).encode());episodes+=1
    return {k:v for k,v in counts.items() if v},dict(corpus_roots=roots,episode_manifests=episodes,
        completed_chunks=chunks,raw_json_and_reference_digest=files.hexdigest(),referenced_artifact_count=len(cache),
        unique_nonzero_source_episodes=len(source_keys),source_episode_keys_unique=True,
        raw_json_index_hashes_verified=True,referenced_artifacts_present=True,
        array_headers_verified=True,native_valid_action_masks_verified=True,full_blob_hash_audit=False)


def validate_ledger_header(snapshot,worker,method,contract):
    header=snapshot['header']
    if (worker['method']!=method or header.get('campaign_id')!=worker['campaign_id']
            or header.get('total_actions')!=worker['maximum_actions']
            or header.get('source_caps')!={method:1100000} or header.get('phase_caps')!={}
            or canonical(header.get('contract'))!=canonical(contract)):
        raise ControlError('Live ledger header identity/caps/contract differ from original worker')


def audit_boundary(experiment,owner,child_exits):
    import torch
    # Use the exact old executor's qualified read-only artifact validator.
    sys.path.insert(0,str(owner.code_root/'src'))
    from fluffy_bassoon.meta_rl.robotwin_trial_resume import load_trial_continuation
    summary_path=experiment/'summary.json';summary=json.loads(summary_path.read_text())
    if summary['status']!='wall_budget_stop':raise ControlError('Latest source child did not publish a wallstop handoff')
    ref=summary.get('continuation_manifest')
    if not ref:raise ControlError('Latest child has no complete continuation manifest')
    manifest=bound_json(ref);checkpoint_path=Path(manifest['checkpoint']['path']).resolve()
    if checkpoint_path!=experiment/'coordinator-recovery.pt' or sha(checkpoint_path)!=manifest['checkpoint']['sha256']:
        raise ControlError('Latest manifest does not bind its own fresh checkpoint')
    with zipfile.ZipFile(checkpoint_path) as archive:
        if archive.testzip() is not None:raise ControlError('Checkpoint CRC failed')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    continuation=load_trial_continuation(ref['path'],ref['sha256'],checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,execution_commit=owner.source_commit,lineage_commit=LINEAGE)
    state=checkpoint['states'][owner.method]
    if (manifest['kind']!='whole_trial_boundary' or manifest['method']!=owner.method
            or state['completed_source_trials']%6 or state.get('pending_rollouts')
            or state['completed_source_trials']!=6*state['completed_source_iterations']
            or (owner.method!='sac' and state['learner']['updates']!=state['completed_source_iterations'])
            or state['schedule']['trials']!=state['completed_source_trials']
            or state['learner']['primitive_steps']!=state['source_steps']
            or canonical(summary['contract'])!=canonical(checkpoint['contract'])):
        raise ControlError('Handoff is not a complete global source round with every action retained')
    frozen_path=experiment/'frozen-motor.json';frozen=json.loads(frozen_path.read_text())
    if not frozen.get('unchanged') or not frozen.get('before') or frozen['before']!=frozen.get('after'):
        raise ControlError('Frozen motor receipt is missing or changed')
    # The same original parent pins the immutable existing method ledger path.
    admission=json.loads((owner.output/'parent-admission.json').read_text())
    parent_path=Path(option(check_parent(owner)['argv'],'--parent-manifest')).resolve()
    if (sha(parent_path)!=PARENT_SHA or admission['parent']['sha256']!=PARENT_SHA
            or Path(admission['parent']['path']).resolve()!=parent_path
            or admission['lineage_source_commit']!=LINEAGE
            or admission['execution_source_commit']!=owner.source_commit or admission['method']!=owner.method):
        raise ControlError('Original parent/lineage/executor admission differs')
    original=json.loads(parent_path.read_text());worker=original['workers'][owner.method]
    ledger_root=Path(admission['existing_ledger_root'])
    if (original['source_commit']!=LINEAGE or ledger_root!=Path(worker['ledger_root'])
            or admission['unchanged_action_cap']!=worker['maximum_actions']):
        raise ControlError('Original method ledger/cap changed')
    before,before_sha=ledger_snapshot(ledger_root)
    validate_ledger_header(before,worker,owner.method,checkpoint['contract'])
    if (before['journal_sha256']!=continuation.receipt['ledger_journal_sha256']
            or before['by_method'].get(f'source/{owner.method}',0)!=state['source_steps']):
        raise ControlError('Live ledger and complete trained source checkpoint differ')
    counts,corpus_receipt=raw_action_counts(corpus_chain(ref))
    ledger_counts={k:v for k,v in before['by_method'].items() if v}
    if counts!=ledger_counts or sum(counts.values())!=before['actual_actions']:
        raise ControlError('Not every executed action is retained in composed closed raw episodes')
    time.sleep(.2);after,after_sha=ledger_snapshot(ledger_root)
    if before_sha!=after_sha:raise ControlError('Ledger advanced while owner supervisor was paused')
    check_parent(owner)
    return dict(experiment=str(experiment),manifest=ref,checkpoint=manifest['checkpoint'],
        summary=bound(summary_path),frozen_motor=bound(frozen_path),checkpoint_crc_ok=True,
        round_boundary=True,pending_ppo_trials=0,completed_source_trials=state['completed_source_trials'],
        completed_source_iterations=state['completed_source_iterations'],source_steps=state['source_steps'],
        all_executed_actions_retained=True,retention_scope='Every action duration indexed; referenced blobs present with declared size/array headers; RGB/blob payloads not fully rehashed',
        raw_json_index_hashes_verified=True,referenced_artifacts_present=True,full_blob_hash_audit=False,
        raw_action_counts=counts,corpus=corpus_receipt,
        ledger_root=str(ledger_root),ledger_snapshot_sha256=after_sha,ledger_journal_sha256=after['journal_sha256'],
        ledger_pending_actions=0,ledger_uncertain_actions=0,ledger_header=after['header'])


def select_latest(owner,child_exits,*,auditor=audit_boundary):
    directories=sorted((owner.output/'slices').glob('[0-9][0-9][0-9][0-9]'),reverse=True)
    if not directories:raise ControlError('No source slices exist')
    known={str(Path(option(value['argv'],'--output')).resolve()):value for value in child_exits}
    inert=[]
    for directory in directories:
        experiment=directory/'experiment';summary=experiment/'summary.json';exit_path=directory/'exit.json'
        if not summary.exists():
            # Before console creation c715 has not called Popen. An empty log
            # is ambiguous and is deliberately not treated as an unstarted child.
            if not (directory/'console.log').exists() and not experiment.exists():inert.append(str(directory));continue
            raise ControlError('Newest potentially executed child has no durable summary; do not select an older cursor')
        evidence=known.get(str(experiment.resolve()));code=None
        if evidence and evidence.get('exit_wait_status') is not None:
            raw=os.waitstatus_to_exitcode(evidence['exit_wait_status']);code=raw if raw>=0 else 128-raw
        if exit_path.exists():
            recorded=json.loads(exit_path.read_text())
            if Path(recorded['output']).resolve()!=experiment.resolve():raise ControlError('Child exit receipt belongs to another slice')
            if code is not None and code!=recorded['exit_code']:raise ControlError('Kernel and durable child exit disagree')
            code=recorded['exit_code']
        if code not in (0,134,139):raise ControlError('Child exit is failed or unknown; owner audit required')
        result=auditor(experiment,owner,child_exits)
        result.update(child_exit_code=code,inert_unstarted_slices=inert)
        return result
    raise ControlError('No newest valid complete source boundary')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('expected-hostname','run-id','method','controller-commit'):parser.add_argument('--'+name,required=True)
    for name in ('supervisor-script','code-root','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--supervisor-pid',type=int,required=True);parser.add_argument('--supervisor-start-ticks',type=int,required=True)
    parser.add_argument('--python',type=Path,default=Path('/mowenjie/fluffy-bassoon-2/envs/robotwin-vlrla-20261029-r1/bin/python'))
    parser.add_argument('--wait-seconds',type=float,default=4000.);parser.add_argument('--rollback-on-timeout',action='store_true')
    args=parser.parse_args();repo=Path(__file__).resolve().parents[2]
    if sys.platform!='linux' or not socket.gethostname().startswith('job-'):
        raise ControlError('Production control requires the actual owned job allocation, never development SSH')
    if (not re.fullmatch(r'[0-9a-f]{40}',args.controller_commit) or not 0<args.wait_seconds<=4500
            or args.method not in ('ppo','sac','ppo_reset') or args.output.name!=args.run_id
            or args.code_root.name!=args.run_id):raise ControlError('Invalid explicit owner/controller scope')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()!=args.controller_commit:
        raise ControlError('Controller is not running from its exact Git pin')
    if subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip():raise ControlError('Controller checkout is dirty')
    owner=Owner(args.expected_hostname,args.supervisor_pid,args.supervisor_start_ticks,
        args.supervisor_script.resolve(),SUPERVISOR_SHA,args.python.resolve(),args.run_id,args.method,EXECUTOR,
        args.code_root.resolve(),args.output.resolve())
    receipt_path=owner.output/'control'/f'owner-handoff-{time.time_ns()}.json'
    try:
        receipt=pause_and_drain(owner,wait_seconds=args.wait_seconds,
            rollback_on_timeout=args.rollback_on_timeout,audit=select_latest)
        result=0
    except Exception as error:
        receipt=getattr(error,'receipt',dict(schema='robotwin_owner_handoff_control_v1',status='rejected_no_signal',
            failure=dict(type=type(error).__name__,message=str(error)),signals=[]))
        result=3
    receipt.update(controller_commit=args.controller_commit,controller=bound(__file__),receipt_path=str(receipt_path))
    # A rejected identity may name an unrelated output: do not write there.
    if receipt.get('signals'):atomic_json(receipt_path,receipt)
    print(json.dumps(receipt,indent=2,allow_nan=False),flush=True)
    return result


if __name__=='__main__':raise SystemExit(main())
