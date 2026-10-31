"""Executable PARTIAL11 cache child; allocation/clock begin+finish stay external.

A continuation reuses completed CASE receipts and captured deterministic inputs.
An interrupted case is retained and restarted in a fresh child directory; no
physical action, optimizer, normalizer, complete-bank seal or fit is available.
"""
import argparse
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import uuid

import torch

from fluffy_bassoon.rl.robotwin_corpus import _atomic_json, _fsync_directory
from fluffy_bassoon.rl.robotwin_noise import _RoboTwinBackend
from fluffy_bassoon.rl.robotwin_steering import validate_interface
from fluffy_bassoon.baige_gpu import verify_selected_cuda, selected_device
from .checkpoint import rng_state,restore_rng
from .warmstart import atomic_save
from .robotwin_demo_capacity import file_sha
from .robotwin_replay_diagnosis import Audit
from . import robotwin_teacher_bank as bank_api

FORMAT='robotwin_partial11_model_cache_v1'
IDS=[identity for identity in bank_api.IDS if identity!='phone-04']
SCOPE_SHA='32ecd37ecde9b4b019387d0df2042c923cc79579ce969722f4bfe51038779447'
GRANT_SHA='b088a6d8d75120971f62b83d3276b65901859f06c5a493ce941a5e28c86fced4'
MOTOR_SHA='de8e03a445f3e0af417a9d61a903d342fca5b6a5b4fdc6f20445e14ad25692eb'
NORM_SHA='1dce5ceb9a98eaa6ae587795e0e8977dc3ce046a7ba0a5ec07858ea896ed1adb'
CACHE_LINEAGE='frozen-pi05-endtoend-20261031-r1/partial11-model-cache'
CLOCK_CONTRACT=dict(format='frozen_pi05_shared_gpu_clock_v1',phase_id='frozen-pi05-endtoend-20261031-r1',
    scope_sha256=SCOPE_SHA,maximum_gpu_seconds=28800,maximum_actions=300000,motor_file_sha256=MOTOR_SHA)


def validate_cache_buffers(cache):
    # Bound saved tensor bytes BEFORE the qualified helper serializes a row.
    # With the fixed18 pairs and small14/32/2062 auxiliaries this leaves ample
    # room inside the declared128MiB per-sample cap, including Torch headers.
    if (len(cache)!=18 or any(len(pair)!=2 or any(not isinstance(t,torch.Tensor)
            or t.ndim!=4 or t.shape[0]!=1 or t.shape[-2:]!=(1,256) for t in pair) for pair in cache)):
        raise ValueError('Exact original18-layer one-lane native cache layout required')
    if sum(t.numel()*t.element_size() for pair in cache for t in pair)>64<<20:
        raise ValueError('Prefix cache exceeds prospective saved-row storage bound')


class PartialCacheBackend(_RoboTwinBackend):
    """Unchanged lazy model/prefix encoder, followed only by a size refusal."""
    def _encode(self,raw):
        value=super()._encode(raw)
        validate_cache_buffers(self._kv_cache)
        return value


def validate_profile(profile):
    if (profile.get('format')!=FORMAT or profile['case_ids']!=IDS
            or profile['maximum_aggregate_gpu_seconds']!=1200 or profile['maximum_output_bytes']!=12<<30
            or profile['maximum_robot_actions']!=0 or profile['maximum_complete_cases']!=11
            or profile['maximum_samples']!=77 or profile['partial_only'] is not True
            or profile['fit_eligible'] is not False or profile['pair_hold_minimum_seconds']!=600
            or profile['bank_implementation_commit']!='061874dfeb6e6d27d5e3551c62c45aeaec98fb7b'):
        raise ValueError('Exact fixed11/PARTIAL/1200 aggregate seconds/12GiB/0actions profile required')


def validate_clock(ticket,clock,*,now,source,run_id,output,cache_root,prior_cache_seconds):
    """Read-only admission of the parent's already registered shared attempt."""
    if (ticket.get('mode')!=FORMAT or ticket['source_commit']!=source or ticket['run_id']!=run_id
            or ticket['hostname']!=socket.gethostname() or ticket['gpu_count']!=1
            or Path(ticket['output']).resolve()!=Path(output).resolve()
            or Path(ticket['cache_root']).resolve()!=Path(cache_root).resolve()
            or ticket['prior_cache_gpu_seconds']!=prior_cache_seconds
            or ticket['maximum_robot_actions']!=0 or ticket['scope_sha256']!=SCOPE_SHA):
        raise ValueError('Actual source/host/GPU/fresh child/cumulative cache ticket mismatch')
    if clock['contract']!=CLOCK_CONTRACT or clock['worker_seconds']!=28800:
        raise ValueError('Use the ORIGINAL shared phase GPU clock; never create another clock')
    attempts=clock['attempts']
    if (len(attempts)<7 or [r.get('charged_seconds') for r in attempts[:5]]!=[632,1138,266,298,318]
            or any(r['status']!='finished' for r in attempts[:5])):
        raise ValueError('Retain the exact five settled attempts/2652 historical GPU seconds')
    rows=[r for r in attempts if r['path']==ticket['resource_attempt']]
    if len(rows)!=1:raise ValueError('Exactly one original registered resource attempt required')
    row=rows[0];seconds=ticket['reserved_gpu_seconds'];start=ticket['scheduler_start_unix'];end=ticket['driver_deadline_unix']
    lineage=[r for r in attempts if r.get('lineage')==CACHE_LINEAGE]
    if (row.get('lineage')!=CACHE_LINEAGE or any(r.get('lineage_seconds')!=1200 for r in lineage)
            or any(r['status']!='finished' for r in lineage if r['path']!=row['path'])
            or sum(r.get('charged_seconds',r['reserved_seconds']) for r in lineage)>1200):
        raise ValueError('ALL original-clock cache-lineage charges/reservations must fit1200; no sibling reservation')
    actual_prior=sum(r.get('charged_seconds',r['reserved_seconds']) for r in lineage if r['path']!=row['path'])
    if prior_cache_seconds!=actual_prior:
        raise ValueError('Selected cursor omits later original-clock cache-lineage costs')
    if (row['status']!='started' or type(seconds) is not int or not 660<=seconds<=1200-prior_cache_seconds
            or row['reserved_seconds']!=seconds or row['slice_seconds']+60!=seconds
            or row['scheduler_allowance_seconds']!=60 or row['started_unix']!=start
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in (start,end,now))
            or not start<=now<end<=start+seconds-180):
        raise ValueError('One all-in residual reservation; original startup/cleanup clock cannot reset')
    charged=sum(r.get('charged_seconds',r['reserved_seconds']) for r in attempts)
    # Keep at least the approved pair600 hold, even if its final cost settles
    # lower. If the actual registered reservation is larger, retain that too.
    pair=[r for r in attempts if r['path']==ticket['pair_resource_attempt']]
    if (len(pair)!=1 or pair[0]['path']==row['path']
            or pair[0]['path'] in {r['path'] for r in attempts[:5]}):raise ValueError('Bind the independent paired04 resource hold')
    pair_charge=pair[0].get('charged_seconds',pair[0]['reserved_seconds'])
    extra_hold=max(0,max(600,pair[0]['reserved_seconds'])-pair_charge)
    if charged>28800 or charged+extra_hold>7200:
        raise ValueError('Shared phase/A1 actual+reserved costs and paired hold must all fit')
    return row


def validate_prior_in_clock(clock,closeout):
    rows=[r for r in clock['attempts'] if r['path']==closeout['attempt']['path']]
    if rows!=[closeout['attempt']]:raise ValueError('Actual predecessor closeout must match the ORIGINAL shared clock row')


def validate_cursor(state,bank,*,source,profile_sha,cache_root):
    expected=IDS[:len(state['complete_cases'])]
    if (state.get('format')!=FORMAT or state['source_commit']!=source or state['profile_sha256']!=profile_sha
            or state['bank_sha256']!=bank_api.digest(bank) or state['complete_cases']!=expected
            or state['new_robot_actions']!=0 or state['fit_eligible'] is not False
            or len(state['records'])!=7*len(expected) or len(expected)==11):
        raise ValueError('Only exact same-source/profile/input closed-case progress may resume')
    root=Path(cache_root).resolve(strict=True);by_id={e['id']:e for e in bank['entries']}
    for identity in expected:
        rows=[r for r in state['records'] if r['context']['teacher_id']==identity]
        if sorted(r['anchor'] for r in rows)!=list(range(7)) or sum(r['native_actual_actions'] for r in rows)!=193:
            raise ValueError('Completed-case cursor is missing/duplicating captured samples')
        entry=by_id[identity]
        wanted=dict(teacher_id=identity,split=entry['split'],producer_result=entry['producer_result'],
            producer_commit=entry['producer_commit'],profile=entry['profile'],record_id=entry['record']['spec']['id'],
            episode_manifest_ref=entry['record']['episode_manifest_ref'],original_source=entry['original_source'])
        for row in rows:
            context=row['context'];path=Path(row['path'])
            if (not path.resolve(strict=True).is_relative_to(root) or path.is_symlink()
                    or context!=wanted
                    or file_sha(path)!=row['sha256']):raise ValueError('Completed-case payload/provenance changed')


def prior_cost(cursor,closeout):
    row=closeout['attempt']
    if (row['path']!=cursor['resource_attempt'] or row['status']!='finished'
            or closeout['source_commit']!=cursor['source_commit']
            or row['charged_seconds']!=closeout['charged_gpu_seconds']
            or row.get('slice_overrun') is not False or type(row['charged_seconds']) is not int):
        raise ValueError('Actual predecessor allocation closure/cost is required before resuming')
    total=cursor['prior_cache_gpu_seconds']+row['charged_seconds']
    if not 0<=total<1200:raise ValueError('Aggregate partial-cache1200 budget exhausted')
    return total


def commit_case(state,identity,records):
    index=len(state['complete_cases'])
    if index>=len(IDS) or identity!=IDS[index] or len(records)!=7:
        raise ValueError('Commit only the next exact complete original case')
    if (sorted(r['anchor'] for r in records)!=list(range(7))
            or any(r['context']['teacher_id']!=identity for r in records)
            or sum(r['native_actual_actions'] for r in records)!=193):
        raise ValueError('Only all7 closed original samples/193 historical actions commit a case')
    state['records'].extend(deepcopy(records));state['complete_cases'].append(identity);state['active_case']=None


def run_cases(bank,state,cache_one,persist,*,should_start):
    """Case-boundary cursor; partial/incomplete case files stay uncommitted."""
    entries={e['id']:e for e in bank['entries']}
    for identity in IDS[len(state['complete_cases']):]:
        if not should_start():state['status']='partial_boundary_stop';persist();return
        state['active_case']=identity;persist()
        records=cache_one(entries[identity])
        commit_case(state,identity,records);persist()
    state['status']='complete_partial11';persist()


def publish_cursor(output,state,global_rng):
    """Immutable durable RNG generation, then one atomic cursor publication."""
    output=Path(output);directory=output/'recoveries'
    directory.mkdir(exist_ok=True)
    _fsync_directory(output.parent.parent);_fsync_directory(output.parent);_fsync_directory(output)
    candidate={k:deepcopy(v) for k,v in state.items() if k!='recovery'}
    candidate['recovery_generation']=state.get('recovery_generation',0)+1
    path=directory/f"{candidate['recovery_generation']:06d}-{uuid.uuid4().hex}.pt"
    sha=atomic_save(path,dict(state=deepcopy(candidate),global_rng=global_rng,optimizer=None,control=None))
    _fsync_directory(directory)  # Make this unique generation durable BEFORE publishing its ref.
    candidate['recovery']=dict(path=str(path),sha256=sha)
    _atomic_json(output/'cursor.json',candidate,replace=True)
    state.clear();state.update(candidate)


def _storage(path):
    total=0
    for directory,dirs,files in os.walk(path,followlinks=False):
        if any((Path(directory)/name).is_symlink() for name in (*dirs,*files)):raise ValueError('Cache storage rejects symlinks')
        total+=sum((Path(directory)/name).stat().st_size for name in files)
    return total


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('profile','inputs','scope','grant','allocation'):
        p.add_argument('--'+name,type=Path,required=True);p.add_argument('--'+name+'-sha256',required=True)
    for name in ('model-path','norm-stats','robotwin-root','rlinf-root','gpu-probe','output','cache-root'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--source-commit',required=True);p.add_argument('--run-id',required=True)
    for name in ('resume-cursor','resume-closeout'):
        p.add_argument('--'+name,type=Path);p.add_argument('--'+name+'-sha256')
    return p


def main(argv=None):
    a=parser().parse_args(argv)
    if sys.platform!='linux':raise RuntimeError('Allocated Linux GPU only; Mac static preparation only')
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if source!=a.source_commit or subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise ValueError('Require the exact clean Git-pinned child')
    audit=Audit(time.time()+30)
    profile=audit.json(dict(path=str(a.profile),sha256=a.profile_sha256));validate_profile(profile)
    scope=audit.json(dict(path=str(a.scope),sha256=a.scope_sha256));grant=audit.json(dict(path=str(a.grant),sha256=a.grant_sha256))
    if (a.scope_sha256!=SCOPE_SHA or a.grant_sha256!=GRANT_SHA or not grant['approved']
            or scope['phase_id']!=CLOCK_CONTRACT['phase_id'] or grant['maximum_gpu_seconds']!=28800):
        raise ValueError('Exact original scope/grant required, no new allowance')
    ticket=audit.json(dict(path=str(a.allocation),sha256=a.allocation_sha256))
    phase=Path(ticket['phase_output_root']).resolve(strict=True);clock_root=phase/'gpu-clock'
    if (Path(ticket['clock_root']).resolve(strict=True)!=clock_root or not a.cache_root.resolve().is_relative_to(phase)
            or a.cache_root.resolve()!=phase/'partial11-cache-r1'
            or a.output.resolve().parent!=a.cache_root.resolve() or a.output.name!=a.run_id):
        raise ValueError('One dedicated cache root/fresh child inside original phase storage')
    if a.inputs_sha256!=profile['inputs_sha256']:raise ValueError('Exact known11 input manifest required')
    inputs=audit.json(dict(path=str(a.inputs),sha256=a.inputs_sha256))
    if inputs['accepted04_ref'] is not None:raise ValueError('This executable may only prepare PARTIAL11, independently of04')
    bank=bank_api.assemble_bank(inputs['demos'],inputs['interface'],inputs['result_refs'],audit=audit,accepted04_ref=None)
    if (bank_api.validate_bank(bank)!=IDS or bank['fixed12_fit_eligible'] is not False
            or any(e['record']['chunks']!=7 or e['record']['primitive_steps']!=193 for e in bank['entries'] if e['record'])):
        raise ValueError('Exact known11/77chunks/2123 historical actions required')
    have_resume=a.resume_cursor is not None
    if any(x is not None for x in (a.resume_cursor,a.resume_cursor_sha256,a.resume_closeout,a.resume_closeout_sha256)) and not all(
            x is not None for x in (a.resume_cursor,a.resume_cursor_sha256,a.resume_closeout,a.resume_closeout_sha256)):
        raise ValueError('Resume requires exact cursor AND actual allocation closeout refs')
    predecessor=None;cost=0;recovery=None
    if have_resume:
        predecessor=audit.json(dict(path=str(a.resume_cursor),sha256=a.resume_cursor_sha256))
        validate_cursor(predecessor,bank,source=source,profile_sha=a.profile_sha256,cache_root=a.cache_root)
        closeout=audit.json(dict(path=str(a.resume_closeout),sha256=a.resume_closeout_sha256));cost=prior_cost(predecessor,closeout)
        ref=predecessor['recovery'];audit.bound(ref['path'],ref['sha256'])
        recovery=torch.load(ref['path'],map_location='cpu',weights_only=False)
        if recovery['state']!={k:v for k,v in predecessor.items() if k!='recovery'}:
            raise ValueError('Full cache cursor/global-RNG recovery differs')
        if predecessor['run_id']==a.run_id:raise ValueError('A continuation needs a fresh output/run identity')
        if predecessor['status']=='failed_partial_preserved':
            diagnosis=audit.json(ticket['resume_diagnosis'])
            if diagnosis['cursor_sha256']!=a.resume_cursor_sha256 or not diagnosis.get('reason','').strip():
                raise ValueError('Diagnose a failed attempt before a fresh residual continuation')
    clock=json.loads((clock_root/'worker-time.json').read_text())
    validate_clock(ticket,clock,now=time.time(),source=source,run_id=a.run_id,output=a.output,cache_root=a.cache_root,prior_cache_seconds=cost)
    if have_resume:validate_prior_in_clock(clock,closeout)
    if file_sha(Path(bank_api.__file__))!=profile['bank_module_sha256']:
        raise ValueError('Require byte-exact carried061874 bank implementation')
    if ticket['profile_sha256']!=a.profile_sha256:raise ValueError('Actual ticket does not bind this exact cache profile')
    audit=Audit(ticket['driver_deadline_unix']-60)
    if not have_resume and a.cache_root.exists() and any(a.cache_root.iterdir()):
        raise ValueError('Existing partial-cache work must resume, never reset its1200-second/storage budget')
    a.cache_root.mkdir(parents=True,exist_ok=True)
    lock=(a.cache_root/'cache.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    backend=None;state=None;persist=None
    try:
        if a.cache_root.stat().st_dev!=phase.stat().st_dev or os.statvfs(a.cache_root).f_bavail*os.statvfs(a.cache_root).f_frsize<16<<30:
            raise ValueError('Shared-device actual free16GiB required before data/model work')
        probe_path=a.cache_root/(a.run_id+'.write-probe')
        with probe_path.open('xb') as stream:stream.write(b'partial11-write-probe');stream.flush();os.fsync(stream.fileno())
        probe_path.unlink()
        a.output.mkdir(exist_ok=False)
        state=dict(format=FORMAT,status='admitted_partial',source_commit=source,profile_sha256=a.profile_sha256,
            bank_sha256=bank_api.digest(bank),run_id=a.run_id,resource_attempt=ticket['resource_attempt'],
            prior_cache_gpu_seconds=cost,complete_cases=[] if predecessor is None else deepcopy(predecessor['complete_cases']),
            records=[] if predecessor is None else deepcopy(predecessor['records']),active_case=None,
            predecessor_cursor=None if predecessor is None else dict(path=str(a.resume_cursor),sha256=a.resume_cursor_sha256),
            new_robot_actions=0,fit_eligible=False,full12_eligible=False,model_qualification=None)
        def guard():
            if time.time()+60>=ticket['driver_deadline_unix']:raise TimeoutError('Original all-in cache clock')
            if _storage(a.cache_root)+(256<<20)>12<<30 or _storage(phase)+(256<<20)>64<<30:
                raise RuntimeError('Aggregate12GiB cache /64GiB phase storage cap')
        def persist():
            publish_cursor(a.output,state,rng_state())
            _atomic_json(a.output/'partial-cache.json',dict(format='robotwin_fixed12_cache_receipt_v1',bank=bank,
                records=state['records'],model_qualification=state['model_qualification'],complete=False,fit_eligible=False,
                cached_teacher_ids=state['complete_cases'],meaning='PARTIAL11 ONLY; never complete-bank or fit admission'),replace=True)
        try:
            guard();probe=audit.json(dict(path=str(a.gpu_probe),sha256=file_sha(a.gpu_probe)))
            if probe.get('status')!='passed' or probe['hostname']!=socket.gethostname() or len(probe['devices'])!=1:
                raise ValueError('Actual one-GPU allocation probe required')
            current_nvml=verify_selected_cuda(torch);current=selected_device()
            if current['uuid']!=current_nvml or current['cuda_uuid']!=probe['devices'][0]['cuda_uuid']:
                raise ValueError('Current selected GPU differs from saved probe')
            interface=validate_interface(audit.json(inputs['interface']))
            if file_sha(a.model_path/'model.safetensors')!=MOTOR_SHA or file_sha(a.norm_stats)!=NORM_SHA:
                raise ValueError('Original motor/norm bytes required')
            guard()
            backend=PartialCacheBackend(a.model_path,a.norm_stats,a.robotwin_root,a.rlinf_root,
                device='cuda',episode_limit=400,action_horizon=32,num_steps=5,output_dir=a.output/'runtime')
            if backend.env is not None:raise RuntimeError('Lazy backend unexpectedly constructed physics')
            frozen=backend.frozen_fingerprint()
            if frozen!=bank_api.FROZEN or backend.contract['norm_stats_sha256']!=interface['normalization_sha256']:
                raise ValueError('Actual loaded frozen motor/norm differs')
            qualification=dict(format='robotwin_model_only_cache_qualification_v1',status='passed',hostname=socket.gethostname(),
                gpu_probe=dict(path=str(a.gpu_probe),sha256=file_sha(a.gpu_probe)),cuda_uuid=current['cuda_uuid'],
                motor_sha256=MOTOR_SHA,normalization_sha256=NORM_SHA,frozen_parameter_sha256=frozen)
            bank_api._bind_current_cuda(qualification,probe)
            path=a.output/'model-qualification.json';_atomic_json(path,qualification)
            state['model_qualification']=dict(path=str(path),sha256=file_sha(path));state['frozen_before']=frozen
            if recovery is not None:restore_rng(recovery['global_rng'])
            state['status']='caching_partial11';persist()
            demos=audit.json(inputs['demos']);by_id={e['id']:e for e in demos['episodes']}
            def cache_one(entry):
                guard()
                rows=bank_api._cache_episode(entry,by_id[entry['id']],demos,backend,a.output/entry['id'],
                    deadline=ticket['driver_deadline_unix']-60,guard=guard)
                for row in rows:
                    if Path(row['path']).stat().st_size>128<<20:raise ValueError('Qualified per-sample storage bound exceeded')
                guard();return rows
            run_cases(bank,state,cache_one,persist,should_start=lambda:time.time()+120<ticket['driver_deadline_unix'])
            if backend.env is not None or backend.frozen_fingerprint()!=frozen:raise RuntimeError('Cache created physics or changed motor')
            state['frozen_after']=frozen;state['output_bytes']=_storage(a.cache_root);persist()
        except BaseException as error:
            state.update(status='failed_partial_preserved',failure=dict(type=type(error).__name__,message=str(error)))
            raise
        finally:
            if state is not None:persist()
    finally:
        try:
            if backend is not None:backend.close()
            if state is not None:
                state['cleanup_error']=None;persist()
        except BaseException as error:
            if state is not None:
                state.update(status='failed_partial_preserved',cleanup_error=dict(type=type(error).__name__,message=str(error)));persist()
            raise
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN);lock.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
