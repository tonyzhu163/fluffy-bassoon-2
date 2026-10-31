"""Fresh-process deployment of the fixed native panel or source teacher K4s.

The execution owner launches this inside its reviewed allocation. This command
does not allocate hardware, expand the panel, or retry an unknown outcome.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[3]
BACKEND_SHA = 'eacc64b4daf8f78aeb5e023a2c4d2a2a040eacfd2e662335c118334c2ab50ff3'
PLAN_SHA = 'b911dd972cc30e6f6b02cacf831eafa84738bd66ebd032824bb5fc94d340287c'
SOURCE_PLAN_SHA = 'ce2d744ed97bb14e5068306807dd829543b898d5c203c7b712ab831b7ce23602'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): digest.update(block)
    return digest.hexdigest()


def write(path, value, *, replace=False):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'-',suffix='.partial',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
        if replace: os.replace(tmp,path)
        else: os.link(tmp,path)
        directory=os.open(path.parent,os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        Path(tmp).unlink(missing_ok=True)


def read_checked(path, expected):
    if sha(path)!=expected: raise ValueError('Artifact checksum differs: '+str(path))
    value=json.loads(Path(path).read_text())
    if sha(path)!=expected: raise ValueError('Artifact changed while reading')
    return value


def read_fixed_plan(path, expected):
    if expected!=PLAN_SHA:
        raise ValueError('Only the immutable fixed-panel PLAN_SHA is admitted; migration requires a reviewed code change')
    return read_checked(path,PLAN_SHA)


def read_source_teacher_plan(path, expected):
    if expected != SOURCE_PLAN_SHA:
        raise ValueError('Only the immutable source100 teacher PLAN_SHA is admitted')
    value = read_checked(path,SOURCE_PLAN_SHA)
    if value.get('format') != 'robotwin_source_teacher_evaluation_v1':
        raise ValueError('Source teacher profile cannot substitute the development panel')
    return value


def process_start_ticks(pid):
    """Linux /proc identity, including PID reuse protection."""
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return int(fields[19]) # Field22: starttime, following state at field3.


def _set_parent_death_signal(value):
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    libc.prctl.restype=ctypes.c_int
    zero=ctypes.c_ulong(0)
    if libc.prctl(ctypes.c_int(1),ctypes.c_ulong(value),zero,zero,zero)!=0:
        raise OSError(ctypes.get_errno(),'PR_SET_PDEATHSIG failed')
    observed=ctypes.c_int()
    if libc.prctl(ctypes.c_int(2),ctypes.byref(observed),zero,zero,zero)!=0:
        raise OSError(ctypes.get_errno(),'PR_GET_PDEATHSIG failed')
    return observed.value


def arm_parent_death_guard(parent_pid, start_ticks, *, configure=None, current_parent=None, identity=None):
    """Arm in the exec'd child, never a multithreaded parent's preexec_fn.

    Check the expected parent both before and after prctl: Linux does not send
    the signal retrospectively if the parent died before it was armed.
    """
    if sys.platform!='linux' or type(parent_pid) is not int or parent_pid<1 or type(start_ticks) is not int or start_ticks<1:
        raise ValueError('Require an explicit allocated Linux parent PID/start identity')
    configure=configure or _set_parent_death_signal
    current_parent=current_parent or os.getppid
    identity=identity or process_start_ticks
    def check():
        try:
            if current_parent()!=parent_pid or identity(parent_pid)!=start_ticks:
                raise RuntimeError('Owned parent died or its identity changed before runtime preparation')
        except (OSError,ValueError,IndexError) as error:
            raise RuntimeError('Owned parent identity is unavailable before runtime preparation') from error
    check()
    if configure(signal.SIGKILL)!=signal.SIGKILL:
        raise RuntimeError('Parent-death signal readback differs; refuse runtime preparation')
    check()
    return dict(parent_pid=parent_pid,parent_start_ticks=start_ticks,parent_death_signal='SIGKILL',
        readback_verified=True,race_checks='expected PPID/start identity before and after prctl')


def unit_order(plan):
    from .robotwin_history_evaluate import trial_profile
    spec=trial_profile(plan)
    return [(condition,task) for condition in spec['conditions'] for task in spec['tasks']]


def ledger(args, plan, suffix):
    from fluffy_bassoon.rl.robotwin_campaign_budget import CampaignBudget
    from .robotwin_history_evaluate import canonical_sha,trial_profile
    spec=trial_profile(plan)
    return CampaignBudget(args.campaign_root,campaign_id=args.campaign_id,total_actions=spec['max_actions'],
        source_caps={},phase_caps={spec['phase']:spec['max_actions']},
        contract=dict(evaluation_plan_sha256=canonical_sha(plan),native_backend_sha256=BACKEND_SHA,
                      deployment='source_teacher_K4_v1' if spec['source'] else 'fresh_K8_fixed100_v1'),
        run_id=f'{args.expected_hostname}:{args.output.name}:{suffix}',source_commit=args.source_commit)


def audit_ledger(args, plan):
    with ledger(args,plan,'audit') as budget: snapshot=budget.snapshot()
    if (snapshot['pending_actions'] or snapshot['uncertain_actions'] or snapshot['unresolved_decisions']
            or snapshot['charged_actions']!=snapshot['actual_actions']):
        raise RuntimeError('Unknown/pending physical outcome: halt without retry')
    return snapshot


def live_group(pgid):
    rows=[]
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rsplit(')',1)[1].split()
            if int(fields[2])==pgid and fields[0] not in ('Z','X'):
                rows.append((int(path.parent.name),int(fields[3])))
        except (OSError,ValueError,IndexError): continue
    return rows


def cleanup_group(pgid, *, grace=5.):
    """Only the new session created for this exact child can be signalled."""
    rows=live_group(pgid)
    if any(sid!=pgid for _,sid in rows): raise RuntimeError('Refuse cleanup outside the owned child session')
    if not rows: return
    try: os.killpg(pgid,signal.SIGTERM)
    except ProcessLookupError: return
    until=time.monotonic()+grace
    while time.monotonic()<until:
        rows=live_group(pgid)
        if any(sid!=pgid for _,sid in rows): raise RuntimeError('Owned session identity changed')
        if not rows: return
        time.sleep(.05)
    try: os.killpg(pgid,signal.SIGKILL)
    except ProcessLookupError: pass
    until=time.monotonic()+2.
    while live_group(pgid) and time.monotonic()<until: time.sleep(.05)
    if live_group(pgid): raise RuntimeError('Owned child processes survived bounded cleanup')


def run_command(command, log, timeout, *, environment=None):
    if timeout<=0: raise TimeoutError('No cumulative runtime remains')
    started=time.monotonic(); timed_out=False
    with Path(log).open('x') as stream:
        process=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,
            start_new_session=True)
        try:
            try: code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True; cleanup_group(process.pid); code=process.wait(timeout=3)
        finally:
            cleanup_group(process.pid)
            if process.poll() is None: process.wait(timeout=3)
    result=dict(command=command,pid=process.pid,returncode=code,timed_out=timed_out,
        elapsed_seconds=time.monotonic()-started,log=str(log))
    if timed_out or code!=0: raise RuntimeError('Child failed; no retry: '+json.dumps(result))
    return result


def verify_unit(record, plan, expected, natives, *, source_admission=None):
    from .robotwin_history_evaluate import canonical_sha,trial_profile,SOURCE_TASK_LIMITS,SOURCE_TEACHER_SHA
    spec=trial_profile(plan);count=spec['episodes']
    condition,task=expected
    limit=SOURCE_TASK_LIMITS[task] if spec['source'] else 400
    receipt=read_checked(record['receipt_path'],record['receipt_sha256'])
    result_path=Path(record['receipt_path']).parent/'trial/result.json'
    result=read_checked(result_path,receipt['result_sha256'])
    if (receipt.get('completed') is not True or receipt.get('condition')!=condition or receipt.get('task')!=task
            or receipt.get('backend_sha256')!=BACKEND_SHA or not receipt.get('parity',{}).get('passed')
            or result.get('completed') is not True or result.get('task')!=task
            or result.get('context',{}).get('method')!=condition
            or result['context'].get('evaluation_plan_sha256')!=canonical_sha(plan)
            or len(result.get('episodes',[]))!=count or len(result.get('reset_cases',[]))!=count
            or result.get('actual_actions')!=sum(row['primitive_steps'] for row in result['episodes'])
            or not 0<=result['actual_actions']<=count*limit):
        raise ValueError('Incomplete or incompatible trial unit receipt')
    for index,(case,episode) in enumerate(zip(result['reset_cases'],result['episodes'],strict=True)):
        if (case.get('task')!=task or case.get('episode')!=index+1
                or case.get('requested_seed')!=plan['cases'][task][index]
                or type(case.get('actual_seed')) is not int or not case.get('observation_sha256')
                or not case.get('features_sha256') or not 0<=episode['primitive_steps']<=limit):
            raise ValueError('Trial reset cases or durations differ')
    if spec['source']:
        context=result['context']
        if (source_admission is None or source_admission.get('teacher_sha256')!=SOURCE_TEACHER_SHA
                or source_admission.get('plan_sha256')!=canonical_sha(plan)
                or context.get('phase')!='teacher_qualification'
                or result.get('source_iteration')!=100
                or result.get('source_teacher_sha256')!=SOURCE_TEACHER_SHA
                or result.get('source_or_target_SGD_updates')!=0 or result.get('paid_sample_reuse') is not False):
            raise ValueError('Source teacher trial lacks its exact source-only admission')
        if condition!='native':
            sampling=result.get('sampling',{})
            if (context.get('policy_checkpoint_sha256')!=SOURCE_TEACHER_SHA
                    or context.get('policy_sha256')!=source_admission['actor_parameter_sha256']
                    or sampling.get('source_checkpoint_sha256')!=SOURCE_TEACHER_SHA
                    or sampling.get('convention')!='restore_fixed_source_private_torch_rng_per_task_K4'
                    or sampling.get('initial_private_rng_sha256')!=source_admission['sample_initial_private_rng_sha256']
                    or sampling.get('deterministic') is not (condition=='sac100-mean')
                    or sampling.get('optimizer_rng_used') is not False):
                raise ValueError('Source teacher actor or private sampling convention changed')
    if condition=='native': natives[task]=(result,str(result_path),receipt['result_sha256'])
    elif task not in natives or result['reset_cases']!=natives[task][0]['reset_cases']:
        raise ValueError('Condition is not paired to the completed native K8 case bank')
    return result


def resume_units(path, expected_sha, plan, source_commit, prior_gpu_seconds, *, source_admission=None):
    from .robotwin_history_evaluate import trial_profile
    expected_plan=SOURCE_PLAN_SHA if trial_profile(plan)['source'] else PLAN_SHA
    previous=read_checked(path,expected_sha)
    if (previous.get('status') not in ('complete','wall_boundary') or previous.get('source_commit')!=source_commit
            or previous.get('plan_sha256')!=expected_plan
            or prior_gpu_seconds<previous.get('cumulative_gpu_seconds_lower_bound',float('inf'))):
        raise ValueError('Only reviewed complete-unit wall-boundary recovery within the cumulative budget is supported')
    units=previous['completed_units']; natives={}; order=unit_order(plan)
    if len(units)>len(order): raise ValueError('Recovery contains extra units')
    for record,expected in zip(units,order):
        verify_unit(record,plan,expected,natives,source_admission=source_admission)
    return units,natives


def prepare_runtime(args, plan):
    """Qualified DB construction, one zero-action parity reset, no learners."""
    import numpy as np
    import torch
    from fluffy_bassoon.baige_gpu import verify_selected_cuda
    from fluffy_bassoon.rl.robotwin_corpus import RobotwinCorpusWriter
    from fluffy_bassoon.rl.robotwin_noise import RobotwinNoiseEnv
    from .robotwin_history_evaluate import assert_runtime,canonical_sha,trial_profile
    spec=trial_profile(plan)
    torch.set_num_threads(4); verify_selected_cuda(torch)
    helper_spec=importlib.util.spec_from_file_location('fb2_qualified_runtime_helpers',ROOT/'scripts/diagnostics/robotwin_noise_compare.py')
    helpers=importlib.util.module_from_spec(helper_spec); helper_spec.loader.exec_module(helpers)
    capture=plan['capture_qualification']; runtime=plan['runtime_contract']
    if sha(capture['path'])!=capture['sha256']: raise ValueError('Capture qualification checksum differs')
    qualified=helpers.validate_capture_qualification(capture['path'],model_path=plan['interface']['model_path'],
        norm_stats=runtime['norm_stats_path'])
    writer=RobotwinCorpusWriter(args.output/'corpus',dict(source_commit=args.source_commit,
        evaluation_plan_sha256=canonical_sha(plan),scope=('source-only frozen teacher qualification'
            if spec['source'] else 'frozen development evaluation only'),
        model_path=plan['interface']['model_path'],native_horizon=32))
    env=RobotwinNoiseEnv(model_path=plan['interface']['model_path'],norm_stats=runtime['norm_stats_path'],
        robotwin_root=args.robotwin_root,rlinf_root=args.rlinf_root,device='cuda',episode_limit=600,
        action_horizon=32,num_steps=5,task_config_name='demo_clean',output_dir=args.output/'runtime',
        final_observation_only=True,corpus_writer=writer,task_episode_limits=plan['interface']['task_episode_limits'])
    try:
        fingerprint=env.frozen_fingerprint()
        helpers.validate_loaded_capture_runtime(qualified,env.contract,fingerprint)
        assert_runtime(env,plan)
        env.set_recording_context(dict(phase='preflight',method=args.condition,
            trial_id=args.run_id+':parity',policy_version='qualified_zero_action_parity'))
        env.reset(args.task,plan['cases'][args.task][0])
        if env.last_reset_info.get('reset_policy_actions')!=0: raise RuntimeError('Unexpected reset policy actions')
        parity=env.check_noise_parity(np.zeros((32,32),np.float32))
        write(args.output/'qualification.json',dict(capture=qualified,parity=parity,
            preflight_reset=dict(env.last_reset_info),zero_physical_actions=True,
            parity_motor_inference_calls=2,parity_direct_native_prefix_calls=1))
        return env,writer,parity
    except BaseException:
        env.close(); raise


def run_child(args, plan, *, prepare=prepare_runtime, agent_loader=None, trial_runner=None):
    from .robotwin_history_evaluate import load_agent,renderer_records,run_trial,trial_profile,SOURCE_TASK_LIMITS
    spec=trial_profile(plan)
    order=unit_order(plan)
    if (type(args.unit_index) is not int or not 0<=args.unit_index<len(order)
            or order[args.unit_index]!=(args.condition,args.task)):
        raise ValueError('Child unit differs from the fixed case order')
    if args.condition!='native' and not args.native_result: raise ValueError('Native K8 receipt is required first')
    references=None if args.condition=='native' else read_checked(args.native_result,args.native_result_sha)
    args.output.mkdir(parents=True,exist_ok=True)
    write(args.output/'intent.json',dict(source_commit=args.source_commit,condition=args.condition,
        task=args.task,unit_index=args.unit_index,plan_sha256=args.plan_sha,run_id=args.run_id,
        parent_death_guard=getattr(args,'parent_death_guard',None)))
    env=None
    with ledger(args,plan,'unit') as budget:
        before=budget.snapshot()
        if before['pending_actions'] or before['uncertain_actions']: raise RuntimeError('Unknown physical outcome: stop')
        allowance=spec['episodes']*(SOURCE_TASK_LIMITS[args.task] if spec['source'] else 400)
        if budget.remaining(phase=spec['phase'],method=args.condition)<allowance:
            label='K4' if spec['source'] else 'K8'
            raise RuntimeError('Insufficient complete '+label+' allowance before model/runtime preparation')
        try:
            agent=(agent_loader or load_agent)(plan,args.condition,device='cuda')
            env,writer,parity=prepare(args,plan)
            result=(trial_runner or run_trial)(env,budget,agent,plan,condition=args.condition,task=args.task,
                output=args.output/'trial',run_id=args.run_id,references=references)
            after=budget.snapshot(); corpus=writer.statistics()
            if (after['pending_actions'] or after['uncertain_actions'] or corpus.get('pending_decision')
                    or corpus.get('unresolved_decisions') or corpus.get('native_completed_chunks')
                    or after['actual_actions']-before['actual_actions']!=result['actual_actions']
                    or corpus['executed_primitive_steps']!=result['actual_actions']):
                raise RuntimeError('Raw corpus or physical ledger did not close at the completed K8 boundary')
            runtime=dict(prefix_calls=env.backend.prefix_calls,backend_motor_queries=env.backend.motor_queries,
                extra_direct_native_parity_queries=1,total_motor_inference_calls=env.backend.motor_queries+1,
                renderer_records=renderer_records(env))
        finally:
            if env is not None: env.close()
        receipt=dict(completed=True,condition=args.condition,task=args.task,source_commit=args.source_commit,
            backend_sha256=BACKEND_SHA,plan_sha256=args.plan_sha,result_sha256=sha(args.output/'trial/result.json'),
            parity=parity,runtime=runtime,corpus=corpus,budget=after)
        write(args.output/'unit-receipt.json',receipt)
    return receipt


def child_command(args, index, condition, task, output, native):
    command=[sys.executable,'-m',__spec__.name,'--mode','child','--plan',str(args.plan),'--plan-sha',args.plan_sha,
        '--output',str(output),'--campaign-root',str(args.campaign_root),'--campaign-id',args.campaign_id,
        '--source-commit',args.source_commit,'--expected-hostname',args.expected_hostname,
        '--robotwin-root',str(args.robotwin_root),'--rlinf-root',str(args.rlinf_root),
        '--condition',condition,'--task',task,'--unit-index',str(index),'--run-id',f'{args.run_id}:u{index:02d}',
        '--parent-pid',str(os.getpid()),'--parent-start-ticks',str(process_start_ticks(os.getpid()))]
    if native: command+=['--native-result',native[1],'--native-result-sha',native[2]]
    if getattr(args,'profile','fixed100')=='source-teacher100':
        command+=['--profile','source-teacher100']
    return command


def run_units(args, plan, *, worker, audit, started=None, source_admission=None):
    from .robotwin_history_evaluate import trial_profile
    spec=trial_profile(plan);order=unit_order(plan)
    started=time.monotonic() if started is None else started
    units=[]; natives={}; status='wall_boundary'
    if args.resume_manifest:
        units,natives=resume_units(args.resume_manifest,args.resume_sha,plan,args.source_commit,args.prior_gpu_seconds,
            source_admission=source_admission)
    snapshot=audit()
    if snapshot['actual_actions']!=sum(read_checked(Path(r['receipt_path']).parent/'trial/result.json',
            read_checked(r['receipt_path'],r['receipt_sha256'])['result_sha256'])['actual_actions'] for r in units):
        raise RuntimeError('Ledger contains interactions outside the admitted completed-unit recovery')
    def save(state,error=None):
        value=dict(status=state,source_commit=args.source_commit,plan_sha256=args.plan_sha,
            completed_units=units,completed_unit_count=len(units),requested_unit_count=len(order),
            elapsed_seconds=time.monotonic()-started,
            cumulative_gpu_seconds_lower_bound=args.prior_gpu_seconds+time.monotonic()-started,
            budget=snapshot,error=error,additional_sgd_updates=0,
            execution=('18 sequential fresh source-only K4 Python processes' if spec['source'] else
                '18 sequential fresh K8 Python processes; operational refinement of the same immutable panel'))
        write(args.output/'progress.json',value,replace=True)
        return value
    save('running')
    try:
        for index,(condition,task) in enumerate(order):
            if index<len(units): continue
            remaining=args.wall_seconds-(time.monotonic()-started)
            if remaining<args.unit_start_reserve+args.final_reserve: break
            native=None if condition=='native' else natives.get(task)
            if condition!='native' and native is None: raise RuntimeError('Native case bank is not complete for this task')
            unit=args.output/'units'/f'{index:02d}-{condition}-{task}'; unit.mkdir(parents=True,exist_ok=False)
            record=worker(index,condition,task,unit,native,remaining-args.final_reserve-10)
            result=verify_unit(record,plan,(condition,task),natives,source_admission=source_admission)
            snapshot=audit()
            if snapshot['actual_actions']!=sum(read_checked(Path(r['receipt_path']).parent/'trial/result.json',
                    read_checked(r['receipt_path'],r['receipt_sha256'])['result_sha256'])['actual_actions'] for r in units)+result['actual_actions']:
                raise RuntimeError('Completed unit and cumulative physical ledger disagree')
            units.append(record); save('running')
        if len(units)==len(order): status='complete'
    except BaseException as error:
        try: snapshot=audit()
        except BaseException as audit_error: snapshot=dict(audit_failed=str(audit_error))
        summary=save('failed',dict(type=type(error).__name__,reason=str(error)))
        write(args.output/'summary.json',summary); raise
    summary=save(status); write(args.output/'summary.json',summary)
    return summary


def run_parent(args, plan):
    from .robotwin_history_evaluate import admit_plan,trial_profile
    spec=trial_profile(plan)
    started=time.monotonic(); args.output.mkdir(parents=True,exist_ok=False)
    write(args.output/'execution.json',dict(source_commit=args.source_commit,plan_sha256=args.plan_sha,
        hostname=socket.gethostname(),wall_seconds=args.wall_seconds,prior_gpu_seconds=args.prior_gpu_seconds,
        max_gpu_seconds=spec['max_gpu_seconds'],processes=len(unit_order(plan)),
        max_nominal_accepted_resets_per_process=spec['episodes']+1))
    environment=os.environ.copy(); environment.update(PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',FLUFFY_GPU_EXECUTION='baige',PYTHONDONTWRITEBYTECODE='1')
    environment.setdefault('CUDA_VISIBLE_DEVICES','0') # Qualified exclusive container fallback; retain any existing visibility.
    def remaining(): return args.wall_seconds-(time.monotonic()-started)-args.final_reserve-10
    checks=['tests/test_robotwin_history_live.py','tests/test_robotwin_history_evaluate.py',
        'tests/test_robotwin_parent_guard.py']
    if spec['source']: checks+=['tests/test_robotwin_source_teacher_profile.py']
    run_command([sys.executable,'-m','pytest','-q',*checks],
        args.output/'wrapper-tests.log',remaining(),environment=environment)
    # CPU-only admission creates no motor and leaves the parent's CUDA context unopened.
    admission=admit_plan(plan)
    write(args.output/'admission.json',admission)
    run_command([sys.executable,str(ROOT/'scripts/[train]gpu_probe.py'),'--expected-count','1',
        '--purpose','evaluation','--baige-container','--output',str(args.output/'gpu-probe.json')],
        args.output/'gpu-probe.log',remaining(),environment=environment)
    probe=json.loads((args.output/'gpu-probe.json').read_text())
    if probe.get('status')!='passed' or len(probe.get('devices',[]))!=1: raise RuntimeError('GPU probe failed')
    environment.update(LYCO_BAIGE_GPU_MAP=str(args.output/'gpu-probe.json'),
        LYCO_BAIGE_ISOLATED_UUID=probe['devices'][0]['uuid'])
    def worker(index,condition,task,unit,native,timeout):
        local=environment.copy(); local['FB2_ROBOTWIN_RENDER_RECORDS']=str(unit/'runtime/renderer-bindings')
        result=run_command(child_command(args,index,condition,task,unit,native),unit/'stdout.log',timeout,environment=local)
        write(unit/'process.json',result)
        receipt=unit/'unit-receipt.json'
        return dict(condition=condition,task=task,receipt_path=str(receipt),receipt_sha256=sha(receipt))
    return run_units(args,plan,worker=worker,audit=lambda:audit_ledger(args,plan),started=started,
        source_admission=admission if spec['source'] else None)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('parent','child'),default='parent')
    p.add_argument('--profile',choices=('fixed100','source-teacher100'),default='fixed100')
    for name in ('plan','output','campaign-root','robotwin-root','rlinf-root'):
        p.add_argument('--'+name,type=Path,required=True)
    for name in ('campaign-id','source-commit','expected-hostname'): p.add_argument('--'+name,required=True)
    p.add_argument('--plan-sha',default=PLAN_SHA); p.add_argument('--wall-seconds',type=float,default=5300.)
    p.add_argument('--prior-gpu-seconds',type=float,default=0.)
    p.add_argument('--unit-start-reserve',type=float,default=300.); p.add_argument('--final-reserve',type=float,default=60.)
    p.add_argument('--resume-manifest',type=Path); p.add_argument('--resume-sha')
    p.add_argument('--condition'); p.add_argument('--task'); p.add_argument('--unit-index',type=int)
    p.add_argument('--native-result',type=Path); p.add_argument('--native-result-sha'); p.add_argument('--run-id')
    p.add_argument('--parent-pid',type=int); p.add_argument('--parent-start-ticks',type=int)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.mode=='child':
        # Before imports/model admission, CUDA, Git checks or any simulator reset.
        args.parent_death_guard=arm_parent_death_guard(args.parent_pid,args.parent_start_ticks)
    source_profile=args.profile=='source-teacher100'
    expected_sha=SOURCE_PLAN_SHA if source_profile else PLAN_SHA
    cap=3600 if source_profile else 5400
    if args.plan_sha!=expected_sha: raise ValueError('Only the immutable selected-profile PLAN_SHA is admitted')
    if (sys.platform!='linux' or socket.gethostname()!=args.expected_hostname
            or not args.expected_hostname.startswith('job-')
            or not 0<=args.prior_gpu_seconds<cap or not 0<args.wall_seconds<=cap-args.prior_gpu_seconds
            or args.final_reserve<60 or args.unit_start_reserve<300
            or not all(math.isfinite(value) for value in (args.wall_seconds,args.prior_gpu_seconds,
                args.final_reserve,args.unit_start_reserve))):
        raise ValueError('Require the actual allocated Linux host and bounded cumulative runtime')
    if (subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()!=args.source_commit
            or subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()):
        raise ValueError('Require the exact clean execution commit')
    if sha(ROOT/'src/fluffy_bassoon/rl/robotwin_noise.py')!=BACKEND_SHA: raise ValueError('Native backend bytes differ')
    for key in ('plan','output','campaign_root','robotwin_root','rlinf_root'):
        setattr(args,key,getattr(args,key).resolve())
    if args.output.is_relative_to(ROOT): raise ValueError('Runtime output must be outside the source checkout')
    if bool(args.resume_manifest)!=bool(args.resume_sha): raise ValueError('Recovery needs its exact SHA')
    args.run_id=args.run_id or f'{args.expected_hostname}:{args.output.name}'
    plan=(read_source_teacher_plan if source_profile else read_fixed_plan)(args.plan,args.plan_sha)
    def terminate(signum,frame): raise TimeoutError('Owned K8 child received termination; no retry')
    signal.signal(signal.SIGTERM,terminate)
    try: return run_parent(args,plan) if args.mode=='parent' else run_child(args,plan)
    except BaseException as error:
        if args.output.exists() and not (args.output/'failure.json').exists():
            write(args.output/'failure.json',dict(type=type(error).__name__,reason=str(error),no_automatic_retry=True))
        raise


if __name__=='__main__': main()
