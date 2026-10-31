"""Allocated-CPU fake regressions for the fresh K8 deployment wrapper."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_robotwin_history_evaluate import plan
from fluffy_bassoon.meta_rl import robotwin_history_live as live
from fluffy_bassoon.meta_rl.robotwin_history_evaluate import canonical_sha


def arguments(tmp_path, *, create=True):
    output=tmp_path/'evaluation'
    if create: output.mkdir(parents=True)
    return SimpleNamespace(output=output,source_commit='a'*40,plan_sha=live.PLAN_SHA,
        plan=tmp_path/'plan.json',resume_manifest=None,resume_sha=None,prior_gpu_seconds=0.,
        wall_seconds=5300.,unit_start_reserve=300.,final_reserve=60.,
        campaign_root=tmp_path/'ledger',campaign_id='cpu-fake-panel',expected_hostname='job-test-master-0',
        robotwin_root=tmp_path/'robotwin',rlinf_root=tmp_path/'rlinf',run_id='cpu-fake',
        condition='native',task='adjust_bottle',unit_index=0,native_result=None,native_result_sha=None)


def result(configuration, condition, task, cases=None):
    cases=cases or [dict(task=task,episode=i+1,requested_seed=seed,actual_seed=seed,
        observation_sha256='b'*64,features_sha256='c'*64) for i,seed in enumerate(configuration['cases'][task])]
    return dict(completed=True,task=task,context=dict(method=condition,evaluation_plan_sha256=canonical_sha(configuration)),
        reset_cases=cases,episodes=[dict(case,primitive_steps=1) for case in cases],actual_actions=8,
        motor_queries=8,offline_weights_unchanged=True)


class Worker:
    def __init__(self,configuration,args,*,actual=0):
        self.configuration=configuration; self.args=args; self.actual=actual; self.calls=[]
    def audit(self):
        return dict(actual_actions=self.actual,charged_actions=self.actual,pending_actions=0,uncertain_actions=0)
    def __call__(self,index,condition,task,unit,native,timeout):
        self.calls.append((index,condition,task)); assert timeout>0
        value=result(self.configuration,condition,task,None if native is None else native[0]['reset_cases'])
        (unit/'trial').mkdir(); live.write(unit/'trial/result.json',value)
        receipt=dict(completed=True,condition=condition,task=task,source_commit=self.args.source_commit,
            backend_sha256=live.BACKEND_SHA,parity=dict(passed=True),result_sha256=live.sha(unit/'trial/result.json'))
        path=unit/'unit-receipt.json'; live.write(path,receipt); self.actual+=8
        return dict(condition=condition,task=task,receipt_path=str(path),receipt_sha256=live.sha(path))


def test_exact18_fresh_k8_units_have_native_first_and_no_extra_cases(tmp_path):
    configuration=plan(); args=arguments(tmp_path); worker=Worker(configuration,args)
    summary=live.run_units(args,configuration,worker=worker,audit=worker.audit)
    assert summary['status']=='complete' and len(worker.calls)==18
    assert [(condition,task) for _,condition,task in worker.calls]==live.unit_order(configuration)
    assert worker.calls[:2]==[(0,'native','adjust_bottle'),(1,'native','move_stapler_pad')]
    assert summary['budget']['actual_actions']==144


def test_wall_boundary_resume_reuses_complete_native_units_without_new_charges(tmp_path,monkeypatch):
    configuration=plan(); args=arguments(tmp_path/'first'); args.wall_seconds=600.
    clock=[0.]; monkeypatch.setattr(live.time,'monotonic',lambda:clock[0])
    worker=Worker(configuration,args)
    def bounded(*values):
        record=worker(*values)
        if len(worker.calls)==2: clock[0]=500.
        return record
    summary=live.run_units(args,configuration,worker=bounded,audit=worker.audit,started=0.)
    assert summary['status']=='wall_boundary' and summary['completed_unit_count']==2
    resumed=arguments(tmp_path/'resumed'); resumed.prior_gpu_seconds=500.; resumed.wall_seconds=4000.
    resumed.resume_manifest=args.output/'summary.json'; resumed.resume_sha=live.sha(resumed.resume_manifest)
    clock[0]=0.; continuation=Worker(configuration,resumed,actual=16)
    final=live.run_units(resumed,configuration,worker=continuation,audit=continuation.audit,started=0.)
    assert final['status']=='complete' and len(continuation.calls)==16 and continuation.actual==144
    assert continuation.calls[0][1]=='native-zero'


def test_unknown_child_outcome_halts_and_failed_unit_is_never_retried(tmp_path):
    configuration=plan(); args=arguments(tmp_path); worker=Worker(configuration,args)
    def fault(index,*values):
        if index==2: raise RuntimeError('unknown native outcome')
        return worker(index,*values)
    with pytest.raises(RuntimeError,match='unknown native outcome'):
        live.run_units(args,configuration,worker=fault,audit=worker.audit)
    failed=args.output/'summary.json'; summary=live.read_checked(failed,live.sha(failed))
    assert summary['status']=='failed' and summary['completed_unit_count']==2 and len(worker.calls)==2
    with pytest.raises(ValueError,match='complete-unit wall-boundary'):
        live.resume_units(failed,live.sha(failed),configuration,args.source_commit,100.)


def test_recovery_rejects_tampered_result_and_underreported_compute(tmp_path):
    configuration=plan(); args=arguments(tmp_path); worker=Worker(configuration,args)
    summary=live.run_units(args,configuration,worker=worker,audit=worker.audit)
    path=args.output/'summary.json'
    with pytest.raises(ValueError,match='cumulative budget'):
        live.resume_units(path,live.sha(path),configuration,args.source_commit,0.)
    receipt=Path(summary['completed_units'][0]['receipt_path'])
    (receipt.parent/'trial/result.json').write_text('{}')
    with pytest.raises(ValueError,match='checksum'):
        live.resume_units(path,live.sha(path),configuration,args.source_commit,100.)


def test_wrong_session_cleanup_is_refused_without_signalling(monkeypatch):
    calls=[]; monkeypatch.setattr(live,'live_group',lambda pgid:[(123,999)])
    monkeypatch.setattr(live.os,'killpg',lambda *values:calls.append(values))
    with pytest.raises(RuntimeError,match='owned child session'): live.cleanup_group(123)
    assert not calls


def test_process_dispatch_creates_new_session_and_records_exit(tmp_path,monkeypatch):
    observed={}; cleaned=[]
    class Process:
        pid=123
        def wait(self,timeout): return 0
        def poll(self): return 0
    def popen(command,**kwargs): observed.update(kwargs); return Process()
    monkeypatch.setattr(live.subprocess,'Popen',popen)
    monkeypatch.setattr(live,'cleanup_group',lambda pid:cleaned.append(pid))
    report=live.run_command(['fake','child'],tmp_path/'stdout.log',10.)
    assert observed['start_new_session'] is True and 'shell' not in observed and cleaned==[123]
    assert report['returncode']==0 and not report['timed_out']


def test_process_timeout_cleans_owned_group_and_never_retries(tmp_path,monkeypatch):
    waited=[]; cleaned=[]
    class Process:
        pid=123
        def wait(self,timeout):
            waited.append(timeout)
            if len(waited)==1: raise live.subprocess.TimeoutExpired('fake',timeout)
            return -15
        def poll(self): return -15
    monkeypatch.setattr(live.subprocess,'Popen',lambda *a,**k:Process())
    monkeypatch.setattr(live,'cleanup_group',lambda pid:cleaned.append(pid))
    with pytest.raises(RuntimeError,match='Child failed; no retry'):
        live.run_command(['fake'],tmp_path/'timeout.log',1.)
    assert len(waited)==2 and cleaned==[123,123]


def test_child_closes_at_real_ledger_k8_boundary_and_no_step_occurs_in_parity(tmp_path):
    configuration=plan(); args=arguments(tmp_path,create=False); events=[]
    class Env:
        backend=SimpleNamespace(prefix_calls=17,motor_queries=9,output_dir=tmp_path/'runtime')
        def close(self): events.append('close')
    env=Env(); statistics=dict(pending_decision=None,unresolved_decisions=[],native_completed_chunks=0,executed_primitive_steps=0)
    def prepare(*values): events.extend(['construct','zero-action-parity']); return env,SimpleNamespace(statistics=lambda:statistics),dict(passed=True)
    def trial(runtime,budget,agent,configuration,**kwargs):
        events.append('K8'); value=result(configuration,'native','adjust_bottle')
        for index in range(8):
            intent=f'fake-physical-{index}'
            budget.reserve(intent,1,phase='target',method='native',task='adjust_bottle',episode=index,chunk=0,policy_version='native')
            budget.resolve(intent,1)
        statistics['executed_primitive_steps']=8
        kwargs['output'].mkdir(); live.write(kwargs['output']/'result.json',value)
        return value
    receipt=live.run_child(args,configuration,prepare=prepare,agent_loader=lambda *a,**k:object(),trial_runner=trial)
    assert receipt['completed'] and receipt['budget']['actual_actions']==8 and events==['construct','zero-action-parity','K8','close']


def test_child_exception_preserves_pending_reservation_and_stops_recovery(tmp_path):
    configuration=plan(); args=arguments(tmp_path,create=False); closed=[]
    env=SimpleNamespace(close=lambda:closed.append(True))
    def trial(runtime,budget,*values,**kwargs):
        budget.reserve('unknown-intent',32,phase='target',method='native',task='adjust_bottle',episode=0,chunk=0,policy_version='native')
        raise RuntimeError('native abort')
    with pytest.raises(RuntimeError,match='native abort'):
        live.run_child(args,configuration,prepare=lambda *a:(env,None,dict(passed=True)),
            agent_loader=lambda *a,**k:object(),trial_runner=trial)
    assert closed==[True] and not (args.output/'unit-receipt.json').exists()
    with pytest.raises(RuntimeError,match='Unknown/pending'): live.audit_ledger(args,configuration)


def test_insufficient_k8_allowance_never_loads_agent_or_prepares_runtime(tmp_path):
    configuration=plan(); args=arguments(tmp_path,create=False); calls=[]
    with live.ledger(args,configuration,'prior-fake-cost') as budget:
        budget.reserve('prior-fake-cost',54401,phase='target',method='native',task='adjust_bottle',
            episode=0,chunk=0,policy_version='native')
        budget.resolve('prior-fake-cost',54401)
        assert budget.remaining(phase='target',method='native')==3199
    with pytest.raises(RuntimeError,match='Insufficient complete K8 allowance'):
        live.run_child(args,configuration,prepare=lambda *a:calls.append('runtime'),
            agent_loader=lambda *a,**k:calls.append('agent'))
    assert not calls and not (args.output/'unit-receipt.json').exists()


def test_parent_orders_fake_tests_admission_gpu_probe_before_any_unit(tmp_path,monkeypatch):
    from fluffy_bassoon.meta_rl import robotwin_history_evaluate
    configuration=plan(); args=arguments(tmp_path,create=False); events=[]
    def command(argv,log,timeout,**kwargs):
        if 'pytest' in argv: events.append('tests')
        else:
            events.append('GPU probe'); live.write(args.output/'gpu-probe.json',dict(status='passed',devices=[dict(uuid='fake')]))
        return dict(returncode=0)
    monkeypatch.setattr(live,'run_command',command)
    monkeypatch.setattr(robotwin_history_evaluate,'admit_plan',lambda value:events.append('admission') or dict(passed=True))
    monkeypatch.setattr(live,'run_units',lambda *a,**k:events.append('units') or dict(status='complete'))
    live.run_parent(args,configuration)
    assert events==['tests','admission','GPU probe','units']
