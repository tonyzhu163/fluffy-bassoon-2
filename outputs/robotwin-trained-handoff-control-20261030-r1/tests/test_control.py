"""Actual Linux fake-process regressions. Run only in an allocated CPU job."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import pytest

MODULE=Path(__file__).resolve().parents[1]/'control.py'
spec=importlib.util.spec_from_file_location('robotwin_owner_control',MODULE)
control=importlib.util.module_from_spec(spec);sys.modules[spec.name]=control;spec.loader.exec_module(control)
pytestmark=pytest.mark.skipif(sys.platform!='linux' or not hasattr(os,'pidfd_open')
    or not hasattr(signal,'pidfd_send_signal'),reason='Actual allocated Linux pidfd regressions')

DRIVER=r'''
import argparse,json,os,time
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path);p.add_argument('--fake-delay',type=float)
a,_=p.parse_known_args();a.output.mkdir(parents=True)
(a.output/'started.json').write_text(json.dumps({'pid':os.getpid()}))
time.sleep(a.fake_delay)
(a.output/'summary.json').write_text(json.dumps({'status':'wall_budget_stop','closed':True}))
(a.output/'finished').write_text('complete')
'''
SUPERVISOR=r'''
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('source-commit','run-id','method','code-root','output'):p.add_argument('--'+name)
p.add_argument('--fake-delay',type=float);a=p.parse_args()
root=Path(a.output);root.mkdir()
for n in range(1,50):
 d=root/'slices'/f'{n:04d}';d.mkdir(parents=True)
 command=[sys.executable,str(Path(a.code_root)/'scripts/diagnostics/robotwin_noise_compare.py'),
 '--output',str(d/'experiment'),'--source-commit',a.source_commit,
 '--lineage-source-commit','db0184089fa88a5f96c153f9b80029bb91b77def',
 '--methods',a.method,'--max-source-trials-per-process','6','--recycle-at-round-boundary',
 '--fake-delay',str(a.fake_delay)]
 (d/'launch.json').write_text(json.dumps({'command':command}))
 with (d/'console.log').open('w') as f:
  child=subprocess.Popen(command,stdout=f,stderr=f,start_new_session=True)
  (d/'child-pid').write_text(str(child.pid))
  code=child.wait()
 (d/'exit.json').write_text(json.dumps({'output':str(d/'experiment'),'exit_code':code}))
'''


def wait_until(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while time.monotonic()<deadline:
        result=predicate()
        if result:return result
        time.sleep(.01)
    raise AssertionError('Owned fake process did not reach its expected state')


@pytest.fixture
def owned(tmp_path):
    running=[]
    def start(delay=.8):
        root=tmp_path/f'fake-{len(running)}';root.mkdir();code=root/'code'
        driver=code/'scripts/diagnostics/robotwin_noise_compare.py';driver.parent.mkdir(parents=True);driver.write_text(DRIVER)
        script=root/'supervise.py';script.write_text(SUPERVISOR);output=root/'owned-output'
        command=[sys.executable,str(script),'--source-commit','e825821a7c2231952bcf1042162f3470bd8ecdea',
            '--run-id','owned-output','--method','ppo','--code-root',str(code),'--output',str(output),
            '--fake-delay',str(delay)]
        parent=subprocess.Popen(command,start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        wait_until(lambda:(output/'slices/0001/experiment/started.json').exists())
        snapshot=control.proc(parent.pid)
        owner=control.Owner(socket.gethostname(),parent.pid,snapshot['start_ticks'],script,control.sha(script),
            Path(sys.executable),'owned-output','ppo','e825821a7c2231952bcf1042162f3470bd8ecdea',code,output,False)
        running.append((parent,owner));return owner
    yield start
    for parent,owner in running:
        if parent.poll() is None:
            fd=os.pidfd_open(parent.pid,0)
            try:signal.pidfd_send_signal(fd,signal.SIGKILL,None,0)
            finally:os.close(fd)
            parent.wait(timeout=5)
        # Fake finite children have their own sessions. Scope cleanup to this
        # unique tmp driver path and recheck start identity through pidfd.
        for pid in control.pids():
            value=control.proc(pid)
            if value is None or len(value['argv'])<2 or Path(value['argv'][1])!=owner.code_root/'scripts/diagnostics/robotwin_noise_compare.py':continue
            try:
                fd=os.pidfd_open(pid,0);again=control.proc(pid)
                if again is not None and again['start_ticks']==value['start_ticks']:
                    signal.pidfd_send_signal(fd,signal.SIGTERM,None,0)
                os.close(fd)
            except ProcessLookupError:pass


def fake_audit(experiment,owner,children):
    value=json.loads((experiment/'summary.json').read_text())
    assert value=={'status':'wall_budget_stop','closed':True}
    assert (experiment/'finished').exists()
    return dict(experiment=str(experiment),fake_only=True)


def test_real_pause_allows_finite_child_to_finish_and_prevents_next(owned):
    owner=owned()
    result=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01,
        audit=lambda scope,children:control.select_latest(scope,children,auditor=fake_audit))
    assert result['status']=='paused_child_complete_handoff_verified'
    assert control.proc(owner.pid)['state']=='T'
    assert (owner.output/'slices/0001/experiment/finished').exists()
    time.sleep(.15)
    assert not (owner.output/'slices/0002').exists()
    assert [row['signal'] for row in result['signals']]==['SIGSTOP']
    assert result['source_children_inactive'] and not result['permission_to_start_new_writer']
    assert result['handoff']['experiment'].endswith('0001/experiment')


@pytest.mark.parametrize('field',['hostname','start_ticks','run_id','method','source_commit','code_root','output'])
def test_wrong_scope_is_rejected_without_signalling_another_process(owned,field):
    owner=owned(delay=2.)
    value=getattr(owner,field)
    wrong=(value+1 if isinstance(value,int) else value/'wrong' if isinstance(value,Path) else value+'-wrong')
    with pytest.raises(control.ControlError):control.pause_and_drain(replace(owner,**{field:wrong}),wait_seconds=.1)
    assert control.proc(owner.pid)['state'] not in ('T','t')
    child=control.direct_children(owner)[0]
    assert child['state'] not in ('T','t')


def test_wrong_pid_rejects_foreign_process_and_preserves_owned_supervisor(owned):
    owner=owned(delay=2.)
    foreign=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
    try:
        snapshot=control.proc(foreign.pid)
        with pytest.raises(control.ControlError):
            control.pause_and_drain(replace(owner,pid=foreign.pid,start_ticks=snapshot['start_ticks']),wait_seconds=.1)
        assert control.proc(foreign.pid)['state'] not in ('T','t')
        assert control.proc(owner.pid)['state'] not in ('T','t')
    finally:foreign.terminate();foreign.wait(timeout=5)


def test_timeout_rollback_revalidates_and_resumes_only_our_supervisor(owned):
    owner=owned(delay=1.)
    with pytest.raises(control.ChildWaitTimeout) as failure:
        control.pause_and_drain(owner,wait_seconds=.05,poll_seconds=.01,rollback_on_timeout=True)
    receipt=failure.value.receipt
    assert receipt['status']=='timeout_rolled_back' and receipt['rollback_verified']
    assert [row['signal'] for row in receipt['signals']]==['SIGSTOP','SIGCONT']
    wait_until(lambda:control.proc(owner.pid)['state'] not in ('T','t'))
    assert control.direct_children(owner)[0]['state'] not in ('T','t')
    wait_until(lambda:(owner.output/'slices/0002/experiment/started.json').exists())


def test_default_timeout_leaves_parent_stopped_without_signalling_source(owned):
    owner=owned(delay=1.)
    with pytest.raises(control.ChildWaitTimeout) as failure:
        control.pause_and_drain(owner,wait_seconds=.05,poll_seconds=.01)
    assert failure.value.receipt['parent_left_stopped']
    assert [row['signal'] for row in failure.value.receipt['signals']]==['SIGSTOP']
    assert control.proc(owner.pid)['state']=='T'
    assert control.direct_children(owner)[0]['state'] not in ('T','t')


def test_newest_missing_proof_never_falls_back_to_stale_boundary(owned):
    owner=owned()
    receipt=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    first=owner.output/'slices/0001'
    (first/'exit.json').write_text(json.dumps({'output':str(first/'experiment'),'exit_code':0}))
    latest=owner.output/'slices/0002';latest.mkdir();(latest/'console.log').write_text('started but no durable proof')
    with pytest.raises(control.ControlError,match='do not select an older cursor'):
        control.select_latest(owner,receipt['owned_children'],auditor=fake_audit)


def test_pause_before_popen_can_select_latest_closed_but_not_first_stale(owned):
    owner=owned()
    receipt=control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    second=owner.output/'slices/0002';(second/'experiment').mkdir(parents=True)
    (second/'experiment/summary.json').write_text(json.dumps({'status':'wall_budget_stop','closed':True}))
    (second/'experiment/finished').write_text('complete')
    (second/'exit.json').write_text(json.dumps({'output':str(second/'experiment'),'exit_code':0}))
    third=owner.output/'slices/0003';third.mkdir();(third/'launch.json').write_text('{}')
    result=control.select_latest(owner,[],auditor=fake_audit)
    assert result['experiment'].endswith('0002/experiment')
    assert result['inert_unstarted_slices']==[str(third)]


def test_stale_start_identity_cannot_send_rollback_continue(owned):
    owner=owned()
    control.pause_and_drain(owner,wait_seconds=4,poll_seconds=.01)
    fd=os.pidfd_open(owner.pid,0);sent=[]
    try:
        with pytest.raises(control.ControlError,match='ownership mismatch'):
            control.checked_signal(replace(owner,start_ticks=owner.start_ticks+1),fd,signal.SIGCONT,sent)
    finally:os.close(fd)
    assert sent==[] and control.proc(owner.pid)['state']=='T'
