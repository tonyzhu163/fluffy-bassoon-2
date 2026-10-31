"""Future wrapper guards; allocated Linux CPU only, no simulator/model calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from fluffy_bassoon.meta_rl import robotwin_history_live as live


def test_caller_checksum_cannot_replace_the_fixed_plan(tmp_path,monkeypatch):
    path=tmp_path/'changed-plan.json';path.write_text('{"different_100_update_endpoint":true}')
    called=[]; monkeypatch.setattr(live,'read_checked',lambda *args:called.append(args))
    with pytest.raises(ValueError,match='immutable fixed-panel PLAN_SHA'):
        live.read_fixed_plan(path,live.sha(path))
    assert not called


def test_original_pinned_plan_remains_readable():
    path=live.ROOT/'outputs/robotwin-native-evaluation-plan-20261030-r1/plan.json'
    value=live.read_fixed_plan(path,live.PLAN_SHA)
    assert value['max_actions']==57600 and value['max_episodes']==144


@pytest.mark.parametrize('actual_pid,ticks',[(99,20),(100,21)])
def test_missing_or_reused_parent_before_arm_never_configures_signal(actual_pid,ticks):
    calls=[]
    with pytest.raises(RuntimeError,match='parent died or its identity changed'):
        live.arm_parent_death_guard(100,20,configure=lambda value:calls.append(value),
            current_parent=lambda:actual_pid,identity=lambda pid:ticks)
    assert not calls


@pytest.mark.parametrize('reparented,reused',[(True,False),(False,True)])
def test_parent_death_or_reuse_during_prctl_is_caught_after_arm(reparented,reused):
    armed=[]; state=dict(pid=100,ticks=20)
    def configure(value):
        armed.append(value)
        if reparented: state['pid']=1
        if reused: state['ticks']=21
        return signal.SIGKILL
    with pytest.raises(RuntimeError,match='parent died or its identity changed'):
        live.arm_parent_death_guard(100,20,configure=configure,
            current_parent=lambda:state['pid'],identity=lambda pid:state['ticks'])
    assert armed==[signal.SIGKILL]


def test_readback_must_confirm_sigkill():
    with pytest.raises(RuntimeError,match='signal readback differs'):
        live.arm_parent_death_guard(100,20,configure=lambda value:signal.SIGTERM,
            current_parent=lambda:100,identity=lambda pid:20)
    receipt=live.arm_parent_death_guard(100,20,configure=lambda value:value,
        current_parent=lambda:100,identity=lambda pid:20)
    assert receipt['readback_verified'] and receipt['parent_death_signal']=='SIGKILL'


def test_fresh_child_command_carries_the_real_parent_start_identity(tmp_path):
    args=type('Args',(),dict(plan=tmp_path/'plan.json',plan_sha=live.PLAN_SHA,
        campaign_root=tmp_path/'ledger',campaign_id='fake',source_commit='a'*40,
        expected_hostname='job-fake-master-0',robotwin_root=tmp_path/'robotwin',
        rlinf_root=tmp_path/'rlinf',run_id='fake'))()
    command=live.child_command(args,0,'native','adjust_bottle',tmp_path/'unit',None)
    assert command[command.index('--parent-pid')+1]==str(os.getpid())
    assert command[command.index('--parent-start-ticks')+1]==str(live.process_start_ticks(os.getpid()))


@pytest.mark.skipif(sys.platform!='linux',reason='Linux kernel parent-death semantics')
def test_sigkill_of_owned_test_parent_kills_independent_session_worker(tmp_path):
    """Kill only this test's parent; no wrapper cleanup can cause the result."""
    child_code='''
import os,sys,time
from pathlib import Path
from fluffy_bassoon.meta_rl.robotwin_history_live import arm_parent_death_guard,process_start_ticks,write
root=Path(sys.argv[1]);parent=int(sys.argv[2]);ticks=int(sys.argv[3])
guard=arm_parent_death_guard(parent,ticks)
write(root/'ready.json',dict(pid=os.getpid(),start_ticks=process_start_ticks(os.getpid()),sid=os.getsid(0),guard=guard))
time.sleep(60)
'''
    parent_code=f'''
import os,subprocess,sys,time
from pathlib import Path
from fluffy_bassoon.meta_rl.robotwin_history_live import process_start_ticks,write
root=Path(sys.argv[1]);child_code={child_code!r}
child=subprocess.Popen([sys.executable,'-c',child_code,str(root),str(os.getpid()),str(process_start_ticks(os.getpid()))],start_new_session=True)
write(root/'spawned.json',dict(pid=child.pid,start_ticks=process_start_ticks(child.pid)))
time.sleep(60)
'''
    environment=os.environ.copy();environment['PYTHONPATH']=str(live.ROOT/'src')
    with (tmp_path/'process.log').open('w') as stream:
        parent=subprocess.Popen([sys.executable,'-c',parent_code,str(tmp_path)],env=environment,
            stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            until=time.monotonic()+15.
            while not (tmp_path/'ready.json').exists() and time.monotonic()<until:
                if parent.poll() is not None: break
                time.sleep(.05)
            assert (tmp_path/'ready.json').exists(),(tmp_path/'process.log').read_text()
            child=json.loads((tmp_path/'ready.json').read_text())
            assert child['sid']==child['pid'] and child['guard']['parent_pid']==parent.pid
            assert child['guard']['readback_verified']
            parent.kill();parent.wait(timeout=5) # This test's owned PID, not the child's PGID.
            until=time.monotonic()+5.
            while live.live_group(child['pid']) and time.monotonic()<until: time.sleep(.05)
            assert not live.live_group(child['pid']),'PDEATHSIG did not stop the independent-session worker'
        finally:
            if parent.poll() is None: parent.kill();parent.wait(timeout=5)
            spawned=tmp_path/'spawned.json'
            if spawned.exists():
                owned=json.loads(spawned.read_text())
                try:
                    if live.process_start_ticks(owned['pid'])==owned['start_ticks'] and os.getsid(owned['pid'])==owned['pid']:
                        live.cleanup_group(owned['pid'])
                except (ProcessLookupError,FileNotFoundError): pass
