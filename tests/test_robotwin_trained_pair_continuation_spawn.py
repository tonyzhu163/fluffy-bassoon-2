"""Prepared real Linux process regression; no models, GPUs or allocations."""
import json
import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import textwrap

import pytest


MODULE=(Path(__file__).parents[1]/"scripts/diagnostics/robotwin_trained_pair_continuation.py").resolve()
pytestmark=pytest.mark.skipif(sys.platform!="linux",reason="Owned Linux process CPU gate only")


WORKER=textwrap.dedent(r'''
import importlib.util,json,os,signal,subprocess,sys,time
from pathlib import Path
module_path=Path(sys.argv[1]);output=Path(sys.argv[2]);case=sys.argv[3]
sys.path.insert(0,str(module_path.parent))
spec=importlib.util.spec_from_file_location("spawn_race_qualified_continuation",module_path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
runtime=sys.modules[module._Guardian.__module__]
class Guardian(module.ContinuationGuardian):
    def parent_alive(self):pass
guardian=Guardian(dict(parent_pid=os.getppid(),output=str(output),deadline=time.monotonic()+20,cleanup_reserve=5))
state=dict(callback=False,delivered=False,spawn_pid=None,identity=None)
def terminated(number,frame):
    assert state["callback"],"guardian termination raised before durable launch callback"
    assert len(guardian.sessions)==1,"guardian termination raised before canonical registration"
    state["delivered"]=True
    raise RuntimeError("pending termination delivered after registration")
signal.signal(signal.SIGTERM,terminated);signal.signal(signal.SIGINT,terminated)

grand_code="""import json,os,signal,time;from pathlib import Path
blocked=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('SigBlk:')),16)
Path(os.environ['SPAWN_TEST_OUTPUT'],'grandchild.json').write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),blocked=blocked,term_default=signal.getsignal(signal.SIGTERM)==signal.SIG_DFL)))
time.sleep(60)
"""
child_code="""import json,os,signal,subprocess,sys,time;from pathlib import Path
blocked=int(next(line.split()[1] for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('SigBlk:')),16)
Path(os.environ['SPAWN_TEST_OUTPUT'],'child.json').write_text(json.dumps(dict(pid=os.getpid(),sid=os.getsid(0),blocked=blocked,term_default=signal.getsignal(signal.SIGTERM)==signal.SIG_DFL)))
subprocess.Popen([sys.executable,'-c',os.environ['SPAWN_TEST_GRAND_CODE']])
time.sleep(60)
"""
original_stat=runtime._stat
def after_popen_before_append(pid):
    if state["spawn_pid"] is None:
        state["spawn_pid"]=pid
        (output/'spawn-pid.txt').write_text(str(pid))
        state['identity']=original_stat(pid)
        (output/'spawn-identity.json').write_text(json.dumps(state['identity']))
        until=time.monotonic()+5
        while not ((output/'child.json').exists() and (output/'grandchild.json').exists()):
            if time.monotonic()>=until:raise RuntimeError("fake native child did not become ready")
            time.sleep(.01)
        assert guardian.sessions==[],"injection must occur after Popen but before canonical append"
        if case!='callback_failure':
            os.kill(os.getpid(),signal.SIGTERM if case=='term' else signal.SIGINT)
    return original_stat(pid)
def registered(entry):
    assert guardian.sessions==[entry]
    module.atom(output/'launch.json',dict(identity=entry['identity'],command=entry['command']))
    state["callback"]=True
    if case=='callback_failure':raise ValueError("test durable callback failure")

runtime._stat=after_popen_before_append
error=None;cleanup=None
try:
    try:
        guardian.launch('fake-native',[sys.executable,'-c',child_code],str(output),
            dict(os.environ,CUDA_VISIBLE_DEVICES='',SPAWN_TEST_OUTPUT=str(output),SPAWN_TEST_GRAND_CODE=grand_code),
            output/'child.log',registered=registered)
    except (RuntimeError,ValueError) as caught:
        error=dict(kind=type(caught).__name__,message=str(caught))
    assert error is not None,"launch must propagate deferred termination or callback failure"
    assert state['callback'] and len(guardian.sessions)==1
    assert signal.getsignal(signal.SIGTERM) is terminated and signal.getsignal(signal.SIGINT) is terminated
    assert state['delivered']==(case!='callback_failure')
    runtime._stat=original_stat
    cleanup=guardian.cleanup(graceful=False)
    assert cleanup['clean'] and cleanup['sessions'][0]['returncode']==-signal.SIGTERM
    for name in ('child','grandchild'):
        row=json.loads((output/(name+'.json')).read_text())
        assert row['sid']==state['spawn_pid'] and row['term_default']
        assert not row['blocked']&(1<<(signal.SIGTERM-1))
        assert not row['blocked']&(1<<(signal.SIGINT-1))
        live=original_stat(row['pid'])
        assert live is None or live['state'] in ('Z','X')
    (output/'result.json').write_text(json.dumps(dict(error=error,cleanup=cleanup,callback=state['callback'],delivered=state['delivered'])))
finally:
    runtime._stat=original_stat
    # Failure cleanup is limited to this newly captured test SID. It cannot
    # turn a failed registration assertion into a passing regression.
    if guardian.sessions:
        guardian.cleanup(graceful=False)
    elif state['identity'] is not None:
        runtime._signal(state['identity'],signal.SIGKILL)
''')


def process_state(pid):
    try:return (Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()[0]
    except FileNotFoundError:return None


@pytest.mark.parametrize("case",["term","int","callback_failure"])
def test_signal_after_popen_stays_owned_and_children_keep_normal_signal_masks(tmp_path,case):
    unrelated=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)
    try:
        completed=subprocess.run([sys.executable,"-c",WORKER,str(MODULE),str(tmp_path),case],
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),capture_output=True,text=True,timeout=25)
        assert completed.returncode==0,completed.stdout+completed.stderr
        receipt=json.loads((tmp_path/"result.json").read_text())
        launch=json.loads((tmp_path/"launch.json").read_text())
        assert receipt["cleanup"]["clean"] and receipt["callback"]
        assert launch["identity"]["pid"]==int((tmp_path/"spawn-pid.txt").read_text())
        assert launch["identity"]["sid"]==launch["identity"]["pid"]
        assert launch["identity"]["start_ticks"]>0
        assert unrelated.poll() is None
        for name in ("child","grandchild"):
            row=json.loads((tmp_path/(name+".json")).read_text())
            assert process_state(row["pid"]) in (None,"Z","X")
    finally:
        unrelated.terminate();unrelated.wait(timeout=5)
        path=tmp_path/"spawn-identity.json"
        if path.exists():
            runtime_path=MODULE.with_name("robotwin_trained_pair_runtime.py")
            spec=importlib.util.spec_from_file_location("spawn_failure_cleanup",runtime_path)
            runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
            runtime._signal(json.loads(path.read_text()),signal.SIGKILL)
