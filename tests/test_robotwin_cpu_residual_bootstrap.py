"""Allocated-Linux fixtures for the residual budget and actual CUA proof bridge."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/robotwin-sac65-cpu-compat-20261031-r2'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


job=load('fixture_residual_gate',ROOT/'gate_job.py')
bridge=load('fixture_actual_proof_bridge',ROOT/'dispatch_bootstrap.py')
PIN='a'*40


def fixture(tmp_path):
    inbox=tmp_path/'inbox';inbox.mkdir()
    ready=dict(pid=123,expires_at_unix=400)
    marker=dict(schema='robotwin_cpu_actual_ui_proof_ready_v1',run_id=bridge.RUN,
        code_commit=PIN,nonce='fixture-nonce',dispatcher_pid=123,captured_at_unix=110)
    ax='''1 text mwj_1031_robotwin_sac65_cpu_r2_aaaaaaa
2 text 运行中
3 text ID:  job-fixture123
4 text job-fixture123-master-0
5 text Worker
6 text 1
7 text 不使用加速芯片
8 text 8 核
9 text 64 GiB
'''
    (inbox/'proof-ready.json').write_text(json.dumps(marker))
    (inbox/'actual-running.png').write_bytes(b'\x89PNG\r\n\x1a\nfixture')
    (inbox/'actual-running.txt').write_text(ax)
    return inbox,ready,marker


def proof(inbox,ready,now=120):
    return bridge.prepare_proof(inbox,ready,nonce='fixture-nonce',code_commit=PIN,armed_at=100,now=now)


def test_exact_prior_charge_and_all_shortened_timers_are_bound():
    values=job.load_budget(REPO)['values']
    assert values['charged_scheduler_seconds']+values['scheduler_cap_seconds']==1649
    assert values['sealed_unused_seconds']==151
    assert values['global_work_seconds']==1400 and values['external_KILL_seconds']==1480


@pytest.mark.parametrize('field,value',[('charged_scheduler_seconds',0),('scheduler_cap_seconds',1800),
    ('global_work_seconds',1700),('external_TERM_seconds',1770),('external_KILL_seconds',1780),
    ('sealed_unused_seconds',0),('gpus',1),('run_id','old-run')])
def test_rehashed_wrong_budget_cannot_restore_spent_time(tmp_path,monkeypatch,field,value):
    root=tmp_path/'outputs'/job.RUN;root.mkdir(parents=True)
    budget=json.loads((ROOT/'budget.json').read_text());budget[field]=value
    raw=json.dumps(budget).encode();(root/'budget.json').write_bytes(raw)
    monkeypatch.setattr(job,'BUDGET_SHA',hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='Wrong original charge'):
        job.load_budget(tmp_path)


def test_predecessor_receipt_tampering_is_rejected(tmp_path):
    root=tmp_path/'outputs'/job.RUN;root.mkdir(parents=True)
    (root/'budget.json').write_bytes((ROOT/'budget.json').read_bytes())
    (root/'predecessor-scheduler-accounting.json').write_text('{}')
    with pytest.raises(ValueError,match='Prior scheduler cost'):
        job.load_budget(tmp_path)


def test_resource_values_and_identity_come_from_actual_ui(tmp_path):
    inbox,ready,_=fixture(tmp_path)
    resource,files=proof(inbox,ready)
    assert resource['allocation_id']=='job-fixture123'
    assert (resource['allocated_gpus'],resource['allocated_cpus'],resource['memory_gib'])==(0,8,64)
    assert resource['proof']['sha256']==hashlib.sha256(files['actual-running.png']).hexdigest()
    assert resource['ax_proof']['sha256']==hashlib.sha256(files['actual-running.txt']).hexdigest()


@pytest.mark.parametrize('case',['nonce','code','old-capture','expired','Failed','old-name',
    'ambiguous-id','wrong-host','4CPU','1GPU','16GiB','image-symlink','empty-image'])
def test_stale_requested_foreign_or_redirected_proof_never_issues_resource(tmp_path,case):
    inbox,ready,marker=fixture(tmp_path)
    if case in ('nonce','code','old-capture'):
        field,value={'nonce':('nonce','wrong'),'code':('code_commit','b'*40),
                     'old-capture':('captured_at_unix',99)}[case]
        marker[field]=value;(inbox/'proof-ready.json').write_text(json.dumps(marker))
    elif case=='expired':ready['expires_at_unix']=119
    elif case=='image-symlink':
        outside=tmp_path/'foreign.png';outside.write_bytes((inbox/'actual-running.png').read_bytes())
        (inbox/'actual-running.png').unlink();(inbox/'actual-running.png').symlink_to(outside)
    elif case=='empty-image':(inbox/'actual-running.png').write_bytes(b'')
    else:
        path=inbox/'actual-running.txt';ax=path.read_text()
        replacements={'Failed':('运行中','失败'),'old-name':('r2_aaaaaaa','r1_aaaaaaa'),
            'ambiguous-id':('3 text ID:  job-fixture123','3 text ID:  job-fixture123\n10 text ID:  job-foreign'),
            'wrong-host':('job-fixture123-master-0','job-foreign-master-0'),
            '4CPU':('8 核','4 核'),'1GPU':('不使用加速芯片','NVIDIA L20 * 1'),
            '16GiB':('64 GiB','16 GiB')}
        before,after=replacements[case];path.write_text(ax.replace(before,after))
    with pytest.raises(ValueError):proof(inbox,ready)


@pytest.mark.parametrize('field,value',[('run_id','old-run'),('code_commit','wrong'),
    ('hostname','job-other-master-0'),('work_deadline_seconds_from_start',1700),
    ('max_wait_seconds',300),('bootstrap_expires_at_unix',119)])
def test_late_or_foreign_actual_intent_is_rejected(tmp_path,field,value):
    inbox,ready,_=fixture(tmp_path);resource,_=proof(inbox,ready)
    intent=dict(run_id=bridge.RUN,code_commit=PIN,hostname=resource['hostname'],output=bridge.REMOTE_OUTPUT,
        max_wait_seconds=120,work_deadline_seconds_from_start=1400,bootstrap_expires_at_unix=200)
    bridge.validate_intent(intent,resource,PIN,120)
    intent[field]=value
    with pytest.raises(ValueError,match='Late or mismatched'):
        bridge.validate_intent(intent,resource,PIN,120)


def test_atomic_publication_cannot_overwrite_an_existing_final(tmp_path):
    path=tmp_path/'ticket.json';bridge.write_new(path,dict(first=True))
    with pytest.raises(FileExistsError):bridge.write_new(path,dict(first=False))
    assert json.loads(path.read_text())==dict(first=True)


@pytest.mark.parametrize('mode',['timeout','TERM','KILL'])
def test_transfer_guard_bounds_descendants_after_dispatcher_loss(tmp_path,mode):
    adapter=load('fixture_transfer_subreaper',ROOT/'owned_cpu_runtime.py');adapter.become_subreaper()
    info=tmp_path/'ids.json'
    child=tmp_path/'child.py'
    child.write_text("import json,os,signal,subprocess,sys,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN)\n"
        "grand=subprocess.Popen([sys.executable,'-I','-B','-S','-c','import time;time.sleep(120)'])\n"
        +f"Path({str(info)!r}).write_text(json.dumps(dict(pid=os.getpid(),grand=grand.pid)))\n"
        "time.sleep(120)\n")
    driver=tmp_path/'driver.py'
    driver.write_text("import importlib.util,sys,signal\nfrom pathlib import Path\n"
        +f"spec=importlib.util.spec_from_file_location('bridge',{str(ROOT/'dispatch_bootstrap.py')!r})\n"
        "m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)\n"
        "def stop(signum,frame):raise RuntimeError('fixture dispatcher signal')\n"
        "signal.signal(signal.SIGTERM,stop)\n"
        +f"m.bounded_command([sys.executable,'-I','-B','-S',{str(child)!r}],timeout={1 if mode=='timeout' else 10},log=Path({str(tmp_path/'transfer.log')!r}))\n")
    process=subprocess.Popen([sys.executable,'-I','-B','-S',str(driver)],start_new_session=True,
                             stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    unrelated=subprocess.Popen([sys.executable,'-I','-B','-S','-c','import time;time.sleep(120)'],start_new_session=True)
    try:
        until=time.monotonic()+5
        while not info.exists() and time.monotonic()<until:time.sleep(.02)
        assert info.exists();identities=json.loads(info.read_text())
        if mode!='timeout':process.send_signal(getattr(signal,'SIG'+mode))
        process.wait(timeout=7)
        until=time.monotonic()+5
        while time.monotonic()<until:
            try:
                while os.waitpid(-1,os.WNOHANG)[0]:pass
            except ChildProcessError:pass
            if not any(Path('/proc',str(pid)).exists() for pid in identities.values()):break
            time.sleep(.02)
        assert not any(Path('/proc',str(pid)).exists() for pid in identities.values())
        receipts=[json.loads(p.read_text()) for p in tmp_path.glob('*.receipt.json')]
        assert len(receipts)==1 and receipts[0]['reason']=={'timeout':'timeout','TERM':'parent_lost','KILL':'parent_lost'}[mode]
        assert unrelated.poll() is None
    finally:
        if process.poll() is None:process.kill();process.wait(timeout=3)
        if unrelated.poll() is None:unrelated.terminate();unrelated.wait(timeout=3)
