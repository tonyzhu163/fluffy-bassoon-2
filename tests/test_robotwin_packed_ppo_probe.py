"""Resource/scientific contracts of the two independent packed qualification forks."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

_path=Path(__file__).parents[1]/'scripts/diagnostics/robotwin_packed_ppo_probe.py'
_spec=importlib.util.spec_from_file_location('packed_probe',_path)
probe=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(probe)


def test_packed_forks_have_disjoint_ledgers_and_unchanged_learning_batches(tmp_path):
    values=probe.commands(Path('/pinned/worktree'),tmp_path,'a'*40,python='/python')
    assert set(values)=={'ppo','ppo_reset'}
    def arg(command,name):return command[command.index(name)+1]
    for method,command in values.items():
        assert arg(command,'--methods')==method
        assert arg(command,'--collector-lanes')=='6'
        assert arg(command,'--ppo-trials-per-update')=='6'
        assert arg(command,'--train-episodes')=='4'
        assert arg(command,'--source-iterations')=='1'
        assert arg(command,'--source-steps')==arg(command,'--campaign-max-actions')=='10800'
        assert arg(command,'--campaign-root')==str(tmp_path/('ledger-'+method))
        assert arg(command,'--native-episodes')==arg(command,'--source-validation-episodes')=='0'
        assert '--stage-source-validation-only' in command and '--resume' not in command
        assert arg(command,'--learning-rate')=='1e-6' and arg(command,'--ppo-epochs')=='5'
    assert probe.ACTIONS_PER_ARM*len(values)==21600


def test_gpu_telemetry_respects_attested_uuid(monkeypatch):
    calls=[]
    def fake(command,**kwargs):
        calls.append(command);return '76800, 98304, 87\n'
    monkeypatch.setattr(probe.subprocess,'check_output',fake)
    row=probe.sample_gpu('GPU-owned')
    assert '--id=GPU-owned' in calls[0]
    assert row['used_gib']==75 and row['total_bytes']==96*(1<<30)
    assert row['utilization_percent']==87


@pytest.mark.skipif(sys.platform!='linux',reason='Linux allocated-process ownership check')
def test_cleanup_terminates_orphan_after_native_parent_abort(tmp_path):
    token=tmp_path/'ppo';child_file=tmp_path/'child.json'
    code=('import subprocess,sys,json;'
          'p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]);'
          'open(sys.argv[1],"w").write(json.dumps(p.pid));sys.exit(7)')
    parent=subprocess.Popen([sys.executable,'-c',code,str(child_file),str(token)],start_new_session=True)
    try:
        assert parent.wait(timeout=10)==7
        child=json.loads(child_file.read_text())
        assert child in probe.owned_members(parent,token)
        probe.stop_owned({'ppo':parent},tmp_path,grace_seconds=0,term_seconds=.2)
        assert not probe.owned_members(parent,token)
        assert parent.returncode==7
    finally:
        if probe.owned_members(parent,token):os.killpg(parent.pid,9)


@pytest.mark.skipif(sys.platform!='linux',reason='Linux allocated-process ownership check')
def test_cleanup_escalates_only_owned_session_and_rejects_wrong_token(tmp_path):
    token=tmp_path/'ppo'
    code='import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);print("ready",flush=True);time.sleep(60)'
    parent=subprocess.Popen([sys.executable,'-c',code,str(token)],stdout=subprocess.PIPE,
                            text=True,start_new_session=True)
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        assert parent.stdout.readline().strip()=='ready'
        with pytest.raises(RuntimeError,match='identity token'):
            probe.owned_members(parent,tmp_path/'wrong')
        probe.stop_owned({'ppo':parent},tmp_path,grace_seconds=.1,term_seconds=.1)
        assert parent.wait(timeout=5)==-9
        assert unrelated.poll() is None
    finally:
        if parent.poll() is None:os.killpg(parent.pid,9);parent.wait(timeout=5)
        unrelated.terminate();unrelated.wait(timeout=5)
