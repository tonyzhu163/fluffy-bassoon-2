"""Prepared Linux CPU fake-process tests; execution belongs to the allocation owner."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest


_path=Path(__file__).parents[1]/"scripts/diagnostics/robotwin_trained_pair_runtime.py"
_spec=importlib.util.spec_from_file_location("trained_pair_runtime_tests",_path)
runtime=importlib.util.module_from_spec(_spec); _spec.loader.exec_module(runtime)
pytestmark=pytest.mark.skipif(sys.platform!="linux",reason="Allocated Linux process tests only")


def wait_for(predicate,seconds=8):
    until=time.monotonic()+seconds
    while time.monotonic()<until:
        if predicate(): return
        time.sleep(.05)
    raise AssertionError("Owned fake process did not reach expected bounded state")


def live(pid):
    value=runtime._stat(pid)
    return value is not None and value["state"] not in ("Z","X")


def test_cpu_preflight_is_cuda_hidden_and_reaped_before_return(tmp_path):
    output=tmp_path/"run"; output.mkdir(); (output/"plan.json").write_text("{}")
    with runtime.TrainedPairRuntime(tmp_path,output,maximum_seconds=30,cleanup_reserve_seconds=10) as owner:
        receipt=owner.run_bounded([sys.executable,"-c",
            "import os;assert os.environ['CUDA_VISIBLE_DEVICES']==''"],cwd=tmp_path,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES="scheduler-token"),
            log_path=output/"preflight.log",timeout_seconds=2)
        assert receipt["returncode"]==0 and receipt["cleanup"]["clean"]
        assert not live(receipt["pid"])
        assert owner.collection_deadline==owner.deadline-10
    assert owner.guardian.returncode==0
    assert json.loads((output/"runtime-owner.json").read_text())["environment_saved"] is False
    with pytest.raises(FileExistsError):
        runtime.TrainedPairRuntime(tmp_path,output,maximum_seconds=30,cleanup_reserve_seconds=10)


@pytest.mark.parametrize("timeout",[False,True])
def test_failed_or_timed_out_preflight_cannot_enter_gpu_stage(tmp_path,timeout):
    output=tmp_path/"run"; forbidden=tmp_path/"GPU-WAS-ENTERED"
    owner=runtime.TrainedPairRuntime(tmp_path,output,maximum_seconds=30,cleanup_reserve_seconds=10)
    command=[sys.executable,"-c","import time;time.sleep(60)" if timeout else "raise SystemExit(7)"]
    try:
        with pytest.raises(RuntimeError):
            owner.run_bounded(command,cwd=tmp_path,env=os.environ,log_path=output/"preflight.log",
                                timeout_seconds=.2 if timeout else 2)
        with pytest.raises(RuntimeError,match="closed/failed"):
            owner.run_bounded([sys.executable,"-c",f"open({str(forbidden)!r},'w').write('bad')"],
                cwd=tmp_path,env=os.environ,log_path=output/"probe.log",timeout_seconds=2,purpose="gpu_probe")
        assert not forbidden.exists()
        assert json.loads((output/"runtime-failure.json").read_text())["cleanup"]["clean"]
    finally: owner.close()


def test_native_command_abort_cleans_orphan_in_changed_pgid_but_not_other_session(tmp_path):
    output=tmp_path/"run"; child_file=tmp_path/"worker.pid"
    unrelated=subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"],start_new_session=True)
    # The orphan changes its process group while retaining the owned SID.
    child_code="import os,time;os.setpgrp();time.sleep(60)"
    code=("import subprocess,sys,time,os,resource;resource.setrlimit(resource.RLIMIT_CORE,(0,0));"
          f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
          f"open({str(child_file)!r},'w').write(str(p.pid));time.sleep(.2);os.abort()")
    try:
        with runtime.TrainedPairRuntime(tmp_path,output,maximum_seconds=30,cleanup_reserve_seconds=10) as owner:
            with pytest.raises(RuntimeError,match="exit-6"):
                owner.run_bounded([sys.executable,"-c",code],cwd=tmp_path,env=os.environ,
                    log_path=output/"preflight.log",timeout_seconds=3)
        assert child_file.exists() and not live(int(child_file.read_text()))
        assert unrelated.poll() is None
    finally:
        unrelated.terminate(); unrelated.wait(timeout=5)
        if child_file.exists() and live(int(child_file.read_text())): os.kill(int(child_file.read_text()),signal.SIGKILL)


@pytest.mark.parametrize("caller_signal",[signal.SIGINT,signal.SIGTERM,signal.SIGKILL])
def test_fatal_launcher_loss_still_has_independent_guardian_cleanup(tmp_path,caller_signal):
    output=tmp_path/"run"; child_file=tmp_path/"child.pid"
    command_code=f"import os,time;open({str(child_file)!r},'w').write(str(os.getpid()));time.sleep(60)"
    caller_code=("import importlib.util,os,sys;"
        f"s=importlib.util.spec_from_file_location('owned_runtime',{str(_path)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
        f"o=m.TrainedPairRuntime({str(tmp_path)!r},{str(output)!r},maximum_seconds=30,cleanup_reserve_seconds=10);"
        f"o.run_bounded([sys.executable,'-c',{command_code!r}],cwd={str(tmp_path)!r},env=os.environ,"
        f"log_path={str(output/'preflight.log')!r},timeout_seconds=20)")
    caller=subprocess.Popen([sys.executable,"-c",caller_code],stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,start_new_session=True)
    try:
        wait_for(child_file.exists)
        child=int(child_file.read_text()); assert live(child)
        caller.send_signal(caller_signal); caller.wait(timeout=8)
        wait_for(lambda:not live(child))
        wait_for(lambda:any(json.loads(p.read_text())["clean"] for p in output.glob("cleanup-*.json")))
    finally:
        if caller.poll() is None: caller.kill(); caller.wait(timeout=5)
        if child_file.exists() and live(int(child_file.read_text())): os.kill(int(child_file.read_text()),signal.SIGKILL)


def test_pair_before_single_fresh_probe_is_rejected_without_dispatch(tmp_path):
    output=tmp_path/"run"; forbidden=tmp_path/"ARM-WAS-ENTERED"
    command=[sys.executable,"-c",f"open({str(forbidden)!r},'w').write('bad')"]
    with runtime.TrainedPairRuntime(tmp_path,output,maximum_seconds=30,cleanup_reserve_seconds=10) as owner:
        with pytest.raises(RuntimeError,match="fresh probe"):
            owner.run_pair({"ppo":command,"ppo_reset":command},tmp_path/"missing.json")
    assert not forbidden.exists()


def probe_value():
    uuid="GPU-fake-owned"
    return dict(status="passed",purpose="training",allocation_type="baige_container",
        hostname=socket.gethostname(),created_at_utc=datetime.now(timezone.utc).isoformat(),
        idle_utilization_required=True,admission_policy="strict_idle_training",compute_processes_before=[],
        gpu_identity_mapping=dict(assigned_nvml_uuids=[uuid]),devices=[dict(uuid=uuid,cuda_total_bytes=96*(1<<30),
            inventory=dict(memory_used_mib=0,utilization_percent=0),
            allocation_check=dict(touched_gpu_uuids=[uuid],own_pid_gpu_uuids=[uuid]))])


@pytest.mark.parametrize("reason",["stale","busy","wrong_uuid","wrong_host"])
def test_probe_identity_or_idle_failure_is_not_accepted(tmp_path,reason):
    value=probe_value(); path=tmp_path/"probe.json"; before=time.time()-1
    if reason=="stale": value["created_at_utc"]="2000-01-01T00:00:00+00:00"
    if reason=="busy": value["devices"][0]["inventory"]["memory_used_mib"]=300
    if reason=="wrong_uuid": value["devices"][0]["allocation_check"]["touched_gpu_uuids"]=["GPU-other"]
    if reason=="wrong_host": value["hostname"]="job-other"
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError): runtime.validate_probe(path,not_before=before)


def test_runtime_never_expands_the_scheduler_envelope(tmp_path):
    with pytest.raises(ValueError,match="2520"):
        runtime.TrainedPairRuntime(tmp_path,tmp_path/"run",maximum_seconds=2700)


def test_guardian_final_cleanup_failure_cannot_return_success(tmp_path,monkeypatch):
    def fail(): raise RuntimeError("owned command remains unkillable")
    guardian=SimpleNamespace(output=tmp_path,parent=os.getppid(),deadline=time.monotonic()+10,cleanup=fail)
    values=iter(["{}",None])
    monkeypatch.setattr(runtime,"_control_line",lambda *args:next(values))
    monkeypatch.setattr(runtime,"_Guardian",lambda configuration:guardian)
    monkeypatch.setattr(runtime.signal,"signal",lambda *args:None)
    with pytest.raises(RuntimeError,match="unkillable"):
        runtime._guardian_main()
    assert json.loads((tmp_path/"cleanup-failure.json").read_text())["clean"] is False
