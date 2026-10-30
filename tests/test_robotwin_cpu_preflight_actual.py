"""Allocated CPU regressions for the actual r2 image/utility failures only."""
import errno
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest
from test_robotwin_cpu_residual_bootstrap import resource_table

REPO=Path(__file__).resolve().parents[1]
ROOT=REPO/'outputs/robotwin-sac65-cpu-compat-20261031-r3'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


bridge=load('fixture_native_proof',ROOT/'dispatch_bootstrap.py')
probe=load('fixture_visibility_probe',ROOT/'visibility_probe.py')
gate=load('fixture_actual_preflight_gate',ROOT/'gate_job.py')
PNG=b'\x89PNG\r\n\x1a\nfixture'
JPEG=b'\xff\xd8\xff\xe0native fixture\xff\xd9'
PIN='a'*40


@pytest.mark.parametrize('raw,extension,media_type',[(PNG,'png','image/png'),(JPEG,'jpg','image/jpeg')])
def test_actual_native_encoding_is_detected_without_conversion(raw,extension,media_type):
    assert bridge.image_format(raw)==(extension,media_type)


@pytest.mark.parametrize('raw',[b'',b'GIF89a'])
def test_unsupported_screenshot_bytes_are_rejected(raw):
    with pytest.raises(ValueError,match='encoding'):bridge.image_format(raw)


@pytest.mark.parametrize('raw,extension,media_type',[(PNG,'png','image/png'),(JPEG,'jpg','image/jpeg')])
def test_complete_actual_proof_retains_byte_hash_and_truthful_suffix(tmp_path,raw,extension,media_type):
    (tmp_path/'proof-ready.json').write_text(json.dumps(dict(schema='robotwin_cpu_actual_ui_proof_ready_v1',
        run_id=bridge.RUN,code_commit=PIN,nonce='native',dispatcher_pid=123,captured_at_unix=110)))
    (tmp_path/'actual-running.image').write_bytes(raw)
    ax='1 AXWebArea actual detail\n  2 text mwj_1031_robotwin_sac65_cpu_r3_aaaaaaa\n  3 text 运行中\n'
    ax+='  4 text ID:  job-native123\n  5 text job-native123-master-0\n'+resource_table()
    (tmp_path/'actual-running.txt').write_text(ax)
    resource,files=bridge.prepare_proof(tmp_path,dict(pid=123,expires_at_unix=200),
        nonce='native',code_commit=PIN,armed_at=100,now=120)
    name='actual-running.'+extension
    assert files[name]==raw and resource['proof']['sha256']==hashlib.sha256(raw).hexdigest()
    assert resource['proof']['path']==bridge.REMOTE_INPUT+'/'+name
    assert resource['proof']['media_type']==media_type


@pytest.mark.parametrize('mode',['png','jpg','wrong-media-type'])
def test_remote_publication_retains_native_bytes_and_rejects_encoding_mismatch(tmp_path,mode):
    base=tmp_path/'input';base.mkdir();stage=base/'upload';stage.mkdir();out=tmp_path/'out';out.mkdir()
    extension='png' if mode=='png' else 'jpg';raw=PNG if extension=='png' else JPEG
    name='actual-running.'+extension;(stage/name).write_bytes(raw);(stage/'actual-running.txt').write_bytes(b'actual AX')
    resource=dict(hostname='job-native-master-0',proof=dict(path=str(base/name),sha256=hashlib.sha256(raw).hexdigest(),
        media_type='image/png' if mode in ('png','wrong-media-type') else 'image/jpeg'),
        ax_proof=dict(path=str(base/'actual-running.txt'),sha256=hashlib.sha256(b'actual AX').hexdigest()))
    rr=json.dumps(resource).encode();(stage/'actual-resource.json').write_bytes(rr)
    intent=dict(ticket_path=str(base/'actual-bootstrap.json'),bootstrap_expires_at_unix=time.time()+30,
                hostname=resource['hostname'],run_id=bridge.RUN)
    ir=json.dumps(intent).encode();(out/'bootstrap-intent.json').write_bytes(ir)
    ticket=dict(code_commit=PIN,hostname=resource['hostname'],run_id=bridge.RUN,output=str(out),
        resource_receipt=dict(path=str(base/'actual-resource.json'),sha256=hashlib.sha256(rr).hexdigest()))
    (stage/'actual-bootstrap.json').write_text(json.dumps(ticket))
    result=subprocess.run([sys.executable,'-I','-B','-S','-c',bridge.REMOTE_PUBLISH,str(stage),str(base),str(out),
        PIN,hashlib.sha256(ir).hexdigest()],capture_output=True,timeout=3)
    assert (result.returncode==0)==(mode!='wrong-media-type')
    assert (base/'actual-bootstrap.json').exists()==(mode!='wrong-media-type')
    if mode!='wrong-media-type':assert (base/name).read_bytes()==raw


def test_both_actual_costs_and_sealed_allowance_are_debited():
    budget=gate.load_budget(REPO)['values']
    assert budget['charged_scheduler_seconds']==299
    assert budget['scheduler_cap_seconds']==1320 and budget['total_sealed_seconds']==181
    assert 299+1320+181==1800 and budget['global_work_seconds']==1220


@pytest.mark.parametrize('field,value',[('charged_scheduler_seconds',149),('total_sealed_seconds',151)])
def test_rehashed_budget_cannot_ignore_second_charge_or_borrow_sealed_time(tmp_path,monkeypatch,field,value):
    directory=tmp_path/'outputs'/gate.RUN;directory.mkdir(parents=True)
    budget=json.loads((ROOT/'budget.json').read_text());budget[field]=value
    raw=json.dumps(budget).encode();(directory/'budget.json').write_bytes(raw)
    monkeypatch.setattr(gate,'BUDGET_SHA',hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='cumulative charge'):gate.load_budget(tmp_path)


def test_absent_utility_is_only_a_diagnostic(monkeypatch):
    monkeypatch.setattr(probe.shutil,'which',lambda _:None)
    result=probe.observe(environment={},device_nodes=[]);probe.validate(result)
    assert result['nvidia_smi']['path'] is None and not result['allocated_GPU_count_inferred']


def test_actual_empty_executable_enoexec_is_recorded_without_poisoning_owned_runtime(tmp_path):
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'');binary.chmod(0o700)
    result=probe.observe(environment={},device_nodes=[],executable=str(binary));probe.validate(result)
    assert result['nvidia_smi']['exec_error']['errno']==errno.ENOEXEC
    assert result['nvidia_smi']['bytes']==0 and not result['allocated_GPU_count_inferred']


@pytest.mark.parametrize('number',[errno.EIO,errno.EACCES])
def test_other_execution_errors_propagate(tmp_path,monkeypatch,number):
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'stub')
    def fail(*args,**kwargs):raise OSError(number,'fixture unexpected execution error')
    monkeypatch.setattr(probe.subprocess,'run',fail)
    with pytest.raises(OSError) as error:probe.observe(environment={},device_nodes=[],executable=str(binary))
    assert error.value.errno==number


def test_utility_timeout_propagates(tmp_path,monkeypatch):
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'stub')
    def fail(*args,**kwargs):raise subprocess.TimeoutExpired(['nvidia-smi'],3)
    monkeypatch.setattr(probe.subprocess,'run',fail)
    with pytest.raises(subprocess.TimeoutExpired):probe.observe(environment={},device_nodes=[],executable=str(binary))


@pytest.mark.parametrize('case',['GPU-output','device-node','namespace'])
def test_independent_pre_mask_visibility_still_rejects_gpus(tmp_path,monkeypatch,case):
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'stub')
    monkeypatch.setattr(probe.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],0,
        'GPU 0: fixture\n' if case=='GPU-output' else '', ''))
    result=probe.observe(environment={'NVIDIA_VISIBLE_DEVICES':'all'} if case=='namespace' else {},
        device_nodes=['/dev/nvidia0'] if case=='device-node' else [],executable=str(binary))
    with pytest.raises(ValueError,match='GPU'):probe.validate(result)


def test_nonzero_driver_diagnostic_does_not_claim_zero_allocation(tmp_path,monkeypatch):
    binary=tmp_path/'nvidia-smi';binary.write_bytes(b'stub')
    monkeypatch.setattr(probe.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a[0],9,'','driver unavailable'))
    result=probe.observe(environment={},device_nodes=[],executable=str(binary));probe.validate(result)
    assert result['nvidia_smi']['returncode']==9 and result['allocated_GPU_count_inferred'] is False


@pytest.mark.parametrize('kind',['ownership-exception','worker-nonzero'])
def test_parent_never_swallows_owned_runtime_or_worker_failure(tmp_path,kind):
    class Runtime:
        def run(self,*args,**kwargs):
            if kind=='ownership-exception':raise RuntimeError('owned cleanup failed')
            return dict(returncode=1,cleanup=dict(clean=True))
    with pytest.raises((RuntimeError,ValueError)):
        gate.pre_mask_visibility(Runtime(),REPO,tmp_path,{})
