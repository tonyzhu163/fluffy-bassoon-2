"""Mocked CPU checks for exclusive Baige training admission; run in allocation."""
import ast
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace

import pytest

from fluffy_bassoon import baige_gpu


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('baige_admission_probe', ROOT / 'scripts/[train]gpu_probe.py')
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)
HOST = 'job-fixture123-master-0'


class Tensor:
    def __init__(self, result=2097152):
        self.result = result

    def __mul__(self, _):
        return self

    def sum(self):
        return self

    def item(self):
        return self.result

    def numel(self):
        return 1024 * 1024


@pytest.fixture
def admission(monkeypatch):
    count = 2
    uuids = [f'GPU-{i+1:08x}-0000-0000-0000-{i+1:012x}' for i in range(count)]
    inventory = [dict(index=i, uuid=uuid, name='NVIDIA RPBZZZ6', memory_total_mib=98304.,
                      memory_used_mib=float(3*i), utilization_percent=100., pci_bus_id=f'0000:{80+i:02x}:00.0')
                 for i, uuid in enumerate(uuids)]
    state = SimpleNamespace(current=0, touched=set(), calls=0, tensor_calls=0,
                            eager=False, foreign_before=False, foreign_during=False,
                            foreign_final=False, missing_touched=False, outside=False,
                            tensor_result=2097152)

    @contextmanager
    def device(ordinal):
        state.current = ordinal
        yield

    def properties(ordinal):
        return SimpleNamespace(uuid=uuids[ordinal], name='NVIDIA RPBZZZ6', total_memory=98304*2**20)

    def ones(*_, **__):
        state.touched.add(state.current)
        state.tensor_calls += 1
        return Tensor(state.tensor_result)

    def processes():
        state.calls += 1
        indices = set(range(count)) if state.eager and state.touched else set(state.touched)
        if state.missing_touched:
            indices.discard(0)
        rows = [dict(gpu_uuid=uuids[i], pid=probe.os.getpid()) for i in sorted(indices)]
        if state.outside and state.touched:
            rows.append(dict(gpu_uuid='GPU-ffffffff-0000-0000-0000-ffffffffffff',pid=probe.os.getpid()))
        if (state.foreign_before and not state.touched) or (state.foreign_during and state.touched) or (
                state.foreign_final and state.calls == count+2):
            rows.append(dict(gpu_uuid=uuids[0],pid=probe.os.getpid()+1))
        return rows

    cuda = SimpleNamespace(is_available=lambda:True,device_count=lambda:count,
                           get_device_properties=properties,device=device,synchronize=lambda:None,empty_cache=lambda:None)
    fake_torch = SimpleNamespace(cuda=cuda,ones=ones,float32='float32')
    monkeypatch.setitem(sys.modules,'torch',fake_torch)
    monkeypatch.setattr(probe.sys,'platform','linux')
    monkeypatch.setattr(probe.socket,'gethostname',lambda:HOST)
    monkeypatch.setenv('KUBERNETES_SERVICE_HOST','fixture')
    monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES',','.join(uuids))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0,1')
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    monkeypatch.delenv('SLURM_STEP_ID',raising=False)
    monkeypatch.setattr(probe,'gpu_inventory',lambda:inventory)
    monkeypatch.setattr(probe,'compute_processes',processes)
    monkeypatch.setattr(probe,'nvidia_query',lambda _: 'fixture topology')
    monkeypatch.setattr(baige_gpu,'cuda_pci',lambda i:f'0000:{80+i:02x}:00.0')
    return state,inventory,uuids


@pytest.mark.parametrize('eager',[False,True])
def test_explicit_training_admission_requires_real_proofs_despite_utilization(admission,eager):
    state,inventory,uuids = admission
    state.eager = eager
    report = probe.run_baige_probe(2,256,'training',exclusive_training=True)
    assert report['purpose']=='training' and report['exclusive_training_admission']
    assert report['admission_policy']=='exclusive_scheduler_cuda_pci_pid_memory'
    assert report['utilization_role']=='telemetry_only' and not report['idle_utilization_required']
    assert report['utilization_telemetry']=={uuid:100. for uuid in uuids}
    assert state.tensor_calls==2 and state.touched=={0,1}
    assert all(row['allocation_check']['result']==2097152 for row in report['devices'])
    assert {row['gpu_uuid'] for row in report['compute_processes_after']}==set(uuids)


def test_default_baige_training_keeps_strict_utilization_guard(admission):
    state,_,_ = admission
    with pytest.raises(RuntimeError,match='not idle'):
        probe.run_baige_probe(2,256,'training')
    assert state.tensor_calls==0


@pytest.mark.parametrize('failure',['memory','foreign_before','foreign_during','foreign_final',
                                   'missing_touched','outside','tensor','pci','allocation'])
def test_exclusive_admission_does_not_bypass_occupancy_or_identity(admission,monkeypatch,failure):
    state,inventory,uuids = admission
    if failure=='memory':
        inventory[0]['memory_used_mib']=257.
    elif failure=='tensor':
        state.tensor_result=0
    elif failure=='pci':
        monkeypatch.setattr(baige_gpu,'cuda_pci',lambda _: '0000:ff:00.0')
    elif failure=='allocation':
        monkeypatch.setenv('NVIDIA_VISIBLE_DEVICES',','.join([uuids[0],'GPU-ffffffff-0000-0000-0000-ffffffffffff']))
    else:
        setattr(state,failure,True)
    with pytest.raises((RuntimeError,ValueError)):
        probe.run_baige_probe(2,256,'training',exclusive_training=True)


def test_training_opt_in_cannot_be_relabelled_evaluation(admission):
    with pytest.raises(ValueError,match='training purpose'):
        probe.run_baige_probe(2,256,'evaluation',exclusive_training=True)


def test_direct_training_still_rejects_high_utilization(admission,monkeypatch):
    state,_,uuids = admission
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',','.join(uuids))
    monkeypatch.setattr(probe.shutil,'which',lambda _:None)
    with pytest.raises(RuntimeError,match='not idle'):
        probe.run_probe(2,purpose='training',direct_hostname=HOST)
    assert state.tensor_calls==0


def test_slurm_training_still_rejects_high_utilization(admission,monkeypatch):
    state,inventory,_ = admission
    monkeypatch.setenv('SLURM_JOB_ID','12')
    monkeypatch.setenv('SLURM_STEP_ID','0')
    monkeypatch.setenv('SLURM_STEP_GPUS','0,1')
    monkeypatch.setenv('SLURM_JOB_GPUS','0,1')
    monkeypatch.setattr(probe,'slurm_gpu_inventory',lambda:(inventory,dict(method='fixture')))
    with pytest.raises(RuntimeError,match='not idle'):
        probe.run_probe(2,purpose='training')
    assert state.tensor_calls==0


@pytest.mark.parametrize('arguments',[['--baige-exclusive-training'],
    ['--baige-exclusive-training','--baige-container','--purpose','evaluation']])
def test_cli_training_opt_in_requires_baige_training(tmp_path,monkeypatch,arguments):
    monkeypatch.setattr(sys,'argv',['gpu_probe.py','--expected-count','2','--output',str(tmp_path/'probe.json')]+arguments)
    with pytest.raises(SystemExit) as error:
        probe.main()
    assert error.value.code==2


def sku_validator():
    source=(ROOT/'scripts/diagnostics/can_dsrl_eight_gpu.sh').read_text().split("<<'PY'\n",1)[1].rsplit('\nPY',1)[0]
    definition=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name=='approved_baige_sku')
    namespace={'re':re}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),'<isolated SKU validator>','exec'),namespace)
    return namespace['approved_baige_sku']


@pytest.mark.parametrize('name',['NVIDIA RPBZZZ6','NVIDIA RTX 6000 Ada Generation','NVIDIA RTX PRO 6000 Blackwell'])
def test_approved_raw_sku_keeps_exact_names_without_marketing_inference(name):
    result=sku_validator()([dict(cuda_name=name) for _ in range(8)])
    assert result['raw_name']==name and result['raw_names']==[name]*8
    assert result['marketing_model_decoded'] is False


@pytest.mark.parametrize('names',[['NVIDIA RPBZZZ60']*8,['NVIDIA unknown']*8,
    ['NVIDIA RPBZZZ6']*7,['NVIDIA RPBZZZ6']*7+['NVIDIA RTX 6000 Ada Generation']])
def test_unapproved_or_mixed_raw_skus_are_rejected(names):
    with pytest.raises(ValueError):
        sku_validator()([dict(cuda_name=name) for name in names])
