"""Core checkpoint ownership, backpressure and error propagation on Linux CPU."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.background_io import BoundedWrites
from fluffy_bassoon.meta_rl.tensor_archive import read_tensor, write_tensor


def test_checkpoint_snapshot_survives_caller_tensor_and_array_mutation(tmp_path):
    gate, started = Event(), Event()
    value = dict(tensor=torch.arange(8.).requires_grad_(),array=np.arange(3),nested=[torch.ones(2)])
    def save(path,payload):
        started.set();gate.wait(10)
        assert not payload['tensor'].requires_grad and payload['tensor'].device.type == 'cpu'
        write_tensor(path,payload)
    writer = BoundedWrites()
    try:
        writer.submit(save,tmp_path/'checkpoint.pt.gz',value)
        assert started.wait(10)
        with torch.no_grad(): value['tensor'].fill_(99)
        value['array'][:] = 99;value['nested'][0].zero_()
    finally:
        gate.set();writer.close()
    restored = read_tensor(tmp_path/'checkpoint.pt.gz')
    assert torch.equal(restored['tensor'],torch.arange(8.))
    assert np.array_equal(restored['array'],np.arange(3))
    assert torch.equal(restored['nested'][0],torch.ones(2))


def test_backpressure_bounds_jobs_and_bytes_and_drains_all_checkpoints(tmp_path):
    first,release,second = Event(),Event(),Event()
    writer = BoundedWrites(maximum_jobs=1,maximum_bytes=4096)
    def save(index,payload):
        if index == 0: first.set();assert release.wait(10)
        else: second.set()
        write_tensor(tmp_path/f'{index}.pt',payload)
    def produce():
        for i in range(8): writer.submit(save,i,torch.full((32,),float(i)))
    with ThreadPoolExecutor(max_workers=1) as producer:
        future = producer.submit(produce)
        assert first.wait(10)
        assert not second.is_set()
        release.set();future.result(timeout=10)
    writer.close()
    assert writer.stats()['pending_jobs'] == writer.stats()['pending_bytes'] == 0
    assert writer.peak_jobs == 1 and writer.peak_bytes <= 4096
    for i in range(8): assert torch.equal(read_tensor(tmp_path/f'{i}.pt'),torch.full((32,),float(i)))


def test_write_failure_blocks_boundary_and_later_submissions():
    writer = BoundedWrites()
    def fail(value): raise OSError('checkpoint write failed')
    future = writer.submit(fail,torch.zeros(1))
    with pytest.raises(OSError): future.result(timeout=10)
    with pytest.raises(OSError,match='checkpoint write failed'): writer.flush()
    with pytest.raises(RuntimeError,match='previous background write failed'): writer.submit(lambda:None)
    with pytest.raises(RuntimeError,match='previous background write failed'): writer.close()
    assert writer.closed


def test_oversized_snapshot_is_rejected_before_enqueue():
    writer = BoundedWrites(maximum_bytes=1024)
    try:
        with pytest.raises(ValueError,match='exceeds'): writer.submit(lambda value:None,torch.zeros(1024))
        with pytest.raises(ValueError,match='numeric NumPy'): writer.submit(lambda value:None,np.array([{}],dtype=object))
        assert writer.stats()['submitted'] == 0
    finally: writer.close()


def test_native_training_and_evaluation_traces_use_owned_bounded_snapshots(tmp_path):
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from fluffy_bassoon.rlinf.native_ppo_runtime import SidecarMixin, _write_trace_record
    class Environment(SidecarMixin):
        pilot_role = 'env'
    worker = Environment()
    worker._rank = 0
    worker.cfg = SimpleNamespace(native_pilot=OmegaConf.create(dict(output=str(tmp_path),
        bounded_io=True,io_max_pending_jobs=1,io_max_pending_bytes=8192)))
    worker._init_pilot_io()
    try:
        for phase in ('training','evaluation'):
            trace = dict(phase=phase,evaluation_data=phase=='evaluation',reward=torch.ones(3),
                         observation=dict(states=torch.arange(14.)[None]))
            worker._submit_pilot_io(_write_trace_record,tmp_path/f'{phase}.pt.gz',trace)
            trace['reward'].zero_();trace['observation']['states'].zero_()
        worker._flush_pilot_io()
        for phase in ('training','evaluation'):
            saved = read_tensor(tmp_path/f'{phase}.pt.gz')
            assert saved['evaluation_data'] == (phase=='evaluation')
            assert torch.equal(saved['reward'],torch.ones(3))
            assert torch.equal(saved['observation']['states'],torch.arange(14.)[None])
    finally: worker._pilot_bounded_io.close()
