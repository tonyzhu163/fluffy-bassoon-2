"""Atomic checkpoint metadata remains valid across transient archive failures."""
import errno
import json
from pathlib import Path

import pytest

from fluffy_bassoon.rlinf import native_ppo as native


def test_busy_publication_retries_same_complete_payload(tmp_path, monkeypatch):
    target=tmp_path/'checkpoint.json'
    native.write(target, {'step':0})
    replace=Path.replace
    calls=[]
    def publish(source,destination):
        assert json.loads(target.read_bytes())=={'step':0}
        assert json.loads(source.read_bytes())=={'step':1,'complete':True}
        calls.append(source)
        if len(calls)==1:raise OSError(errno.EBUSY,'archive busy')
        return replace(source,destination)
    monkeypatch.setattr(Path,'replace',publish)
    monkeypatch.setattr(native.time,'sleep',lambda seconds:None)
    native.write(target,{'step':1,'complete':True})
    assert len(calls)==2 and calls[0]==calls[1]
    assert json.loads(target.read_bytes())=={'step':1,'complete':True}
    assert not calls[0].exists()


@pytest.mark.parametrize('code',[errno.EACCES,errno.ENOSPC])
def test_permanent_publication_error_preserves_prior_boundary(tmp_path,monkeypatch,code):
    target=tmp_path/'checkpoint.json';native.write(target,{'step':0})
    calls=[]
    def fail(source,destination):
        calls.append(source)
        raise OSError(code,'permanent')
    monkeypatch.setattr(Path,'replace',fail)
    with pytest.raises(OSError) as error:native.write(target,{'step':1})
    assert error.value.errno==code and len(calls)==1
    assert json.loads(target.read_bytes())=={'step':0}
    assert json.loads(calls[0].read_bytes())=={'step':1}


def test_busy_publication_has_bounded_retry_time(tmp_path,monkeypatch):
    target=tmp_path/'checkpoint.json';native.write(target,{'step':0})
    ticks=iter([0.,31.]);monkeypatch.setattr(native.time,'monotonic',lambda:next(ticks))
    def busy(source,destination):raise OSError(errno.EBUSY,'still busy')
    monkeypatch.setattr(Path,'replace',busy)
    with pytest.raises(OSError) as error:native.write(target,{'step':1})
    assert error.value.errno==errno.EBUSY
    assert json.loads(target.read_bytes())=={'step':0}
