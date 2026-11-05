import gzip
import errno
import hashlib
import io
import json
import os
import tarfile

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.flow_history import DirectActionCodec
from fluffy_bassoon.meta_rl.flow_source_queries import FlowSourceQueries
from fluffy_bassoon.meta_rl.vla_history import file_sha256


def test_source_query_replays_exact_rank_lane_version_and_rejects_changed_truth(tmp_path):
    rng = np.random.default_rng(34)
    codec = DirectActionCodec(tuple([0.] * 12), tuple([1.] * 12))
    states = rng.normal(size=(16, 25)).astype(np.float32)
    actions = rng.normal(size=(16, 50, 12)).astype(np.float32)
    actions[..., [6, 11]] = np.where(actions[..., [6, 11]] > 0, 1., -1.)
    images = torch.arange(16)[:, None, None, None].expand(16, 2, 2, 3)
    payload = dict(current_observation=dict(states=states, main_images=images,
        task_descriptions=['task'] * 16, private_label=['forbidden'] * 16), decoded_actions=actions)
    buffer = io.BytesIO(); torch.save(payload, buffer)
    raw = gzip.compress(buffer.getvalue())
    archive = tmp_path / 'raw.tar'
    with tarfile.open(archive, 'w') as tf:
        member = tarfile.TarInfo('source.pt.gz'); member.size = len(raw)
        tf.addfile(member, io.BytesIO(raw))
    with tarfile.open(archive) as tf:
        offset = tf.getmember('source.pt.gz').offset_data
    index = tmp_path / 'index.json'
    index.write_text(json.dumps([dict(rank=2, version=2, chunk=4,
        bytes=len(raw), offset_data=offset, sha256=hashlib.sha256(raw).hexdigest())]))
    sources = dict(Task=dict(index_path=str(index), index_sha256=file_sha256(index), archive=str(archive)))
    observations = np.zeros((36, 2073), np.float32); observations[27, :25] = states[1]
    executed = np.zeros((36, 600), np.float32); executed[27] = codec.encode(actions[1])
    trial = dict(observations=observations, executed_noise=executed,
                 teacher_steps=np.repeat(np.arange(3), 12), rewards=np.zeros(36))
    reader = FlowSourceQueries(sources, codec, retry_delay=0)
    try:
        found = reader.observations([(0, 27)], [trial], ['Task/flow-source-slot-033'])
        np.testing.assert_array_equal(found['states'], states[1:2])
        assert torch.equal(found['main_images'], images[1:2])
        assert 'private_label' not in found
        trial['executed_noise'][27, 0] += 1
        with pytest.raises(ValueError, match='source state, behavior action and version'):
            reader.observations([(0, 27)], [trial], ['Task/flow-source-slot-033'])
    finally:
        reader.close()
    data = bytearray(archive.read_bytes()); data[offset] ^= 1; archive.write_bytes(data)
    reader = FlowSourceQueries(sources, codec, retry_delay=0)
    try:
        with pytest.raises(ValueError, match='hash verification'):
            reader.read_member('Task', 2, 2, 4)
    finally:
        reader.close()


def source_reader(tmp_path, **kwargs):
    buffer = io.BytesIO(); torch.save({'value': torch.arange(12)}, buffer)
    raw = gzip.compress(buffer.getvalue())
    archive = tmp_path / 'source'; archive.write_bytes(raw)
    index = tmp_path / 'index.json'
    index.write_text(json.dumps([dict(rank=0, version=0, chunk=1, bytes=len(raw),
        offset_data=0, sha256=hashlib.sha256(raw).hexdigest())]))
    reader = FlowSourceQueries(dict(Task=dict(index_path=str(index),
        index_sha256=file_sha256(index), archive=str(archive))), None, retry_delay=0, **kwargs)
    return reader


def test_short_reads_are_completed_before_verification(tmp_path, monkeypatch):
    reader = source_reader(tmp_path)
    original = os.pread
    monkeypatch.setattr(os, 'pread', lambda fd, size, offset: original(fd, min(size, 31), offset))
    try:
        assert torch.equal(reader.read_member('Task', 0, 0, 1)['value'], torch.arange(12))
    finally:
        reader.close()


@pytest.mark.parametrize('failure', ['io_error', 'wrong_bytes', 'early_eof'])
def test_transient_source_failure_reopens_and_returns_only_verified_data(tmp_path, monkeypatch, failure):
    reader = source_reader(tmp_path)
    original, calls = os.pread, []
    def read(fd, size, offset):
        calls.append((fd, size, offset))
        if len(calls) == 1:
            if failure == 'io_error':
                raise OSError(errno.EIO, 'injected transient read failure')
            return b'x' * size if failure == 'wrong_bytes' else b''
        return original(fd, size, offset)
    monkeypatch.setattr(os, 'pread', read)
    try:
        with pytest.warns(RuntimeWarning, match='retrying the same member'):
            found = reader.read_member('Task', 0, 0, 1)
        assert torch.equal(found['value'], torch.arange(12))
    finally:
        reader.close()


def test_persistent_bad_bytes_never_enter_the_cache(tmp_path, monkeypatch):
    reader = source_reader(tmp_path, read_attempts=3)
    calls = []
    def read(fd, size, offset):
        calls.append(offset)
        return b'x' * size
    monkeypatch.setattr(os, 'pread', read)
    try:
        with pytest.warns(RuntimeWarning), pytest.raises(ValueError, match='hash verification'):
            reader.read_member('Task', 0, 0, 1)
        assert len(calls) == 3 and not reader.cache
    finally:
        reader.close()


def test_read_only_close_error_does_not_discard_a_completed_segment(tmp_path, monkeypatch):
    reader = source_reader(tmp_path)
    reader.read_member('Task', 0, 0, 1)
    original, calls = os.close, []
    def close(fd):
        calls.append(fd)
        original(fd)
        raise OSError(errno.EBUSY, 'injected close failure after descriptor release')
    monkeypatch.setattr(os, 'close', close)
    with pytest.warns(RuntimeWarning, match='descriptor retired without retry'):
        reader.close()
    reader.close()
    assert len(calls) == 1 and not reader.cache and not reader.sources


def cached_source_reader(tmp_path):
    original = source_reader(tmp_path)
    row = original.members['Task'][0, 0, 1]
    root = tmp_path / 'pfs'; (root / 'Task').mkdir(parents=True)
    path = root / 'Task' / (row['sha256'] + '.pt.gz')
    path.write_bytes((tmp_path / 'source').read_bytes())
    manifest = dict(format='flow_source_member_cache_v1', completed=True,
        sources=dict(Task=dict(index_sha256=file_sha256(tmp_path / 'index.json'),
                               members=1, bytes=row['bytes'])))
    (root / 'CACHE.json').write_text(json.dumps(manifest))
    sources = dict(Task=dict(index_path=str(tmp_path / 'index.json'),
        index_sha256=file_sha256(tmp_path / 'index.json'), archive=str(tmp_path / 'source')))
    original.close()
    return root, path, sources


def test_complete_pfs_cache_replays_without_accessing_bos_archive(tmp_path, monkeypatch):
    root, path, sources = cached_source_reader(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError('PFS reader must not open the BOS archive')
    monkeypatch.setattr(os, 'pread', forbidden)
    (tmp_path / 'source').unlink()
    reader = FlowSourceQueries(sources, None, member_cache=root, retry_delay=0)
    assert torch.equal(reader.read_member('Task', 0, 0, 1)['value'], torch.arange(12))
    reader.close()


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'index', 'incomplete'])
def test_pfs_cache_fails_closed_without_bos_fallback(tmp_path, failure):
    root, path, sources = cached_source_reader(tmp_path)
    if failure == 'missing':
        path.unlink()
    elif failure == 'corrupt':
        path.write_bytes(b'x' * path.stat().st_size)
    else:
        manifest = json.loads((root / 'CACHE.json').read_text())
        if failure == 'index':
            manifest['sources']['Task']['index_sha256'] = '0' * 64
        else:
            manifest['completed'] = False
        (root / 'CACHE.json').write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        reader = FlowSourceQueries(sources, None, member_cache=root, read_attempts=1, retry_delay=0)
        try:
            reader.read_member('Task', 0, 0, 1)
        finally:
            reader.close()
