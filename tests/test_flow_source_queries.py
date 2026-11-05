import gzip
import hashlib
import io
import json
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
    reader = FlowSourceQueries(sources, codec)
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
    reader = FlowSourceQueries(sources, codec)
    try:
        with pytest.raises(ValueError, match='hash verification'):
            reader.read_member('Task', 2, 2, 4)
    finally:
        reader.close()
