"""Hash-bound motor replay semantics; allocated Linux tests without a backbone."""
from copy import deepcopy
import gzip
import json
import pickle
from types import SimpleNamespace

import numpy as np
import pytest

import fluffy_bassoon.meta_rl.vla_motor_replay as module
from fluffy_bassoon.meta_rl.vla_libero import write_json
from fluffy_bassoon.meta_rl.vla_source40_corpus import PI05_RECEIPT


def inputs_manifest(tmp_path, *, change_observation=False, change_noise=False):
    entries = []
    for task in (0, 1):
        target = dict(benchmark='libero90', case_id=f'libero90:{task:02d}/state-0/seed-123',
            suite='libero_90', task_id=task, initial_state=0, scene_seed=123)
        observation = dict(image=np.zeros((4, 4, 3), np.uint8), wrist=np.zeros((4, 4, 3), np.uint8), state=np.zeros(8, np.float32))
        first = dict(motor_observation=observation, native_initial_noise=np.ones((10, 32), np.float32),
            executed_actions=np.zeros((10, 7), np.float32), done=True)
        second = deepcopy(first)
        second['executed_actions'] += .002
        if change_observation:
            second['motor_observation']['state'][0] = 1.
        if change_noise:
            second['native_initial_noise'][0, 0] = 0.
        receipts = []
        for name, data in (('reference', [first]), ('dsrl', dict(rows=[second]))):
            directory = tmp_path / f'{name}-{task}'
            directory.mkdir()
            path = directory / 'trace.pkl.gz'
            with gzip.open(path, 'wb') as stream:
                pickle.dump(data, stream)
            receipt = dict(trace=path.name, trace_sha256=module.sha256(path), chunks=1, completed=True,
                initial_state_vector_sha256='actual-state', episode_random_seed=456)
            if name == 'reference':
                receipt.update(method='native', support_budget=0, role='query')
                index = dict(completed=True, case=target, weights_unchanged=True, episodes=[receipt])
                contract = dict(motor_receipt_sha256=PI05_RECEIPT, simulator=dict(instruction='do the task'))
            else:
                receipt.update(milestone=0, query_behavior='native', target_optimizer_updates=0)
                index = dict(completed=True, case=target, target_only=True, query_receipts=[receipt],
                    training_receipts=[dict(instruction='do the task')])
                contract = dict(motor_receipt_sha256=PI05_RECEIPT)
            write_json(directory / 'index.json', index)
            write_json(directory / 'contract.json', contract)
            receipts.append((directory, module.sha256(directory / 'index.json'), module.sha256(directory / 'contract.json')))
        entries.append(dict(task_id=task, reference_index=str(receipts[0][0] / 'index.json'),
            reference_index_sha256=receipts[0][1], reference_contract_sha256=receipts[0][2],
            dsrl_index=str(receipts[1][0] / 'index.json'), dsrl_index_sha256=receipts[1][1], dsrl_contract_sha256=receipts[1][2]))
    path = tmp_path / 'inputs.json'
    write_json(path, dict(format='openpi_motor_replay_inputs_v1', cases=entries))
    return path, module.sha256(path)


def test_same_real_inputs_and_different_reference_actions_are_preserved(tmp_path):
    path, digest = inputs_manifest(tmp_path)
    inputs, manifest = module.load_inputs(path, digest)
    assert [value['task_id'] for value in inputs] == [0, 1]
    assert len(manifest['cases']) == 2
    for value in inputs:
        np.testing.assert_array_equal(value['reference_observation']['state'], value['dsrl_observation']['state'])
        delta = module.action_differences(value['reference_actions'], value['dsrl_actions'])
        assert delta['first_primitive']['maximum_absolute_error'] == pytest.approx(.002)
        assert not delta['horizon']['exactly_equal']


@pytest.mark.parametrize('changed', ('observation', 'noise'))
def test_different_initial_inputs_cannot_be_reported_as_runtime_effects(tmp_path, changed):
    path, digest = inputs_manifest(tmp_path, change_observation=changed == 'observation', change_noise=changed == 'noise')
    with pytest.raises(ValueError, match='Cannot isolate runtime effects'):
        module.load_inputs(path, digest)


def test_manifest_and_trace_checksum_changes_are_rejected(tmp_path):
    path, digest = inputs_manifest(tmp_path)
    manifest = json.loads(path.read_text())
    trace = module.Path(manifest['cases'][0]['reference_index']).parent / 'trace.pkl.gz'
    trace.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        module.load_inputs(path, digest)
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        module.load_inputs(path, digest)


def test_key_types_repetitions_and_both_input_origins_are_compared(tmp_path, monkeypatch):
    path, digest = inputs_manifest(tmp_path)
    inputs, _ = module.load_inputs(path, digest)
    class Motor:
        policy = SimpleNamespace(_rng=None)
        def infer(self, observation, instruction, noise):
            seed, kind = self.policy._rng
            assert seed == 456 and instruction == 'do the task' and noise.shape == (10, 32)
            return np.full((10, 7), .003 if kind == 'legacy' else 0., np.float32)
    monkeypatch.setattr(module, 'key', lambda seed, kind: (seed, kind))
    monkeypatch.setattr(module, 'runtime_configuration', lambda: dict(test=True))
    result, arrays = module.replay_stage(Motor(), inputs, 'plain', deadline=float('inf'), key_order=('legacy', 'typed'))
    assert len(result['predictions']) == len(arrays) == 16
    for row in result['comparisons']:
        expected = .003 if row['kind'] == 'typed_vs_legacy' else 0.
        assert row['delta']['horizon']['maximum_absolute_error'] == pytest.approx(expected)
    with pytest.raises(RuntimeError, match='elapsed-time'):
        module.replay_stage(Motor(), inputs, 'plain', deadline=0.)
