"""Exactly four synthetic allocated-Linux tests; no real payload/model opened."""
from copy import deepcopy
import gzip
import hashlib
import importlib.util
from pathlib import Path
import pickle
import sys
import time

import pytest

pytestmark = pytest.mark.skipif(sys.platform != 'linux', reason='Allocated Linux CPU only')


@pytest.fixture
def inspector():
    path = Path(__file__).resolve().parents[1] / 'scripts/diagnostics/libero_assessment_reset_pair_diagnosis.py'
    spec = importlib.util.spec_from_file_location('reset_pair_inspector', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def saved(tmp_path):
    import numpy as np
    observation = dict(image=np.zeros((2, 2, 3), np.uint8), wrist=np.zeros((2, 2, 3), np.uint8),
                       state=np.zeros(8, np.float32))
    physics = np.asarray([1., 2., 3.], np.float64)
    initial = dict(observation=observation, initial_state=40, initial_simulator_state=physics,
        initial_observation_sha256=hashlib.sha256(b''.join(observation[k].tobytes() for k in ('image', 'wrist', 'state'))).hexdigest(),
        initial_state_vector_sha256='a' * 64, initial_simulator_state_sha256=hashlib.sha256(physics.tobytes()).hexdigest())
    reference = dict(initial=deepcopy(initial), features=np.zeros(4104, np.float32), instruction='saved public instruction')
    failed = dict(initial_state=40, scene_seed=123, reset_result=deepcopy(initial))
    def write(failed_value=failed):
        paths = []
        for name, value in (('reference', reference), ('failed', failed_value)):
            path = tmp_path / (name + '.pkl.gz')
            with gzip.open(path, 'wb') as stream: pickle.dump(value, stream, protocol=5)
            paths.extend([path, hashlib.sha256(path.read_bytes()).hexdigest()])
        return paths
    return reference, failed, write


def test_matching_available_saved_records_are_exact_but_full_guard_terms_remain_unavailable(inspector, saved):
    _, _, write = saved
    result = inspector.diagnose(*write(), deadline=time.time() + 10)
    assert result['pair_equal'] is True and result['classification'] == 'available_saved_evidence_exact'
    assert result['saved_same_public_value_term'] is True
    assert all(v['exact'] and v['mismatching_elements'] == 0 for v in result['public'].values())
    assert all(v['equal'] is True for v in result['initial_hash_terms'].values())
    assert result['preserved_vectors']['initial_simulator_state']['exact'] is True
    assert result['preserved_vectors']['initial_state_vector']['status'] == 'UNAVAILABLE'
    assert result['instruction']['status'] == result['reset_request']['scene_seed']['status'] == 'UNAVAILABLE'
    assert result['features']['reference_shape'] == [4104] and result['features']['failed_status'] == 'UNAVAILABLE'
    assert result['features']['compared'] is False and result['full_guard_reconstructable'] is False


def test_actual_array_difference_is_reported_even_when_identity_hash_strings_agree(inspector, saved):
    _, failed, write = saved
    failed['reset_result']['observation']['state'][0] = .125
    failed['reset_result']['observation']['wrist'][0, 0, 0] = 255
    result = inspector.diagnose(*write(failed), deadline=time.time() + 10)
    assert result['pair_equal'] is False and result['classification'] == 'actual_saved_array_difference'
    assert all(v['equal'] is True for v in result['initial_hash_terms'].values())
    state, wrist = result['public']['state'], result['public']['wrist']
    assert state['dtype_equal'] and state['shape_equal'] and not state['byte_equal'] and not state['value_equal']
    assert state['mismatching_elements'] == 1 and state['maximum_absolute_delta'] == .125
    assert state['first_differences'][0] == dict(index=[0], reference=0., failed=.125)
    assert wrist['mismatching_elements'] == 1 and wrist['maximum_absolute_delta'] == 255.
    assert result['recorded_hash_integrity']['failed']['recorded_public_hash_matches'] is False
    assert result['full_guard_reconstructable'] is False


def test_hash_metadata_difference_with_identical_arrays_is_distinguished_without_a_cause_claim(inspector, saved):
    _, failed, write = saved
    failed['reset_result']['initial_observation_sha256'] = 'f' * 64
    result = inspector.diagnose(*write(failed), deadline=time.time() + 10)
    assert result['pair_equal'] is False
    assert result['classification'] == 'metadata_or_hash_only_difference_in_available_evidence'
    assert result['saved_same_public_value_term'] is True and all(v['exact'] for v in result['public'].values())
    assert result['initial_hash_terms']['initial_observation_sha256']['equal'] is False
    assert result['preserved_vectors']['initial_simulator_state']['exact'] is True
    assert result['features']['failed_status'] == 'UNAVAILABLE' and result['features']['compared'] is False
    assert result['full_guard_reconstructable'] is False


def test_all_hashes_are_checked_before_any_unpickle_and_inputs_remain_byte_identical(inspector, saved, monkeypatch):
    _, _, write = saved
    paths = write(); originals = {Path(paths[i]): Path(paths[i]).read_bytes() for i in (0, 2)}
    loads = []
    def forbidden_load(*args, **kwargs):
        loads.append(True); raise AssertionError('Unpickle occurred before both SHA checks')
    with monkeypatch.context() as context:
        context.setattr(inspector.pickle, 'load', forbidden_load)
        with pytest.raises(ValueError, match='SHA mismatch before unpickle: failed'):
            inspector.diagnose(paths[0], paths[1], paths[2], '0' * 64, deadline=time.time() + 10)
    assert loads == []
    result = inspector.diagnose(*paths, deadline=time.time() + 10)
    assert set(result['read_only_proof']) == {'reference', 'failed'}
    for value in result['read_only_proof'].values():
        assert value['unchanged'] is True
        assert value['declared_sha256'] == value['before_sha256'] == value['after_sha256']
    assert all(path.read_bytes() == payload for path, payload in originals.items())
