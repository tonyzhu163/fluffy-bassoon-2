#!/usr/bin/env python3
"""Read exactly two SHA-bound reset records; no replay, model or guard change."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import pickle
import sys
import time

EXPECTED = {'reference': '50f9dd77c52fb39b08763d5d5adaacb31a50b810c9bc767f5f594e6d59022d94',
            'failed': '1361cdf53df3ae1422990628c3513502bd86bc33dfda8f9be67a4509090a3e47'}
HASHES = ('initial_observation_sha256', 'initial_state_vector_sha256', 'initial_simulator_state_sha256')
PUBLIC = ('image', 'wrist', 'state')


def guard(deadline):
    if time.time() >= deadline:
        raise TimeoutError('Original owner-bound CPU deadline exhausted')


def sha(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            guard(deadline); digest.update(block)
    return digest.hexdigest()


def term(left, right):
    available = left is not None and right is not None
    return dict(status='AVAILABLE' if available else 'UNAVAILABLE', reference=left, failed=right,
                equal=(left == right) if available else None)


def compare_array(left, right, deadline):
    import numpy as np
    guard(deadline); a, b = np.asarray(left), np.asarray(right)
    if a.dtype.kind not in 'buif' or b.dtype.kind not in 'buif':
        raise ValueError('Saved comparison arrays must be real numeric values')
    same_shape, same_dtype = a.shape == b.shape, a.dtype == b.dtype
    result = dict(reference_dtype=str(a.dtype), failed_dtype=str(b.dtype), reference_shape=list(a.shape),
                  failed_shape=list(b.shape), shape_equal=same_shape, dtype_equal=same_dtype,
                  reference_finite=bool(np.isfinite(a).all()), failed_finite=bool(np.isfinite(b).all()),
                  value_equal=bool(same_shape and np.array_equal(a, b)),
                  byte_equal=bool(same_shape and same_dtype and a.tobytes() == b.tobytes()),
                  mismatching_elements=None, byte_mismatching_elements=None, maximum_absolute_delta=None, first_differences=[])
    if same_shape:
        different = (a != b).reshape(-1)
        result['mismatching_elements'] = int(np.count_nonzero(different))
        if same_dtype:
            byte_diff = np.any(np.frombuffer(a.tobytes(), np.uint8).reshape(-1, a.itemsize) !=
                               np.frombuffer(b.tobytes(), np.uint8).reshape(-1, b.itemsize), axis=1)
            result['byte_mismatching_elements'] = int(np.count_nonzero(byte_diff)); different = different | byte_diff
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        if result['reference_finite'] and result['failed_finite']:
            result['maximum_absolute_delta'] = float(delta.max()) if delta.size else 0.
        for index in np.flatnonzero(different)[:5]:
            scalar = lambda x: x.item() if np.isfinite(x) else str(x)
            result['first_differences'].append(dict(index=list(map(int, np.unravel_index(int(index), a.shape))),
                reference=scalar(a.reshape(-1)[index]), failed=scalar(b.reshape(-1)[index])))
    result['exact'] = result['shape_equal'] and result['dtype_equal'] and result['value_equal'] and result['byte_equal']
    return result


def diagnose(reference_path, reference_sha, failed_path, failed_sha, *, deadline):
    """Synthetic callers may use their own hashes; CLI locks the two real hashes."""
    if sys.platform != 'linux':
        raise RuntimeError('Deserialization and array analysis require allocated Linux CPU')
    refs = {'reference': (Path(reference_path), reference_sha), 'failed': (Path(failed_path), failed_sha)}
    proof = {}
    # Verify BOTH input hashes before unpickling EITHER record.
    for role, (path, expected) in refs.items():
        guard(deadline)
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise ValueError('Explicit absolute immutable regular input required')
        actual = sha(path, deadline)
        if actual != expected:
            raise ValueError('Input SHA mismatch before unpickle: ' + role)
        proof[role] = dict(path=str(path), declared_sha256=expected, before_sha256=actual,
                           bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    with gzip.open(refs['reference'][0], 'rb') as stream: reference = pickle.load(stream)
    with gzip.open(refs['failed'][0], 'rb') as stream: failed = pickle.load(stream)
    import numpy as np
    if (not isinstance(reference, dict) or not {'initial', 'features', 'instruction'} <= reference.keys()
            or not isinstance(failed, dict) or not {'initial_state', 'scene_seed', 'reset_result'} <= failed.keys()
            or not isinstance(reference['initial'], dict) or not isinstance(failed['reset_result'], dict)):
        raise ValueError('Expected case-reference {initial,features,instruction} and captured raw reset schema')
    first, second = reference['initial'], failed['reset_result']
    if any(not isinstance(value.get('observation'), dict) or set(value['observation']) != set(PUBLIC) for value in (first, second)):
        raise ValueError('Expected saved public image/wrist/state arrays')
    public = {key: compare_array(first['observation'][key], second['observation'][key], deadline) for key in PUBLIC}
    hashes = {key: term(first.get(key), second.get(key)) for key in HASHES}
    vectors = {key: (dict(status='AVAILABLE', **compare_array(first[key], second[key], deadline))
                    if key in first and key in second else dict(status='UNAVAILABLE', reference_present=key in first, failed_present=key in second))
               for key in ('initial_state_vector', 'initial_simulator_state')}
    request = dict(initial_state=term(first.get('initial_state'), failed.get('initial_state')),
                   returned_initial_state=term(first.get('initial_state'), second.get('initial_state')),
                   scene_seed=term(first.get('scene_seed'), failed.get('scene_seed')))
    instruction = term(reference.get('instruction'), failed.get('instruction', second.get('instruction')))
    integrity = {}
    for role, value in (('reference', first), ('failed', second)):
        computed = hashlib.sha256(b''.join(np.asarray(value['observation'][k]).tobytes() for k in PUBLIC)).hexdigest()
        integrity[role] = dict(computed_public_sha256=computed,
            recorded_public_hash_matches=(computed == value['initial_observation_sha256']) if value.get('initial_observation_sha256') else None)
        if 'initial_simulator_state' in value:
            digest = hashlib.sha256(np.asarray(value['initial_simulator_state']).tobytes()).hexdigest()
            integrity[role]['recorded_simulator_hash_matches'] = (digest == value['initial_simulator_state_sha256']) if value.get('initial_simulator_state_sha256') else None
    available_terms = [v['equal'] for v in [*hashes.values(), *request.values(), instruction] if v['equal'] is not None]
    arrays_equal = all(v['exact'] for v in public.values()) and all(v['exact'] for v in vectors.values() if v['status'] == 'AVAILABLE')
    metadata_equal = all(available_terms)
    report = dict(format='libero_assessment_two_record_reset_diagnosis_v1', public=public, initial_hash_terms=hashes,
        preserved_vectors=vectors, reset_request=request, instruction=instruction, recorded_hash_integrity=integrity,
        saved_same_public_value_term=all(v['shape_equal'] and v['dtype_equal'] and v['value_equal'] for v in public.values()),
        pair_equal=bool(arrays_equal and metadata_equal), pair_equal_scope='Available saved reset evidence only; not a reconstruction or PASS of the full production guard',
        classification=('actual_saved_array_difference' if not arrays_equal else 'metadata_or_hash_only_difference_in_available_evidence' if not metadata_equal else 'available_saved_evidence_exact'),
        features=dict(reference_present=True, reference_shape=list(np.asarray(reference['features']).shape),
                      reference_dtype=str(np.asarray(reference['features']).dtype), failed_status='UNAVAILABLE', compared=False),
        full_guard_reconstructable=False,
        limits=['FAILED computed4104 features were not saved; no feature equality or causal attribution can be inferred.',
                'Missing instruction/scene seed/nominal vector remains UNAVAILABLE; no values inferred from filenames or other episodes.',
                'No native160 raw record, source/actor/checkpoint/model, replay, simulator or VLA recomputation read.',
                'No tolerance relaxation, prescribed repair, failed-role replay or recovery/admission decision.'], read_only_proof=proof)
    for role, (path, _) in refs.items():
        after = sha(path, deadline); proof[role].update(after_sha256=after, unchanged=after == proof[role]['before_sha256'])
        if not proof[role]['unchanged'] or path.stat().st_mtime_ns != proof[role]['mtime_ns']:
            raise ValueError('Input changed during read-only comparison')
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference-path', 'reference-sha256', 'failed-path', 'failed-sha256', 'output', 'deadline-unix'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args(argv); deadline = float(args.deadline_unix)
    if not math.isfinite(deadline): raise ValueError('Finite actual owner CPU deadline required')
    if args.reference_sha256 != EXPECTED['reference'] or args.failed_sha256 != EXPECTED['failed']:
        raise ValueError('CLI is restricted to the TWO admitted record identities')
    output = Path(args.output)
    if not output.is_absolute() or output.exists(): raise ValueError('Fresh absolute JSON output required')
    result = diagnose(args.reference_path, args.reference_sha256, args.failed_path, args.failed_sha256, deadline=deadline)
    encoded = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    if len(encoded) > 1 << 20: raise ValueError('Small diagnosis JSON bound exceeded')
    guard(deadline)
    with output.open('xb') as stream: stream.write(encoded)


if __name__ == '__main__': main()
