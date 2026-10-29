"""Allocated diagnostic: real source recovery reproduces the next SAC update.

Reads trusted, locally generated recoveries without changing their files.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_arrays(value, prefix=''):
    import numpy as np
    if isinstance(value, dict):
        return {key: array for name in sorted(value) for key, array in
                state_arrays(value[name], prefix + '/' + str(name)).items()}
    if isinstance(value, (list, tuple)):
        return {key: array for index, child in enumerate(value) for key, array in
                state_arrays(child, prefix + '/' + str(index)).items()}
    return {prefix: np.asarray(value)}


def compare_arrays(left, right):
    import numpy as np
    if left.keys() != right.keys():
        return dict(exact=False, numerical=False, reason='State keys differ')
    changes = []
    for key in left:
        a, b = left[key], right[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            return dict(exact=False, numerical=False, reason='Shape/dtype differs', key=key)
        if not np.array_equal(a, b):
            floating = np.issubdtype(a.dtype, np.floating)
            changes.append(dict(key=key, maximum_absolute_delta=float(np.max(np.abs(
                a.astype(np.float64) - b.astype(np.float64)))),
                numerical=bool(floating and np.allclose(a, b, rtol=1e-4, atol=1e-5))))
    return dict(exact=not changes, numerical=all(row['numerical'] for row in changes),
        changed_arrays=len(changes), largest_changes=sorted(changes,
            key=lambda row: row['maximum_absolute_delta'], reverse=True)[:8],
        rtol=1e-4, atol=1e-5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--recovery', type=Path, required=True)
    ap.add_argument('--dsrl-root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if sys.platform != 'linux':
        raise ValueError('Model diagnostics require allocated Linux')
    from fluffy_bassoon.gpu_execution import validate_execution
    validate_execution(os.environ.get('FLUFFY_GPU_EXECUTION', 'baige'), os.environ)
    import jax
    import numpy as np
    from flax import serialization
    from fluffy_bassoon.meta_rl.vla_libero import OfficialSAC
    from fluffy_bassoon.meta_rl.vla_source40_corpus import restore_motor_rng
    before = digest(args.recovery)
    with args.recovery.open('rb') as stream:
        payload = pickle.load(stream)
    contract = payload['contract']
    assert contract['source_only'] and not contract['source_data_from_targets']
    assert contract['motor']['receipt_sha256'] == 'ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170'
    outputs, state_values, before_checks = [], [], []
    original_state = state_arrays(serialization.msgpack_restore(payload['agent']))
    for _ in range(2):
        learner = OfficialSAC(args.dsrl_root, contract['learner_seed'])
        sampler = np.random.default_rng(0)
        progress = learner.restore(args.recovery, sampler, contract)
        restored_state = state_arrays(serialization.to_state_dict(learner.agent._save_dict))
        check = compare_arrays(original_state, restored_state)
        before_checks.append(check)
        assert check['exact'], 'Saved model/optimizer state was not restored exactly'
        metrics = learner.update(256, sampler)
        assert all(np.isfinite(value) for value in metrics.values())
        assert learner.updates == payload['updates'] + 1
        outputs.append(dict(agent=hashlib.sha256(serialization.to_bytes(learner.agent._save_dict)).hexdigest(),
            agent_rng=np.asarray(learner.agent._rng).tolist(), sampler=sampler.bit_generator.state,
            next_motor_key=np.asarray(jax.random.key_data(jax.random.split(
                restore_motor_rng(progress['motor_rng'])))).tolist(), metrics=metrics))
        state_values.append(state_arrays(serialization.to_state_dict(learner.agent._save_dict)))
    comparison = compare_arrays(*state_values)
    metrics_check = compare_arrays(*[state_arrays(row['metrics']) for row in outputs])
    rng_equal = all(outputs[0][key] == outputs[1][key] for key in
                    ('agent_rng', 'sampler', 'next_motor_key'))
    assert digest(args.recovery) == before, 'Diagnostic changed source recovery'
    result = dict(passed=comparison['numerical'] and metrics_check['numerical'] and rng_equal,
        recovery_sha256=before, restored_updates=payload['updates'],
        checked_next_update=payload['updates'] + 1, replay_rows=len(payload['replay']),
        source_unchanged=True, restored_states_exact=all(row['exact'] for row in before_checks),
        rng_and_sampler_exact=rng_equal, next_update_arrays=comparison,
        next_update_metrics=metrics_check, serialized_bytes_equal=outputs[0]['agent'] == outputs[1]['agent'],
        serialized_hashes=[row['agent'] for row in outputs],
        interpretation='Saved state and RNG must restore exactly; independent GPU update compilations are checked numerically, without a bitwise continuation claim',
        compared=['actor/critic/target/temperature/optimizer state', 'SAC RNG',
                  'replay sampler', 'next motor PRNG split', 'finite update metrics'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result['passed']:
        raise AssertionError('Restored SAC continuation differs; see numerical diagnostic receipt')


if __name__ == '__main__':
    main()
