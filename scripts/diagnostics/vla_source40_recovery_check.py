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
    outputs = []
    for _ in range(2):
        learner = OfficialSAC(args.dsrl_root, contract['learner_seed'])
        sampler = np.random.default_rng(0)
        progress = learner.restore(args.recovery, sampler, contract)
        metrics = learner.update(256, sampler)
        assert all(np.isfinite(value) for value in metrics.values())
        assert learner.updates == payload['updates'] + 1
        outputs.append(dict(agent=hashlib.sha256(serialization.to_bytes(learner.agent._save_dict)).hexdigest(),
            agent_rng=np.asarray(learner.agent._rng).tolist(), sampler=sampler.bit_generator.state,
            next_motor_key=np.asarray(jax.random.key_data(jax.random.split(
                restore_motor_rng(progress['motor_rng'])))).tolist(), metrics=metrics))
    assert outputs[0] == outputs[1], 'Restored SAC continuation differs'
    assert digest(args.recovery) == before, 'Diagnostic changed source recovery'
    result = dict(passed=True, recovery_sha256=before, restored_updates=payload['updates'],
        checked_next_update=payload['updates'] + 1, replay_rows=len(payload['replay']),
        source_unchanged=True, next_update_sha256=outputs[0]['agent'],
        compared=['actor/critic/target/temperature/optimizer state', 'SAC RNG',
                  'replay sampler', 'next motor PRNG split', 'finite update metrics'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
