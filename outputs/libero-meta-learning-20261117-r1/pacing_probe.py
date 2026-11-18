"""Compare one timing-aware source strategy to 16 already measured fitting resets."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from openpi.policies import policy_config
from openpi.training import config

from job import ROOT
from reference import CHECKPOINT, observation, write
from simulator import Simulator


def replay(policy, baseline, output):
    case = baseline['case']
    assert case['variant'] == 'S20' and baseline['attempt'] == 1
    simulator = Simulator()
    output.mkdir()
    try:
        spec, first = simulator.call('create', case)
        initial = first['environment_reproducibility']['initial_state_fingerprint']
        assert initial == baseline['initial_state']
        raw = first['observation']
        rng = np.random.default_rng(case['seed'] + 817117)
        issued, wait_mask, query_offsets = [], [], []
        done = False
        while not done:
            query_offsets.append(len(issued))
            item = observation(raw, baseline['prompt'])
            noise = rng.standard_normal((10, 32)).astype(np.float32)
            chunk = policy.infer(item, noise=noise)['actions']
            assert chunk.shape == (10, 7) and np.isfinite(chunk).all()
            wait = np.zeros((14, 7), dtype=chunk.dtype)
            wait[:, -1] = chunk[-1, -1]
            for index, action in enumerate(np.concatenate((chunk, wait))):
                step = simulator.call('step', action)
                issued.append(action.copy())
                wait_mask.append(index >= len(chunk))
                raw = step['observation']
                done = step['terminated'] or step['truncated']
                if done:
                    break
        assert len(issued) <= spec['max_steps']
        np.savez_compressed(output/'commands.npz', issued=issued, is_wait=wait_mask, query_offsets=query_offsets)
        row = dict(case=case, source_trial=baseline['trial'], initial_state=initial,
                   baseline_success=baseline['success'], baseline_controls=baseline['controls'],
                   paced_success=bool(step['success']), paced_controls=len(issued), waits=sum(wait_mask))
        write(output/'result.json', row)
        print(json.dumps(row), flush=True)
        return row
    finally:
        simulator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out = parser.parse_args().output
    out.mkdir()
    assert len(jax.devices()) == 1 and jax.devices()[0].platform == 'gpu'
    baseline = ROOT/'outputs'/Path(__file__).parent.name/'devcollect01/reference/progress.json'
    rows = [r for r in json.loads(baseline.read_text()) if r['attempt'] == 1 and r['case']['variant'] == 'S20']
    assert len(rows) == 16 and len({(r['case']['suite'], r['case']['task_id']) for r in rows}) == 8
    policy = policy_config.create_trained_policy(config.get_config('pi05_libero'), str(CHECKPOINT))
    results = []
    for row in rows:
        results.append(replay(policy, row, out/f"trial-{row['trial']:03d}"))
        write(out/'progress.json', results)


if __name__ == '__main__':
    main()
