"""Independent Linux CPU reduction of the completed physical controller pilot."""
import argparse
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*2**20), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    # Torch's ZIP reader seeks repeatedly; seeking through a gzip/FUSE stream
    # repeatedly decompresses and re-reads the object. Decode once into RAM.
    return torch.load(io.BytesIO(gzip.decompress(path.read_bytes())),
                      map_location='cpu', weights_only=False)


def array(value):
    return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    assert os.uname().sysname == 'Linux' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not torch.cuda.is_initialized() and not args.output.exists()
    result = json.loads((args.root/'RESULT.json').read_bytes())
    exit_receipt = json.loads((args.root/'EXIT.json').read_bytes())
    assert result['status'] == 'completed' and exit_receipt['exit_code'] == 0
    cases = result['cases']
    count = len(cases)
    assert count in (16, 64) and count == len(set(cases)) and result['optimizer_updates'] == 0
    outcomes, files, initial_commands, initial_observations = {}, {}, {}, {}
    total_decisions = total_delivered = total_physics = total_requested = 0
    from fluffy_bassoon.rl.robotwin_noise import _observation_hash
    for name, profile in result['profiles'].items():
        assert profile['owned_workers_closed'] and profile['frozen_model_unchanged']
        assert len(profile['resets']) == len(profile['episodes']) == len(cases)
        assert [r['actual_seed'] for r in profile['resets']] == cases
        episodes = {case: dict(success=False, decisions=0, delivered_targets=0, physics_steps=0) for case in cases}
        freezes = failures = empty = 0
        for trace in profile['traces']:
            paths = {key: args.root/trace[key] for key in ('intent', 'transitions')}
            for key, path in paths.items():
                files[trace[key]] = dict(sha256=digest(path), bytes=path.stat().st_size)
            intent, transitions = load(paths['intent']), load(paths['transitions'])
            metadata = intent['native_batch']
            assert metadata['batch_size'] == 2 and metadata['fixed_native_batch']
            assert [t['native_batch']['row'] for t in transitions] == metadata['active_lanes']
            total_requested += 32*len(transitions)
            for row, decision_id in enumerate(metadata['decision_ids']):
                assert decision_id.startswith('paired-controller/case-')
                if decision_id.endswith('/chunk-0'):
                    key = decision_id
                    signature = _observation_hash(intent['observations'][row])
                    if key in initial_observations:
                        assert initial_observations[key] == signature
                    else:
                        initial_observations[key] = signature
                    if profile['num_steps'] == 5:
                        command = array(intent['commands'][row])
                        if key in initial_commands:
                            np.testing.assert_array_equal(initial_commands[key], command)
                        else:
                            initial_commands[key] = command.copy()
            for transition in transitions:
                row = transition['native_batch']['row']
                case = int(transition['case_id'].split('case-')[1])
                episode = episodes[case]
                assert transition['chunk_index'] == episode['decisions']
                np.testing.assert_array_equal(array(transition['base_actions']), array(intent['commands'][row]))
                np.testing.assert_array_equal(array(transition['requested_actions']), array(intent['commands'][row]))
                assert _observation_hash(transition['observation']) == _observation_hash(intent['observations'][row])
                episode['decisions'] += 1
                episode['success'] |= bool(transition['audit']['success'])
                episode['complete'] = bool(transition['terminated'] or transition['truncated'])
                for key in ('delivered_targets', 'physics_steps'):
                    episode[key] += transition['execution'][key]
                for calls in transition['audit']['controller_calls'].values():
                    freezes += int(calls['full_chunk_freeze'])
                    failures += calls['failures']
                    empty += calls['empty_plans']
                    if profile['waypoint_controller']:
                        assert all(n == 2 for n in calls['path_points'])
        for saved in profile['episodes']:
            actual = episodes[saved['case']]
            assert actual['complete']
            for key, value in actual.items():
                assert saved[key] == value, (name, saved['case'], key)
            total_decisions += actual['decisions']
            total_delivered += actual['delivered_targets']
            total_physics += actual['physics_steps']
        assert freezes == profile['full_chunk_arm_freezes']
        assert failures == profile['planner_failures'] and empty == profile['empty_plans']
        assert sum(ep['success'] for ep in episodes.values()) == profile['successes']
        outcomes[name] = {str(case): ep['success'] for case, ep in episodes.items()}
    assert len(initial_observations) == len(initial_commands) == count
    assert total_requested == result['charged_requested_targets']
    comparisons = {}
    base = outcomes['legacy_k5_h192']
    for name, values in outcomes.items():
        wins = sum(values[k] and not base[k] for k in base)
        losses = sum(base[k] and not values[k] for k in base)
        n = wins+losses
        p = min(1., 2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
        comparisons[name] = dict(successes=sum(values.values()), total=len(cases), wins=wins,
                                  losses=losses, exact_two_sided_mcnemar_p=p)
    gate = (comparisons['waypoint_k10_h384']['successes'] >= count//4 and
            all(p['full_chunk_arm_freezes'] == 0 for p in result['profiles'].values() if p['waypoint_controller']))
    assert gate == result['gate_passed']
    report = dict(status='passed_independent_linux_cpu_physical_trace_reduction',
                  auditor_sha256=digest(Path(__file__)),
                  result_sha256=digest(args.root/'RESULT.json'), source_commit=result['source_commit'],
                  comparisons=comparisons, gate_passed=gate, paired_initial_observations=count,
                  paired_k5_initial_commands_exact=True, complete_episodes=3*count,
                  decisions=total_decisions, delivered_targets=total_delivered,
                  physics_steps=total_physics, charged_requested_targets=total_requested,
                  files=files, optimizer_updates=0, gpu_seconds=0)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
