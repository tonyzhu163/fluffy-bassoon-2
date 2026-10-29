"""Read-only audit of saved robot data; constructs no model or simulator.

Run on Baige with the saved-data Python, passing the immutable output root.
The report is written to stdout. By default all28 groups must be terminal.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import warnings

import numpy as np

warnings.filterwarnings('ignore', category=DeprecationWarning)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def unpack(path):
    with (gzip.open(path, 'rb') if str(path).endswith('.gz') else Path(path).open('rb')) as stream:
        return pickle.load(stream)


def equal(a, b):
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, type(a)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    output = args.root / 'experiment'
    contract = read(output / 'contract.json')
    plan = contract['groups']
    assert len(plan) == 28 and contract['episode_ceiling'] == 76
    assert contract['maximum_total_actions'] == 32000
    summary = []
    seen = {}
    trajectories = {}
    candidate_files = {}
    total_actions = episodes_count = chunks_count = contexts_count = 0
    runtime_ids = set()
    for ordinal, group in enumerate(plan):
        path = output / f'group-{ordinal:03d}'
        if not (path / 'index.json').exists():
            assert args.allow_incomplete, f'Missing completed group{ordinal}'
            break
        result = read(path / 'index.json')
        local_contract = read(path / 'contract.json')
        for candidate in local_contract['candidates'].values():
            if candidate.get('path') and candidate['path'] not in candidate_files:
                assert sha(candidate['path']) == candidate['sha256']
                candidate_files[candidate['path']] = candidate['sha256']
        assert result['completed'] and result['case'] == group['case'] and result['weights_unchanged']
        assert result['budget_matched_primary_eligible']
        assert [(t['method'], t['support_budget']) for t in result['trials']] == [
            (group['method'], budget) for budget in group['support_budgets']]
        expected_episodes = sum(b + 1 for b in group['support_budgets'])
        assert len(result['episodes']) == expected_episodes
        assert sha(path / 'evaluation-recovery.pkl') == result['latest_recovery_sha256']
        recovery = unpack(path / 'evaluation-recovery.pkl')
        assert recovery['trial_index'] == len(group['support_budgets']) and recovery['next_episode'] == 0
        assert len(recovery['records']) == expected_episodes
        assert recovery['motor_rng'] is not None and recovery['sampler'] is not None
        contexts = read(path / 'contexts/index.json')
        assert len(contexts) == expected_episodes
        for ctx in contexts:
            assert Path(ctx['path']).name == ctx['path']
            assert sha(path / 'contexts' / ctx['path']) == ctx['sha256']
            saved = unpack(path / 'contexts' / ctx['path'])
            assert saved['global_rng'].keys() >= {'python', 'numpy', 'torch', 'cuda'}
            if group['method'] != 'native':
                assert saved['motor_rng'] is not None
                assert 'controller_before_episode' in saved and 'controller_after_episode' in saved
            contexts_count += 1
        journal = [read(p) for p in (path / 'physical-journal').glob('*.json')]
        assert all(row['returned'] for row in journal), 'Unresolved RPC must be charged and reported'
        charge = sum(row['actual_steps'] for row in journal)
        assert charge == sum(e['primitive_steps'] + e['settling_steps'] for e in result['episodes'])
        assert sum(row['kind'] == 'reset' for row in journal) == expected_episodes
        total_actions += charge
        invocations = read(path / 'runtime-invocations.json')
        assert len(invocations) == 1
        runtime_ids.add(invocations[0]['runtime_id'])
        for episode in result['episodes']:
            trace = Path(episode['trace'])
            assert not trace.is_absolute() and '..' not in trace.parts
            assert sha(path / trace) == episode['trace_sha256']
            rows = unpack(path / trace)
            assert rows and len(rows) == episode['chunks']
            assert sum(row['primitive_steps'] for row in rows) == episode['primitive_steps']
            assert rows[-1]['done'] and bool(rows[-1]['success']) == episode['success']
            assert all(not row['done'] for row in rows[:-1])
            assert all(equal(a['motor_next_observation'], b['motor_observation']) for a, b in zip(rows, rows[1:]))
            native_rng = np.random.default_rng(episode['episode_random_seed'])
            for row in rows:
                assert row['executed_actions'].shape == (row['primitive_steps'], 7)
                assert len(row['raw_rewards']) == row['primitive_steps']
                assert np.isfinite(row['executed_actions']).all()
                assert row['native_initial_noise'].shape == (10, 32)
                assert row['reward'] == (0. if row['success'] else -1.)
                if group['method'] != 'native':
                    assert row['executed_noise'].shape == (32,)
                    assert np.array_equal(row['native_initial_noise'], np.repeat(row['executed_noise'][None], 10, axis=0))
                    assert np.max(np.abs(row['executed_noise'])) <= 1.
                else:
                    assert np.array_equal(row['native_initial_noise'], native_rng.standard_normal((10, 32)).astype(np.float32))
            key = group['case']['case_id']
            identity = (episode['initial_state_vector_sha256'], episode['initial_observation_sha256'])
            old = seen.setdefault(key, (identity, rows[0]['motor_observation']))
            assert old[0] == identity and equal(old[1], rows[0]['motor_observation'])
            qkey = (key, group['source_condition'], group['method'], episode['role'], episode['support_budget'])
            trajectories[qkey] = (episode, rows)
            if episode['role'] == 'query':
                expected = hashlib.sha256(json.dumps(dict(case=group['case'], role='query', ordinal=0),
                    sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
                assert episode['episode_random_seed'] == int(expected[:8], 16)
            episodes_count += 1
            chunks_count += len(rows)
        summary.append(dict(group=ordinal, task_id=group['case']['task_id'], source=group['source_condition'],
            method=group['method'], charged_actions=charge, trials=result['trials']))
    pair_audits = []
    for key in seen:
        for source in ('source1', 'source2'):
            for objective in ('ad', 'dpt'):
                prefix = (key, source)
                full = prefix + (objective + '_full',)
                reset = prefix + (objective + '_episode_reset',)
                if all(k in trajectories for k in (full + ('support', 1), reset + ('support', 1))):
                    support_same = equal(trajectories[full + ('support', 1)][1], trajectories[reset + ('support', 1)][1])
                    query0_same = equal(trajectories[full + ('query', 0)][1], trajectories[reset + ('query', 0)][1])
                    assert support_same and query0_same
                    pair_audits.append(dict(case_id=key, source=source, objective=objective,
                        actual_support_and_empty_query_exact=True))
                static = prefix + (objective + '_static',)
                if all(static + ('query', k) in trajectories for k in (0, 1)):
                    assert equal(trajectories[static + ('query', 0)][1], trajectories[static + ('query', 1)][1])
    completed = len(summary) == 28
    if completed:
        final = read(output / 'index.json')
        assert final['completed'] and final['motor_parameters_unchanged']
        assert final['strict_common_runtime_primary_eligible'] and len(runtime_ids) == 1
        assert final['deployment_optimizer_updates'] == 0 and episodes_count == 76
        assert final['charged_total_actions'] == total_actions <= 32000
        assert len(pair_audits) == 8
        assert [row['group'] for row in final['completed_groups']] == plan
        for row in final['completed_groups']:
            assert sha(row['result_path']) == row['result_sha256']
        assert (args.root / 'exit-code.txt').read_text().strip() == '0'
        assert len(candidate_files) == 8
        for metadata in contract['target_final_candidates'].values():
            for name, path in metadata['paths'].items():
                assert sha(path) == metadata['hashes'][name]
    print(json.dumps(dict(read_only_saved_artifact_audit=True, completed=completed,
        source_commit=contract['source_commit'], completed_groups=len(summary), actual_episodes=episodes_count,
        actual_chunks=chunks_count, complete_contexts_hashed=contexts_count, charged_actions=total_actions,
        runtime_ids=sorted(runtime_ids), immutable_student_files_hashed=candidate_files,
        paired_memory_audits=pair_audits, groups=summary), indent=2))


if __name__ == '__main__':
    main()
