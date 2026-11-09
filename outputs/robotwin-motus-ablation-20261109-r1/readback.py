"""Independently audit a completed paired ablation allocation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root, reference):
    load = lambda path: json.loads(path.read_bytes())
    result = load(root / 'RESULT.json')
    exit_ = load(root / 'EXIT.json')
    probe = load(root / 'gpu-probe.json')
    assert result['status'] == 'complete' and exit_['exit_code'] == 0
    assert result['frozen_parameters_unchanged']
    assert result['manifest'] == load(reference / 'REFERENCE-CPU-QUALIFICATION.json')['manifest']
    assert probe['status'] == 'passed' and len(probe['devices']) == 1
    assert not any(s in (root / 'evaluation.log').read_text()
                   for s in ('OIDN Error', 'unsupported device type', 'invalid handle'))
    arms = ('raw_take_action', 'default_prompt_take_action',
            'raw_continuous_chunk', 'default_prompt_continuous_chunk')
    assert set(result['arms']) == set(arms)
    pairs = result['paired_seeds']
    assert len(pairs) == result['target_episodes']
    seeds = [p['seed'] for p in pairs]
    rejected = [p['seed'] for p in result['rejected_candidates']]
    assert seeds == sorted(set(seeds)) and not set(seeds) & set(rejected)
    assert set(seeds + rejected) == set(range(100000, max(seeds) + 1))
    summaries, action_hashes, modes, state_diffs = {}, {}, {}, []
    for name in arms:
        arm = result['arms'][name]
        assert arm['completed'] == len(arm['episodes']) == len(pairs)
        assert arm['successes'] == sum(e['success'] for e in arm['episodes'])
        durations, physics = [], 0
        for i, episode in enumerate(arm['episodes']):
            folder = root / f'{name}-{i:03d}-seed-{seeds[i]}'
            assert load(folder / 'episode.json') == episode
            assert episode['complete'] and episode['expert_feasible']
            assert episode['instruction'] == pairs[i]['instruction']
            assert episode['seed'] == seeds[i] and episode['step_limit'] == 400
            assert 0 < episode['delivered_targets'] <= 400
            assert episode['success'] or episode['delivered_targets'] == 400
            assert sum(d['delivered'] for d in episode['decisions']) == episode['delivered_targets']
            assert pairs[i]['outcomes'][name]['success'] == episode['success']
            paths = sorted(folder.glob('actions-*.npy'))
            assert len(paths) == len(episode['decisions'])
            for j, path in enumerate(paths):
                assert path.name == f'actions-{j:03d}.npy'
                action = np.load(path, allow_pickle=False)
                assert action.shape == (32, 14) and np.isfinite(action).all()
                decision = episode['decisions'][j]
                assert decision['noise_seed'] == seeds[i] * 1000 + j
                assert 0 < decision['delivered'] <= decision['requested'] <= 32
                for side in ('left', 'right'):
                    mode = decision.get(side + '_mode')
                    if mode is not None:
                        modes[mode] = modes.get(mode, 0) + 1
                physics += decision.get('physics_steps', 0)
                action_hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            assert (folder / 'initial.jpg').is_file() and (folder / 'final.jpg').is_file()
            durations.append(episode['ended_unix'] - episode['started_unix'])
        summaries[name] = dict(successes=arm['successes'], episodes=len(pairs),
                               episode_seconds=durations, recorded_physics_steps=physics)
    for i, pair in enumerate(pairs):
        states = [np.asarray(result['arms'][a]['episodes'][i]['initial_state']) for a in arms]
        assert all(s.shape == (14,) and np.isfinite(s).all() for s in states)
        difference = max(float(np.max(np.abs(s - states[0]))) for s in states)
        assert difference <= 1e-8
        state_diffs.append(difference)
        for left, right in ((arms[0], arms[2]), (arms[1], arms[3])):
            paths = [root / f'{a}-{i:03d}-seed-{pair["seed"]}' / 'actions-000.npy'
                     for a in (left, right)]
            chunks = [np.load(p, allow_pickle=False) for p in paths]
            assert chunks[0].dtype == chunks[1].dtype
            assert chunks[0].tobytes() == chunks[1].tobytes()
        assert pair['decision0_chunk_l2']['raw_controller_diff'] == 0
        assert pair['decision0_chunk_l2']['default_prompt_controller_diff'] == 0
    renderers = [load(p) for p in (root / 'renderers').glob('*.json')]
    assert renderers and all(r['expected_cuda_pci'] == r['actual_render_pci'] for r in renderers)
    return dict(status='passed', source_commit=result['source_commit'], paired_seeds=seeds,
                arms=summaries, decision0_controller_chunks_bit_identical=True,
                max_initial_state_differences=state_diffs, frozen_parameters_unchanged=True,
                denoiser_errors=0, controller_modes=modes, action_sha256=action_hashes,
                result_sha256=hashlib.sha256((root / 'RESULT.json').read_bytes()).hexdigest(),
                worker_gpu_seconds=exit_['worker_gpu_seconds'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    value = audit(args.root, args.reference)
    args.output.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({k: v for k, v in value.items() if k != 'action_sha256'}))
