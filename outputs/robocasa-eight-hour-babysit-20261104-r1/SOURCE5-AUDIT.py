"""Recount fixed endpoint results from raw chunks and native receipts on Linux CPUs."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import torch
from scipy.stats import binomtest
from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash

RUN = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-eight-hour-babysit-20261104-r1')
SEEDS = dict(PickPlaceCounterToCabinet=73220000, CoffeeSetupMug=73250000, TurnOnSinkFaucet=73280000)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(task, *, qualification=False, folder=None):
    assert sys.platform == 'linux' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    folder = folder or RUN / 'source5-tests' / task / 'attempt-01'
    manifest = folder / 'MANIFEST.json'
    m = read(manifest)
    assert m['task'] == task and m['case_base'] == SEEDS[task] and m['cases'] == 512
    worker = folder / 'node-00'
    collection = read(folder / 'COLLECTION.json') if (folder / 'COLLECTION.json').exists() else None
    if collection:
        assert collection['manifest_sha256'] == sha(manifest)
        for source in collection['sources']:
            for key in ('probe', 'parity'):
                assert sha(Path(source[key]['path'])) == source[key]['sha256']
            probe = read(Path(source['probe']['path']))
            assert probe['status'] == 'passed' and len(probe['devices']) == 1
            assert probe['hostname'] == source['job_id'] + '-master-0'
            assert read(Path(source['parity']['path']))[0]['passed']
            if 'execution_manifest' in source:
                path = Path(source['execution_manifest']['path'])
                assert sha(path) == source['execution_manifest']['sha256']
                execution = read(path)
                assert {k:v for k,v in execution.items() if k != 'sequential_order'} == {
                    k:v for k,v in m.items() if k != 'sequential_order'}
                index = collection['sources'].index(source)
                assert execution['sequential_order'] == [m['sequential_order'][bank]
                    if collection['bank_sources'][str(bank)] == index else [] for bank in range(64)]
    else:
        probe = read(worker / 'GPU-PROBE.json')
        assert probe['status'] == 'passed' and len(probe['devices']) == 1
        assert read(worker / 'initial-parity.json')[0]['passed']
    cases = {key: {} for key in ('SFT', 'teacher')}
    ends = {key: 0 for key in cases}
    panels = controls = resets = chunks = images = image_exact = 0
    raw_digest = hashlib.sha256()
    for bank in range(1 if qualification else 64):
        base = worker / 'banks' / f'bank-{bank:03d}'
        if collection:
            source = collection['sources'][collection['bank_sources'][str(bank)]]
            assert base.resolve() == (Path(source['worker']) / 'banks' / f'bank-{bank:03d}').resolve()
        seeds = list(range(SEEDS[task] + bank*8, SEEDS[task] + bank*8 + 8))
        canonical = base / 'canonical-eval/rank-00/complete.json'
        c = read(canonical)
        assert sha(canonical.parent / 'initial.pt') == c['snapshot_sha256']
        assert c['case']['seeds'] == seeds
        reference = read(base / 'rng-reference.json')
        assert len(reference) == 12
        expected_order = ['SFT', 'teacher'] if bank % 2 == 0 else ['teacher', 'SFT']
        expected_order += ['repeat'] if bank == 0 else []
        assert m['sequential_order'][bank] == expected_order
        for label in expected_order:
            panel = read(base / f'{label}.json')
            if collection:
                source_panel = Path(source['worker']) / 'banks' / f'bank-{bank:03d}' / f'{label}.json'
                assert sha(source_panel) == collection['panel_sha256'][str(source_panel)]
            policy = 'SFT' if label == 'repeat' else label
            assert panel['load']['strict_load'] and panel['load']['sha256'] == m['policies'][policy]['sha256']
            assert panel['sampler_seed'] == SEEDS[task] + 10000000 + bank and panel['seeds'] == seeds
            path = Path(panel['episodes_path'])
            assert sha(path) == panel['episodes_sha256']
            initial = read(path.with_name(path.name.replace('-episodes.json', '-initial.json')))
            assert initial['canonical'] == panel['canonical'] and initial['report']['matched']
            assert (Path(initial['canonical']).resolve() == canonical.resolve())
            assert initial['report']['physical_state_exact'] and initial['report']['public_state_exact']
            image_exact += int(initial['report']['images_exact'])
            rows = read(path)['rows']
            assert panel['episodes'] == rows and [row['seed'] for row in rows] == seeds
            rewards = []
            final = None
            for chunk in range(1, 13):
                sampler = read(worker / 'sampler' / f'{panel["phase"]}-{chunk:02d}.json')
                assert [sampler['before_rng_sha256'], sampler['after_rng_sha256']] == reference[chunk-1]
                assert sampler['actions_shape'] == [8, 50, 12]
                chunk_path = Path(str(path).removesuffix('-episodes.json') + f'-chunk{chunk:02d}.pt.gz')
                with gzip.open(chunk_path, 'rb') as stream:
                    raw = torch.load(stream, map_location='cpu', weights_only=False)
                assert raw['phase'] == panel['phase'] and raw['seeds'] == seeds
                assert raw['iteration'] == m['policies'][policy]['step']
                assert raw['episode_ids'] == [f'{task}/{seed}' for seed in seeds]
                actions = torch.as_tensor(raw['decoded_actions'])
                assert tuple(actions.shape) == (8, 50, 12) and torch.isfinite(actions).all()
                assert sampler['actions_sha256'] == panel['sampler_actions'][chunk-1]
                # The sampler hash covers released model scores. EnvWorker's
                # qualified prepare_actions thresholds6/11 before this archive
                # records physical commands, so these stage hashes differ.
                assert torch.isin(actions[..., [6, 11]], torch.tensor([-1., 1.])).all()
                raw_digest.update(str(chunk_path.relative_to(RUN if collection else folder)).encode())
                raw_digest.update(tensor_hash(actions).encode())
                observation = raw['current_observation']
                assert tensor_hash(observation) == raw['current_observation_sha256']
                raw_digest.update(raw['current_observation_sha256'].encode())
                state = torch.as_tensor(observation['states'])
                assert state.shape[0] == 8 and torch.isfinite(state).all()
                for key in ('main_images', 'wrist_images', 'extra_view_images'):
                    value = torch.as_tensor(observation[key])
                    assert value.shape[0] == 8 and value.shape[-1] == 3 and value.dtype == torch.uint8
                    assert value.ndim == (5 if key == 'extra_view_images' else 4)
                    assert key != 'extra_view_images' or value.shape[1] == 1
                    images += 8
                reward = torch.as_tensor(raw['rewards'])
                assert tuple(reward.shape) == (8, 50) and torch.isfinite(reward).all()
                rewards.append(reward)
                final = torch.as_tensor(raw['final_info']['success']).bool().reshape(8)
                chunks += 1
            rewards = torch.cat(rewards, dim=1)
            assert tuple(rewards.shape) == (8, 600)
            for lane, row in enumerate(rows):
                hits = torch.nonzero(rewards[lane] > 0).flatten()
                first = int(hits[0]) + 1 if len(hits) else None
                assert len(hits) <= 1 and first == row['first_success_control']
                assert bool(len(hits)) == bool(row['success_once'])
                assert bool(final[lane]) == bool(row['success_at_end'])
                count = row['native_counts']
                assert count['controls'] - count['reset_controls'] == 600
                controls += 600
                resets += count['reset_controls']
                if label != 'repeat':
                    assert row['seed'] not in cases[label]
                    cases[label][row['seed']] = int(row['success_once'])
                    ends[label] += int(row['success_at_end'])
            panels += 1
    result = dict(verified=True, qualification_only=qualification, task=task, manifest_sha256=sha(manifest),
        cases_per_policy=len(cases['SFT']), policy_episodes=panels*8, matched_physical_public_starts=panels*8,
        matched_sampler_chunk_calls=chunks, hashed_raw_action_observation_chunks=chunks,
        decision_camera_frames=images, panels_with_exact_initial_RGB=image_exact, accounted_policy_controls=controls,
        reset_controls=resets, first_successes={key: sum(value.values()) for key, value in cases.items()},
        end_successes=ends, raw_first_success_and_end_outcomes_recounted=True, GPU_hours=0)
    result.update(raw_physical_command_observation_digest=raw_digest.hexdigest(),
        sampler_and_physical_stages_checked_separately=True,
        native_binary_mapping='8f9bfa7661928ef8c17494b04f4d7bf3125634e1:rlinf/envs/action_utils.py:203-222',
        limitation='Sampler hashes bind pre-threshold model scores; raw archives retain post-threshold physical commands, so their whole-tensor hashes cannot be directly equated.')
    if not qualification:
        combined = read(folder / 'RESULT.json')
        done = read(worker / 'RESULT.json')
        assert done['completed'] and done['banks'] == list(range(64))
        assert done['manifest_sha256'] == combined['manifest_sha256'] == sha(manifest)
        calls = read(worker / 'accounting/rank-00/node-00.json')['calls']
        if collection:
            selected = set()
            for bank in range(64):
                source_index = collection['bank_sources'][str(bank)]
                for label in m['sequential_order'][bank]:
                    panel = read(worker / 'banks' / f'bank-{bank:03d}' / f'{label}.json')
                    selected.add((source_index, panel['phase']))
            normalized = []
            for index, source in enumerate(collection['sources']):
                journal = Path(source['journal']['path'])
                assert sha(journal) == source['journal']['sha256']
                for call_index, row in enumerate(read(journal)['calls']):
                    normalized.append(dict(row, source_index=index, source_call_index=call_index,
                        executed_or_conservative=row.get('executed_or_conservative', row['requested']),
                        selected_for_readout=(index, row['phase'].split('/')[0]) in selected))
            assert calls == normalized
            assert all(row['resolved'] for row in calls if row['selected_for_readout'])
            discarded = sum(row['executed_or_conservative'] for row in calls if not row['selected_for_readout'])
        else:
            assert all(row['resolved'] for row in calls)
            discarded = 0
        charged = sum(row['executed_or_conservative'] for row in calls)
        assert panels == 129 and controls == 619200 and charged == controls + resets + discarded == combined['charged_controls']
        assert all(set(rows) == set(range(SEEDS[task], SEEDS[task]+512)) for rows in cases.values())
        assert result['first_successes'] == combined['first_successes'] and ends == combined['end_successes']
        differences = Counter(cases['teacher'][seed] - cases['SFT'][seed] for seed in cases['SFT'])
        wins, losses = differences[1], differences[-1]
        exact = binomtest(wins, wins+losses, .5).pvalue if wins+losses else 1.
        assert (wins, losses) == (combined['primary']['wins'], combined['primary']['losses'])
        assert abs(exact-combined['primary']['exact_McNemar_p']) < 1e-12
        result.update(combined_sha256=sha(folder / 'RESULT.json'), charged_controls=charged,
            independent_paired_test=dict(wins=wins, losses=losses, exact_McNemar_p=exact))
        if collection:
            result.update(input_collection_sha256=sha(folder / 'COLLECTION.json'),
                          discarded_recovery_controls_upper=discarded)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(SEEDS), required=True)
    parser.add_argument('--qualification', action='store_true')
    parser.add_argument('--folder', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.task, qualification=args.qualification, folder=args.folder), indent=2))
