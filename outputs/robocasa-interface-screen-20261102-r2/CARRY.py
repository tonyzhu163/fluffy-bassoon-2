"""Independently audit immutable native outcomes before skipping any rollout."""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fluffy_bassoon.meta_rl.robocasa_pairing import compare_native_pair

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
SOURCE = ROOT / 'robocasa-interface-screen-20261102-r1'
OUT = ROOT / 'robocasa-interface-screen-20261102-r2'
TASKS = ('OpenDrawer', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def motor_plan(case, draw):
    return np.random.default_rng(900000000 + case['native_seed'] * 100 + draw + 1).integers(0, 2**63 - 100000000, 12).tolist()


def verify(row, directory, case, draw, *, videos):
    truth_ref = row['truth']
    assert sha(truth_ref['path']) == truth_ref['sha256']
    truth = json.loads(Path(truth_ref['path']).read_text())
    assert truth['case'] == row['case'] == case and bool(truth['simulator_final_success']) == bool(row['success'])
    assert row['draw'] == draw
    assert truth['native_counts']['controls'] - truth['native_counts']['reset_controls'] == 600
    episode = row['episode']; key = episode['id']
    assert json.loads((directory / 'videos' / (key + '.json')).read_text()) == episode
    assert episode['completed'] and episode['horizon'] == episode['frame_clock']['physical_steps'] == 600
    assert episode['frame_clock']['frame_count'] == 601 and len(episode['videos']) == 3
    if videos:
        for video in episode['videos']:
            assert sha(video['path']) == video['sha256']
    pair_path = directory / 'videos' / (key + '-pairing.json')
    pair = json.loads(pair_path.read_text())
    assert sha(pair['snapshot']['path']) == pair['snapshot']['sha256']
    marker = json.loads(Path(pair['canonical']).read_text())
    canonical = Path(pair['canonical']).parent / 'initial.pt'
    assert sha(canonical) == marker['snapshot_sha256'] and marker['case'] == case
    before = torch.load(canonical, map_location='cpu', weights_only=False)
    after = torch.load(pair['snapshot']['path'], map_location='cpu', weights_only=False)
    result = compare_native_pair(before, after)
    assert result['matched'] and result['images'] == pair['report']['images']
    assert result['physical_state_exact'] and result['public_state_exact']
    return dict(truth=truth_ref, episode=ref(directory / 'videos' / (key + '.json')),
                pairing=ref(pair_path), canonical=ref(canonical), comparison=result)


def load_anchors(task):
    root = SOURCE / task
    bank = json.loads((root / 'FAILURE-BANK.json').read_text())
    records, refs = [], []
    for path in sorted(root.glob('anchor-*/records.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        assert len(payload['rows']) == 8 and len(payload['records']) == 96
        seeds = [seed for row in payload['rows'] for seed in motor_plan(row['case'], -1)]
        assert seeds == payload['planned_motor_seeds']
        for row, seed in zip(payload['records'], seeds, strict=True):
            assert row['noise_seed'] == seed and not np.any(row['control'])
            assert np.array_equal(row['base_chunk'], row['executed_chunk'])
        records.extend(payload['records']); refs.append(ref(path))
    assert len(records) == 12 * len(bank['all_anchors'])
    return bank, records, refs


def audit_task(task):
    root = SOURCE / task
    bank, records, anchor_refs = load_anchors(task)
    anchors = bank['all_anchors']
    anchor_evidence = []
    for index, row in enumerate(anchors):
        assert row['case']['slot'] == index and row['draw'] == -1
        anchor_evidence.append(verify(row, root / f'anchor-{index // 8 * 8:03d}', row['case'], -1, videos=True))
    assert [row['case'] for row in anchors if not row['success']][:20] == bank['selected']
    cases = [case for case in bank['selected'] for _ in range(8)]
    draws = list(range(8)) * 20
    retained, indices, evidence = [], [], []
    directory = root / 'native8'
    for truth_path in sorted((directory / 'monitoring').glob('*.json')):
        truth = json.loads(truth_path.read_text()); key = truth['id']
        assert key.startswith('native8-episode-')
        index = int(key.removeprefix('native8-episode-'))
        assert 0 <= index < 160 and index not in indices
        episode = json.loads((directory / 'videos' / (key + '.json')).read_text())
        assert episode['round'] == index
        row = dict(case=cases[index], draw=draws[index], success=bool(truth['simulator_final_success']),
                   truth=ref(truth_path), episode=episode, source_run='robocasa-interface-screen-20261102-r1',
                   original_slot=index, chunk_records_available=False,
                   source_contract_motor_seeds=motor_plan(cases[index], draws[index]))
        evidence.append(verify(row, directory, cases[index], draws[index], videos=True))
        retained.append(row); indices.append(index)
    qualification = root / 'INTERFACE-QUALIFICATION.json'
    qualified = json.loads(qualification.read_text()); assert qualified['completed']
    assert all(row['native_zero_bitwise_equal'] for row in qualified['rows'])
    return dict(task=task, bank=ref(root / 'FAILURE-BANK.json'), selected=bank['selected'],
        anchor_episodes=len(anchors), anchor_record_refs=anchor_refs, anchor_evidence=anchor_evidence,
        native_rows=retained, native_evidence=evidence, native_completed_indices=indices,
        native_missing_indices=[index for index in range(160) if index not in indices],
        interface_qualification=ref(qualification), anchor_chunk_records=len(records),
        limitation='Native8 outcomes have truth/video/reset evidence and immutable source seed contract, but no retained actual chunk records; never AD/DPT training data')


if __name__ == '__main__':
    assert platform.system() == 'Linux' and not torch.cuda.is_available()
    tick = time.time()
    tasks = []
    for task in TASKS:
        result = audit_task(task); tasks.append(result)
        print(json.dumps(dict(task=task, anchors=result['anchor_episodes'], carried_native=len(result['native_rows']),
                              remaining_native=len(result['native_missing_indices']))), flush=True)
    assert sum(task['anchor_episodes'] for task in tasks) == 136
    assert sum(len(task['native_rows']) for task in tasks) == 209
    OUT.mkdir(parents=True, exist_ok=True)
    value = dict(completed=True, source_run='robocasa-interface-screen-20261102-r1',
        original_source_commit='ca5037dac770f3fa1d77293b3c2b02409b8cba66', tasks=tasks,
        carried_complete_episodes=345, new_native_episodes=431, PPO_updates=0,
        full_truth_video_and_native_pairing_verified=True, seconds=time.time()-tick)
    path = OUT / 'CARRY-AUDIT.json'
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(path=str(path), sha256=sha(path), completed=True, seconds=value['seconds'])), flush=True)
