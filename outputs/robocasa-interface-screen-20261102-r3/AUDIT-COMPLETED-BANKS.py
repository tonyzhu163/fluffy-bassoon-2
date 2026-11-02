"""Allocated-Linux CPU audit of immutable completed Stage0 banks; no model inference."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from fluffy_bassoon.rl.action_residual import ResidualBound

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-interface-screen-20261102-r3')
PIN = 'd126c0a2c07984dfac2a0151c1506bf9e374a2b6'
TASKS = ('OpenDrawer', 'PickPlaceDrawerToCounter', 'CoffeeSetupMug', 'TurnOffStove')
spec = importlib.util.spec_from_file_location('carry', Path('/mowenjie/fluffy-bassoon-2/worktrees/robocasa-interface-screen-20261102-r3/outputs/robocasa-interface-screen-20261102-r3/CARRY.py'))
carry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carry)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit(task, name):
    directory = ROOT / task / name
    path = ROOT / 'audits' / (task + '-' + name + '.json')
    result_path = directory / 'RESULT.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous['completed'] and carry.sha(result_path) == previous['result']['sha256']
        return dict(task=task, bank=name, cached=True, audit=carry.ref(path))
    result = json.loads(result_path.read_text())
    assert result['completed'] and result['source_commit'] == PIN and result['task'] == task
    plan = load(directory / 'PLAN.pt')
    assert plan['source_commit'] == PIN and plan['arm'] == result['arm'] and plan['fraction'] == result['fraction']
    seeds = [seed for case, draw in zip(plan['cases'], plan['draws'], strict=True) for seed in carry.motor_plan(case, draw)]
    assert seeds == plan['planned_motor_seeds']
    stochastic = plan['actor_standard_normal'] is not None
    action_noise = []
    for seed in seeds:
        rng = np.random.default_rng(seed + 17)
        code = rng.normal(0, .5, 32).astype(np.float32)
        assert np.array_equal(plan['motor_controls'][seed], code)
        action_noise.append(rng.normal(0, 1, 6))
    expected_noise = np.asarray(action_noise, dtype=np.float32)
    if stochastic:
        assert np.array_equal(plan['actor_standard_normal'], expected_noise)
    initializer = json.loads((ROOT / task / 'manifest.json').read_text())['initializer']
    stats = json.loads(Path(initializer['norm_path']).read_text())
    stats = stats.get('norm_stats', stats)
    bound = ResidualBound.from_std(stats['actions']['std'], indices=range(6),
                                  fraction=result['fraction'] if result['arm'] == 'action' else .25, action_dim=12)
    payload = load(directory / 'records.pt')
    assert payload['arm'] == plan['arm'] and payload['fraction'] == plan['fraction']
    assert payload['planned_motor_seeds'] == seeds
    assert len(payload['records']) == 12 * len(plan['cases'])
    assert payload['carried_rows'] == plan['carried_rows']
    assert len(result['rows']) == len(plan['carried_rows']) + len(plan['cases'])
    carried = {row['original_slot']:row for row in plan['carried_rows']}
    original = dict(zip(plan['original_indices'], range(len(plan['cases'])), strict=True))
    assert not (set(carried) & set(original))
    assert sorted(set(carried) | set(original)) == list(range(len(result['rows'])))
    row_evidence = []
    changed = 0
    code_nonzero = 0
    image_max = {}
    reset_controls = 0
    for row in result['rows']:
        index = row['original_slot']
        if index in carried:
            assert row == carried[index]
            continue
        slot = original[index]
        checkpoint_path = directory / 'episodes' / f'slot-{slot:06d}.pt'
        checkpoint = load(checkpoint_path)
        assert checkpoint['source_commit'] == PIN
        assert checkpoint['original_index'] == index
        assert checkpoint['case'] == row['case'] == plan['cases'][slot]
        assert checkpoint['draw'] == row['draw'] == plan['draws'][slot]
        assert checkpoint['episode'] == row['episode']
        episode_seeds = seeds[slot*12:(slot+1)*12]
        assert checkpoint['planned_motor_seeds'] == episode_seeds
        assert len(checkpoint['records']) == 12
        for chunk, (record, seed) in enumerate(zip(checkpoint['records'], episode_seeds, strict=True)):
            aggregate = payload['records'][slot*12+chunk]
            for key, value in record.items():
                assert np.array_equal(value, aggregate[key])
            assert record['noise_seed'] == seed
            assert record['episode_start'] == (chunk == 0) and record['done'] == (chunk == 11)
            assert record['reward'] == 0 and record['features'].shape == (2088,)
            for field in ('features','base_chunk','executed_chunk','control'):
                assert np.isfinite(record[field]).all()
            assert record['base_chunk'].shape == record['executed_chunk'].shape == (50,12)
            expected_control = expected_noise[slot*12+chunk] * np.float32(.5) if stochastic else np.zeros(6,dtype=np.float32)
            assert np.array_equal(record['control'], expected_control)
            assert np.array_equal(record['executed_chunk'], bound.apply(record['base_chunk'], record['control']))
            changed += int(not np.array_equal(record['base_chunk'], record['executed_chunk']))
            code_nonzero += int(np.any(plan['motor_controls'][seed]))
            if result['arm'] == 'select':
                selected = checkpoint['selector_draws'][seed]
                candidates = [seed] + [int(np.random.default_rng(seed + k*1000003).integers(0,2**63-1)) for k in range(1,8)]
                assert selected['candidate_seeds'] == candidates
                assert 0 <= selected['selected_candidate'] < 8
                assert selected['actual_motor_seed'] == candidates[selected['selected_candidate']]
        verified = carry.verify(row, directory, row['case'], row['draw'], videos=True)
        for view, metric in verified['comparison']['images'].items():
            image_max[view] = max(image_max.get(view,0),metric['max_absolute_delta'])
        truth = json.loads(Path(row['truth']['path']).read_text())
        reset_controls += truth['native_counts']['reset_controls']
        row_evidence.append(dict(original_slot=index, new_slot=slot, native_seed=row['case']['native_seed'], draw=row['draw'],
            success=bool(row['success']), checkpoint=carry.ref(checkpoint_path), truth=row['truth'],
            pairing=verified['pairing'], canonical=verified['canonical']))
    n = len(plan['cases'])
    expected_inference = n * 12 * (8 if result['arm'] == 'select' else 1)
    assert result['new_episodes'] == n and result['new_policy_controls'] == n*600
    assert result['new_inference_draws'] == payload['inference_draws'] == expected_inference
    outcomes = {str(row['case']['native_seed'])+':'+str(row['draw']):bool(row['success']) for row in result['rows']}
    assert len(outcomes) == len(result['rows'])
    per_case = {}
    for row in result['rows']:
        per_case.setdefault(str(row['case']['native_seed']),[]).append(bool(row['success']))
    audit_result = dict(completed=True, checked_unix=time.time(), task=task, bank=name, source_commit=PIN,
        result=carry.ref(result_path), plan=carry.ref(directory/'PLAN.pt'), aggregate_records=carry.ref(directory/'records.pt'),
        carry_audit=carry.ref(ROOT/'CARRY-AUDIT.json'), new_episodes=n, carried_episodes=len(carried),
        complete_chunks_verified=n*12, changed_physical_residual_chunks=changed, planned_32d_codes_nonzero=code_nonzero,
        physical_and_public_pairing_exact=True, all_new_video_truth_snapshot_checkpoint_hashes_verified=True,
        image_deltas_are_diagnostic=image_max, policy_controls=n*600, reset_controls=reset_controls,
        inference_draws=expected_inference, combined_outcomes=outcomes,
        combined_successes=sum(outcomes.values()), combined_best_of8_cases=sum(any(v) for v in per_case.values()),
        cases=len(per_case), row_evidence=row_evidence,
        limitation='32D motor intervention is verified by saved plan, immutable qualified source and reset audit; no new GPU inference/replay in this CPU audit. Carried native rows retain original missing-chunk-record limitation.')
    write_new(path,audit_result)
    return dict(task=task,bank=name,cached=False,audit=carry.ref(path),new_episodes=n,successes=sum(outcomes.values()))


def main():
    assert platform.system() == 'Linux' and not torch.cuda.is_available()
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    banks=[(task,path.parent.name) for task in TASKS for path in sorted((ROOT/task).glob('*/RESULT.json'))]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda values:audit(*values),banks))
    snapshot={'completed':True,'checked_unix':time.time(),'job_id':'job-8ppjn68jxa9e','audits':results,'progress':[],'nodes':[]}
    for task in TASKS:
        progress=json.loads((ROOT/task/'progress.json').read_text());progress.pop('completed_slots',None)
        snapshot['progress'].append(dict(task=task,**progress))
    for rank in range(4):
        node=ROOT/f'job/node-{rank}'
        data={'rank':rank}
        for name in ('worker-terminal.json','clock.json'):
            if (node/name).exists():data[name]=json.loads((node/name).read_text())
        lines=(node/'console.log').read_text(errors='replace').splitlines()
        data['errors']=[line for line in lines if 'Traceback' in line or 'Error:' in line or 'Exception:' in line][-4:]
        snapshot['nodes'].append(data)
    write_new(ROOT/args.output,snapshot)
    print(json.dumps(snapshot,indent=2))


if __name__=='__main__':
    main()
