"""Verify diagnostic rollouts, camera interventions and paired differences."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def load(path):
    return json.loads(path.read_bytes())


def pairs(left, right):
    counts = dict(both=0, first_only=0, second_only=0, neither=0)
    labels = {(True,True): 'both', (True,False): 'first_only',
              (False,True): 'second_only', (False,False): 'neither'}
    for a,b in zip(left,right,strict=True):
        counts[labels[a,b]] += 1
    n = counts['first_only']+counts['second_only']
    counts['exact_p_two_sided'] = min(1., 2*sum(math.comb(n,i) for i in range(
        min(counts['first_only'],counts['second_only'])+1)) / 2**n)
    return counts


def group_flags(rows, horizon):
    return [r['success'] and r['targets'] <= horizon for r in rows]


def audit(root, reference, partial):
    result = load(root/'RESULT.json')
    probe = load(root/'gpu-probe.json')
    assert result['optimizer_updates'] == 0 and probe['status'] == 'passed'
    assert load(root/'READY.json')['strict_weight_load']
    if not partial:
        assert load(root/'EXIT.json')['exit_code'] == 0 and result['status'] == 'complete'
        assert not any(x in (root/'evaluation.log').read_text(errors='replace')
                       for x in ('OIDN Error','invalid handle','unsupported device type'))
    renderers = list((root/'renderers').glob('*.json'))
    assert renderers and all(load(p)['actual_render_pci'] == probe['devices'][0]['pci_bus_id'] for p in renderers)
    baseline = load(reference/'RESULT.json')['settings']['demo_clean']
    wrist = 'wrist_near_m' in result
    report = dict(source_commit=result['source_commit'],partial=partial,settings={},renderers=len(renderers))
    total_chunks = 0
    initial_diffs, actions_changed, prefix_equal = [], 0, 0
    for mode,rows in result['settings'].items():
        if not partial:
            assert len(rows) == result['episodes_per_domain']
        complete = []
        for i, row in enumerate(rows):
            assert row['index'] == i
            if row.get('reset_error'):
                assert mode == 'short_sequential' and row['success'] is None
                continue
            directory = root/f"{mode}-{i:03d}-seed-{row['seed']}"
            assert load(directory/'episode.json') == row
            complete.append(row)
            horizon = 400 if wrist else 1200
            assert row['success'] or row['targets'] == horizon
            assert sum(c['delivered'] for c in row['chunks']) == row['targets']
            for j, chunk in enumerate(row['chunks']):
                data = np.load(directory/f'actions-{j:02d}.npy',allow_pickle=False)
                assert data.shape == (50,14) and np.isfinite(data).all()
                assert chunk['noise_seed'] == 1000+i*100+j
                assert hashlib.sha256(data.tobytes()).hexdigest() == chunk['actions_sha256']
            total_chunks += len(row['chunks'])
            if not wrist:
                assert row['success_by_400'] == bool(row['success'] and row['targets'] <= 400)
                assert (row['at_400'] is not None) == (row['targets'] >= 400)
            if mode == 'short_sequential':
                assert row['seed'] == 1000+i and row['instruction'] == 'place phone stand'
                continue
            ref = baseline[i]
            assert row['seed'] == ref['seed'] and row['actors'] == ref['actors']
            ref_dir = reference/f"demo_clean-{i:03d}-seed-{row['seed']}"
            with np.load(directory/'initial.npz') as a, np.load(ref_dir/'initial.npz') as b:
                assert np.array_equal(a['state'], b['state']) and np.array_equal(a['images'][0],b['images'][0])
                assert a['instruction'].item() == row['instruction']
                if mode == 'short_frozen':
                    assert row['instruction'] == 'place phone stand'
                else:
                    assert row['instruction'] == ref['instruction']
                if wrist:
                    assert np.allclose(row['wrist_near_m'], .01)
                    delta = np.abs(a['images'][1:].astype(float)-b['images'][1:])
                    initial_diffs.append(delta.mean(axis=(1,2,3)).tolist())
                    actions_changed += int(row['chunks'][0]['actions_sha256'] != ref['chunks'][0]['actions_sha256'])
                else:
                    assert np.array_equal(a['images'],b['images'])
                    if mode == 'scene_frozen':
                        prefix_equal += int([c['actions_sha256'] for c in row['chunks'][:len(ref['chunks'])]]
                                            == [c['actions_sha256'] for c in ref['chunks']])
        stats = dict(completed=len(complete), reset_errors=len(rows)-len(complete),
                     success_by_400=sum(group_flags(complete,400)),final_successes=sum(r['success'] for r in complete))
        if not wrist:
            for cutoff in (400,1200):
                failures = [r for r in complete if not (r['success'] and r['targets']<=cutoff)]
                traces = [(r['at_400'] if cutoff==400 else r['final_trace']) for r in failures]
                stats[f'failed_at_{cutoff}'] = dict(n=len(traces),
                    lift_under_2cm=sum(t['max_phone_lift_m'] < .02 for t in traces),
                    lifted_2cm_but_never_aligned=sum(t['max_phone_lift_m'] >= .02 and not t['inside_position_tolerance'] for t in traces),
                    reached_position_tolerance=sum(t['inside_position_tolerance'] for t in traces))
            stats['late_success_targets']=[r['targets'] for r in complete if r['success'] and r['targets']>400]
        report['settings'][mode] = stats
    if wrist:
        rows=result['settings']['demo_clean']
        if rows:
            report['compared_to_native_400'] = pairs(group_flags(baseline[:len(rows)],400),group_flags(rows,400))
            report['wrist_reset_rgb_mae'] = np.mean(initial_diffs,axis=0).tolist()
            report['different_first_action_chunks']=actions_changed
    else:
        settings=result['settings']
        n=min(len(settings['scene_frozen']),len(settings['short_frozen']))
        report['scene_vs_short_same_seeds_1200']=pairs(group_flags(settings['scene_frozen'][:n],1200),
                                                        group_flags(settings['short_frozen'][:n],1200))
        report['exact_native_action_prefixes_at_400']=prefix_equal
    report['verified_action_chunks'] = total_chunks
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    p.add_argument('reference',type=Path)
    p.add_argument('--partial',action='store_true')
    args=p.parse_args()
    print(json.dumps(audit(args.root,args.reference,args.partial),indent=2))
