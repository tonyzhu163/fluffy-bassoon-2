"""Audit update, simulator and pairing receipts for the matched OpenDrawer training."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs')
RUN = ROOT/'robocasa-opendrawer-ppo-ablation-3m-20261104-r1'


def read(path):
    return json.loads(path.read_text())


def summarize(folder, label, complete=False):
    def files(pattern):
        return sorted(folder.glob(pattern))
    expected_steps = 24 if label == 'E' else 6
    updates = [read(p) for p in files('updates/iteration-*-rank-*.json')]
    replay = [read(p) for p in files('logprob-parity/iteration-*-rank-*.json')]
    match = [read(p) for p in files('first-collection-match-rank-*.json')]
    assert all(j['passed'] and j['value']['passed'] and j['before_backward'] for j in replay)
    assert all(j['passed'] and j['globally_joined']==128 for j in match)
    by_rank = {}
    for rank in range(2):
        rows = sorted((j for j in updates if j['rank']==rank),key=lambda j:j['producing_policy_iteration'])
        for i,j in enumerate(rows):
            assert j['completed'] and j['producing_policy_iteration']==i
            assert (j['optimizer_steps_before'],j['optimizer_steps_after'])==(expected_steps*i,expected_steps*(i+1))
            assert j['frozen_vlm_unchanged'] and all(math.isfinite(v) for v in j['metrics'].values())
        by_rank[rank]=rows
    n=min(map(len,by_rank.values()))
    train={}
    for path in files('environment/rank-*/training-*-episodes.json'):
        panel=read(path)
        assert panel['completed']
        for row in panel['rows']:
            case=row['seed']
            assert row['native_counts']['controls']-row['native_counts']['reset_controls']==600
            assert case not in train and row['episode_id']==f'OpenDrawer/{case}'
            train[case]=int(row['success_once'])
    assert all(2100000+i in train for i in range(n*128))
    first = by_rank[0][0] if n else {}
    if complete:
        assert n==40 and len(updates)==len(replay)==80 and len(match)==2
        assert set(train)==set(range(2100000,2105120))
        assert read(folder/'progress.json')['completed_iterations']==40
        assert read(folder/'after-evaluation.json')['completed']
        assert read(folder/'OpenDrawer/checkpoints/global_step_40/pilot-boundary.json')['completed']
    return dict(label=label,progress_updates=n,matched_optimizer_steps=n*expected_steps,
        actual_training_controls=600*len(train),first_rollout_cases=128 if len(match)==2 else 0,
        first_images_bitwise_equal=sum(j['unchanged_rgb_first_lanes'] for j in match),
        first_max_action_gap=max((j['max_first_action_gap'] for j in match),default=None),
        first_max_gap_on_equal_images=max((j['max_first_action_gap_equal_rgb'] for j in match if j['max_first_action_gap_equal_rgb'] is not None),default=None),
        selected_indices=match[0]['selected_indices'] if match else None,
        replay_checks=len(replay), first_sgd_seconds=first.get('training_seconds'),
        first_sgd_metrics=first.get('metrics',{}),recent_sgd_seconds=[round(j['training_seconds'],1) for j in by_rank[0][-3:]],
        training_success_first10=sum(train.get(2100000+i,0) for i in range(1280)) if len(train)>=1280 else None,
        training_success_last10=sum(train.get(2100000+i,0) for i in range(3840,5120)) if len(train)==5120 else None,
        completed=complete)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('label',choices=('E','C','D'));p.add_argument('--complete',action='store_true')
    a=p.parse_args()
    folder=RUN/a.label/('attempt-02' if a.label=='E' else 'attempt-01')
    print(json.dumps(summarize(folder,a.label,a.complete),indent=2))
