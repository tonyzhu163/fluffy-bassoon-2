"""Independently recount completed native evaluation cases and sampler calls."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes, count):
    z = 1.959963984540054
    rate = successes/count
    denominator = 1+z*z/count
    center = (rate+z*z/(2*count))/denominator
    radius = z*math.sqrt(rate*(1-rate)/count+z*z/(4*count*count))/denominator
    return [center-radius, center+radius]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.manifest.read_text())
    tasks = []
    cases = []
    ranks = []
    for shard in plan['shards']:
        folder = args.output/f'rank-{shard["rank"]:02d}'
        identity = json.loads((folder/'IDENTITY.json').read_text())
        exit_record = json.loads((folder/'EXIT.json').read_text())
        sampler = json.loads((folder/'server/SAMPLER-STATE.json').read_text())
        probe = json.loads((folder/'GPU-PROBE.json').read_text())
        assert identity['manifest_sha256'] == sha(args.manifest)
        assert exit_record['completed'] and exit_record['training_updates'] == 0
        assert exit_record['server_exit'] == 0 and probe['status'] == 'passed'
        assert sampler['normalization_sha256'] == '4aed1af411bd0e0f49d2b0e6d6832b11b8682917231a07173fbc30fd493bbdee'
        rank_cases = []
        for stage in plan['stages']:
            for task in shard[stage]:
                root = folder/stage/task
                result = json.loads((root/'RESULT.json').read_text())
                records = [json.loads(f.read_text()) for f in sorted(root.glob(f'{task}-*.json'))]
                assert result['completed'] and result['training_updates'] == 0
                assert len(records) == result['episodes'] == plan['trials_per_task']
                assert {x['episode'] for x in records} == set(range(plan['trials_per_task']))
                for record in records:
                    assert record['task'] == task and record['split'] == plan['split']
                    assert record['seed'] == plan['seed']
                    assert 0 < record['controls'] <= record['horizon']
                    assert record['predictions'] == math.ceil(record['controls']/plan['replan'])
                    assert record['success'] == (record['success_step'] is not None)
                    assert record['success_original_horizon'] == (record['success'] and record['success_step'] <= record['original_registry_horizon'])
                    assert record['horizon']*2 == record['original_registry_horizon']*3
                    if record['success']:
                        assert record['success_step'] == record['controls']
                    else:
                        assert record['controls'] == record['horizon']
                    prefix = root/f'{task}-{record["episode"]:03d}'
                    assert sha(prefix.with_name(prefix.name+'-first.npz')) == record['first_inputs_sha256']
                    if record['video_sha256']:
                        assert sha(prefix.with_suffix('.mp4')) == record['video_sha256']
                n = len(records)
                wins = sum(x['success'] for x in records)
                old_wins = sum(x['success_original_horizon'] for x in records)
                assert result['tasks'][task]['successes'] == wins
                assert result['tasks'][task]['original_horizon_successes'] == old_wins
                assert result['controls'] == sum(x['controls'] for x in records)
                tasks.append(dict(task=task, stage=stage, episodes=n, successes=wins,
                                  old_cutoff_successes=old_wins, success_rate=wins/n))
                rank_cases.extend(records)
        predictions = sum(x['predictions'] for x in rank_cases)
        assert sampler['calls'] == predictions and sampler['training_updates'] == 0
        ranks.append(dict(rank=shard['rank'], uuid=probe['devices'][0]['uuid'],
                          identity=identity, exit=exit_record, predictions=predictions))
        cases.extend(rank_cases)
    assert len({x['uuid'] for x in ranks}) == len(ranks)
    assert len({x['identity']['job'] for x in ranks}) == 1
    assert len({x['identity']['source_commit'] for x in ranks}) == 1
    n = len(cases)
    assert n == len(tasks)*plan['trials_per_task']
    wins = sum(x['success'] for x in cases)
    old_wins = sum(x['success_original_horizon'] for x in cases)
    controls = sum(x['controls'] for x in cases)
    assert controls <= plan['maximum_policy_controls']
    value = dict(passed=True, episodes=n, successes=wins, success_rate=wins/n,
                 descriptive_wilson_95=wilson(wins,n), old_cutoff_successes=old_wins,
                 old_cutoff_rate=old_wins/n, old_cutoff_wilson_95=wilson(old_wins,n),
                 controls=controls, predictions=sum(x['predictions'] for x in cases),
                 inference_batch_size=1, training_updates=0, tasks=tasks, ranks=ranks,
                 note='Unpaired aggregate reproduction; Wilson interval is descriptive, not a formal equivalence claim')
    target = args.output/'INDEPENDENT-AUDIT.json'
    assert not target.exists()
    target.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({k:v for k,v in value.items() if k not in ('tasks','ranks')}))


if __name__ == '__main__':
    main()
