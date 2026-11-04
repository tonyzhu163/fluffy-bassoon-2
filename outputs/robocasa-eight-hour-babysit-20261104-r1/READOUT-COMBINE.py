"""Join complete fixed-cohort shards; keep individual context queries distinct."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(a, b, seed):
    import numpy as np
    from scipy.stats import binomtest
    assert set(a) == set(b)
    delta = np.asarray([int(b[k])-int(a[k]) for k in sorted(a)], np.float64)
    wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
    counts = np.asarray([(delta == value).sum() for value in (-1,0,1)])
    bootstrap = np.random.default_rng(seed).multinomial(len(delta),counts/len(delta),size=10000)
    interval = np.quantile((bootstrap[:,2]-bootstrap[:,0])/len(delta),[.025,.975])*100
    return dict(cases=len(delta), delta_percentage_points=float(delta.mean()*100),
        wins=wins, losses=losses,
        exact_McNemar_p=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
        paired_bootstrap95_percentage_points=interval.tolist(),
        bootstrap_seed=seed, bootstrap_draws=10000, inference='descriptive; marginal intervals, no multiplicity-adjusted adaptation claim')


def main():
    assert sys.platform == 'linux'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', type=Path)
    args = parser.parse_args()
    nodes = sorted(args.group.glob('node-*/RESULT.json'))
    assert nodes
    first = read(nodes[0])['binding']
    method, split = first['selection']['kind'], first['split']
    anchor = method in ('native_initial','native_final')
    assert len(nodes) == (2 if anchor else 4)
    tasks = ('OpenDrawer','CloseToasterOvenDoor','CoffeeSetupMug') if split == 'ID' else ('CloseFridge','TurnOnMicrowave')
    modes = ('full',) if anchor else ('full','episode')
    cases, references, checked_banks = {}, [], []
    node_membership = set()
    selected_sha = first['selection'].get('sha256')
    for path in nodes:
        node = path.parent
        top = read(path)
        assert top['completed'] and top['deployment_optimizer_updates'] == 0
        binding = top['binding']
        assert binding['stage'] == 'final' and binding['cases_per_task'] == 50
        assert binding['selection']['kind'] == method and binding['split'] == split
        assert binding['selection'].get('sha256') == selected_sha
        for key in ('source_commit','source_assembly_sha256','support_episodes','batch_contract','render_contract'):
            assert binding[key] == first[key]
        for key in ('model_sha256','norm_sha256','runtime_commit'):
            assert binding['signature'][key] == first['signature'][key]
        assert len(binding['memory_modes']) == 1 and binding['memory_modes'][0] in modes
        mode = binding['memory_modes'][0]
        membership = (mode,binding['shard_index'])
        assert membership not in node_membership and binding['shard_count'] == 2
        node_membership.add(membership)
        references.append(dict(path=str(path),sha256=sha(path)))
        for task in tasks:
            for bank in sorted((node/mode/task).glob('bank-*/RESULT.json')):
                row = read(bank)
                assert row['method'] == method and row['mode'] == mode and row['task'] == task
                slots = [x['case']['slot'] for x in row['cases']]
                assert slots == list(range(slots[0],min(50,slots[0]+8))) and slots[0]%8 == 0
                assert row['denominators'] == len(slots)
                horizon=600 if split=='ID' else 900 if task=='CloseFridge' else 450
                assert row['actual_batch_sizes'] == [len(slots)]*(6*horizon//50)
                assert len(row['episodes']) == len(row['references']) == 6*len(slots)
                for ref in row['references']:
                    assert sha(Path(ref['path'])) == ref['sha256']
                for episode in row['episodes']:
                    case = episode['case'];slot=case['slot'];attempt=episode['attempt']
                    key=(mode,task,slot,attempt)
                    assert key not in cases and slot in slots and 1 <= attempt <= 6
                    assert episode['completed'] and case['task'] == task
                    assert episode['horizon'] == horizon
                    index=('OpenDrawer','CloseToasterOvenDoor','CoffeeSetupMug','CloseFridge','TurnOnMicrowave').index(task)
                    assert case['native_seed'] == 64105000 + index*10000 + slot
                    assert case['bank'] == 'joint_meta_final_'+split
                    cases[key]=episode
                for support in range(6):
                    assert row['first_success_counts_by_support_episodes'][str(support)] == sum(
                        ep['success_once'] for ep in row['episodes'] if ep['attempt']==support+1)
                checked_banks.append(dict(path=str(bank),sha256=sha(bank)))
    assert node_membership == {(mode,index) for mode in modes for index in range(2)}
    expected={(mode,task,slot,attempt) for mode in modes for task in tasks for slot in range(50) for attempt in range(1,7)}
    assert set(cases) == expected
    summaries={}
    for mode in modes:
        summaries[mode]={}
        for task in tasks:
            summaries[mode][task]=dict(cases=50,first_success_by_support_episodes={},
                success_at_end_by_support_episodes={},ever_success_through_query={})
            for support in range(6):
                rows=[cases[(mode,task,slot,support+1)] for slot in range(50)]
                summaries[mode][task]['first_success_by_support_episodes'][str(support)]=sum(ep['success_once'] for ep in rows)
                summaries[mode][task]['success_at_end_by_support_episodes'][str(support)]=sum(ep['success_at_end'] for ep in rows)
                summaries[mode][task]['ever_success_through_query'][str(support)]=sum(
                    any(cases[(mode,task,slot,j)]['success_once'] for j in range(1,support+2)) for slot in range(50))
    comparisons={}
    if not anchor:
        for task in tasks:
            comparisons[task]={}
            for support in range(6):
                outcome=lambda mode:{slot:cases[(mode,task,slot,support+1)]['success_once'] for slot in range(50)}
                identity=f'{method}/{split}/{task}/{support}'
                seed=int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4],'little')
                comparisons[task][str(support)]=paired(outcome('episode'),outcome('full'),seed)
    result=dict(completed=True,method=method,split=split,stage='final',cases_per_task=50,
        source_nodes=references,source_banks=checked_banks,selected_candidate_sha256=selected_sha,
        episodes=len(cases),summaries=summaries,descriptive_full_minus_episode=comparisons,
        meaning='Query episode j uses j-1 complete actual prior episodes in full mode. Ever-success is a separate cumulative measure.',
        native_truth_audit='Pending independent raw command/draw/frame audits; this reduction verifies exact coverage and retained complete-episode hashes.',
        deployment_optimizer_updates=0)
    target=args.group/'RESULT.json'
    assert not target.exists()
    temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)
    print(json.dumps({k:v for k,v in result.items() if k not in ('source_nodes','source_banks')}))


if __name__ == '__main__':
    main()
