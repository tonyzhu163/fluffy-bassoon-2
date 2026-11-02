"""Summarize audited finite Stage0 panels on allocated Linux CPU, without inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import time

ROOT = Path('/mowenjie/fluffy-bassoon-2/outputs/robocasa-interface-screen-20261102-r3')
TASKS = ('OpenDrawer','PickPlaceDrawerToCounter','CoffeeSetupMug','TurnOffStove')
REQUIRED = ('native8','action025','action050','noise32','conditioning','velocity','select8')


def ref(path):
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def paired(a,b):
    assert len(a)==len(b)
    wins=sum(x and not y for x,y in zip(a,b))
    losses=sum(y and not x for x,y in zip(a,b))
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.
    return dict(wins=wins,losses=losses,ties=len(a)-n,exact_two_sided_p=p)


def cluster_test(a,b):
    # Swap all eight arm/native draws together within each independent case.
    differences=[sum(x)-sum(y) for x,y in zip(a,b)]
    distribution={0:1}
    for delta in differences:
        if delta:
            updated={}
            for value,count in distribution.items():
                for signed in (delta,-delta):
                    updated[value+signed]=updated.get(value+signed,0)+count
            distribution=updated
    observed=abs(sum(differences))
    p=sum(count for value,count in distribution.items() if abs(value)>=observed)/sum(distribution.values())
    return dict(per_case_success_count_differences=differences,
        exact_case_cluster_swap_p=p,
        test='Two-sided exact within-case arm/native swap, preserving eight-draw clusters; exploratory, unadjusted')


def vectors(outcomes):
    result={}
    for key,value in outcomes.items():
        seed,draw=map(int,key.split(':'))
        result.setdefault(seed,{})[draw]=bool(value)
    return result


def main():
    assert platform.system()=='Linux'
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    report=dict(completed=True,checked_unix=time.time(),job_id='job-8ppjn68jxa9e',
        headroom_only=True,PPO_updates=0,horizon=600,chunks_per_episode=12,tasks=[],
        limitations=[
            'Twenty selected TRAIN anchor failures per task; no independent population or PPO-learning claim.',
            'Each continuous arm uses eight fixed draws per case. Repeated draws share a case; primary exploratory rate tests swap complete case clusters.',
            'Native best-of-eight requires simulator-truth oracle selection across complete episodes. It is neither a deployed selector nor an upper bound on all chunkwise selectors.',
            'Selector uses eight candidate inferences per chunk and one actual rollout per case, scored by a frozen source-PPO value heuristic rather than qualified Q-values.',
            'Tied-noise changes native independent Gaussian noise to bounded 32D repeated noise; its results apply to this declared family and scale.',
            'Direct-head qualification failure leaves that family incomplete; it does not rule out direct PPO.',
            'The 213 inherited native outcomes retain truth/video/reset/seed-contract evidence but lack actual chunk records and are not AD/DPT training histories.',
            'AD/DPT pipeline uses separate 750/900-control horizons. Improving simulator- and Flora-reward PPO curves remain later gates.'
        ])
    for task in TASKS:
        path=ROOT/task/'RESULT.json'
        final=json.loads(path.read_text())
        assert final['completed'] and final['headroom_only'] and final['PPO_updates']==0
        panels={}
        for name in REQUIRED + (('direct',) if 'direct' in final['panels'] else ()):
            audit_path=ROOT/'audits'/(task+'-'+name+'.json')
            audit=json.loads(audit_path.read_text())
            assert audit['completed']
            result_path=ROOT/task/name/'RESULT.json'
            assert ref(result_path)==audit['result']
            grouped=vectors(audit['combined_outcomes'])
            assert len(grouped)==20
            expected_draws=set(range(8)) if name!='select8' else {0}
            assert all(set(draws)==expected_draws for draws in grouped.values())
            selected=sorted(grouped)
            successes=sum(sum(values.values()) for values in grouped.values())
            recovered=sum(any(values.values()) for values in grouped.values())
            saved=final['panels'][name]
            assert successes==saved['successes'] and recovered==saved['recovered_cases']
            assert len(audit['combined_outcomes'])==saved['episodes']
            result=json.loads(result_path.read_text())
            assert result['inference_draws']==saved['inference_draws']
            panels[name]=dict(episodes=saved['episodes'],successes=successes,
                success_rate=successes/saved['episodes'],recovered_cases=recovered,cases=20,
                per_case={str(seed):[grouped[seed][draw] for draw in sorted(grouped[seed])] for seed in selected},
                inference_draws=result['inference_draws'],new_inference_draws=result['new_inference_draws'],
                physical_policy_controls=result['policy_controls'],new_physical_policy_controls=result['new_policy_controls'],
                wall_seconds=result['seconds'],audit=ref(audit_path))
        native=panels['native8']
        keys=sorted(native['per_case'],key=int)
        native_vectors=[native['per_case'][key] for key in keys]
        comparisons={}
        for name,panel in panels.items():
            if name=='native8':continue
            assert set(panel['per_case'])==set(keys)
            arm=[panel['per_case'][key] for key in keys]
            if name=='select8':
                comparisons[name]=dict(
                    versus_native_draw0=paired([values[0] for values in arm],[values[0] for values in native_vectors]),
                    versus_finite_native_oracle=paired([values[0] for values in arm],[any(values) for values in native_vectors]),
                    native_oracle_comparison_caveat='Different deployment/search procedure; both use 96 motor candidates per case, but native oracle spends eight actual simulator trajectories and truth selection.')
            else:
                comparisons[name]=dict(
                    versus_native_best_of8=paired([any(values) for values in arm],[any(values) for values in native_vectors]),
                    versus_native_per_draw_descriptive=paired([v for values in arm for v in values],[v for values in native_vectors for v in values]),
                    per_draw_test_caveat='Descriptive exploratory test treats draws separately; use the case-cluster swap for rate inference.',
                    rate_by_case_cluster=cluster_test(arm,native_vectors))
        report['tasks'].append(dict(task=task,source_commit=final['source_commit'],task_result=ref(path),
            panels=panels,comparisons=comparisons,qualification=final['qualification'],
            incomplete_families=final['incomplete_families']))
    output=ROOT/args.output
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(completed=True,path=str(output),tasks=len(report['tasks']),PPO_updates=0)))


if __name__=='__main__':
    main()
