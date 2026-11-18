"""Score all supported policies on the matched bottle trials."""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from design import BUDGETS, VARIANTS, curve_metrics, lag24, native_record, scored_trials, use_pacing


def load(path):
    return json.loads(path.read_text())


def key(record):
    return record['initial_physics'],json.dumps(record['initial_public'],sort_keys=True)


def summarize(root):
    fit=load(root/'frozen-policy.json')
    model=fit['learned']['model']
    constant=fit['fixed']['model']
    blocks=[]
    rows=[]
    resets={}
    supports_by_pose=[]
    reach={}
    for idx in range(6):
        support,query={},{}
        for v in VARIANTS:
            b=root/'bottle'/f's{idx}-{v}'
            for path in sorted(b.glob('support-*')):
                rec=native_record(path)
                opt,mode=rec['goal'],rec['paced']
                assert (v,opt,mode) not in support
                with np.load(path/'public.npz') as data:
                    assert lag24(data['issued'],data['state_after'])==rec['lag']
                    prefix=[data[k][:24].copy() for k in ('issued','state_after')]
                if (v,opt) in resets:
                    for a,z in zip(prefix,resets[v,opt]):
                        np.testing.assert_array_equal(a,z)
                else:
                    resets[v,opt]=prefix
                choice=load(path/'decision.json')
                assert choice==dict(control=24,lag=rec['lag'],paced=mode,option=opt)
                assert (path/'decision.json').stat().st_mtime_ns<=(path/'result.json').stat().st_mtime_ns
                if rec['mode']=='learned':
                    assert opt==model['option'] and mode==use_pacing(model,rec['lag'])
                rows.append(dict(role='support',scene=rec['scene'],variant=v,option=opt,paced=mode,
                                 lag=rec['lag'],first=rec['first'],simultaneous=rec['simultaneous']))
                support[v,opt,mode]=rec
            for opt in (0,1):
                path=b/f'query-g{opt}'
                rec=native_record(path)
                assert rec['paced'] and rec['scene']==idx+6 and rec['goal']==opt
                assert (b/'retry-decisions.json').stat().st_mtime_ns <= (path/'result.json').stat().st_mtime_ns
                query[v,opt]=rec
                rows.append(dict(role='query',scene=rec['scene'],variant=v,option=opt,paced=True,
                                 lag=rec['lag'],first=rec['first'],simultaneous=rec['simultaneous']))
            for i in load(b/'retry-decisions.json'):
                opt=model['option']
                rec=support[v,opt,use_pacing(model,support[v,opt,True]['lag'])]
                assert i['reward']==(rec['first'][i['assigned']] is not None)
                assert i['chosen']==(opt if i['reward'] else 1-opt)
        assert len(support)==8 and len(query)==4
        assert len({key(r) for r in support.values()})==1 and len({key(r) for r in query.values()})==1
        supports_by_pose.append(dict(scene=idx,initial_support=next(iter(support.values()))['initial_physics'],
                                    initial_query=next(iter(query.values()))['initial_physics']))
        resets.clear()
        blocks.append((support,query))
    assert len(rows)==load(root/'complete.json')['bottle']==72
    policies={'learned':(model,False,False),'bowl_fixed':(constant,False,False),
              'paced_same_target':(dict(model,threshold=None,low_paced=True),False,False),
              'balanced_always_pace':(dict(model,threshold=None,low_paced=True),True,False),
              'balanced_learned_execution':(model,True,False),'cleared_uniform_retry':(model,False,True)}
    for option in range(2):
        for paced in (True,False):
            policies[f'fixed_target{option}_{"paced" if paced else "switch"}']=(dict(option=option,threshold=None,low_paced=paced),False,False)
    panels={}
    raw={}
    for name,(policy,uniform,clear) in policies.items():
        a=scored_trials(policy,blocks,uniform_option=uniform,clear=clear)
        raw[name]=a
        panels[name]=dict(overall=curve_metrics(a),by_timing={v:curve_metrics(a.reshape(6,2,2,-1)[:,j].reshape(12,-1))
                                                            for j,v in enumerate(VARIANTS)})
    live=load(root/'bottle/live.json')
    assert len(live)==24
    actual=[r['first'] is not None for r in live]
    np.testing.assert_array_equal(actual,raw['learned'][:,-1])
    diff=raw['learned'][:,-1]-raw['bowl_fixed'][:,-1]
    for v,role,mode,opt in product(VARIANTS,('support','query'),(True,False),range(2)):
        subset=[r for r in rows if (r['variant'],r['role'],r['paced'],r['option'])==(v,role,mode,opt)]
        if subset:
            reach[f'{role}-{v}-{opt}-{"paced" if mode else "switch"}']=dict(
                intended=sum(r['first'][opt] is not None for r in subset),
                wrong=sum(r['first'][1-opt] is not None for r in subset),episodes=len(subset))
    # A single exploratory choice must cover both possible preferences. Even the
    # descriptive oracle cannot pick a different first option for each preference.
    fixed_actions = [raw[name][:,-1].reshape(6,2,2) for name in policies if name.startswith('fixed_target')]
    per_block = np.stack(fixed_actions).sum(axis=-1)
    bounds=dict(best_constant_on_bottles=float(per_block.sum(axis=(1,2)).max()),
                known_timing=float(per_block.sum(axis=1).max(axis=0).sum()),
                known_pose_and_timing=float(per_block.max(axis=0).sum()))
    curves={name:raw[name].mean(axis=0).tolist() for name in
            ('learned','bowl_fixed','paced_same_target','balanced_always_pace','cleared_uniform_retry')}
    return dict(fit=fit,models=panels,descriptive_menu_ceiling=bounds,budgets=BUDGETS.tolist(),curves=curves,
                learned_only=int((diff>0).sum()),fixed_only=int((diff<0).sum()),
                by_pose_gain_vs_fixed=diff.reshape(6,4).sum(axis=1).tolist(),native_reach=reach,
                double_hits=sum(all(t is not None for t in r['first']) for r in rows),branches=rows,initial_states=supports_by_pose)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('episodes',type=Path)
    args=p.parse_args()
    print(json.dumps(summarize(args.episodes),indent=2))
