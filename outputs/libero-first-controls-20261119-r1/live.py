"""Execute the missing branches after observing the first two query replans."""
from pathlib import Path
import sys
import json
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'libero-decision-pretraining-20261119-r1'))
from collect import FrozenMotor,Simulator,issued_chunk,reset,rollout,write,CAMPAIGN
import jax

ARCHIVE=Path('/mowenjie/fluffy-bassoon-2/outputs/libero-decision-pretraining-20261119-r1/menu02/episodes')


def response(tape):
    x=np.linalg.norm(tape['positions'][:7].astype(float)-tape['initial_position'],axis=1)
    return float(x[6]/max(1e-12,x[:6].max()))


def calibration():
    scores={'BASE':[],'S20':[]}
    for case in json.loads((ARCHIVE/'fitting/progress.json').read_text()):
        for mode in ('default','paced','support'):
            with np.load(ARCHIVE/'fitting'/case['trial']/mode/'public.npz') as tape:
                scores[case['support']['case']['variant']].append(response(tape))
    lo,hi=max(scores['S20']),min(scores['BASE'])
    assert lo<hi
    return dict(cut=(lo+hi)/2,slow_max=lo,fast_min=hi)


def hybrid(sim,motor,case,identity,oldpath,out,ten_first,mode,cut):
    raw,new=reset(sim,case,identity['prompt'])
    assert new==identity
    out.mkdir()
    with np.load(oldpath/'public.npz') as tape:
        prefix=tape['issued'][:10]
        assert len(prefix)==10 and not np.any(tape['rewards'][:10]>0)
        step=sim.call('chunk',prefix)
        assert step['steps']==10 and not step['terminated'] and not step['truncated']
        np.testing.assert_array_equal(step['positions'],tape['positions'][:10])
        np.testing.assert_array_equal(step['gripper_positions'],tape['gripper'][:10])
        observed=response(dict(positions=np.asarray(step['positions']),initial_position=raw['robot0_eef_pos']))
        assert ('paced' if observed<cut else 'default')==mode
    write(out/'decision.json',dict(after=10,observed=observed,mode=mode))
    rng=np.random.default_rng(case['seed']+817117)
    for _ in range(1 if ten_first else 2):
        rng.standard_normal((10,32))
    actions=prefix.tolist()
    rewards=list(step['rewards'])
    while not (step['terminated'] or step['truncated']):
        prop=motor.proposal(step['observation'],identity['prompt'],rng.standard_normal((10,32)).astype(np.float32))
        cmds,_,_=issued_chunk(prop,mode,None)
        step=sim.call('chunk',cmds)
        actions.extend(cmds[:step['steps']])
        rewards.extend(step['rewards'])
    assert len(actions)==len(rewards)<=identity['horizon'] and bool(step['success'])==bool(max(rewards)>0)
    np.savez_compressed(out/'executed.npz',issued=actions,rewards=rewards)
    rec=dict(success=bool(step['success']),controls=len(actions),mode=mode,score=observed)
    write(out/'result.json',rec)
    return rec


def main(out):
    assert len(jax.devices())==1 and jax.devices()[0].platform=='gpu'
    out.mkdir()
    cal=calibration()
    write(out/'frozen.json',cal)
    tasks=json.loads((CAMPAIGN/'SPLITS.json').read_text())['development']
    rows=[]
    motor=FrozenMotor()
    with Simulator() as sim:
        for i,(suite,ids) in enumerate(sorted(tasks.items())):
            for task in ids:
                prompt=None
                for g in ('BASE','S20'):
                    folder=out/f'{suite}-{task}-{g}'
                    folder.mkdir()
                    seed=19197000+i*10000+task*100
                    prior=dict(suite=suite,task_id=task,variant=g,init_state_id=6,seed=seed)
                    query=dict(prior,init_state_id=7,seed=seed+1)
                    raw,sid=reset(sim,prior,prompt)
                    prompt=sid['prompt']
                    support=rollout(sim,motor,raw,prior,sid,'default',folder/'support')
                    with np.load(folder/'support/public.npz') as t:
                        prior_score=response(t)
                    prior_mode='paced' if prior_score<cal['cut'] else 'default'
                    write(folder/'prior-choice.json',dict(mode=prior_mode,score=prior_score))
                    results={}
                    for mode in ('default','paced'):
                        raw,new=reset(sim,query,prompt)
                        if mode=='default':
                            qid=new
                            rng=np.random.default_rng(query['seed']+817117)
                            opening=motor.proposal(raw,prompt,rng.standard_normal((10,32)).astype(np.float32))
                        assert qid==new
                        results[mode]=rollout(sim,motor,raw,query,qid,mode,folder/mode,opening)
                    results['prior']=results[prior_mode]
                    choices={}
                    for base in ('paced','default'):
                        with np.load(folder/base/'public.npz') as tape:
                            score=response(tape)
                        mode='paced' if score<cal['cut'] else 'default'
                        key='ten_first' if base=='paced' else 'default_first'
                        choices[key]=dict(score=score,mode=mode,alias=base==mode)
                        results[key]=results[base] if base==mode else hybrid(sim,motor,query,qid,folder/base,folder/key,base=='paced',mode,cal['cut'])
                    row=dict(suite=suite,task=task,timing=g,prior_mode=prior_mode,prior_score=prior_score,
                             choices=choices,controls={k:v['controls'] for k,v in results.items()},
                             hits={k:v['success'] for k,v in results.items()},support_hit=support['success'])
                    rows.append(row)
                    write(out/'rows.json',rows)
                    print(folder.name,row['hits'],flush=True)
    write(out/'complete.json',dict(rows=len(rows),episodes=sum(3+sum(not c['alias'] for c in r['choices'].values()) for r in rows)))


if __name__=='__main__':
    main(Path(sys.argv[1]))
