"""Read-only CPU diagnostics of retained PPO runs; never call an optimizer."""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic,ResidualPPOConfig
from fluffy_bassoon.meta_rl.ppo import generalized_advantages
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import sha,write_new

torch.set_num_threads(4)
assert not torch.cuda.is_available(), 'Hide GPUs for this read-only CPU analysis'
root=Path('/mowenjie/fluffy-bassoon-2/outputs')
out=root/'robocasa-ppo-reward-gap-investigation-20261102-r1';out.mkdir(exist_ok=True)
S=root/'robocasa-eight-gpu-speed-20261102-r1/S8-r2'
F0=root/'robocasa-opendrawer-flora-first-20261102-r1/policy'
F1=root/'robocasa-opendrawer-flora-resume-20261102-r1/policy'
O=root/'robocasa-original-pi05-eval50-20261102-r1'
read=lambda p:json.loads(Path(p).read_text())
load=lambda p:torch.load(p,map_location='cpu',weights_only=False)
def ref(p):return dict(path=str(p),sha256=sha(p))
def steps(saved):return sorted({int(v['step'].item()) for v in saved['optimizer']['state'].values()})
def model(saved):
    p=ResidualActorCritic(ResidualPPOConfig(**saved['config']['ppo']));p.load_state_dict(saved['policy'],strict=True)
    return p.eval()
def auc(y,s):
    y,s=np.asarray(y,dtype=bool),np.asarray(s,dtype=float)
    if not y.any() or y.all():return None
    d=s[y,None]-s[None,~y]
    return float(((d>0)+.5*(d==0)).mean())
def stats(v):
    v=np.asarray(v,dtype=float)
    return dict(n=len(v),mean=float(v.mean()),std=float(v.std()),minimum=float(v.min()),
                q25=float(np.quantile(v,.25)),median=float(np.median(v)),q75=float(np.quantile(v,.75)),maximum=float(v.max()))
def advantages(records,rewards):
    values=torch.tensor([v['value'] for v in records],dtype=torch.float32)[:,None]
    ends=torch.tensor([v['done'] for v in records],dtype=torch.bool)[:,None]
    adv,_=generalized_advantages(torch.tensor(rewards,dtype=torch.float32)[:,None],values,ends,gamma=1.,gae_lambda=.95)
    adv=adv.flatten();return (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
def gradient_comparison(saved,records,a,b):
    p=model(saved);features=torch.tensor(np.asarray([v['features'] for v in records]),dtype=torch.float32)
    controls=torch.tensor(np.asarray([v['control'] for v in records]),dtype=torch.float32)
    parameters=[*p.actor.parameters(),p.log_std]
    def grad(adv):
        lp,_,_=p.evaluate(features,controls)
        return torch.cat([v.flatten() for v in torch.autograd.grad(-(lp*adv).mean(),parameters)])
    ga,gb=grad(a),grad(b);cos=float(torch.dot(ga,gb)/(ga.norm()*gb.norm()).clamp_min(1e-15))
    return dict(full_batch_actor_gradient_cosine=cos,soft_gradient_norm=float(ga.norm()),binary_gradient_norm=float(gb.norm()),
        normalized_advantage_correlation=float(np.corrcoef(a.numpy(),b.numpy())[0,1]),
        normalized_advantage_sign_disagreement=float(((a>0)!=(b>0)).float().mean()),
        scope='Labels changed on same saved Flora records/critic, before update; diagnostic gradient only, no optimizer steps or robot counterfactual')

initialS=load(root/'robocasa-residual-approved-20261101-r1/S/boundary-000.pt');initialF=load(F0/'boundary-000.pt')
assert initialS['config']==initialF['config']
assert all(torch.equal(v,initialF['policy'][k]) for k,v in initialS['policy'].items())
finals={'S':load(S/'final.pt'),'F':load(F1/'final.pt')}
assert finals['S']['config']==finals['F']['config']
assert all(steps(v)==[360] and v['iteration']==6 for v in finals.values())
profiles={name:read(p/'profile.json') for name,p in [('S',S),('F0',F0),('F1',F1)]}
training=[];episodes=[];all_y=[];all_score=[];gradients=[]
for arm in ['S','F']:
    for i in range(1,7):
        base=S if arm=='S' else F0 if i<=4 else F1
        profile=profiles['S' if arm=='S' else 'F0' if i<=4 else 'F1']
        request=read(base/f'labels-{i:03d}/request.json');saved=load(request['pending_checkpoint']['path'])
        assert sha(request['pending_checkpoint']['path'])==request['pending_checkpoint']['sha256']
        assert saved['iteration']==i-1 and len(saved['pending_episodes'])==64 and len(saved['pending_records'])==960
        assert steps(saved)==([] if i==1 else [(i-1)*60])
        labels={} if arm=='S' else {v['id']:(v['expected_score']-1)/4 for v in read(read(base/f'labels-{i:03d}/ready.json')['outputs']['path'])}
        y=[];score=[]
        for ep in request['episodes']:
            truth=read(Path(profile['monitoring_root'])/'train'/(ep['id']+'.json'))
            success=bool(truth['simulator_final_success']);reward=float(success) if arm=='S' else labels[ep['id']]
            y.append(success);score.append(reward)
            episodes.append(dict(arm=arm,update=i,id=ep['id'],case=truth['case'],task=ep['task'],success=success,reward=reward,videos=ep.get('videos',[])))
        row=dict(arm=arm,update=i,successes=sum(y),episodes=64,true_success_rate=sum(y)/64,
            reward=stats(score),reward_on_success=stats(np.asarray(score)[y]) if any(y) else None,
            reward_on_failure=stats(np.asarray(score)[np.logical_not(y)]) if not all(y) else None,
            reward_AUROC_vs_terminal_success=auc(y,score),metrics=read(base/f'update-{i:03d}.json')['metrics'],
            Adam_steps_before=steps(saved),Adam_steps_after=steps(load(base/f'boundary-{i:03d}.pt')))
        if arm=='F':
            rec=saved['pending_records'];ends=np.flatnonzero([v['done'] for v in rec]);assert len(ends)==64
            soft=np.zeros(960,np.float32);binary=soft.copy();soft[ends]=score;binary[ends]=y
            g=gradient_comparison(saved,rec,advantages(rec,soft),advantages(rec,binary));g['update']=i;gradients.append(g)
            row['failures_reward_at_least_075']=int(np.sum((np.asarray(score)>=.75)&~np.asarray(y)))
            all_y.extend(y);all_score.extend(score)
        training.append(row)
        print(json.dumps(dict(arm=arm,update=i,successes=sum(y),mean_reward=float(np.mean(score)),AUC=row['reward_AUROC_vs_terminal_success'])),flush=True)

models={k:model(v) for k,v in finals.items()};fixed={}
for phase in ['ID','OOD']:
    features=[]
    for p in sorted((O/f'policy/{phase}-000/videos').glob('parallel-*/rank-*/result.pt')):
        value=load(p);features.extend(v['features'] for v in value['records'])
    x=torch.tensor(np.asarray(features),dtype=torch.float32);fixed[phase]={}
    for arm,p in models.items():
        with torch.no_grad():control=torch.cat([p.act(b,deterministic=True)[0] for b in x.split(256)]).numpy()
        fraction=np.tanh(control);scale=np.asarray(finals[arm]['config']['bound']['scale'])
        fixed[phase][arm]=dict(decisions=len(x),mean_abs_bound_fraction=float(np.abs(fraction).mean()),
            signed_mean_bound_fraction_per_dimension=fraction.mean(0).tolist(),
            mean_abs_bound_fraction_per_dimension=np.abs(fraction).mean(0).tolist(),
            p95_abs_bound_fraction_per_dimension=np.quantile(np.abs(fraction),.95,axis=0).tolist(),
            bound_saturation_fraction=float((np.abs(fraction)>.9).mean()),
            mean_abs_physical_residual_per_dimension=np.abs(fraction*scale).mean(0).tolist())

dev={}
for arm in ['S','F']:
    dev[arm]=[dict(update=0,**read(root/'robocasa-residual-approved-20261101-r1/common-before/initial_DEV-000/evaluation.json'))]
    for i in [2,4,6]:
        base=S if arm=='S' else F0 if i<=4 else F1
        dev[arm].append(dict(update=i,**read(base/f'DEV-{i:03d}/evaluation.json')))
y=np.asarray(all_y,dtype=bool);score=np.asarray(all_score)
false_high=sorted([e for e in episodes if e['arm']=='F' and not e['success']],key=lambda e:e['reward'],reverse=True)
true_low=sorted([e for e in episodes if e['arm']=='F' and e['success']],key=lambda e:e['reward'])
report=dict(completed=True,analysis='retained-data investigation; CPU only; no new episodes or optimizer steps',
    original_S_F_comparison=read(O/'TERMINAL-REVIEW.json'),
    initial_policy_tensors_identical=True,learner_configs_identical=True,final_Adam_steps={k:steps(v) for k,v in finals.items()},
    final_pre_tanh_std={k:torch.exp(v['policy']['log_std'].clamp(-5,2)).tolist() for k,v in finals.items()},
    final_feature_normalization_equal=all(torch.equal(finals['S']['policy'][k],finals['F']['policy'][k]) for k in ['feature_mean','feature_std']),
    training_batches=training,DEV_curves=dev,
    Flora_training_reward_alignment=dict(episodes=len(y),successes=int(y.sum()),AUROC=auc(y,score),
        reward_on_success=stats(score[y]),reward_on_failure=stats(score[~y]),
        failures_reward_at_least_075=int(np.sum((score>=.75)&~y)),failures=int((~y).sum()),
        successes_below_median_failed_reward=int(np.sum(score[y]<np.median(score[~y])))),
    label_only_gradient_diagnostics=gradients,
    fixed_original_feature_residual_predictions=fixed,
    residual_prediction_limit='Same original trajectory features for both heads, no simulator counterfactual; intervention sizes do not establish which actions cause wins/losses.',
    highest_reward_failures=false_high[:12],lowest_reward_successes=true_low[:6],
    calibration_reference=dict(path=str(root/'robocasa-flora-five-task-fifty-score-20261102-r1/exchange/cohorts/OpenDrawer/baseline-validation.json'),data=read(root/'robocasa-flora-five-task-fifty-score-20261102-r1/exchange/cohorts/OpenDrawer/baseline-validation.json')),
    source_checkpoints={k:ref(p) for k,p in [('S',S/'final.pt'),('F',F1/'final.pt')]})
write_new(out/'EPISODES.json',dict(episodes=episodes))
write_new(out/'REPORT.json',report)
print('ALIGNMENT',json.dumps(report['Flora_training_reward_alignment']),flush=True)
print('GRADIENTS',json.dumps(gradients),flush=True)
print('FIXED_FEATURE_RESIDUALS',json.dumps(fixed),flush=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
for arm,color in [('S','#2563eb'),('F','#ea580c')]:
    rows=[v for v in training if v['arm']==arm]
    axes[0].plot(range(1,7),[v['true_success_rate'] for v in rows],'-o',color=color,label=arm+' true success')
rows=[v for v in training if v['arm']=='F']
axes[0].plot(range(1,7),[v['reward']['mean'] for v in rows],'--',color='#ea580c',label='F mean reward')
axes[0].set(xlabel='Completed PPO update',ylabel='Fraction / normalized reward',ylim=(0,1),title='On-policy training batches (64 each)');axes[0].legend(fontsize=8)
axes[1].boxplot([score[~y],score[y]],showfliers=True)
axes[1].set_xticks([1,2],['Terminal failure','Terminal success'])
axes[1].set(ylabel='Official Flora normalized reward',ylim=(0,1),title=f'Flora TRAIN discrimination: AUC {auc(y,score):.3f}')
axes[2].bar(range(1,7),[v['full_batch_actor_gradient_cosine'] for v in gradients],color='#7c3aed')
axes[2].axhline(0,color='black',linewidth=.7);axes[2].set(xlabel='PPO update',ylabel='Gradient cosine',ylim=(-1,1),title='Same F records: Flora vs binary labels')
fig.savefig(out/'reward-alignment.png',dpi=180);plt.close(fig)
