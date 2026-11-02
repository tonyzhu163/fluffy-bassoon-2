"""Allocated-CPU final comparison of independently completed one-shot phases."""
import json,os,runpy,sys
from pathlib import Path
import torch
assert sys.platform=='linux' and not torch.cuda.is_available()
ROOT=Path('/mowenjie/fluffy-bassoon-2');OUT=ROOT/'outputs/opendrawer-pairing-recovery-20261102-r1'
os.environ['TASK_READOUT_METHODS']=json.dumps(['ppo_initial','ppo_final','ad_full','ad_static','dpt_full','dpt_static'])
os.environ['TASK_READOUT_FRESH_ID']='1'
m=runpy.run_path(str(ROOT/'worktrees/opendrawer-paired-eval-20261102-r1/outputs/opendrawer-ad-dpt-k1-id-ood-20261102-r1/READOUT.py'))
sha=m['sha'];ref=m['ref'];read=m['read_ref'];write=m['write']
phases={'ad':ROOT/'outputs/opendrawer-ad-query-recovery-20261102-r1','dpt_full':ROOT/'outputs/opendrawer-dpt-full-query-recovery-20261102-r1','dpt_static':ROOT/'outputs/opendrawer-dpt-static-id-ood-20261102-r1'}
methods=['ppo_initial','ppo_final','ad_full','ad_static','dpt_full','dpt_static']
bindings={k:json.loads((p/'binding.json').read_text()) for k,p in phases.items()}
for k,p in phases.items():
 summary=json.loads((p/'RESULT.json').read_text());assert summary['completed'] and summary['binding']==ref(p/'binding.json')
 assert bindings[k]['pairing']==bindings['ad']['pairing']
 for rank in range(8):assert json.loads((p/'workers'/f'rank-{rank:02d}.json').read_text())['completed']
cache={};episodes=[];tables={}
for cohort in ('ID','OOD'):
 zero={name:[] for name in methods};one={name:[] for name in ('ad_full','dpt_full')};support={name:[] for name in one}
 for index in range(50):
  canonical=None
  for name in methods:
   key=name if name.startswith('dpt') else 'ad';b=bindings[key];directory=phases[key]/'episodes'/cohort/f'case-{index:03d}'/name
   selected=b['initial'] if name=='ppo_initial' else b['teacher'] if name=='ppo_final' else b['selected'][name]['candidate']
   for shot,kind in [('K0','query')]+([('K1','support'),('K1','query')] if name.endswith('_full') else []):
    path=directory/shot/kind/'result.json';r=json.loads(path.read_text())
    assert r['complete'] and r['method']==name and r['candidate']==selected and r['shot']==shot and r['kind']==kind
    assert r['native_case']==m['native_case'](cohort,index) and r['motor_seeds']==m['motor_seeds'](cohort,index,kind=='support')
    assert r['horizon']==m['horizon'](cohort) and r['training_source']=='68f2c5aa33218709453a7821f6c6223ac3803035'
    assert r['evaluation_source']=='63a173362461d206ad65f39a1134342c3537f195'
    assert sha(r['records']['path'])==r['records']['sha256']
    truth=read(r['truth']);assert truth['case']==r['native_case'] and int(truth['simulator_final_success'])==r['success']
    a=r['initial_pairing'];marker=read(a['canonical']);p=Path(a['canonical']['path']).parent/'initial.pt'
    if canonical is None:canonical=a['canonical']
    assert a['canonical']==canonical
    if canonical['sha256'] not in cache:
     assert sha(p)==marker['snapshot_sha256'];cache[canonical['sha256']]=torch.load(p,map_location='cpu',weights_only=False)
    assert sha(a['snapshot']['path'])==a['snapshot']['sha256']
    report=m['PAIRING']['compare_native_pair'](cache[canonical['sha256']],torch.load(a['snapshot']['path'],map_location='cpu',weights_only=False))
    assert report==a['report'] and report['matched']
    (zero[name] if shot=='K0' else support[name] if kind=='support' else one[name]).append(int(r['success']))
    episodes.append(ref(path))
 primary=dict(zero);primary.update(one)
 tables[cohort]={'cases':50,'primary_successes':{k:sum(v) for k,v in primary.items()},'zero_shot_successes':{k:sum(v) for k,v in zero.items()},'support_successes':{k:sum(v) for k,v in support.items()},'primary_cases':primary,'zero_cases':zero,'support_cases':support,
  'student_vs_PPO':{f'{s}-vs-{p}':m['paired'](primary[s],primary[p]) for s in methods[2:] for p in methods[:2]},
  'full_K1_vs_K0':{s:m['paired'](one[s],zero[s]) for s in one},'full_vs_static':{s:m['paired'](one[s],zero[s.replace('_full','_static')]) for s in one}}
result={'completed':True,'native_episodes_verified':len(episodes),'canonical_cases':len(cache),'independent_phase_results':{k:ref(p/'RESULT.json') for k,p in phases.items()},'bindings':{k:ref(p/'binding.json') for k,p in phases.items()},'ID':tables['ID'],'OOD':tables['OOD'],'verified_episode_results':episodes,'pairing':bindings['ad']['pairing'],'weights_frozen':True,'separate_support_query':True,'tests':'unadjusted exploratory two-sided exact paired tests','limitations':['single source task and fit seed','source learning streams span different scenes','OOD held out from student training, not necessarily foundation pretraining','qualified <=1uint8 rendering roundoff with original policy pixels']}
write(OUT/'COMBINED-RESULT.json',result)
print(json.dumps({k:tables[k] for k in ('ID','OOD')},indent=2))
