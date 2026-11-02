"""Combine independently verified actual-attempt results on allocated Linux CPU."""
import hashlib,json,math,sys
from pathlib import Path
assert sys.platform=='linux'
root=Path('/mowenjie/fluffy-bassoon-2/outputs');out=root/'opendrawer-attempts50-20261102-r1'
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def paired(a,b):
 wins=sum(x>y for x,y in zip(a,b));losses=sum(x<y for x,y in zip(a,b));n=wins+losses
 return dict(wins=wins,losses=losses,ties=len(a)-n,exact_p=min(1.,2*sum(math.comb(n,j) for j in range(min(wins,losses)+1))/2**n) if n else 1.)
models={};bindings={};sources={}
for family in ('ad','dpt'):
 p=root/f'opendrawer-{family}-attempts50-20261102-r1';r=json.loads((p/'RESULT.json').read_text());b=json.loads((p/'binding.json').read_text())
 assert r['completed'] and r['actual_attempts_only'] and r['binding']==ref(p/'binding.json')
 assert b['driver_commit']=='26033f08515b4ae14a6372614225deaaf0d333b9' and b['attempt_numbers']==[1,2,3,4,5]
 assert r['total_actual_episodes']==500 and r['newly_collected_episodes']+r['retained_actual_episodes']==500
 models[family]=r;bindings[family]=b;sources[family]=ref(p/'RESULT.json')
assert bindings['ad']['pairing']==bindings['dpt']['pairing'] and bindings['ad']['assembly']==bindings['dpt']['assembly']
comparisons={}
for cohort in ('ID','OOD'):
 for r in models.values():
  c=r['cohorts'][cohort];assert c['cases']==50
  assert all(len(c['attempt_cases'][str(k)])==50 and sum(c['attempt_cases'][str(k)])==c['attempt_successes'][str(k)] for k in range(1,6))
 comparisons[cohort]={str(k):paired(models['ad']['cohorts'][cohort]['attempt_cases'][str(k)],models['dpt']['cohorts'][cohort]['attempt_cases'][str(k)]) for k in range(1,6)}
result=dict(completed=True,independent_results=sources,actual_attempts_only=True,cases_per_cohort=50,attempt_numbers=[1,2,3,4,5],successes={f:{c:models[f]['cohorts'][c]['attempt_successes'] for c in ('ID','OOD')} for f in models},paired_AD_vs_DPT=comparisons,paired_vs_attempt1={f:{c:models[f]['cohorts'][c]['paired_vs_attempt1'] for c in ('ID','OOD')} for f in models},pairing=bindings['ad']['pairing'],original_PPO_comparison=ref(root/'opendrawer-pairing-recovery-20261102-r1/COMBINED-RESULT.json'),frozen_weights=True,memory_retained_within_case=True,memory_cleared_between_cases=True,environment_reset_between_attempts=True,per_attempt_success_not_cumulative=True,limitations=['one source task and fit seed','different fixed motor streams across attempts; matched between models','no repeated static control so attempt trends alone are not a causal memory effect','75 ID/90 OOD tokens fit128 capacity but later exceed64-token training windows','original unmodified pixels with qualified <=1uint8 reset roundoff','OOD held out from student training, not necessarily foundation pretraining'],p_values='exploratory unadjusted exact paired tests')
(out/'COMBINED-RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
