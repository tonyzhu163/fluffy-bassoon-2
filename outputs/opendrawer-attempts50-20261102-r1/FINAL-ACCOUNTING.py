"""Close completed actual-only allocations without counting retained episodes twice."""
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs');OUT=ROOT/'opendrawer-attempts50-20261102-r1'
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def read(p):return json.loads(p.read_text())
scheduler=read(OUT/'FINAL-SCHEDULER.json');phases=[]
for family,job,expected_new,expected_hash in [('ad','job-pyhz34s90cnv',273,'7361cf4c32e5573fd8acfde138cf11341cdacba4bb68e66b140cde2611f9ac07'),('dpt','job-ci8hnzbt5ur9',181,'610b0c3a2eb8a01ff725dae3395b1738c0c0e08426736b04a8cb77f39693dd86')]:
 p=ROOT/f'opendrawer-{family}-attempts50-20261102-r1';b=read(p/'binding.json');r=read(p/'RESULT.json');clock=[];uuids=[]
 assert r['completed'] and r['newly_collected_episodes']==expected_new and ref(p/'binding.json')['sha256']==expected_hash
 assert scheduler['jobs'][job]['state']=='Succeeded' and scheduler['jobs'][job]['gpus']==8
 for node in range(4):
  a=p/'attempts'/f'{family}-actual-attempts-r1'/f'node-{node:02d}'
  t=read(a/'workers-terminal.json');assert t['exit_codes']==[0,0]
  c=read(a/'clock.json');assert c['exit_code']==0 and c['allocated_gpus']==2;clock.append(dict(file=ref(a/'clock.json'),clock=c))
  m=read(a/'manifest.json');assert m['source']==b['driver_commit'] and m['binding']==ref(p/'binding.json');uuids.extend(x['uuid'] for x in m['devices'])
 assert len(set(uuids))==8
 for rank in range(8):
  w=read(p/'workers'/f'rank-{rank:02d}.json');assert w['completed'] and w['weights_unchanged'] and w['binding_sha256']==expected_hash
 journals=[]
 for q in p.glob('accounting/rank-*/*.json'):
  calls=read(q)['calls'];assert all(c['resolved'] for c in calls)
  journals.append(dict(file=ref(q),conservative_controls=sum(c.get('executed_or_conservative',c['requested']) for c in calls),calls=len(calls),unresolved=0))
 assert len(journals)==8 and len(list(p.glob('episodes/**/result.json')))==expected_new and len(list(p.glob('trials/**/*.json')))==100
 s=scheduler['jobs'][job];duration=(datetime.fromisoformat(s['terminal'])-datetime.fromisoformat(s['created'])).total_seconds()
 phases.append(dict(method=b['method'],job=job,binding=ref(p/'binding.json'),verified_summary=ref(p/'RESULT.json'),new_actual_episodes=expected_new,reused_actual_episodes=r['retained_actual_episodes'],total_admitted_actual_episodes=500,new_controls=sum(j['conservative_controls'] for j in journals),unresolved_calls=0,journals=journals,clocks=clock,all_process_clocks_complete=True,measured_process_GPU_seconds=sum(c['clock']['observed_gpu_seconds'] for c in clock),creation_to_terminal_GPU_seconds_upper=int(duration)*8,scheduler_receipt=s,terminal_success_before_publication=True))
old=read(OUT/'OLD-CURVE-TERMINAL-ACCOUNTING.json');original=read(ROOT/'opendrawer-pairing-recovery-20261102-r1/RESOURCE-LEDGER.json')
report=dict(completed=True,checked_at=datetime.now(timezone.utc).isoformat(),all_owned_experiment_allocations_terminal=True,source_and_original_scope=ref(ROOT/'opendrawer-pairing-recovery-20261102-r1/RESOURCE-LEDGER.json'),old_curve_attempts=ref(OUT/'OLD-CURVE-TERMINAL-ACCOUNTING.json'),actual_only_phases=phases,combined_result=ref(OUT/'COMBINED-RESULT.json'),new_actual_controls=sum(x['new_controls'] for x in phases),new_actual_episodes=sum(x['new_actual_episodes'] for x in phases),actual_only_process_GPU_seconds=sum(x['measured_process_GPU_seconds'] for x in phases),actual_only_creation_to_terminal_GPU_seconds_upper=sum(x['creation_to_terminal_GPU_seconds_upper'] for x in phases),inherited_source_receipt_GPU_seconds=original['inherited_source']['inherited_GPU_seconds'],original_incremental_creation_to_terminal_upper=original['incremental_original_creation_to_terminal_GPU_seconds_upper'],old_curves_creation_to_terminal_upper=original['old_curve_creation_to_terminal_GPU_seconds_upper'],cost_rules='Retained episodes cost only their source allocation; process clocks and creation-to-terminal bounds are alternatives, never additive for one allocation; bounds include queue and are not prices or scientific caps')
report['source_receipts_plus_all_incremental_creation_to_terminal_GPU_seconds_upper']=report['inherited_source_receipt_GPU_seconds']+report['original_incremental_creation_to_terminal_upper']+report['old_curves_creation_to_terminal_upper']+report['actual_only_creation_to_terminal_GPU_seconds_upper']
(OUT/'FINAL-ACCOUNTING.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='actual_only_phases'},indent=2))
