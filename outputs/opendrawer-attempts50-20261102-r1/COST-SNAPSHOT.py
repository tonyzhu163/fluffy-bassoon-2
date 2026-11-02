"""Preserve terminal R1/R2 curve accounting without dropping failed/partial work."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
root=Path('/mowenjie/fluffy-bassoon-2/outputs');rows=[]
config=[('ad',1,'job-y7j4b4n61ubv','16:38:15','16:53:21','Failed'),('dpt',1,'job-hqbjh4rfhxfi','16:34:20','16:51:44','Failed'),('ad',2,'job-io98kf4jtqh4','17:12:49','18:06:51','Stopped'),('dpt',2,'job-5s1gfdfcqh6b','17:12:49','18:06:38','Stopped')]
def ref(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
for family,version,job,start,end,status in config:
 p=root/f'opendrawer-{family}-practice-curve50-20261102-r{version}';episodes=[json.loads(x.read_text()) for x in p.glob('episodes/**/result.json')]
 journals=[]
 for q in p.glob('accounting/rank-*/*.json'):
  d=json.loads(q.read_text());calls=d['calls'];journals.append(dict(file=ref(q),calls=len(calls),unresolved=sum(not c['resolved'] for c in calls),conservative_controls=sum(c.get('executed_or_conservative',c['requested']) for c in calls)))
 clocks=[dict(file=ref(q),clock=json.loads(q.read_text())) for q in p.glob('attempts/*/node-*/clock.json')]
 probes=[ref(q) for q in p.glob('attempts/*/node-*/gpu-probe.json')]
 dt=lambda t:datetime.fromisoformat('2026-11-02T'+t+'+08:00')
 rows.append(dict(family=family,version=version,job=job,scheduler_created=dt(start).isoformat(),scheduler_terminal=dt(end).isoformat(),status=status,complete_episodes=len(episodes),complete_supports=sum(x['kind']=='support' for x in episodes),complete_queries=sum(x['kind']=='query' for x in episodes),journals=journals,conservative_controls=sum(x['conservative_controls'] for x in journals),unresolved_calls=sum(x['unresolved'] for x in journals),probes=probes,process_clocks=clocks,known_process_gpu_seconds=sum(x['clock']['observed_gpu_seconds'] for x in clocks),clock_count_incomplete=len(clocks)!=4,creation_to_terminal_gpu_seconds_upper=int((dt(end)-dt(start)).total_seconds())*8,upper_bound_includes_queue=True))
report=dict(checked_at=datetime.now(timezone.utc).isoformat(),terminal_curves=rows,new_actual_only_allocations_ongoing=True,all_experiment_costs_complete=False,retained_queries='Preserved/charged but excluded from future practice history',original_one_shot_accounting='Separate existing original phase/failure/fit/source ledger; no double counting retained records')
out=root/'opendrawer-attempts50-20261102-r1/OLD-CURVE-TERMINAL-ACCOUNTING.json';out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k not in ('journals','probes','process_clocks')} for row in rows],indent=2))
