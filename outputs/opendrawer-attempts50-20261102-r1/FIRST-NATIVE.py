"""Audit first actual continuation against its frozen prefix and native bank."""
import argparse,json,os,runpy,sys
from pathlib import Path
import torch
assert sys.platform=='linux' and not torch.cuda.is_available()
p=argparse.ArgumentParser();p.add_argument('family',choices=('ad','dpt'));a=p.parse_args()
root=Path('/mowenjie/fluffy-bassoon-2');out=root/'outputs'/f'opendrawer-{a.family}-attempts50-20261102-r1'
os.environ['TASK_CURVE_METHOD']=a.family+'_full';os.environ['TASK_CURVE_OUTPUT']=str(out)
m=runpy.run_path(str(root/'worktrees/opendrawer-attempts50-20261102-r1/outputs/opendrawer-attempts50-20261102-r1/ATTEMPTS.py'))
b=json.loads((out/'binding.json').read_text());assert b['driver_commit']=='26033f08515b4ae14a6372614225deaaf0d333b9'
assert m['BASE']['head'](root/'worktrees/opendrawer-attempts50-20261102-r1')==b['driver_commit']
uuids=[]
for node in range(4):
 d=out/'attempts'/f'{a.family}-actual-attempts-r1'/f'node-{node:02d}'
 manifest=json.loads((d/'manifest.json').read_text());probe=json.loads((d/'gpu-probe.json').read_text())
 assert manifest['source']==b['driver_commit'] and manifest['binding']==m['ref'](out/'binding.json')
 assert len(probe['devices'])==2
 uuids.extend(x['uuid'] for x in probe['devices'])
assert len(set(uuids))==8
path=sorted(out.glob('episodes/**/result.json'))[0];r=json.loads(path.read_text());c=r['cohort'];i=r['index'];k=r['K']
assert k in range(2,6) and r['kind']=='support' and r['complete'] and r['evaluation_source']==b['driver_commit']
assert r['candidate']==b['selected']['candidate']
assert r['native_case']==m['native_case'](c,i) and r['motor_seeds']==m['motor_seeds'](c,i,k)
assert r['memory_entry']['cache_length']==(k-1)*m['horizon'](c)//50
prefix=b['practice_prefixes'][f'{c}-{i:03d}']
previous=m['read_ref'](prefix[str(k-1)]) if str(k-1) in prefix else json.loads((out/'episodes'/c/f'case-{i:03d}'/f'K{k-1}'/'support/result.json').read_text())
assert r['memory_entry']['previous_done'] and r['memory_entry']['previous_reward']==previous['success']
assert m['sha'](r['records']['path'])==r['records']['sha256'];rows=torch.load(r['records']['path'],map_location='cpu',weights_only=False)
assert len(rows)==m['horizon'](c)//50 and rows[-1]['done'] and rows[-1]['reward']==r['success']
truth=m['read_ref'](r['truth']);assert truth['case']==r['native_case'] and int(truth['simulator_final_success'])==r['success']
x=r['initial_pairing'];marker=m['read_ref'](x['canonical']);native=Path(x['canonical']['path']).parent/'initial.pt'
assert m['sha'](native)==marker['snapshot_sha256'] and m['sha'](x['snapshot']['path'])==x['snapshot']['sha256']
comparison=m['PAIRING']['compare_native_pair'](torch.load(native,map_location='cpu',weights_only=False),torch.load(x['snapshot']['path'],map_location='cpu',weights_only=False))
assert comparison==x['report'] and comparison['matched']
assert not list(out.glob('episodes/**/query/result.json'))
report=dict(passed=True,method=b['method'],binding=m['ref'](out/'binding.json'),first_native_result=m['ref'](path),all_four_probes=True,unique_uuids=uuids,native_pair=comparison,terminal_truth_verified=True,previous_actual_terminal_feedback_verified=True,original_motor_stream=True,no_new_query_probes=True,completed_new_episodes=len(list(out.glob('episodes/**/result.json'))))
m['write'](out/'FIRST-NATIVE.json',report);print(json.dumps(report,indent=2))
