"""Read-only CPU metadata/hash audit. No learner, forward/backward or CUDA."""
import hashlib,json,zipfile
from pathlib import Path
import torch
root=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-staged-20261030-r2/optimizer-calibration')
result=json.loads((root/'result.json').read_text())
def sha(p):
 with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
checks=[]
for candidate in result['candidates']:
 path=Path(candidate['recovery']); saved=torch.load(path,map_location='cpu',weights_only=False)
 state=saved['learner']; rate=candidate['learning_rate']
 with zipfile.ZipFile(path) as z: crc=z.testzip() is None
 steps=sorted(set(float(v['step'].item()) for v in state['optimizer']['state'].values()))
 opt_rates=[v['lr'] for v in state['optimizer']['param_groups']]
 row=dict(learning_rate=rate,recovery_sha256=sha(path),recovery_bytes=path.stat().st_size,zip_crc_ok=crc,
  updates=state['updates'],optimizer_steps=state['optimizer_steps'],per_parameter_optimizer_steps=steps,
  optimizer_state_count=len(state['optimizer']['state']),config_lr=state['config']['learning_rate'],
  optimizer_rates=opt_rates,pending_trials=len(saved['pending_rollouts']),consumed_trials=len(saved['consumed_rollouts']),
  consumed_episodes=sum(len(r[1]['episodes']) for r in saved['consumed_rollouts']),
  sampler_present='sampler' in state,schedule_present='schedule' in saved,global_rng_keys=sorted(state['rng']),
  declared_gate_pass=candidate['provisional_stable'])
 row['execution_integrity_pass']=(row['recovery_sha256']==candidate['recovery_sha256'] and crc
  and row['updates']==1 and row['optimizer_steps']==5 and steps==[5.]
  and row['config_lr']==rate and opt_rates==[rate] and row['pending_trials']==0
  and row['consumed_trials']==3 and row['consumed_episodes']==12 and row['sampler_present'] and row['schedule_present'])
 checks.append(row)
arrays=[]
for entry in [result['reference'],*result['candidates']]:
 data=entry['post_divergence']; p=Path(data['decision_arrays'])
 with zipfile.ZipFile(p) as z: crc=z.testzip() is None
 arrays.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,zip_crc_ok=crc,
  hash_matches=sha(p)==data['decision_arrays_sha256']))
print(json.dumps(dict(inspection='Read-only CPU metadata, CRC and SHA; no model computation',candidates=checks,
 decision_arrays=arrays,input_unchanged=sha(result['input_checkpoint'])==result['input_sha256'],
 all_execution_integrity_checks_pass=all(r['execution_integrity_pass'] for r in checks)
  and all(r['zip_crc_ok'] and r['hash_matches'] for r in arrays),
 stable_learning_claim=False,main_comparison_advanced=False),indent=2))
