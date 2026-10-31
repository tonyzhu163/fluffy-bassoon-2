"""Read-only CPU recovery inspection; no learner or model computation."""
import hashlib,json,zipfile
from pathlib import Path
import torch
root=Path('/mowenjie/fluffy-bassoon-2/outputs/robotwin-saved-ppo-20261030-r2')
result=json.loads((root/'diagnostic/result.json').read_text())
path=root/'diagnostic/diagnostic-recovery.pt'
state=torch.load(path,map_location='cpu',weights_only=False)
learner=state['learner']
def sha_file(p):
 with p.open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def weights_sha(weights):
 h=hashlib.sha256()
 for name,value in sorted(weights.items()):
  h.update(name.encode()); h.update(value.detach().contiguous().numpy().tobytes())
 return h.hexdigest()
with zipfile.ZipFile(path) as archive: crc_ok=archive.testzip() is None
steps=sorted(set(float(value['step'].item()) for value in learner['optimizer']['state'].values()))
original=Path(result['input_checkpoint'])
audit=dict(inspection='Read-only CPU hash, CRC and metadata; no learner, forward/backward, optimizer step or CUDA call',
 status=result['status'],retained_ppo_updates=learner['updates'],optimizer_steps=learner['optimizer_steps'],
 optimizer_state_count=len(learner['optimizer']['state']),per_parameter_optimizer_steps=steps,
 recovery_bytes=path.stat().st_size,recovery_sha256=sha_file(path),zip_crc_ok=crc_ok,
 policy_sha256=weights_sha(learner['policy']),original_input_sha256=sha_file(original),
 pending_trials=len(state['pending_rollouts']),consumed_trials=len(state['consumed_rollouts']),
 consumed_episodes=sum(len(row[1]['episodes']) for row in state['consumed_rollouts']),
 schedule_present='schedule' in state,sampler_present='sampler' in learner,
 global_rng_keys=sorted(learner['rng']),full_contract=state['contract'],
 metrics=result['metrics'],robot_performance_claim=False,main_comparison_advanced=False)
audit['all_checks_pass']=(audit['recovery_sha256']==result['recovery_sha256'] and crc_ok
 and audit['policy_sha256']==result['policy_after_sha256']
 and audit['original_input_sha256']==result['input_sha256']
 and audit['retained_ppo_updates']==1 and audit['optimizer_steps']==5 and steps==[5.]
 and audit['pending_trials']==0 and audit['consumed_trials']==3 and audit['consumed_episodes']==12
 and audit['schedule_present'] and audit['sampler_present'])
print(json.dumps(audit,indent=2))
