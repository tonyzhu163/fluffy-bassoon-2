"""Freeze completed source recoveries; CPU metadata only, no model computation."""
from pathlib import Path
import datetime,hashlib,json,shutil
import torch
base=Path('/mowenjie/fluffy-bassoon-2')
out=base/'exports/robotwin-native-prefix-20261030-r1'
out.mkdir(parents=True,exist_ok=False)
manifest={'schema':'robotwin_source_prefix_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':'db0184089fa88a5f96c153f9b80029bb91b77def','source_phase_only':True,'teacher_competence_qualified':False,'methods':{}}
for method in ('sac','ppo','ppo_reset'):
 root=base/'outputs'/f'robotwin-wholetrial-{method}-20261030-r2'/'experiment'
 target=out/f'{method}-source-recovery.pt'
 shutil.copyfile(root/f'{method}-recovery.pt',target)
 saved=torch.load(target,map_location='cpu',weights_only=False)
 rounds=saved['completed_source_iterations'];trials=saved['completed_source_trials'];pending=saved['pending_rollouts']
 qualified=not pending and trials==rounds*6 and rounds>0 and saved['source_steps']==saved['learner']['primitive_steps']
 with target.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
 manifest['methods'][method]={'recovery_path':str(target),'recovery_sha256':sha,'recovery_bytes':target.stat().st_size,'eligible_complete_round':qualified,'completed_rounds':rounds,'completed_trials':trials,'source_actions':saved['source_steps'],'learner_updates':saved['learner']['updates'],'pending_trials':len(pending),'corpus_root':str(root/'corpus'),'source_contract':saved['contract'],'selection':{'phase':'source','maximum_source_iteration':rounds,'require_four_complete_episodes_per_trial':True,'exclude_validation_target_preflight_native':True,'require_committed_records_and_hashes':True},'dataset_materialization':'converter must enumerate and hash selected complete records before fitting; recovery snapshot is immutable, live corpus contains later excluded records'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'path':str(out),'methods':{m:{k:v for k,v in row.items() if k not in ('source_contract','selection')} for m,row in manifest['methods'].items()}}))
