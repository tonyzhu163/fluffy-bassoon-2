"""Read saved tensors and selected NPZ label arrays; no models or forward pass."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs')
index=json.loads((ROOT/'libero40-target-firstwave-20261030-w00-r1/experiment/index.json').read_text())
registry=index['fixed500_models']
report={'controllers':[],'training_labels':[]}
for row in registry:
 path=Path(row['final_path'])
 if hashlib.sha256(path.read_bytes()).hexdigest()!=row['final_sha256']:
  raise ValueError('Frozen candidate hash mismatch')
 saved=torch.load(path,map_location='cpu',weights_only=True)
 log_std=saved['policy']['log_std'].detach().numpy()
 scale=np.exp(np.clip(log_std,-5,2))
 report['controllers'].append(dict(objective=row['objective'],mode=row['mode'],final_path=str(path),
  final_sha256=row['final_sha256'],updates=row['updates'],initial_std=saved['config']['initial_std'],
  std_min=float(scale.min()),std_mean=float(scale.mean()),std_max=float(scale.max()),
  log_std_min=float(log_std.min()),log_std_mean=float(log_std.mean()),log_std_max=float(log_std.max()),
  learning_rate=saved['manifest']['contract']['learning_rate'],
  training_objective='Gaussian sequence NLL; per-channel log_std global, not state-conditioned'))
assembly_sha=index['source_provenance']['assembly_file_sha256']
assemblies=[p for name in ('libero40-all40-offline-20261030-r1','libero40-all40-offline-20261030-r2')
 for p in (ROOT/name).rglob('assembly.json') if hashlib.sha256(p.read_bytes()).hexdigest()==assembly_sha]
if len(assemblies)!=1:
 raise ValueError('Need exact bound assembly file')
assembly=json.loads(assemblies[0].read_text())
report['assembly_path']=str(assemblies[0]);report['assembly_file_sha256']=assembly_sha
for objective in ('ad','dpt'):
 spec=assembly['datasets']['train'][objective]
 data_path=Path(spec['path'])
 if not data_path.is_absolute():data_path=assemblies[0].parent/data_path
 with np.load(data_path,allow_pickle=False) as data:
  labels, executed, stages=data['target_noise'],data['executed_noise'],data['teacher_steps']
  for stage,mask in (('all',np.ones(len(stages),bool)),('warmup',stages==0),('post_update',stages>0)):
   if not mask.any():continue
   a,b=labels[mask],executed[mask]
   report['training_labels'].append(dict(objective=objective,stage=stage,chunks=int(mask.sum()),
    label_mean=float(a.mean()),label_std=float(a.std()),label_rms=float(np.sqrt(np.mean(a*a))),
    executed_std=float(b.std()),executed_rms=float(np.sqrt(np.mean(b*b))),
    coordinate_std_mean=float(a.std(0).mean()),dataset_path=spec['path'],dataset_manifest_sha256=spec['sha256']))
print(json.dumps(report,indent=2))
