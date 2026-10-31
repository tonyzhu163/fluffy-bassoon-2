"""Saved metadata/arrays only; no model imports, inference or new interactions."""
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np

ROOT=Path('/mowenjie/fluffy-bassoon-2/outputs')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def natives(prefix,count):
 result={}
 for worker in range(count):
  root=ROOT/f'{prefix}-w{worker:02d}-r1/experiment'
  index=json.loads((root/'index.json').read_text())
  assert index['completed']
  for saved in index['completed_groups']:
   g=saved['group']
   if g['case']['benchmark']!='libero90' or g['method']!='native' or g['support_budgets']!=[0]:continue
   path=Path(saved['result_path'])
   assert sha(path)==saved['result_sha256']
   receipt=json.loads(path.read_text())
   assert receipt['completed'] and receipt['weights_unchanged'] and len(receipt['episodes'])==1
   e=receipt['episodes'][0]
   task=g['case']['task_id'];assert task not in result
   result[task]=dict(case=g['case'],success=e['success'],episode_random_seed=e['episode_random_seed'],
    nominal_state_sha256=e['initial_state_vector_sha256'],initial_observation_sha256=e['initial_observation_sha256'],
    initial_public_arrays_sha256=saved['audit']['initial_public_arrays_sha256'],
    motor_parameters_sha256=index['motor_parameters_before_sha256'],result_path=str(path),
    result_sha256=saved['result_sha256'],trace=str(path.parent/e['trace']),trace_sha256=e['trace_sha256'])
 assert set(result)==set(range(90))
 return result


def trace_first(meta):
 p=Path(meta['trace']);assert sha(p)==meta['trace_sha256']
 with gzip.open(p,'rb') as f:rows=pickle.load(f)
 return rows[0]


def compare(a,b):
 a,b=np.asarray(a),np.asarray(b)
 assert a.shape==b.shape and a.dtype==b.dtype
 delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
 return dict(shape=list(a.shape),dtype=str(a.dtype),bitwise_equal=bool(np.array_equal(a,b)),
  unequal_values=int((a!=b).sum()),max_abs_delta=float(delta.max()),mean_abs_delta=float(delta.mean()),
  first_sha256=hashlib.sha256(a.tobytes()).hexdigest(),second_sha256=hashlib.sha256(b.tobytes()).hexdigest())


first=natives('libero40-target-firstwave-20261030',6)
noise=natives('libero40-noise-controls-20261030',2)
fields=('case','episode_random_seed','nominal_state_sha256','initial_observation_sha256','initial_public_arrays_sha256','motor_parameters_sha256')
metadata_mismatches={key:[task for task in range(90) if first[task][key]!=noise[task][key]] for key in fields}
discordant=[task for task in range(90) if first[task]['success']!=noise[task]['success']]
details=[]
for task in discordant:
 a,b=trace_first(first[task]),trace_first(noise[task])
 cameras={key:compare(a['motor_observation'][key],b['motor_observation'][key]) for key in a['motor_observation']}
 details.append(dict(task_id=task,firstwave=first[task],noise_run=noise[task],initial_arrays=cameras,
  native_initial_noise=compare(a['native_initial_noise'],b['native_initial_noise']),
  first_decoded_actions=compare(a['executed_actions'],b['executed_actions'])))
print(json.dumps(dict(format='libero90_native_crossrun_saved_comparison_v1',
 firstwave_successes=sum(r['success'] for r in first.values()),noise_run_successes=sum(r['success'] for r in noise.values()),
 queries_per_run=90,all90_metadata_mismatch_ids=metadata_mismatches,
 earlier_only=[t for t in discordant if first[t]['success']],later_only=[t for t in discordant if noise[t]['success']],
 outcome_discordant_ids=discordant,discordant_firstchunk_comparisons=details,
 method='All90 native metadata checked; saved arrays inspected only for outcome-discordant cases; no models/rollouts/tests',
 pooling='Keep runs separate; contemporary within-run native is primary'),indent=2))
