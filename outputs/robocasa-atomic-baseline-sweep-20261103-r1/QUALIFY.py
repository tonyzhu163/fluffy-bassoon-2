"""One-run CPU admission evidence; no accelerator work."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

code=Path(sys.argv[1]); output=Path(sys.argv[2])
run=code/'outputs/robocasa-atomic-baseline-sweep-20261103-r1'
spec=importlib.util.spec_from_file_location('atomic_sweep',run/'EVAL.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
root=Path('/mowenjie/fluffy-bassoon-2')
native=root/'setups/robocasa365-native-20261101-r1/sources/RoboCasa'
registry=native/'robocasa/utils/dataset_registry.py'
tree=ast.parse(registry.read_text())
values={node.targets[0].id:node.value for node in tree.body if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name)}
targets=next(ast.literal_eval(k.value) for k in values['TARGET_TASKS'].keywords if k.arg=='atomic_seen')
horizons={k.arg:next(ast.literal_eval(h.value) for h in k.value.keywords if h.arg=='horizon')
          for k in values['ATOMIC_TASK_DATASETS'].keywords}
measured={'OpenDrawer','CloseFridge','PickPlaceCounterToCabinet','PickPlaceDrawerToCounter','TurnOffStove','CoffeeSetupMug'}
assert set(targets)-measured=={t for t,h in module.TASKS}
assert all(horizons[t]==h for t,h in module.TASKS)
assert subprocess.check_output(['git','-C',str(native),'rev-parse','HEAD'],text=True).strip()=='3443cdf3dd20f9ae8ed41adba442059b506891d0'
bank=module.cases();seeds=module.motor_seeds(bank)
assert len(bank)==len({(c['task'],c['native_seed']) for c in bank})==240
assert len(seeds)==len(set(seeds))==2940
assert sum(dict(module.TASKS)[c['task']] for c in bank)==147000
assert all(len(bank[rank::8])==30 for rank in range(8))
module.register_evaluation_tasks()
from fluffy_bassoon.meta_rl.robocasa_residual_parallel import native_chunk_offsets
assert native_chunk_offsets(bank)[-1]==2940
from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic,ResidualPPOConfig
from fluffy_bassoon.rl.action_residual import ResidualBound
import numpy as np
import torch
policy=ResidualActorCritic(ResidualPPOConfig(feature_dim=2088,residual_dim=6)).eval()
controls,_,_=policy.act(torch.randn(4,2088),deterministic=True)
chunk=np.random.default_rng(42).normal(size=(4,50,12)).astype('float32')
assert torch.count_nonzero(controls)==0
assert np.array_equal(ResidualBound(12,tuple(range(6)),(1.,)*6).apply(chunk,controls.numpy()),chunk)
for n,ref in [(0,[0.,.16112515805281938]),(20,[.8388748419471806,1.])]:
 assert np.allclose(module.interval(n),ref,atol=1e-12)
reference=json.loads((root/'outputs/robocasa-residual-approved-20261101-r1/S-job/pipeline-profile.json').read_text())['training']
initializer=reference['initializer']
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(16<<20),b''):h.update(block)
 return h.hexdigest()
assert sha(Path(initializer['model_directory'])/'model.safetensors')==initializer['model_sha256']
assert sha(initializer['norm_path'])==initializer['normalization_sha256']
storage=os.statvfs(root);free=storage.f_bavail*storage.f_frsize
assert free>=100*2**30
output.mkdir(parents=True,exist_ok=True)
probe=output/'write-probe'
with probe.open('x') as f:f.write('atomic baseline sweep\n');f.flush();os.fsync(f.fileno())
probe.unlink()
quota=subprocess.run(['quota','-s'],capture_output=True,text=True) if __import__('shutil').which('quota') else None
result=dict(completed=True,source_commit=subprocess.check_output(['git','-C',str(code),'rev-parse','HEAD'],text=True).strip(),
 native_source='3443cdf3dd20f9ae8ed41adba442059b506891d0',native_registry=str(registry),
 native_registry_sha256=sha(registry),tasks=module.TASKS,episodes=240,policy_controls=147000,
 chunks=2940,motor_seeds_unique=True,zero_residual_batch_identity=True,wilson_reference_pass=True,
 initializer=initializer,rlinf_execution_commit=reference['rlinf_execution_commit'],
 core_files={str(p.relative_to(code)):sha(p) for p in [
 code/'src/fluffy_bassoon/meta_rl/robocasa_residual.py',
 code/'src/fluffy_bassoon/meta_rl/rlinf_robocasa_ppo.py',
 code/'src/fluffy_bassoon/meta_rl/robocasa_residual_parallel.py',
 code/'src/fluffy_bassoon/meta_rl/robocasa_video.py',run/'EVAL.py',run/'COMMAND.sh']},
 storage_free_bytes=free,write_probe_pass=True,quota='unavailable' if quota is None else quota.stdout.strip() or quota.stderr.strip(),
 allocated_CPU_max=Path('/sys/fs/cgroup/cpu.max').read_text().strip(),
 allocated_RAM_bytes=int(Path('/sys/fs/cgroup/memory.max').read_text()),new_GPU_seconds=0,
 created_unix=time.time())
module.write(output/'QUALIFICATION.json',result)
print(json.dumps(result),flush=True)
