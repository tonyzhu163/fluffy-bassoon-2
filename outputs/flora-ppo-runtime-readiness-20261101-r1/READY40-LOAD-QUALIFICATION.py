"""One allocated-GPU strict-load observation; no forward, optimizer or simulator."""
import hashlib,json,os,random,socket,sys,time
from pathlib import Path
import numpy as np
import torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
from fluffy_bassoon.meta_rl.rlinf_flora_ppo import INITIALIZER_REPO,INITIALIZER_REVISION,INITIALIZER_WEIGHT_SHA,NORM_SHA
from fluffy_bassoon.meta_rl.rlinf_native_ppo import tensor_hash
import rlinf.models.embodiment.openpi as official
from rlinf.models.embodiment.openpi.checkpoint import _missing_pi0_keys
out=Path(os.environ['TASK_OUT']);source=Path(os.environ['TASK_WT'])
assert socket.gethostname().startswith('job-') and torch.cuda.device_count()==1
assert Path(official.__file__).resolve().is_relative_to(Path(os.environ['TASK_RLINF']))
with initialize_config_dir(version_base=None,config_dir=str(source/'configs/native_ppo')):
 cfg=compose(config_name='libero44_flora')
OmegaConf.resolve(cfg)
assert cfg.actor.model.openpi.model_action_dim==32 and cfg.actor.model.openpi.action_horizon==10
assert cfg.actor.model.num_steps==3 and cfg.actor.model.openpi.action_chunk==5
assert cfg.actor.model.openpi.discrete_state_input is False and cfg.actor.model.openpi_data.extra_delta_transform is False
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
observed_seed=torch.initial_seed();assert observed_seed==42
before_rng_sha=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
original=torch.nn.Module.load_state_dict;loads=[]
def observe(self,state_dict,*args,**kwargs):
 result=original(self,state_dict,*args,**kwargs)
 loads.append(dict(model_class=type(self).__module__+'.'+type(self).__qualname__,missing_keys=list(result.missing_keys),unexpected_keys=list(result.unexpected_keys),loaded_tensor_count=len(state_dict),strict=kwargs.get('strict',True)))
 return result
torch.nn.Module.load_state_dict=observe
started=time.time()
try:model=official.get_model(cfg.actor.model,torch_dtype=torch.bfloat16)
finally:torch.nn.Module.load_state_dict=original
assert len(loads)==1,loads
row=loads[0];missing_backbone=_missing_pi0_keys(row['missing_keys'])
assert not missing_backbone and not row['unexpected_keys']
assert all(key.startswith('value_head.') for key in row['missing_keys']),row
head={k:v for k,v in model.state_dict().items() if k.startswith('value_head.')}
assert head and hasattr(model,'value_head')
head_sha=tensor_hash(head)
model=model.to('cuda');torch.cuda.synchronize()
result=dict(completed=True,repository_id=INITIALIZER_REPO,revision=INITIALIZER_REVISION,model_sha256=INITIALIZER_WEIGHT_SHA,normalization_sha256=NORM_SHA,
 backbone_missing_keys=missing_backbone,unexpected_keys=row['unexpected_keys'],actual_load_state_dict=row,expected_fresh_value_head_missing_keys=row['missing_keys'],value_head_seed=observed_seed,value_head_tensor_sha256=head_sha,
 cpu_rng_sha256_immediately_before_construction=before_rng_sha,cuda_device_name=torch.cuda.get_device_name(0),cuda_device_uuid=str(torch.cuda.get_device_properties(0).uuid),model_class=type(model).__module__+'.'+type(model).__qualname__,
 parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),model_parameter_count=sum(p.numel() for p in model.parameters()),cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_peak_bytes=torch.cuda.max_memory_allocated(),
 source_commit=os.environ['TASK_SOURCE'],rlinf_commit='c605b9d55ed93a180119b099e6f836abff293b08',official_builder_path=official.__file__,literal_model_config=OmegaConf.to_container(cfg.actor.model,resolve=True),elapsed_seconds=time.time()-started,
 forwards=0,optimizer_steps=0,simulator_actions=0,initialization_namespace='released_LIBERO40_PyTorch_no_C565_equivalence')
(out/'loading-qualification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(event='strict_ready40_load_complete',backbone_missing=len(missing_backbone),unexpected=len(row['unexpected_keys']),expected_fresh_head_missing=len(row['missing_keys']),value_head_sha256=head_sha,seconds=result['elapsed_seconds'])),flush=True)
