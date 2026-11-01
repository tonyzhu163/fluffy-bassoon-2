"""Observe actual official no-value-head CPU loader incompatibility and bound norm."""
import hashlib,json,os,random,sys
from pathlib import Path
import numpy as np,torch
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf
out=Path(os.environ['TASK_OUT']);converted=Path(os.environ['TASK_CONVERTED']);payload=Path(os.environ['TASK_PAYLOAD']);meta=Path(os.environ['TASK_META']);source=Path(os.environ['TASK_RLINF'])
assert torch.cuda.device_count()==0 and not torch.cuda.is_initialized()
random.seed(42);np.random.seed(42);torch.manual_seed(42)
with initialize_config_dir(config_dir=str(meta/'configs/native_ppo'),version_base=None):cfg=compose(config_name='robocasa365_pi05_eval')
m=cfg.rollout.model;assert m.openpi.task=='eval' and not m.add_value_head and m.openpi.action_horizon==50 and m.openpi.model_action_dim==32 and m.openpi.max_token_len==200 and m.openpi.num_images_in_input==3 and m.num_steps==10
OmegaConf.save(m,out/'actual-model-config.yaml',resolve=True)
from rlinf.models.embodiment.openpi import get_model
original=torch.nn.Module.load_state_dict;observed=[]
def observe(self,state_dict,*args,**kwargs):
 actual=original(self,state_dict,*args,**kwargs)
 observed.append(dict(model_class=type(self).__module__+'.'+type(self).__name__,strict=kwargs.get('strict',True),missing_keys=list(actual.missing_keys),unexpected_keys=list(actual.unexpected_keys),checkpoint_tensor_keys=len(state_dict)))
 return actual
torch.nn.Module.load_state_dict=observe
try:model=get_model(m,torch_dtype=torch.bfloat16)
finally:torch.nn.Module.load_state_dict=original
assert len(observed)==1 and not observed[0]['missing_keys'] and not observed[0]['unexpected_keys'] and not hasattr(model,'value_head')
assert not torch.cuda.is_initialized() and all(p.device.type=='cpu' for p in model.parameters())
# Use the actual published stats with the exact native opt-in factory; no forward/sampling.
from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from openpi import transforms
kwargs=OmegaConf.to_container(m.openpi_data,resolve=True)
train=get_openpi_config(m.openpi.config_name,model_path=str(converted),data_kwargs=kwargs);data=train.data.create(train.assets_dirs,train.model)
assert data.use_quantile_norm is False
stats=data.norm_stats;assert stats is not None
state=np.zeros(32,np.float32);normalized=transforms.Normalize(stats,use_quantiles=False)({'state':state.copy()})['state']
expected=(state-np.asarray(stats['state'].mean))/(np.asarray(stats['state'].std)+1e-6)
np.testing.assert_allclose(normalized,expected,rtol=0,atol=1e-6)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=dict(completed=True,model_class=type(model).__module__+'.'+type(model).__name__,actual_loader=observed,has_value_head=hasattr(model,'value_head'),model_parameter_count=sum(p.numel() for p in model.parameters()),model_devices=sorted({str(p.device) for p in model.parameters()}),model_dtypes=sorted({str(p.dtype) for p in model.parameters()}),actual_seed_before_construction=42,configuration_sha256=sha(out/'actual-model-config.yaml'),normalization_sha256=sha(payload/'assets/norm_stats.json'),converted_norm_identical=sha(payload/'assets/norm_stats.json')==sha(converted/'robocasa365/norm_stats.json'),state_stat_shape=list(np.asarray(stats['state'].mean).shape),action_stat_shape=list(np.asarray(stats['actions'].mean).shape),actual_mean_std_reference_passed=True,model_forward_calls=0,physical_actions=0,optimizer_updates=0,rlinf_origin=__import__('rlinf').__file__,torch_origin=torch.__file__)
(out/'strict-loading-qualification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
