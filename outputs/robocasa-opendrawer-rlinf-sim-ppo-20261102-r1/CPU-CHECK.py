"""Allocated Linux input/loader qualification; no simulator or GPU inference."""
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time

assert platform.system() == 'Linux' and not os.environ.get('CUDA_VISIBLE_DEVICES')
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors import safe_open

root = Path(os.environ['TASK_META'])
output = Path(os.environ['TASK_CPU_OUTPUT'])
output.mkdir(parents=True, exist_ok=False)
with initialize_config_dir(config_dir=str(root / 'configs/native_ppo'), version_base='1.1'):
    cfg = compose(config_name='robocasa365_opendrawer_pi05_sim_ppo')
OmegaConf.resolve(cfg)
from rlinf.config import validate_cfg
cfg = validate_cfg(cfg)
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo_runtime import FlowActor, FlowEnvironment, FlowRollout, FlowRunner

tick = time.monotonic()
from rlinf.models.embodiment.openpi import get_model
from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL
model = get_model(cfg.actor.model, torch_dtype=torch.bfloat16)
assert isinstance(model, Pi0RL) and model.rl_cfg.train_expert_only
assert model.rl_cfg.noise_method == 'flow_sde' and model.rl_cfg.noise_level == .3
assert model.action_horizon == model.action_chunk == 50 and model.action_env_dim == 12 and model.num_steps == 10
with safe_open(str(Path(cfg.actor.model.model_path) / 'model.safetensors'), framework='pt', device='cpu') as f:
    original_keys = set(f.keys())
model_keys = set(model.state_dict())
missing = sorted(model_keys - original_keys)
unexpected = sorted(original_keys - model_keys)
assert missing and all('value_head' in key for key in missing), missing
assert not unexpected, unexpected
trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
frozen = [(name, parameter.numel()) for name, parameter in model.named_parameters() if not parameter.requires_grad]
assert trainable and frozen and any('value_head' in name for name, _ in trainable)
assert not any(name.startswith('img.') for name, _ in trainable)
result = subprocess.run([os.sys.executable, '-B', '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
    'tests/test_robocasa_flow_ppo.py'], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
value = dict(completed=result.returncode == 0, source=subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
    runtime=subprocess.check_output(['git', '-C', os.environ['RLINF_PATH'], 'rev-parse', 'HEAD'], text=True).strip(),
    model_loading_seconds=time.monotonic() - tick, missing_only_fresh_value_head=missing,
    unexpected_checkpoint_keys=unexpected, trainable_elements=sum(n for _, n in trainable),
    frozen_elements=sum(n for _, n in frozen), trainable_names=[name for name, _ in trainable],
    tests=result.stdout, exit_code=result.returncode, new_simulator_controls=0, inference_calls=0, GPU_computation=False)
(output / 'RESULT.json').write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps({k:v for k,v in value.items() if k != 'trainable_names'}, indent=2))
raise SystemExit(result.returncode)
