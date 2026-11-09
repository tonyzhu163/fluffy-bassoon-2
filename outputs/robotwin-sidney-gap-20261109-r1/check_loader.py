"""Check which checkpoint tensors the model-card loader actually installs."""
import json
import os
from pathlib import Path
import sys

import torch
from safetensors import safe_open
from lerobot.configs import PreTrainedConfig
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

assert os.uname().nodename.startswith('job-') and torch.cuda.device_count() == 1
bundle = Path(sys.argv[1])
config = PreTrainedConfig.from_pretrained(str(bundle))
config.compile_model = False
config.gradient_checkpointing = False
policy = PI05Policy.from_pretrained(str(bundle), config=config, local_files_only=True).eval()
actual = policy.state_dict()
mismatch = []
with safe_open(bundle/'model.safetensors', framework='pt', device='cpu') as stored:
    keys = set(stored.keys())
    missing = sorted(keys - set(actual))
    for key in sorted(keys & set(actual)):
        tensor = actual[key].detach().cpu()
        if not torch.equal(tensor, stored.get_tensor(key).to(tensor.dtype)):
            mismatch.append(key)
report = dict(expected_tensors=len(keys), missing=missing, unequal=mismatch, extra=sorted(set(actual)-keys),
              comparison='all original tensors compared after casting to the loaded parameter dtype',
              compile_model=False, torch_version=torch.__version__)
Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
