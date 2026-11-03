"""GPU replay of retained Q1 inputs; zero simulator controls and zero SGD."""
from collections import defaultdict
import gc
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model
from fluffy_bassoon.meta_rl.rlinf_robocasa_flow_ppo import flow_logprob_parity

root = Path(os.environ['TASK_Q1'])
output = Path(os.environ['TASK_OUTPUT'])
cfg = OmegaConf.create(json.loads((root / 'resolved-config.json').read_text()))
torch.set_num_threads(4)
batch = torch.load(root / 'training-rollouts/iteration-0001-policy-0000-rank-00.pt',
                   map_location='cpu', weights_only=False)['rollout_batch']
inputs = batch['forward_inputs']
T, B = batch['prev_logprobs'].shape[:2]
flat = {key: value.reshape(T * B, *value.shape[2:]) for key, value in inputs.items()}
old = batch['prev_logprobs'].reshape(T * B, 50, 12)


def hash_row(value):
    return hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


chain_index = {hash_row(value): index for index, value in enumerate(flat['chains'])}
original_indices, record = None, None
for path in sorted((root / 'sampling').glob('*/*.pt.gz')):
    with gzip.open(path, 'rb') as stream:
        value = torch.load(stream, map_location='cpu', weights_only=False)
    if value['mode'] != 'train' or value['terminal_value_only']:
        continue
    chains = value['forward_sampling_fields']['chains']
    indexes = [chain_index.get(hash_row(row)) for row in chains]
    if all(index is not None for index in indexes):
        original_indices, record = torch.tensor(indexes), value
        break
assert original_indices is not None and len(original_indices) == 16
assert torch.equal(old[original_indices], record['old_logprobs'])
generator = torch.Generator().manual_seed(1234)
shuffled_indices = torch.randperm(T * B, generator=generator)[:16]

model = get_model(cfg.actor.model, torch_dtype=torch.bfloat16).to('cuda:0')
model.set_global_step(0)
results = []
for name, indexes, gradient, frozen_image_eval in [('original_cohort_eval', original_indices, False, False),
        ('original_cohort_train_grad', original_indices, True, False),
        ('shuffled_cohort_eval', shuffled_indices, False, False),
        ('shuffled_cohort_train_grad', shuffled_indices, True, False),
        ('original_cohort_train_frozen_image_eval', original_indices, True, True),
        ('shuffled_cohort_train_frozen_image_eval', shuffled_indices, True, True)]:
    model.train(gradient)
    if frozen_image_eval:
        assert not any(parameter.requires_grad for parameter in model.img.parameters())
        model.img.eval()
    selected = {key: value[indexes].to('cuda:0') for key, value in flat.items()}
    tick = time.monotonic()
    with torch.set_grad_enabled(gradient):
        result = model(forward_inputs=selected, compute_logprobs=True,
            compute_values=True, compute_entropy=False, use_cache=False)
    report = flow_logprob_parity(result['logprobs'], old[indexes])
    results.append(dict(name=name, report=report, seconds=time.monotonic() - tick,
        indices=indexes.tolist(), tf32=torch.backends.cuda.matmul.allow_tf32,
        matmul_precision=torch.get_float32_matmul_precision()))
    print(json.dumps(results[-1]), flush=True)
    del result, selected
    gc.collect(); torch.cuda.empty_cache()
    (output / 'RESULT.json').write_text(json.dumps(dict(completed=False,
        results=results, SGD=0, new_controls=0), indent=2) + '\n')
(output / 'RESULT.json').write_text(json.dumps(dict(completed=True,
    results=results, SGD=0, new_controls=0), indent=2) + '\n')
