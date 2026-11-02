"""Allocated-Linux qualification against the committed pre-change learner."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import torch

from fluffy_bassoon.meta_rl.residual_ppo import ResidualActorCritic, ResidualPPOConfig, update


root = Path(__file__).resolve().parents[2]
source = subprocess.check_output([
    "git", "-C", str(root), "show", "545a528355099245c82a56837d45d79d443693e5"
], text=True)
name = "fluffy_bassoon.meta_rl._pre_kl_reference"
module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
sys.modules[name] = module
exec(compile(source, "pre_kl_reference.py", "exec"), module.__dict__)
torch.set_num_threads(1)
torch.manual_seed(891)
policy = ResidualActorCritic(ResidualPPOConfig(3, 2, width=16, epochs=4, minibatch_size=5))
reference = deepcopy(policy)
features = torch.randn(17, 3)
with torch.no_grad():
    controls, log_probs, values = policy.act(features)
batch = dict(features=features[:, None], controls=controls[:, None],
    log_probs=log_probs[:, None], values=values[:, None],
    rewards=torch.linspace(0, 1, 17)[:, None], ends=torch.ones(17, 1, dtype=torch.bool))
optimizers = [torch.optim.Adam(model.parameters(), lr=3e-4) for model in (reference, policy)]
generators = [torch.Generator().manual_seed(892) for _ in range(2)]
global_rng = torch.get_rng_state()
old = module.update(reference, optimizers[0], batch, generator=generators[0])
new = update(policy, optimizers[1], batch, generator=generators[1])
for key, value in reference.state_dict().items():
    torch.testing.assert_close(value, policy.state_dict()[key], rtol=0, atol=0)
for key, state in optimizers[0].state_dict()["state"].items():
    for field, value in state.items():
        torch.testing.assert_close(value, optimizers[1].state_dict()["state"][key][field], rtol=0, atol=0)
for key, value in old.items():
    assert new[key] == value, (key, new[key], value)
torch.testing.assert_close(generators[0].get_state(), generators[1].get_state(), rtol=0, atol=0)
torch.testing.assert_close(torch.get_rng_state(), global_rng, rtol=0, atol=0)
print(json.dumps(dict(completed=True, source_commit=subprocess.check_output(
    ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
    reference_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    model_state_bitwise_equal=True, adam_state_bitwise_equal=True,
    original_metrics_exactly_equal=True, update_rng_bitwise_equal=True,
    global_rng_unchanged=True, optimizer_steps=new["optimizer_steps"],
    epochs_completed=new["epochs_completed"], final_metrics=new["final_metrics"]),
    indent=2, allow_nan=False))
