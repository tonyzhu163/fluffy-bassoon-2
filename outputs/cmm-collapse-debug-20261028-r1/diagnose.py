"""Read-only forensic replay of a copied CMM checkpoint, on allocated Linux CPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from fluffy_bassoon.meta_rl import fewshot_regression as fs


assert sys.platform == "linux" and os.environ.get("SLURM_STEP_ID")
torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
out = Path(sys.argv[1])
source = Path(sys.argv[2])
snapshot = out / "snapshot.pt"
with source.open("rb") as incoming, snapshot.open("xb") as target:
    shutil.copyfileobj(incoming, target)
saved = torch.load(snapshot, map_location="cpu", weights_only=False)
contract = saved["contract"]
assert contract["source_commit"] == "b6919d6b95a02704dee97cd03813a4c2ac1749de"
assert contract["variant"] == "cmm" and contract["shots"] == 20
model = fs.build("cmm", contract["width"])
model.load_state_dict(saved["policy"], strict=True)
model.eval()
assert all(torch.isfinite(p).all() for p in model.parameters())


def report(name, value):
    row = {"section": name, "data": value}
    print(json.dumps(row, allow_nan=False), flush=True)
    with (out / "results.jsonl").open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


report("snapshot", dict(iteration=saved["iteration"], contract=contract,
    sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(), job=os.environ["SLURM_JOB_ID"]))
rates = model.core.decay_params.detach()
report("decay", dict(min=float(rates.min()), median=float(rates.median()),
    max=float(rates.max()), at_zero=int((rates == 0).sum()), count=rates.numel(),
    effective_query_weight=(1 - torch.exp(-rates)).tolist()))

with torch.no_grad():
    x, y, query = fs.regression_batch(np.random.default_rng(fs.TEST_SEED + 1), 32, 20, .1)
    prediction = model(x)
    support_mse = float((prediction[~query] - y[~query]).square().mean())
    report("evaluation_32", dict(fs.query_metrics(prediction, y, query), support_mse=support_mse,
        prediction_std=float(prediction[query].std()), target_std=float(y[query].std())))
    design = torch.cat((x[:20, :, :12].double(), torch.ones(20, 32, 1)), -1).permute(1, 0, 2)
    coefficients = torch.linalg.lstsq(design, x[:20, :, 12].double().T.unsqueeze(-1)).solution
    test_design = torch.cat((x[20:, :, :12].double(), torch.ones(20, 32, 1)), -1).permute(1, 0, 2)
    oracle = (test_design @ coefficients).squeeze(-1).T
    report("least_squares_32", fs.query_metrics(torch.cat((y[:20], oracle.float())), y, query))

# Restore the saved generator and construct the exact next training batch, then
# use its first four lanes for inexpensive, matched gradient-path interventions.
rng = np.random.default_rng()
rng.bit_generator.state = saved["environment_rng"]
batch = fs.training_batch_size(saved["iteration"], contract["batch_size"])
x, y, query = fs.regression_batch(rng, batch, 20, .1)
x, y = x[:, :4], y[:, :4]
reference = None
for sever in ((), ("long_term",), ("activated",), ("trace",), ("alpha", "beta"),
              ("long_term", "activated", "trace", "alpha", "beta")):
    model.zero_grad(set_to_none=True)
    state, predictions, states = None, [], []
    for observation in x:
        if state is not None:
            state = {key: value.detach() if key in sever else value for key, value in state.items()}
        features, state = model.core(model.input(observation), state)
        for value in state.values():
            if value.requires_grad:
                value.retain_grad()
        states.append(state)
        predictions.append(model.output(features).squeeze(-1))
    prediction = torch.stack(predictions)
    if reference is None:
        reference = prediction.detach().clone()
    else:
        torch.testing.assert_close(prediction, reference, atol=0, rtol=0)
    loss = (prediction - y).square().mean()
    loss.backward()
    gradients = sorted(((name, float(p.grad.norm())) for name, p in model.named_parameters()
                        if p.grad is not None), key=lambda item: item[1], reverse=True)
    state_grads = [{key: float(value.grad.norm()) for key, value in state.items()
                    if value.grad is not None} for state in states]
    norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
    clipped = float(torch.stack([p.grad.norm().square() for p in model.parameters()
                                 if p.grad is not None]).sum().sqrt())
    assert clipped <= 5.00001
    report("gradient_paths", dict(detach=sever, loss=float(loss), preclip_norm=norm,
        postclip_norm=clipped, top_parameters=gradients[:8], state_gradients=state_grads,
        final_state_std={key: float(value.detach().std()) for key, value in states[-1].items()}))
    if not sever:
        parameters = dict(model.named_parameters())
        before = {name: p.detach().clone() for name, p in parameters.items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=contract["learning_rate"], weight_decay=0.)
        optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
        optimizer.step()
        updates = sorted(((name, float((p - before[name]).norm()),
                            float((p - before[name]).norm() / before[name].norm().clamp_min(1e-12)))
                          for name, p in parameters.items()), key=lambda item: item[2], reverse=True)
        report("optimizer_step", dict(largest_relative_updates=updates[:10]))
        model.load_state_dict(saved["policy"], strict=True)

report("complete", True)
