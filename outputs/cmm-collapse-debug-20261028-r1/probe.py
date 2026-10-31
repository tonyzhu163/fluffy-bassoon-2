"""Matched frozen-checkpoint gradient probes; never changes a training run."""
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

from fluffy_bassoon.meta_rl import fewshot_regression as fs


assert sys.platform == "linux" and os.environ.get("SLURM_STEP_ID")
torch.set_num_threads(int(os.environ["SLURM_CPUS_PER_TASK"]))
saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
out = Path(sys.argv[2])
model = fs.build("cmm", saved["contract"]["width"])
model.load_state_dict(saved["policy"], strict=True)
model.eval()
rng = np.random.default_rng()
rng.bit_generator.state = saved["environment_rng"]
rows = []
for batch_index in range(16):
    # Generate each complete original-size batch before selecting eight lanes.
    batch_size = fs.training_batch_size(saved["iteration"] + batch_index, 64)
    x, y, query = fs.regression_batch(rng, batch_size, 20, .1)
    x, y = x[:, :8], y[:, :8]
    reference = None
    for sever in ((), ("long_term",), ("activated", "trace")):
        model.zero_grad(set_to_none=True)
        state, predictions = None, []
        for observation in x:
            if state is not None:
                state = {key: value.detach() if key in sever else value for key, value in state.items()}
            features, state = model.core(model.input(observation), state)
            predictions.append(model.output(features).squeeze(-1))
        prediction = torch.stack(predictions)
        if reference is None:
            reference = prediction.detach().clone()
        else:
            torch.testing.assert_close(prediction, reference, atol=0, rtol=0)
        loss = (prediction - y).square().mean()
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True))
        row = dict(batch=batch_index, detach=sever, loss=float(loss), gradient_norm=norm)
        rows.append(row)
        print(json.dumps(row), flush=True)
    with (out / "probes.json").open("w") as stream:
        json.dump(dict(snapshot_iteration=saved["iteration"], lanes=8, rows=rows), stream, indent=2)

# A forward-only memory intervention tests whether support information reaches
# the queries, independently of the gradient-path interventions above.
with torch.no_grad():
    x, y, query = fs.regression_batch(np.random.default_rng(2_100_000_000), 256, 20, .1)
    scores = {}
    for lesion in ("normal", "erase_ltm_at_query", "erase_all_at_query", "support_values_zero"):
        sequence = x.clone()
        if lesion == "support_values_zero":
            sequence[:20, :, 12] = 0
        state, predictions = None, []
        for step, observation in enumerate(sequence):
            if step == 20 and lesion == "erase_ltm_at_query":
                state = dict(state, long_term=model.core.initial_state(256)["long_term"])
            if step == 20 and lesion == "erase_all_at_query":
                state = None
            features, state = model.core(model.input(observation), state)
            predictions.append(model.output(features).squeeze(-1))
        scores[lesion] = fs.query_metrics(torch.stack(predictions), y, query)
    (out / "lesions.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores), flush=True)
