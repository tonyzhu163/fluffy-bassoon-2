"""Measure actual stock and zero-injection proposal differences on source demonstrations."""
import json
import sys
from pathlib import Path

import jax
import numpy as np
from cotrain_policy import array_sha, load_stock, transformed
from flow_head import bind, full_params, initial_params, live_sample
from source_data import SourceDataset
from ur5e_motor import to_openpi_observation


def measure(data_root, output):
    stock = load_stock()
    model = stock._model
    sampler = bind(model, live_sample)
    template = initial_params(model.action_out_proj.in_features, "joint")
    zero_params = full_params({}, template)
    examples = SourceDataset(data_root, split="train")
    report = []
    for ix in (0, len(examples) // 3, 2 * len(examples) // 3):
        row = examples[ix]
        item = to_openpi_observation(row["raw"], row["prompt"], canonicalize_quat=False)
        noise = np.random.default_rng(401 + ix).normal(size=(10, 32)).astype(np.float32)
        observations, _ = transformed(stock, item)
        a = stock.infer(item, noise=noise)["actions"]
        repeat = stock.infer(item, noise=noise)["actions"]
        proposed = sampler(jax.random.key(0), observations, noise[None], zero_params, row["history"][None])
        b = stock._output_transform({"actions": np.asarray(proposed[0]), "state": np.asarray(observations.state[0])})["actions"]
        report.append(dict(example=ix, tape=row["trajectory_id"], t=int(row["t"]),
                           stock=array_sha(a), repeat=array_sha(repeat), zero=array_sha(b),
                           repeat_max=float(np.abs(a - repeat).max()), zero_max=float(np.abs(a - b).max()),
                           issued_rms=float(np.sqrt(np.mean(np.square(a[:5] - b[:5]))))))
        output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    if len(jax.devices()) != 1 or jax.devices()[0].platform != "gpu":
        raise RuntimeError("Expected one allocated production GPU")
    measure(Path(sys.argv[1]), Path(sys.argv[2]))
