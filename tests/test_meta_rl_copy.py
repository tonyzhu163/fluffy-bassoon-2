"""Copy targets must require memory, with gradients across the input/query boundary."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl.cmm import CMMCore
from fluffy_bassoon.meta_rl.copy_task import CopyModel, copy_batch
from fluffy_bassoon.meta_rl.ppo import GRUCore, LoopedTransformerCore


def test_copy_batch_preserves_order_and_hides_targets_at_query_time():
    inputs, targets = copy_batch(np.random.default_rng(7), 3, 4, 2)
    assert inputs.shape == (8, 3, 4) and targets.shape == (4, 3, 2)
    torch.testing.assert_close(inputs[:4, :, :2], targets)
    assert not inputs[4:, :, :2].any()
    assert inputs[:4, :, 2].all() and not inputs[4:, :, 2].any()
    assert not inputs[:4, :, 3].any() and inputs[4:, :, 3].all()
    repeated, _ = copy_batch(np.random.default_rng(7), 3, 4, 2)
    torch.testing.assert_close(inputs, repeated, rtol=0, atol=0)
    with pytest.raises(ValueError, match="positive"):
        copy_batch(np.random.default_rng(7), 0, 4, 2)


@pytest.mark.parametrize("kind", ["gru", "ctm", "cmm", "looped_transformer"])
def test_query_gradients_require_input_phase_state(kind):
    torch.manual_seed(12)
    core = (GRUCore(6, width=8) if kind == "gru" else
            LoopedTransformerCore(6, width=8, memory_tokens=2) if kind == "looped_transformer" else CMMCore(
        6, width=8, stm_length=3, ltm_tokens=0 if kind == "ctm" else 2,
        layers=0 if kind == "ctm" else 1, synapse_depth=2))
    model = CopyModel(core, 2)
    inputs, targets = copy_batch(np.random.default_rng(7), 2, 4, 2)
    inputs.requires_grad_()
    for reset in (False, True):
        loss = (model(inputs, reset_before_query=reset) - targets).square().mean()
        gradient, = torch.autograd.grad(loss, inputs)
        assert bool(gradient[:4, :, :2].abs().sum() > 0) != reset


@pytest.mark.skipif(not os.environ.get("SLURM_STEP_ID"), reason="Copy CLI requires an allocated Linux step")
def test_copy_cli_saves_complete_checkpoint_and_query_metrics(tmp_path):
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output = tmp_path / "copy"
    arguments = [sys.executable, "-m", "fluffy_bassoon.meta_rl.copy_task",
                             "--source-commit", source, "--output-dir", str(output),
                             "--core", "gru", "--width", "8", "--length", "3", "--dimensions", "2",
                             "--updates", "1", "--batch-size", "2", "--eval-trials", "4"]
    result = subprocess.run(arguments, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["completed"] and summary["last_evaluation"]["iteration"] == 1
    assert summary["counters"]["training_steps"] == 12
    saved = torch.load(output / "latest.pt", weights_only=False)
    assert saved["boundary"] == "between_trials" and saved["iteration"] == 1
    assert saved["optimizer"]["state"] and saved["environment_rng"]
    arguments[arguments.index("--updates") + 1] = "2"
    for name, extra in (("resumed", ["--resume", str(output / "latest.pt")]), ("direct", [])):
        arguments[arguments.index("--output-dir") + 1] = str(tmp_path / name)
        run = subprocess.run(arguments + extra, capture_output=True, text=True, timeout=60)
        assert run.returncode == 0, run.stdout + run.stderr
    resumed = torch.load(tmp_path / "resumed/latest.pt", weights_only=False)
    direct = torch.load(tmp_path / "direct/latest.pt", weights_only=False)
    for key in resumed["policy"]:
        torch.testing.assert_close(resumed["policy"][key], direct["policy"][key], rtol=0, atol=0)
