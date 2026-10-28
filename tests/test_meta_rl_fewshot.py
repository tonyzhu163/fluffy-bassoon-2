"""CPU checks of the CMM few-shot regression calibration task."""
import numpy as np
import pytest
import torch

from fluffy_bassoon.meta_rl import fewshot_regression as fs


def test_episode_layout_standardization_and_support_only_noise():
    inputs, targets, query = fs.regression_batch(np.random.default_rng(3), 5, 10, .1)
    assert inputs.shape == (30, 5, 14) and targets.shape == (30, 5) and query.shape == (30,)
    assert query[10:].all() and not query[:10].any()
    torch.testing.assert_close(targets.mean(0), torch.zeros(5), atol=1e-5, rtol=0)
    torch.testing.assert_close(targets.std(0), torch.ones(5), atol=1e-5, rtol=0)
    assert inputs[:10, :, 13].eq(1).all() and inputs[10:, :, 12:].eq(0).all()
    noise = inputs[:10, :, 12] - targets[:10]
    assert noise.abs().max() > 0 and noise.std() < .2
    assert inputs[..., :12].abs().max() <= 1
    clean, _, _ = fs.regression_batch(np.random.default_rng(3), 5, 10, 0.)
    torch.testing.assert_close(clean[:10, :, 12], targets[:10])


def test_targets_are_linear_in_the_episode_inputs():
    inputs, targets, _ = fs.regression_batch(np.random.default_rng(4), 2, 20, 0.)
    for lane in range(2):
        design = torch.cat((inputs[:, lane, :12], torch.ones(40, 1)), 1).double()
        solution = torch.linalg.lstsq(design, targets[:, lane:lane + 1].double()).solution
        torch.testing.assert_close(design @ solution, targets[:, lane:lane + 1].double(), atol=1e-5, rtol=0)


def test_query_accuracy_threshold_and_scoring_only_queries():
    targets = torch.zeros(4, 2)
    predictions = torch.tensor([[9., 9.], [9., 9.], [.04, .06], [-.01, 1.]])
    query = torch.tensor([False, False, True, True])
    metrics = fs.query_metrics(predictions, targets, query)
    assert metrics["queries"] == 4 and metrics["query_accuracy"] == pytest.approx(.5)


@pytest.mark.parametrize("variant", sorted(fs.VARIANTS))
def test_parameter_matched_width_is_the_largest_within_the_paper_count(variant):
    width, count = fs.select_width(variant)
    assert width % 4 == 0 and count <= fs.VARIANTS[variant]["target"]
    assert fs.parameters(fs.build(variant, width + 4)) > fs.VARIANTS[variant]["target"]


def test_one_training_step_is_finite_and_reaches_memory():
    torch.manual_seed(0)
    model = fs.build("cmm", 16)
    inputs, targets, _ = fs.regression_batch(np.random.default_rng(5), 3, 10, .1)
    predictions = model(inputs)
    assert predictions.shape == targets.shape
    (predictions - targets).square().mean().backward()
    assert model.core.initial_ltm.grad.abs().sum() > 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
def test_partial_last_batch_preserves_exact_paper_epoch_budget():
    from fluffy_bassoon.meta_rl.fewshot_regression import training_batch_size
    for epoch in range(3):
        batches = [training_batch_size(epoch*157+i, 64) for i in range(157)]
        assert sum(batches) == 10_000
        assert batches[:-1] == [64]*156
        assert batches[-1] == 16
