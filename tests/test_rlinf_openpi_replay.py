"""Continuous likelihood partition/mask numerics on allocated Linux CPU."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from fluffy_bassoon.meta_rl.rlinf_openpi_replay import recompute_openpi_logprobs


class Model:
    def __init__(self, failure=None):
        self.calls = []
        self.failure = failure

    def __call__(self, *, forward_inputs, **kwargs):
        assert kwargs == dict(compute_logprobs=True,compute_entropy=False,compute_values=False,use_cache=False)
        rows = forward_inputs['row_ids']
        self.calls.append(rows.tolist())
        # Deliberate batch-shape dependence models native BF16 partition drift.
        result = rows.float()[:,None,None].expand(-1,3,2)+len(rows)
        if self.failure == 'shape': result = result[:,0]
        if self.failure == 'nan': result = result * float('nan')
        return dict(logprobs=result)


def actor(mask, failure=None):
    order = torch.tensor([8,1,5,0,10,4,11,2,7,3,9,6])
    batch = dict(prev_logprobs=torch.zeros(12,3,2),loss_mask=mask,
                 advantages=torch.arange(12.),forward_inputs=dict(row_ids=order))
    return SimpleNamespace(rollout_batch=batch,device=torch.device('cpu'),amp_context=nullcontext(),
        model=Model(failure),cfg=SimpleNamespace(actor=SimpleNamespace(micro_batch_size=2,
        model=SimpleNamespace(num_action_chunks=3,action_dim=2))))


@pytest.mark.parametrize('columns', [1,3])
def test_reference_uses_exact_shuffled_sgd_microbatches_and_preserves_native_masks(columns):
    mask = torch.ones(12,columns,dtype=torch.bool);mask[0] = False
    worker = actor(mask)
    original = {key:worker.rollout_batch[key] for key in ('loss_mask','advantages','forward_inputs')}
    rng = torch.get_rng_state().clone()
    result = recompute_openpi_logprobs(worker,4)
    expected = worker.rollout_batch['forward_inputs']['row_ids'].float()+2
    assert worker.model.calls == [[8,1],[5,0],[10,4],[11,2],[7,3],[9,6]]
    assert torch.equal(worker.rollout_batch['prev_logprobs'],expected[:,None,None].expand(-1,3,2))
    assert result['actor/rollout_train_logprob_gap'] == float(expected[1:].mean())
    assert all(worker.rollout_batch[key] is value for key,value in original.items())
    assert torch.equal(torch.get_rng_state(),rng)


def test_all_masked_reference_has_finite_zero_gap_and_still_replays_all_rows():
    worker = actor(torch.zeros(12,3,dtype=torch.bool))
    assert recompute_openpi_logprobs(worker,4)['actor/rollout_train_logprob_gap'] == 0
    assert len(worker.model.calls) == 6


@pytest.mark.parametrize('failure', ['shape','nan'])
def test_bad_reference_never_replaces_the_previous_likelihood(failure):
    worker = actor(torch.ones(12,3,dtype=torch.bool),failure)
    original = worker.rollout_batch['prev_logprobs']
    with pytest.raises(ValueError,match='shape or finiteness'):
        recompute_openpi_logprobs(worker,4)
    assert worker.rollout_batch['prev_logprobs'] is original


@pytest.mark.parametrize('bad', ['partial_batch','mask'])
def test_reference_rejects_partial_optimizer_batches_and_incompatible_masks(bad):
    worker = actor(torch.ones(12,4 if bad == 'mask' else 3,dtype=torch.bool))
    original = worker.rollout_batch['prev_logprobs']
    with pytest.raises(ValueError):
        recompute_openpi_logprobs(worker,5 if bad == 'partial_batch' else 4)
    assert worker.rollout_batch['prev_logprobs'] is original
