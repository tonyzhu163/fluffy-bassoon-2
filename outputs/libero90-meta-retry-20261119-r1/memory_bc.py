"""Offline success-conditioned cadence imitation for the LIBERO-90 retry Reader.

In learner.py, initialize a separate Reader with the outcome scorer's seed and
fitting statistics. Reuse fit_model's optimizer, minibatch IDs and update count.
For the memory imitation arm, replace its objective with

    objective = memory_bc_batch_loss(model, batch, actions[ids], labels[ids],
                                     forward=forward)

Pass the same actions and labels supplied by designs and fit_model. Both are
shaped [queries, revealed cadences], with two columns for paired trials and one
for single-logged trials. Each label is the native retry success for its cadence
ID. Pass the entire Public batch, including unsuccessful practice episodes.
The runner's forward handles both the practice tape and its terminal reward.

Save the Reader state_dict under a distinct checkpoint name in the frozen
manifest and include this module in code_identity. For scoring, call
memory_bc_logits(model.eval(), batch, forward=forward) under torch.inference_mode().
The keys 'own' (retained practice) and 'cleared' match learner.predict. For
each view, average raw logits across seeds, take argmax in ARMS order, and
score that branch's native outcome on every trial.
"""

import torch


def memory_bc_loss(
    logits: torch.Tensor, actions: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Mean negative log likelihood per sampled query, with equal weight per success within it."""
    if (logits.ndim != 2 or actions.ndim != 2 or actions.shape != labels.shape
            or actions.shape[0] != logits.shape[0] or logits.shape[0] == 0
            or not 1 <= actions.shape[1] <= logits.shape[1]):
        raise ValueError("Expected menu logits [B, A] and aligned revealed cadence IDs and labels [B, K]")
    if not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Each revealed cadence needs a binary native success label")

    # Normalize over the whole menu even when only one cadence was logged.
    log_prob = logits.log_softmax(dim=1).gather(1, actions)
    successes = labels == 1
    log_likelihood = log_prob.masked_fill(~successes, 0).sum(dim=1)
    # Failed queries contribute zero and stay in the batch denominator and graph.
    return -(log_likelihood / successes.sum(dim=1).clamp_min(1)).mean()


def memory_bc_logits(model, batch, *, forward):
    """Return per-cadence logits for both history views of the same Reader and query."""
    return {
        "own": forward(model, batch, retain=True),
        "cleared": forward(model, batch, retain=False),
    }


def memory_bc_batch_loss(model, batch, actions, labels, *, forward):
    """Exact expectation under the runner's 50% complete-history dropout."""
    views = memory_bc_logits(model, batch, forward=forward)
    retained = memory_bc_loss(views["own"], actions, labels)
    cleared = memory_bc_loss(views["cleared"], actions, labels)
    return 0.5 * (retained + cleared)
