"""Native flow-matching updates for frozen-expert history steering.

The logical batch contains one causally selected query per complete history
window. Execution batches change memory use only. DIT/IQL use independently
fitted, frozen Q/V estimates; action labels never enter the current query token.
"""
import math

import torch

from .flow_history import HistoryFlowSteeringPolicy
from .vla_history_train import _restore_rng, _rng_state
from .warmstart import atomic_save


RESUME_FORMAT = "native_flow_history_bridge_resume_v1"


def query_context(policy, batch):
    indices = batch["query_rel"]
    width = batch["tokens"].shape[1]
    if (indices.shape != (width,) or indices.dtype != torch.long
            or (indices < 0).any() or (indices >= len(batch["tokens"])).any()
            or not batch["valid"][indices, torch.arange(width, device=indices.device)].all()):
        raise ValueError("Require one valid causal query per history window")
    end = int(indices.max()) + 1
    contexts = policy.encode_sequence(batch["tokens"][:end], batch["valid"][:end],
                                      batch["starts"][:end], mode="full")
    return contexts[indices, torch.arange(width, device=indices.device)]


@torch.no_grad()
def query_weights(batch, objective, values=None):
    width = len(batch["query_rel"])
    if objective in ("ad", "dpt"):
        if values is not None:
            raise ValueError("AD/DPT do not use value reweighting")
        return HistoryFlowSteeringPolicy.meta_objective_weights(
            objective, batch_size=width, device=batch["tokens"].device)
    if values is None or values.config.objective != objective:
        raise ValueError("Require the matching independently fitted frozen value objective")
    if any(p.requires_grad for model in (values.q, values.v) for p in model.parameters()):
        raise ValueError("Value estimators must stay frozen during policy extraction")
    lanes = torch.arange(width, device=batch["query_rel"].device)
    q = values.q.sequence(batch)[batch["query_rel"], lanes]
    v = values.v.sequence(batch)[batch["query_rel"], lanes]
    return HistoryFlowSteeringPolicy.meta_objective_weights(
        objective, q_values=q, v_values=v, value_config=values.config)


def slice_batch(batch, begin, end):
    sequence = ("tokens", "actions", "rewards", "dones", "starts", "returns", "valid")
    result = {k: batch[k][:, begin:end] for k in sequence}
    for key in ("query_rel", "target_actions", "noise", "timesteps", "weights"):
        result[key] = batch[key][begin:end]
    result["query_indices"] = batch["query_indices"][begin:end]
    return result


def flow_batch_loss(policy, model, batch, prepare_queries):
    """prepare_queries binds actual saved images to this batch's selected queries."""
    context = query_context(policy, batch)
    observation, cache, mask = prepare_queries(batch["query_indices"])
    return policy.flow_matching_loss(model, observation, batch["target_actions"],
        batch["noise"], batch["timesteps"], cache, mask, context,
        sample_weights=batch["weights"])


def bridge_update(policy, model, optimizer, batch, prepare_queries, *, execution_batch_size):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Freeze the native action expert and VLM before bridge fitting")
    native_ids = {id(p) for p in model.parameters()}
    policy_ids = {id(p) for p in policy.parameters() if p.requires_grad}
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimized != policy_ids or optimized & native_ids:
        raise ValueError("Optimize exactly the history model and steering adapter")
    width = len(batch["query_rel"])
    if type(execution_batch_size) is not int or execution_batch_size < 1:
        raise ValueError("Require a positive execution batch")
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.
    for begin in range(0, width, execution_batch_size):
        end = min(width, begin + execution_batch_size)
        loss = flow_batch_loss(policy, model, slice_batch(batch, begin, end), prepare_queries)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite frozen-expert flow-matching objective")
        factor = (end - begin) / width
        (loss * factor).backward()
        total += float(loss.detach()) * factor
    # Analytically redundant key bias: avoid Adam amplifying cancellation noise.
    for block in policy.core.layers:
        bias = block.key_value.bias
        if bias.grad is not None:
            bias.grad[:bias.numel() // 2].zero_()
    norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True))
    optimizer.step()
    if any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Native expert acquired gradients during bridge fitting")
    if any(not torch.isfinite(p).all() for p in policy.parameters()):
        raise FloatingPointError("Nonfinite history/steering parameter")
    return dict(loss=total, gradient_norm=norm, logical_batch_size=width,
                execution_batch_size=execution_batch_size,
                weight_mean=float(batch["weights"].mean()), weight_max=float(batch["weights"].max()))


def save_resume(path, *, contract, policy, optimizer, sampler, generator, progress):
    return atomic_save(path, dict(format=RESUME_FORMAT, contract=contract,
        policy=policy.state_dict(), optimizer=optimizer.state_dict(),
        rng=_rng_state(sampler), flow_generator=generator.get_state(), progress=progress))


def restore_resume(path, *, contract, policy, optimizer, sampler, generator):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != RESUME_FORMAT or saved.get("contract") != contract:
        raise ValueError("Resume must preserve the exact data, frozen motor and learner contract")
    policy.load_state_dict(saved["policy"], strict=True)
    optimizer.load_state_dict(saved["optimizer"])
    _restore_rng(saved["rng"], sampler)
    generator.set_state(saved["flow_generator"])
    return saved["progress"]


@torch.no_grad()
def validate_batches(policy, model, batches, prepare_queries):
    policy.eval()
    losses = [float(flow_batch_loss(policy, model, batch, prepare_queries)) for batch in batches]
    if not losses or not all(math.isfinite(x) for x in losses):
        raise FloatingPointError("Require finite held-out native flow losses")
    return sum(losses) / len(losses)
