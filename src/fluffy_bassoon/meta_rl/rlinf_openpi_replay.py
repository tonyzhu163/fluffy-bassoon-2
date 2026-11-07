"""Actor-partition likelihood references for native continuous OpenPi actions.

RLinf c605's reference loop is restricted to OpenVLA: its diagnostic reshape
assumes flattened action tokens. Keep the same two-level SGD partition and
forward arguments, but preserve OpenPi's [sample, chunk, action] coordinates.
This changes the reference/gap measurement, never the learner loss or masks.
"""
import torch


def recompute_openpi_logprobs(actor, batch_size_per_rank, *, require_behavior_match=False):
    from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
    old = actor.rollout_batch['prev_logprobs']
    micro = int(actor.cfg.actor.micro_batch_size)
    horizon, dimension = int(actor.cfg.actor.model.num_action_chunks), int(actor.cfg.actor.model.action_dim)
    if (batch_size_per_rank < 1 or micro < 1
            or old.ndim != 3 or tuple(old.shape[1:]) != (horizon,dimension)
            or len(old) == 0 or len(old) % batch_size_per_rank
            or batch_size_per_rank % micro or not torch.isfinite(old).all()):
        raise ValueError('Require finite continuous OpenPi likelihoods and whole SGD partitions')
    pieces = []
    with torch.no_grad():
        for global_batch in split_dict_to_chunk(actor.rollout_batch,len(old)//batch_size_per_rank):
            for batch in split_dict_to_chunk(global_batch,batch_size_per_rank//micro):
                batch = put_tensor_device(batch,actor.device)
                with actor.amp_context:
                    result = actor.model(forward_inputs=batch['forward_inputs'],compute_logprobs=True,
                        compute_entropy=False,compute_values=False,use_cache=False)
                values = result['logprobs'].detach().cpu()
                if values.shape != batch['prev_logprobs'].shape or not torch.isfinite(values).all():
                    raise ValueError('OpenPi replay likelihood shape or finiteness changed')
                pieces.append(values)
    replay = torch.cat(pieces,dim=0).to(old.dtype)
    # Diagnostic only: one mean absolute coordinate difference per decision.
    # Preserve the learner's full [N,H,A] tensor and its existing masks.
    coordinate_error = (replay-old.cpu()).float().abs()
    gap = coordinate_error.mean(dim=(-2,-1))
    valid = torch.ones(len(old), dtype=torch.bool)
    mask = actor.rollout_batch.get('loss_mask')
    if mask is not None:
        if mask.ndim != 2 or mask.shape[0] != len(old) or mask.shape[1] not in (1,horizon):
            raise ValueError('OpenPi reference needs native decision/action masks')
        valid = mask.cpu().bool().any(dim=-1)
        gap = gap[valid]
    mean_gap = float(gap.mean()) if gap.numel() else 0.
    maximum = float(coordinate_error[valid].max()) if bool(valid.any()) else 0.
    # Check the actual behavior reference before replacing it. A zero ratio
    # against a replacement reference does not establish on-policy consistency.
    joint = (replay.double()-old.cpu().double()).sum(dim=(-2,-1))[valid]
    ratio = joint.exp()
    if not torch.isfinite(ratio).all():
        raise ValueError('Original behavior-policy joint ratio is not finite')
    if require_behavior_match and maximum != 0.:
        raise ValueError('Original behavior-policy likelihood differs before SGD')
    actor.rollout_batch['prev_logprobs'] = replay
    if isinstance(actor.device,int) or torch.device(actor.device).type == 'cuda':
        from rlinf.utils.utils import clear_memory
        clear_memory()
    return {'actor/rollout_train_logprob_gap': mean_gap,
            'actor/original_behavior_logprob_max_abs': maximum,
            'actor/original_behavior_ratio_outside_clip_fraction':
                float(((ratio<.8)|(ratio>1.2)).float().mean()) if ratio.numel() else 0.}
