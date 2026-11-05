"""Read-only FSDP gradient-layout evidence; persist before enforcing gates."""
from pathlib import Path

import torch
from fluffy_bassoon.meta_rl.rlinf_native_ppo import write


def audit(actor,*,is_last,ordinal):
    optimizer_ids={id(p) for group in actor.optimizer.param_groups for p in group['params']}
    owners={}
    for module in actor.model.modules():
        handle=getattr(module,'_handle',None)
        if handle is None:continue
        flat=handle.flat_param
        for index,param in enumerate(getattr(flat,'_params',None) or []):
            owners[id(param)]=(flat,index)
    rows=[]
    for name,param in actor.model.named_parameters():
        if param.requires_grad:continue
        flat,index=owners.get(id(param),(None,None));grad=param.grad
        original=getattr(param,'_fqns',None)
        if original is None and flat is not None:original=[flat._fqns[index]]
        row=dict(name=name,original_names=original,requires_grad=param.requires_grad,
            parameter_shape=list(param.shape),parameter_numel=param.numel(),
            parameter_dtype=str(param.dtype),parameter_device=str(param.device),optimizer_member=id(param) in optimizer_ids,
            gradient_is_none=grad is None)
        if grad is not None:
            finite=bool(torch.isfinite(grad).all())
            ptr=grad.untyped_storage().data_ptr()
            candidates={} if flat is None else {
                'flat_grad':flat.grad,'saved_grad_shard':getattr(flat,'_saved_grad_shard',None)}
            row.update(gradient_shape=list(grad.shape),gradient_numel=grad.numel(),
                gradient_dtype=str(grad.dtype),gradient_device=str(grad.device),finite=finite,
                nonzero_elements=int(torch.count_nonzero(grad)),
                maximum_absolute=(float(grad.abs().max()) if finite and grad.numel() else 0.0 if finite else None),
                gradient_storage_ptr=ptr,gradient_storage_offset=grad.storage_offset(),
                gradient_has_base=getattr(grad,'_base',None) is not None,
                parameter_storage_ptr=param.untyped_storage().data_ptr(),
                flat_parameter_requires_grad=None if flat is None else flat.requires_grad,
                shares_flat_gradient_storage={k:v is not None and v.untyped_storage().data_ptr()==ptr
                    for k,v in candidates.items()})
        rows.append(row)
    report=dict(rank=actor._rank,optimizer_steps_before=actor.optimizer_steps,
        accumulation_microbatch=ordinal,synchronized_boundary=is_last,
        use_orig_params=bool(actor.cfg.actor.fsdp_config.use_orig_params),frozen_parameters=rows,
        frozen_optimizer_members=sum(r['optimizer_member'] for r in rows),
        frozen_non_none_views=sum(not r['gradient_is_none'] for r in rows),
        frozen_nonzero_elements=sum(r.get('nonzero_elements',0) for r in rows),
        all_finite=all(r.get('finite',True) for r in rows))
    root=Path(actor.cfg.native_pilot.output)
    write(root/'gradient-audit'/f'rank-{actor._rank:02d}-micro-{ordinal:02d}.json',report)
    assert report['frozen_optimizer_members']==0,'Frozen parameter included in native optimizer'
    assert report['all_finite'] and report['frozen_nonzero_elements']==0,'Nonzero/nonfinite frozen gradient'
    if is_last:
        assert report['frozen_non_none_views']==0,'Frozen gradient persists at synchronized SGD boundary'
    return report
