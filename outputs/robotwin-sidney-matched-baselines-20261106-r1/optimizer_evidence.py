"""Describe native optimizer storage without mutating or normalizing its state."""
import torch
from recovery_signature import digest


def describe(value):
    if isinstance(value,torch.Tensor):
        result=dict(kind='tensor',shape=list(value.shape),numel=value.numel(),
            dtype=str(value.dtype),device=str(value.device),sha256=digest(value))
        finite=bool(torch.isfinite(value).all())
        result.update(finite=finite,nonzero_elements=int(torch.count_nonzero(value)),
            maximum_absolute=(float(value.abs().max()) if finite and value.numel() else 0.0 if finite else None))
        if value.numel()<=8:result['values']=value.detach().cpu().reshape(-1).tolist()
        return result
    if isinstance(value,dict):return {str(k):describe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return dict(kind=type(value).__name__,items=[describe(v) for v in value])
    return dict(kind=type(value).__name__,value=value)


def evidence(actor):
    names={id(p):name for name,p in actor.model.named_parameters()}
    states={};groups=[]
    for group in actor.optimizer.param_groups:
        ordered=[]
        for index,p in enumerate(group['params']):
            name=names.get(id(p),f'<unregistered-parameter-{len(groups)}-{index}>');ordered.append(name)
            states[name]=dict(parameter_shape=list(p.shape),parameter_numel=p.numel(),
                parameter_dtype=str(p.dtype),requires_grad=p.requires_grad,
                registered_model_parameter=id(p) in names,
                local_nonempty_parameter=p.numel()>0,state_present=p in actor.optimizer.state,
                state=describe(actor.optimizer.state.get(p,{})))
        groups.append(dict(parameters=ordered,hyperparameters=describe({k:v for k,v in group.items() if k!='params'})))
    return dict(optimizer_sha256=digest(actor.optimizer.state_dict()),rank=actor._rank,
        optimizer_steps=actor.optimizer_steps,is_optimizer_offloaded=actor.is_optimizer_offloaded,
        state_entries=len(actor.optimizer.state),groups=groups,parameters=states)
