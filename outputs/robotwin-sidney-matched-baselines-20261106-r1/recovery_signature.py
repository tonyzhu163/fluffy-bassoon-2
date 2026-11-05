"""Allocated-worker signatures for actual native checkpoint readback."""
import hashlib

import numpy as np
import torch


def digest(value):
    result=hashlib.sha256()
    def visit(item):
        if isinstance(item,dict):
            result.update(b'dict')
            for key in sorted(item,key=lambda x:(type(x).__name__,repr(x))):visit(key);visit(item[key])
        elif isinstance(item,(tuple,list)):
            result.update(type(item).__name__.encode())
            for child in item:visit(child)
        elif isinstance(item,torch.Tensor):
            array=item.detach().cpu().contiguous()
            result.update(str(array.dtype).encode());result.update(str(tuple(array.shape)).encode())
            result.update(array.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item,np.ndarray):
            result.update(str(item.dtype).encode());result.update(str(item.shape).encode());result.update(item.tobytes())
        else:result.update((type(item).__name__+':'+repr(item)).encode())
    visit(value)
    return result.hexdigest()


def actor_signature(actor,rng):
    return dict(optimizer_steps=actor.optimizer_steps,
        trainable_parameters_sha256=digest({n:p for n,p in actor.model.named_parameters() if p.requires_grad}),
        optimizer_sha256=digest(actor.optimizer.state_dict()),
        lr_scheduler_sha256=digest(actor.lr_scheduler.state_dict()),
        grad_scaler_sha256=digest(actor.grad_scaler.state_dict()),rng_sha256=digest(rng))
