"""Exactly the PR7 factor Perceiver with nine additional public calibration coefficients."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
import eval_wider_family as pr7
from fluffy_bassoon.xemb.equiv_v2 import (
    _apply_perceiver, _apply_res_cnn, _normalize_cs, _vis_inputs_v2, init_reader_v2,
)

CFG = pr7.PERCEIVER
OBS = ('img_before','img_after','dis_flow','pixel_valid','calibration','grid_h',
       'issued_mean','delta_eef','delta_jaw','valid')
LABELS = pr7.LABEL_KEYS


def init(seed):
    key = jax.random.key(seed)
    p = init_reader_v2(key, CFG)
    kernel = p['cnn']['c1_w']
    p['cnn']['c1_w'] = jnp.concatenate([kernel, jnp.zeros_like(kernel[:, :, :3])], axis=2)
    old = p['perceiver']['in_w']
    extra = jax.random.normal(jax.random.fold_in(key, 941), (9, old.shape[1])) / np.sqrt(old.shape[0])
    p['perceiver']['in_w'] = jnp.concatenate((old, extra), axis=0)
    return p


def tokens(p, b):
    x, n, k = _vis_inputs_v2(b, CFG.visual_mode)
    mask = jnp.asarray(b['pixel_valid'], bool).reshape(x.shape[:-1] + (1,))
    x = jnp.concatenate([jnp.where(mask, x[..., :7], 0.), x[..., 7:]], axis=-1)
    xy = x[..., 7:9].reshape(n,k,*x.shape[1:3],2)[:,0]
    grid = jnp.concatenate([xy, jnp.ones_like(xy[..., :1])], axis=-1)
    other = jnp.einsum('bij,bhwj->bhwi',b['grid_h'].reshape(n,3,3),grid)
    xy_other = other[..., :2] / other[..., 2:3]
    local = jnp.broadcast_to(xy_other[:,None],(n,k,*xy_other.shape[1:])).reshape(x.shape[:-1]+(2,))
    x = jnp.concatenate([x, jnp.where(mask, local, 0.), mask.astype(x.dtype)],axis=-1)
    v = _apply_res_cnn(p['cnn'], x).reshape(n,k,-1)
    pose = jnp.broadcast_to(b['calibration'][:,None], (n, k, 9))
    return jnp.concatenate([v, b['issued_mean'], 100*b['delta_eef'], 100*b['delta_jaw'], pose], axis=-1)


def forward(p, b):
    valid = b['valid']
    raw = _apply_perceiver(p['perceiver'], tokens(p,b), valid, CFG.perceiver_heads)
    good = jnp.any(valid, axis=1)
    c, s = _normalize_cs(raw[:,0], raw[:,1], good)
    return c, s, jnp.where(good, raw[:,2], 0.), jnp.where(good, raw[:,3], 10.)


def loss(p, b, labels):
    c, s, g, grip = forward(p,b)
    yaw = jnp.deg2rad(labels['yaw_deg'])
    angle_loss = jnp.mean(1 - c*jnp.cos(yaw) - s*jnp.sin(yaw))
    gain_loss = jnp.mean(jnp.square(g - jnp.log(labels['xy_scale'])))
    grip_loss = jnp.mean(jax.nn.softplus(grip) - (labels['gripper_sign'] > 0)*grip)
    return angle_loss + gain_loss + grip_loss


def schedule(step):
    # Retain the optimizer moments across matched cycles. Restart only the scalar LR.
    return optax.warmup_cosine_decay_schedule(0., 3e-4, 200, 2500, end_value=.05*3e-4)(step % 2500)


OPT = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-4))


@jax.jit
def step(p, st, b, labels):
    result, grad = jax.value_and_grad(loss)(p,b,labels)
    changes, st = OPT.update(grad,st,p)
    return optax.apply_updates(p,changes), st, result


@jax.jit
def infer(p, b):
    return forward(p,b)


def decode(output):
    c,s,g,grip = map(np.asarray, output)
    if not all(np.isfinite(x).all() for x in (c,s,g,grip)):
        raise ValueError("Nonfinite factor")
    return np.rad2deg(np.arctan2(s,c)), np.exp(np.clip(g, np.log(.25),np.log(4.))), np.where(grip >= 0,1,-1)


def restore(path, seed):
    return serialization.from_bytes(init(seed), path.read_bytes())
