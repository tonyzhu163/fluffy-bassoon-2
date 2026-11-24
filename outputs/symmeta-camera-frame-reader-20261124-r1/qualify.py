"""Run-local source fixture. Never scores the target."""
from pathlib import Path
import sys
import cv2
import jax
import numpy as np
from camera import digital, encoding, frames_of, homography, render, view_inputs
from data import ROOT, SENSOR, write, sha
from model import CFG, LABELS, decode, forward, init, loss, tokens
from fluffy_bassoon.xemb.equiv_v2 import apply_reader_v2, init_reader_v2, reader_loss_v2, _vis_inputs_v2

cv2.setNumThreads(1)
assert jax.default_backend()=='cpu'
path=ROOT/'xemb-icml-campaign-20261121-r1/gpu12scale/stage3/diverse_shards/shard-0/diverse.npz'
with np.load(path) as z:
    original={k:z[k][:2,:3].copy() for k in ('img_before','img_after','dis_flow',*SENSOR)}
    labels={k:z[k][:2] for k in LABELS}

def batch(arm,pose):
    rows=[digital(original,i,pose,arm) for i in range(2)]
    return {**{k:np.stack([x[k] for x in rows]) for k in rows[0]},**{k:original[k] for k in SENSOR}}

nom=batch('A',(0,0))
raw=batch('B',(0,0))
for k in raw:
    np.testing.assert_array_equal(nom[k],raw[k])
for k in original:
    np.testing.assert_array_equal(original[k],nom[k])
for angle in (-90,90):
    inv=batch('A',(angle,0))
    for k in original:
        np.testing.assert_array_equal(original[k],inv[k])
    np.testing.assert_array_equal(_vis_inputs_v2(inv,CFG.visual_mode)[0],_vis_inputs_v2(nom,CFG.visual_mode)[0])
p=init(41)
base=init_reader_v2(jax.random.key(41),CFG)
blind=dict(nom,calibration=np.zeros_like(nom['calibration']))
np.testing.assert_allclose(forward(p,blind),apply_reader_v2(base,CFG,original),rtol=2e-5,atol=2e-5)
np.testing.assert_allclose(float(loss(p,blind,labels)),float(reader_loss_v2(base,CFG,{**original,**labels})[0]),rtol=2e-5)
t=np.array(tokens(p,nom))
assert t.shape==(2,3,CFG.cnn_width+12+9)
np.testing.assert_equal(t[:,:,-9:],np.broadcast_to(nom['calibration'][:,None],t[:,:,-9:].shape))
r=batch('B',(35,0))
fake=dict(r,calibration=np.repeat(encoding(homography(-35,0))[None],2,axis=0))
delta=np.max(np.abs(np.asarray(forward(p,r))-np.asarray(forward(p,fake))))
assert delta>0
f=frames_of(original,0)
obs,mask,h=render(f,35,18)
altered=obs.copy()
altered[:,24:38,24:38]=255-altered[:,24:38,24:38]
a=view_inputs(obs,mask,h,'A')
other=view_inputs(altered,mask,h,'A')
assert not np.array_equal(a['img_before'],other['img_before']) and not np.array_equal(a['dis_flow'],other['dis_flow'])
assert not np.array_equal(a['img_before'],f[:-1])
write(Path(sys.argv[1])/'qualification.json',dict(source=sha(path),nominal_rgb_dis_exact=True,
      both_quarter_inverse_rgb_dis_features_exact=True,reference_output_loss_agree=True,
      token_shape=t.shape,source_parameter_count=sum(x.size for x in jax.tree.leaves(p)),
      metadata_only_output_change=float(delta),lossy_input_from_observed_only=True,
      support_mask_mean=float(a['pixel_valid'].mean()),label_keys=LABELS))
