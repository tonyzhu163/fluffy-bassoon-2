"""Fit only public source image/annotation files. Model files carry the matched timing controls."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from common import save


def load_public(path):
    with np.load(path) as data:
        assert set(data.files) == {"t","tiles"}
        t, brightness = data["t"], data["tiles"]
    assert np.array_equal(t, 5*np.arange(len(t))) and brightness.shape == (len(t),256)
    assert np.isfinite(brightness).all() and np.all((brightness >= 0) & (brightness <= 1))
    return t, brightness - brightness[0]


def fit(directory, annotations):
    ids = json.loads((directory/"split.json").read_text())["fit"]
    assert set(ids) == set(annotations)
    values, labels, weights, switches, ticks = [], [], [], [], []
    for i in ids:
        t, x = load_public(directory/f"public-{i}.npz")
        switch = annotations[i]
        assert switch in t
        values.extend(x)
        labels.extend(t >= switch)
        weights.extend(np.full(len(t), 1/(len(ids)*len(t))))
        switches.append(switch)
        ticks.extend(t)
    x,y,w = np.array(values),np.array(labels,dtype=float),np.array(weights)
    mean = w @ x
    scale = np.sqrt(w @ ((x-mean)**2))
    z = (x-mean) / np.maximum(scale,1e-8)
    rate = float(w @ y)
    correlation = np.abs((w*(y-rate)) @ z)
    selected = []
    for k in np.argsort(-correlation):
        # Avoid counting the same small image region in adjacent tiles.
        if all(max(abs(k//16 - j//16), abs(k%16 - j%16)) > 1 for j in selected):
            selected.append(int(k))
        if len(selected) == 4:
            break
    design = np.column_stack([np.ones(len(x)), z[:,selected]])
    def objective(beta):
        logit = design @ beta
        loss = w @ (np.logaddexp(0,logit) - y*logit) + 0.005*np.sum(beta[1:]**2)/2
        grad = design.T @ (w*(expit(logit)-y)) + 0.005*np.r_[0,beta[1:]]
        return loss, grad
    model = minimize(objective,np.zeros(5),jac=True,method="BFGS",tol=1e-7)
    assert model.success, model.message
    ticks = np.asarray(ticks)
    # Empirical pointwise Brier-optimal clock on the *same* fit observations and weights.
    clock = [np.average(y[ticks == k],weights=w[ticks == k]) for k in range(0,max(ticks)+1,5)]
    return dict(ids=ids, annotation=annotations, signed_luminance_tile_ids=selected,
                mean=mean[selected].tolist(),scale=scale[selected].tolist(),beta=model.x.tolist(),
                constant=rate, clock=clock, source_switch_times=switches,
                regularization=0.005, clock_extrapolation="carry last fitted probability")


def predict(t, x, model):
    z = (x[:,model["signed_luminance_tile_ids"]] - model["mean"]) / model["scale"]
    instant = expit(model["beta"][0] + z @ model["beta"][1:])
    visual = np.maximum.accumulate(instant)
    clock = np.asarray(model["clock"])[np.minimum(t//5, len(model["clock"])-1)]
    return dict(visual=visual, instant=instant, clock=clock, constant=np.full(len(t),model["constant"]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source",type=Path)
    p.add_argument("annotations",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    labels = {int(k):v for k,v in json.loads(a.annotations.read_text()).items()}
    save(a.output,fit(a.source,labels))
