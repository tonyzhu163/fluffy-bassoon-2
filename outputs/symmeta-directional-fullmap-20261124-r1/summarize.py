"""Whole-task paired summaries of the frozen matched replay. No model selection."""
from collections import defaultdict
import itertools

import numpy as np

from common import SEEDS, TARGET_ARMS, TARGET_TASKS

METHODS = ("directional", "direct", "incumbent", "scalar", "trace", "reversed", "ridge", "true_map")
ENDPOINTS = ("command", "motion")
ARMS = TARGET_ARMS
TASKS = tuple(f"{s}/{t}" for s,t in TARGET_TASKS.items())


def average(values):
    return float(np.mean(values)) if len(values) else None


def paired_tasks(rows, endpoint, method=None):
    cells = defaultdict(list)
    for r in rows:
        a = r["scores"]["directional"][endpoint]
        b = r["scores"][method][endpoint] if method else None
        if a is not None:
            value = b-a if method else a
            cells[r["task"],r["arm"],r["family"],r["strategy"]].append(value)
    task = defaultdict(list)
    for (t,arm,family,strategy), vals in cells.items():
        task[t].append(average(vals))
    return {t:average(v) for t,v in sorted(task.items())}


def paired(rows, endpoint, method):
    data = paired_tasks(rows, endpoint, method)
    values = np.array(list(data.values()))
    if not len(values):
        return dict(mean=None, ci95=None, tasks=data)
    idx = np.array(list(itertools.product(range(len(values)), repeat=len(values))))
    return dict(mean=float(values.mean()), ci95=np.quantile(values[idx].mean(1), [.025,.975]).tolist(), tasks=data)


def table(rows):
    result = {}
    for metric in ENDPOINTS:
        cols = {}
        for method in METHODS:
            # Paired levels using the same nested task weighting as paired differences.
            cells = defaultdict(list)
            for r in rows:
                v = r["scores"][method][metric]
                if v is not None:
                    cells[r["task"],r["arm"],r["family"],r["strategy"]].append(v)
            ts = defaultdict(list)
            for (task,*_), values in cells.items():
                ts[task].append(average(values))
            cols[method] = dict(level=average([average(v) for v in ts.values()]))
            if method != "directional":
                cols[method]["improvement"] = paired(rows, metric, method)
        result[metric] = cols
    return result


def simultaneous(primary):
    names = [(m,b) for m in ENDPOINTS for b in ("direct","incumbent","scalar")]
    pairs = [paired_tasks(primary,m,b) for m,b in names]
    if any(set(p)!=set(TASKS) for p in pairs):
        return dict(bands=None, reason="Missing whole task for at least one endpoint.")
    x = np.array([[p[t] for t in TASKS] for p in pairs])
    center = x.mean(1)
    se = x.std(1,ddof=1)/np.sqrt(len(TASKS))
    resamples = np.array(list(itertools.product(range(4), repeat=4)))
    draws = x[:,resamples].mean(-1)
    variable = np.ptp(x,axis=1)>0
    critical = (float(np.quantile((np.abs(draws[variable]-center[variable,None])/se[variable,None]).max(0),.95))
                if variable.any() else 0.)
    return dict(max_deviation_factor=critical, draws=len(resamples),
                bands={f"{m}/{b}":[float(c-critical*s),float(c+critical*s)]
                       for (m,b),c,s in zip(names,center,se)})


def association(rows, metric, field):
    # Each task, arm, map draw, strategy has equal weight. Signs and seeds share a cell's weight.
    cells = defaultdict(list)
    for r in rows:
        a = r["scores"]["scalar"][metric]
        b = r["scores"]["directional"][metric]
        if a is not None:
            cells[r["task"],r["arm"],r["family"],r["strategy"]].append((r[field],a-b))
    if not cells:
        return None
    xy = np.array([v for cell in cells.values() for v in cell])
    w = np.array([1/(len(cells)*len(cell)) for cell in cells.values() for v in cell])
    centered = xy-np.average(xy,weights=w,axis=0)
    var = (centered**2*w[:,None]).sum(0)
    if not all(np.ptp(xy,axis=0)>0):
        return dict(slope=None, correlation=None, reason="constant predictor or contrast")
    cov = (centered.prod(1)*w).sum()
    return dict(slope=float(cov/var[0]), correlation=float(cov/np.sqrt(var.prod())))


def summarize(rows):
    general = [r for r in rows if r["family"].startswith("general")]
    primary = [r for r in general if r["strategy"] in ("passive","lateral")]
    panels = {"primary":primary}
    for name in ("passive","lateral","along"):
        panels[f"general/{name}"] = [r for r in general if r["strategy"]==name]
    for field, categories in (("arm",ARMS),("seed",SEEDS),("family",("general+","general-"))):
        for v in categories:
            panels[f"primary/{field}/{v}"] = [r for r in primary if r[field]==v]
    for mode in ("orth+","orth-","along+","along-"):
        panels[f"general/mode/{mode}"] = [r for r in general if r["mode"]==mode]
    for fam in ("native","family","swap","general+","general-"):
        for strategy in ("passive","along","lateral"):
            panels[f"family/{fam}/{strategy}"] = [r for r in rows if r["family"]==fam and r["strategy"]==strategy]
    spectra = {}
    for strat in ("passive","along","lateral"):
        sub = [r for r in general if r["strategy"]==strat and r["seed"]==41]
        spectra[strat] = {k: np.mean([r[k] for r in sub],axis=0).tolist()
                          for k in ("g","weights","pulse_energy","effort","rank_ratio","pulse2","pulse3")}
    cells = {(r["task"],r["arm"],r["family"]) for r in primary}
    observed = {(r["task"],r["arm"],r["family"]) for r in primary if r["held_chunks"]>0}
    return dict(tables={n:table(sub) for n,sub in panels.items()}, simultaneous=simultaneous(primary),
        primary_held_cells=len(cells), primary_observed_motion_cells=len(observed), spectra_general=spectra,
        continuous={metric:{field:association(primary,metric,field) for field in
                   ("log_min_information","weight_range","pulse_energy")} for metric in ENDPOINTS})
