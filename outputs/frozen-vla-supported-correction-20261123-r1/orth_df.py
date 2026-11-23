"""Second-axis control with exactly the same effective df as the first axis."""
import json
import os
from pathlib import Path
import sys

import numpy as np

import experiment as ex
from adapter import issue, line_inverse, spectral_fit, spectrum
from compare import ci, digest, SOURCE_SHA

root, split, out = map(Path, sys.argv[1:])
old = root/'frozen-vla-motion-models-20261123-r1/cpu01/source-fit.json'
assert digest(old) == SOURCE_SHA
source = json.loads(old.read_bytes())
prior = np.array(source['prior'])
results = {}
for arm, rows in ex.target_traces(root,json.loads(split.read_bytes())).items():
    inputs, response = ex.query_panel(rows)
    x = np.stack([np.load(r['support']['path'])['issued_mean'][:3] for r in rows])
    leading, orthogonal, change = [np.full((8,8), np.nan) for _ in range(3)]
    ds = []
    for i,r in enumerate(rows):
        spec = spectrum(prior, r)
        _, s, vt, _ = spec
        fit, df = spectral_fit(prior, spec, 1., 0)
        assert s[1] > 0 and 0 < df < 1
        coef, actual = spectral_fit(prior, spec, s[1]**2 * (1-df) / df, 1)
        np.testing.assert_allclose(actual, df, atol=1e-13, rtol=1e-13)
        v = vt[1]
        mat = line_inverse(prior, coef, source['residual_power'], v)
        keep = np.arange(8)!=i
        leading[i,keep]=ex.errors(fit,inputs[keep],response[keep])
        orthogonal[i,keep]=ex.errors(coef,inputs[keep],response[keep])
        changed = issue(x[keep],mat,v)
        change[i,keep]=np.linalg.norm(changed[...,:3]-x[keep,:,:3],axis=-1).mean(-1)
        ds.append(dict(task=r['task'],df=df,orthogonal_df=actual))
    results[arm]=dict(rank1_error_mm=ci(leading),orthogonal_equal_df_error_mm=ci(orthogonal),
                      leading_minus_orthogonal_mm=ci(leading-orthogonal),orthogonal_command_change=ci(change),
                      degrees_of_freedom=ds)
ex.dump(out,dict(results=results,affinity=sorted(os.sched_getaffinity(0))))
