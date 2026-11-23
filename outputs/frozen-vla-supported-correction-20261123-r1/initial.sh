#!/usr/bin/env bash
set -euo pipefail
root=/mowenjie/fluffy-bassoon-2
sibling="$root/outputs/frozen-vla-motion-models-20261123-r1"
export PYTHONPATH="$sibling/source" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=''
taskset -c 0 /mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python -B - "$root" "$sibling" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
from experiment import target_traces, query_panel, errors
from motion import fit_target
root, sibling = map(Path, sys.argv[1:])
splits = json.loads((root/'setups/xemb-icml-campaign-20261121-r1/source-ff2ddf0a/outputs/libero-meta-learning-20261117-r1/SPLITS.json').read_text())
frozen = json.loads((sibling/'cpu01/source-fit.json').read_text())
prior = np.array(frozen['prior'])
lam = frozen['choice']['jacobian']
for arm, rows in target_traces(root/'outputs', splits).items():
    x, y = query_panel(rows)
    scores = []
    for i, row in enumerate(rows):
        design = row['inputs'][:, :3]
        _, s, vt = np.linalg.svd(design, full_matrices=False)
        v = vt[0]
        z = design @ v
        rank1 = prior.copy()
        rank1[:3] += np.outer(v, z @ (row['response'] - row['inputs'] @ prior)) / (s[0]**2 + lam)
        full = fit_target('jacobian', prior, row, lam)
        scores.append([errors(f, x[np.arange(8)!=i], y[np.arange(8)!=i]).mean() for f in (rank1, full, prior)])
    print(arm, 'rank1 / full ridge / source mm =', np.round(np.mean(scores, axis=0), 6), 'source lambda =', lam)
PY
