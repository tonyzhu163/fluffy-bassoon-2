# Subset optimizer-scale wall-time correction

Prepared revision only; no job was launched by this change.

Baige `job-nhd0iek20qwx`, immutable source `80c48bb`, passed70 Linux tests.
The width128,2-layer,batch8,window128 AD/full controller measured24.26s at
update25 and90.77s at update100, approximately0.9 seconds/update. A500-update
trajectory therefore projects to roughly450 seconds, before uncertainty.
The original300-second per-controller guard cannot reach the intended endpoint.

The follow-up launcher uses600 cumulative training seconds per controller and
2700 seconds overall, with an absolute module ceiling of3000 seconds. This
retains the same architecture, dataset, optimizer settings and25/100/500-update
targets. It changes only the runtime allowance. A fresh unique source-pinned
run must carry the revised contract; the active run, its partial recoveries and
immutable milestones remain unchanged.

The parent will let the original attempt reach its existing guard, preserve
its evidence, and decide the next allocated launch. A fresh rerun must be
reported as a rerun rather than a continuation of its optimizer trajectory.
Early termination under an undersized runtime limit is not a negative method
result or evidence against adaptation.
