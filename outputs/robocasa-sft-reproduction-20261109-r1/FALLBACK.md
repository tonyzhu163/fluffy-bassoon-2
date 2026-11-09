# Conditional 1.0.0 comparison

Prepared while campaign01 runs; not yet scheduled. Launch only if its complete
900-case aggregate falls outside the predeclared practical screen.

Use the actual v1.0 tag 8f3c96ec8d1bfcd8126cad2bca887da98d30e997 in the fresh
sources/robocasa-v100 checkout. Native Linux import passed with version1.0.0,
OpenDrawer500 and CloseToasterOvenDoor300. Existing shared source/env/cache
and campaign01 remain unchanged. The separate V100-MANIFEST.json preserves
all18 tasks,50 cases/task,pretrain,seed7,three shards and at most four clients.

Original JAX checkpoint/OpenPI code/H50/C5/K10 and batch-one inference remain
fixed. Effective allowance remains old registry×1.5, with old cutoff success
saved separately. The source diff predicts matching simulator behavior under
these effective horizons. Independent native sampler streams mean differences
between aggregates cannot by themselves establish a causal version effect.

Before scheduling, recheck available GPUs and cumulative closed+reserved GPU
cost. The prepared size is three exclusive1GPU/21CPU/224GiB/16SHM containers,
four-hour maximum,retryOFF. Worker stops after this full atomic comparison;
reference_mismatch is a completed failed reproduction attempt, not a request
to repeat the same version indefinitely. Retain all completed cases and RNG.
