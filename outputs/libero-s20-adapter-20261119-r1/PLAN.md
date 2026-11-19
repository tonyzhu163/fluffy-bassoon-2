# Timing-conditioned π0.5 motor on released LIBERO-Pro2.0

Can the frozen policy execute a stronger correction under S20? Fit a
two-layer velocity adapter on the frozen JAX OpenPI π0.5 action expert.
Use normalized action-expert features, current public state and a one-bit
BASE/S20 setting. Add the learned velocity correction on the seven physical
action coordinates and keep the original ten denoising steps. Start with
zero correction. Check exact command agreement with the same solver's
forced-zero control and measure the decoded difference from the original
release solver before the first optimizer update. [Preceding replay](../libero-s20-motor-20261119-r1/RESULT.md)

Use the released successful demonstrations on the 24 fitting tasks at
`6c4c5524`, dropping declared duplicates and holding out task/reset groups
for offline validation. Give both timings the same task wording. Pair stored
observation `t` with the next ten actions and mask an incomplete end chunk.
Use official π0.5 preprocessing and the frozen checkpoint's normalization.
Train with the standard OpenPI noise/time schedule and mean squared flow
error on the physical coordinates. Change only the adapter weights. Start
with 12 updates on the two already cached bowl tasks to qualify data,
nonzero gradients, checkpoint reload and one live evaluation panel. Fit the
full corpus with 2,048 updates, balanced batches of four, AdamW at 3e-4,
gradient norm clipping at 1 and seed 19112026. Retain the final checkpoint.
The validation readout compares the same windows under true timing, swapped
timing and zero correction. In the full fit the draw count is 8,192 windows.
Report distinct trajectories and task/reset groups actually sampled.

Evaluate on the eight project development tasks, each with two fresh resets
under BASE and S20. Compare default, original paced execution, the adapter's
solver with its correction forced to zero, and the learned adapter forced
to BASE, adapter forced to S20, and adapter using public command–motion lag,
as well as from-reset `adapter_base_full` and `adapter_s20_full` arms
(running without the 40-control default opening) to measure the cross-episode
informed motor ceiling separately from the 40-control online-lag switching
arms. Force both flags independently to measure the informed adapter and the fixed
flag selected on separate fitting resets. The 40-control adapter arms and the default
share a charged 40-control opening. The lag estimator then uses that opening
with threshold eight, falling back to the source-selected flag if undefined.
The paced control and `*_full` adapter arms run from reset. Compare the paired
native successes and BASE regressions. Timing-conditioned improvement over
the strongest fixed setting establishes motor coverage. Improvement over the
lag-estimate arm measures the remaining opportunity for learned inference.
The official test tasks remain reserved. This is a phase-one diagnostic.

Reserve up to four incremental GPU-hours on Baige. Use the development GPU
for smoke tests and submit the full fit and benchmark panel through
**分布式训练**. Save the adapter, optimizer, step and sampler RNG
in the checkpoint. Reassess the fit after the held-out action and native
success readouts. [Frozen action menu pilot](../libero-decision-pretraining-20261119-r1/RESULT.md)
