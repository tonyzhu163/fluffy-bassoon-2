# Source-only PPO learning-rate and distribution diagnostic

2026-10-30 Asia/Shanghai. The corrected real update matched every saved old
likelihood and value exactly, and completed one PPO update/five optimizer steps
in5.58seconds. Its output distribution was unstable: reported average sampled
KL8.86e19, clip fraction0.794 at learning rate3e-4. It is optimizer-execution
evidence, not stable learning or robot performance.

Use exactly the same immutable three K4 source trials, original zero-update
weights and full source optimizer/RNG recovery. No new robot actions, VLA or
simulator. Evaluate the retained bad-update checkpoint's exact diagonal-Gaussian
KL(old||new), per-decision means/std/logprob ratios and reconstruction. Train
independent3e-6 and3e-5 candidates from the original state, keeping five epochs,
batch3, minibatch16, clipping, value coefficient and all other settings.
The learning-rate override is explicitly written to both config and optimizer.
Keep all candidates and complete recoveries; do not use target outcomes or
claim three source tasks establish the six-task main comparison.

Record per-decision distribution diagnostics, mean-shift/variance KL components,
per-optimizer gradient norms when available, repeated empty-history likelihood
reconstruction, actual counters and timings. A provisional stability gate is
mean analytic joint KL<=0.02 and postupdate clip fraction<=0.2, with finite
nonzero gradients and changed weights. This tests an optimizer setting on a
small source batch, not recovery, adaptation or generalization performance.

The originalr1 startup reserve has been reconciled to its verified49second
scheduler receipt; all other job charges are retained. Corrected PPO r2 took
32scheduler seconds, leaving78seconds of the original2700. Request exactly one
final one-minute allocation, no automatic retries,45s process alarm and50s
outer timeout/5s kill grace. Two measured5.58s updates plus shared preparation
and extra inference should fit; skip unfinished work if the deadline prevents
safe completion and preserve completed candidate checkpoints. Do not launch
more GPU allocations without the separate larger resource decision.
