# Production critic parity and bounded repair qualification

Owner chat `01a1a4a2`. The user authorized the proposed critic validation on
Nov4 after reviewing the synthetic mismatch and its uncertain production
impact. This run is separate from the completed dimension replay.

## Phase 1: unchanged-policy production replay

Use the original Drawer-to-Counter checkpoints at policy steps 0, 1, 10 and
120, matched to saved collection iterations 1, 2, 11 and 121. Step 0 uses q1;
the others use production. For each, take actor ranks 0 and 1, chunk indices
0 and 6, and actor lanes 0–15: sixteen B16 cohorts / 256 decisions. Verify
every actor/sampling/environment join and member hash against the closed
corpus audit; restore omitted images and chains from their exact joined raw
records. Preserve saved old values, log-probabilities, returns and masks.

At each matching frozen checkpoint measure native actor replay likelihood
parity; all ten suffix critic values; their mean versus the selected-step
prediction; agreement with the saved rollout mean; pre-SGD value-clip
frequency; and the change in prediction/head gradients through the actual
PPO critic loss when its reference is made consistent. No optimizer step,
new sampling, simulator control or counterfactual learning claim belongs to
this phase. Reject mismatched checkpoint versions or reconstructed inputs.

Use one verified, exclusive Baige GPU for at most 45 minutes (2,700
GPU-seconds), with no automatic retry. Check actual queue capacity, limits,
accelerator type, device probe, storage capacity and a write probe first.
Keep active runtime/vendor worktrees immutable; stage only private checkpoint
copies and reclaim only those copies after verified consumption.

## Phase 2: qualify a consistent critic

The candidate retains the suffix critic and selects its rollout value at the
same saved denoising index that the native actor recomputes. Preserve joint
likelihood behavior, action/noise draws, model weights and optimizer recipe.
Qualify unchanged-weight value parity and likelihood/action equality on CPU
core regressions and the real saved cohorts before enabling any training.
Keep the historical mean-value path explicit for the controlled comparison.

## Phase 3: conditional short comparison

If measured production gaps affect value clipping or gradients, compare the
original and qualified implementation on Drawer under matched initial
weights/RNG, the same global batch and loss schedule, and disjoint output
identities. Start with three updates per arm; this is an optimization
diagnostic, not evidence of convergence or a replacement 10M-control result.
Predeclare exact evaluation cases, resource/controls ceilings and stopping
criteria after phase-1 timing and numerical evidence are available, before
launch. Otherwise document why a live comparison is not warranted yet.

The cumulative ceiling for this new validation, including any comparison and
failed attempts, is **6 GPU-hours**. The preceding dimension diagnostic cost
1.215 conservative GPU-hours and is closed. Do not start a full campaign,
repeat all five sources, change critic architecture, or exceed this ceiling.
Monitor launched jobs every 20 minutes and close scheduler release/costs.
Do not conclude that a synthetic or saved-batch gradient difference caused
the historical teacher-gate failures.
