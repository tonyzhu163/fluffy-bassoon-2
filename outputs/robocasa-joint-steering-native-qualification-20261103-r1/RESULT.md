# Native frozen steering qualification

Baige `job-peaxl7mua1os`, source `ccaea60b`, completed successfully2026-11-03 **00:33:04 Shanghai**; its only GPU is released, zero restarts. Conservative creation-to-terminal cost122GPU-seconds (0.0339GPU-h), charged to the velocity arm preparation budget. [Scheduler closeout](SCHEDULER-CLOSEOUT.json), [actual result](RESULT.json), [GPU probe](GPU-PROBE.json).

The original Human300 VLM and action expert remained frozen. Both velocity32 and independent all-layer K/V1152 passed original-native zero-code parity, finite nonzero decoded-action influence, and exact same-device next-Adam restoration. The actual native cache had18layers; both K and V changed in **all36targets**, with the original cache unchanged. The pipeline used the allocated CUDA device; peak allocated VRAM7.75GB. All five referenced initial/cached-output/synthetic-boundary artifact hashes were independently checked.

One actual TRAIN OpenDrawer reset supplied the public observations. The probe repeated those observations for B4 inference and used24 repeated public-feature rows with synthetic rewards for numerical learner tests. **No policy control or scientific PPO update occurred**, and no scoreable episode completed. The reset integrated250internal physics ticks; the exact native control counter was0 and all reservations were resolved. Temporary synthetic actor/value checkpoints are numerical evidence.

## Parameter calibration still required before scientific launch

The public-observation reference RMS was0.08496. Velocity's largest probed fraction0.1 produced normalized action RMS0.06393; KV's largest fraction0.08 produced0.01479. These grids do not bracket the declared action-RMS target. They establish functional influence; the final eight-worker scale has not been selected.

The24-row synthetic test at learning rate3e-4 ended with joint old-behavior KL proxy0.02147 after5steps for velocity and0.75348 after1step for KV. The1152-dimensional recipe needs its learning rate and KL-check cadence qualified before real training. This is a synthetic numerical stress result, not measured robot-policy improvement or cumulative drift from the frozen VLA. No hyperparameters were changed in an active scientific run.

## Human launch hold

A subsequent user instruction, explicitly relayed by owner chat01a199fb, holds **both scientific steering PPO launches until iteration count and environments per GPU are finalized**. The103x80/four-lanes-per-GPU draft remains provisional. The parent also stopped further GPU requalification pending that discussion; existing evidence and accounting are preserved. Neither scientific arm has been submitted.
