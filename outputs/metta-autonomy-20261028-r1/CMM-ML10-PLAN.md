# CMM addition to the primary ML10 comparison

2026-10-29 Asia/Shanghai. User seeks CTM/CMM performance, generalization or
training-efficiency evidence during the ten-hour campaign. CMM is absent from
the ongoing four-method ML10 comparison, so add one independent CMM lane.

Budget revision: overlap this one-GPU run with the one-GPU complementary
screen, retaining the same total additional campaign cap of ten GPU-hours.
The initial screen has a 200-minute scheduler cap; this CMM run has a
180-minute cap. Including completed qualification jobs, those maxima stay
below seven GPU-hours. Confirmation work must fit the remaining cap and the
09:28:11 deadline. Existing allocations/contracts remain unchanged; no
extra cards per policy, no duplicate run or calibration, no automatic retry.

Review latest main before launch and pin the source; executable learner,
policy, environment and PPO code are byte-identical to the qualified 47a37b7
source. The new shell launcher and saved-array reporter add no model changes.
Run on Baige RPBZZZ6 x1 /16 CPU/64GiB/8GiB SHM, eight simulator workers,
with allocated probe/tests and one actual default-shaped CMM ML10 update
before fresh training. Preserve failed admissions under their own identity.

Match the existing ML10 task/data/schedule: seed1000, benchmark-seed0, all
ten balanced training families, goal-visible MetaWorldV3, family-fixed trials
with instance resampling per episode, 50lanes/K5/H500, 80rounds=10Msteps,
ten PPO epochs, minibatch10trial windows, cached100 TBPTT. CMM uses trace32,
LTM32, T1, width matched to a250k policy cap shared with CTM. Other method
baselines retain their independently specified architectures and algorithms.
Eval every5rounds, ten balanced lanes/K25 on fresh train-family tasks and
held-out families; final normal/reset/reward-blind on paired seeds. Preserve
training clock encoding. Native returns and binary success are both saved.

One-seed exploratory result: no confirmed architecture advantage without
follow-up evidence. CMM paper-calibration state remains unverified while
the required school alias is unavailable. The data may inform implementation
diagnostics; do not claim qualified paper parity or a VLA admission from it.
Check progress every20minutes after the first advancing update.
