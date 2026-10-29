# Frozen RoboTwin noise steering: initial matched-budget experiment

User-approved design, 2026-10-29 Asia/Shanghai. Launch actual source training
as soon as preflight passes; do not substitute a shortened-recipe pilot.

## Contract

- One common frozen `motus-robotics/pi0.5_robotwin2` motor, immutable revision
  `effc6e1a99bd4d0edf9447e8ba313def67649cfe`; native 32-action chunks,
  32-dimensional model noise per action. It was trained on all 50 RoboTwin
  tasks; held-out tasks below are new to the steering learner, not the motor.
- Identical current public frozen-VLM/proprioceptive features and bounded
  learned noise support [-1.5,1.5] for all learners. Native Gaussian noise is
  the unbounded motor reference. Preserve checkpoint delta-joint transforms.
- Uniform balanced blocks over six VLRLA source tasks. No target interactions
  enter source training. Keep successful and failed experience.
- Primary arms: persistent GTrXL/PPO and feature-conditioned feedforward
  DSRL-SAC-style. Separately train episode-reset GTrXL/PPO as a memory control.
  This SAC baseline implements the noise-steering principle with common
  frozen features; it is not an exact PixelSAC or noise-aliasing reproduction.
- Train K4 trials, each with one task identity and fresh scene resets. Memory
  persists only in the persistent arm; no mid-trial PPO policy updates.
- Each arm's source ceiling is **10,240,000 executed primitive actions**.
  VLRLA's 25x256x4x400 expression is a nominal capacity reference. Actual
  early termination means preserving 256 complete K4 trials per PPO update
  does not necessarily produce exactly 25 updates under this actual-data cap.
- Native H32 gives up to 13 chunk decisions per 400-action episode; the last
  chunk is shortened. Never equate it with the H50 paper's 8,192-slot tensor.
- PPO uses GTrXL and the recorded recurrent loss settings; source SAC uses
  batch256, warmup500 chunks, UTD20. All exact sizes/optimizers are saved in
  the executable contract. Do not lower warmup or batch merely to obtain a
  quick update. Optimizer counts and time are reported separately.
- Evaluate K8, indices1/2/4/8 and cumulative outcomes, on the six held-out
  task identifiers in the VLRLA protocol. Report trial-length extrapolation
  in episodes5–8. Include source-validation outcomes separately.

## Transfer and target adaptation

Transfer the learned steering parameters, not an interaction credit balance.
Both methods receive the same source interaction ceiling and target episode /
horizon allowance. Report actual target costs, because early success changes
executed actions. Frozen recurrent weights adapt through context. Evaluate
both frozen source-SAC and source-SAC with target updates, using identical
stochastic action conventions.

For target SGD, independently restore source actor, critics, targets,
temperature, optimizers and sampler state for each target trial. The primary
fine-tuning convention clears source replay and samples a target-only batch
with replacement from at least one observed target transition; it does not
repeat source warmup. This is predeclared so a 100k source buffer does not
dilute few target observations, and so a batch256 threshold does not prevent
updates when K8 supplies at most104 chunks. Record all target optimizer costs.
Held-out updates never carry between independent tasks/trials.

## Initial allocation and continuation

Initial submission: Baige, one exclusive scheduler GPU, 16 CPU cores,
128 GiB RAM, 8 GiB shared memory, **45-minute scheduler cap** (at most0.75
GPU-hours). Run allocation/CUDA/renderer attestation, focused tests, native
sampler parity and a real source-task episode as preflight, then begin actual
source training automatically. No extra standalone qualification allocation.

The driver interleaves methods and checkpoints after complete trials. Keep
pending PPO rollout sequences, every learner/optimizer/replay, balanced
samplers and all RNG state. A wall-boundary stop is a continuation point,
not completion of 10.24M training steps or a failed scientific result.
Measure real episode/update throughput before committing to a materially
larger continuation allocation. Use a new output identity for every job and
preserve the immutable source/learning contract when resuming.

Launcher: [robotwin_noise_compare.sh](../../scripts/diagnostics/robotwin_noise_compare.sh).
Published reference: [VLRLA protocol](../../docs/vlrla-protocol.md).
