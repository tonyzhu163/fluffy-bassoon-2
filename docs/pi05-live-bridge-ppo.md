# Live native pi0.5 bridge PPO

`python -m fluffy_bassoon.meta_rl.pi05_bridge_train` trains the feature bridge
on live `PutOnPlateInScene25Main-v3` simulator outcomes. The complete native
pi0.5 VLM, prefix cache, action expert and decoder remain frozen. CTM T4 and
Transformer-XL T1 read the same detached final-layer native K/V and eleven
public feedback fields. Both predict a zero-initialized residual on native
flow velocities, plus a zero-initialized critic on their recurrent features.

## Trial and observation contract

The initial bounded learner batch contains two complete trials, each with four
80-step attempts. Each chunk executes five physical steps and four Euler passes.
Theta is sampled once per trial from `[-45, 0, 45]` degrees and rotates prepared
world xy commands after native action decoding. Scene objects remain within the
declared object set; each attempt gets a freshly sampled overlay, position and
orientation. Training uses overlays 0–11. Evaluation uses overlays 12–15 and
held-out theta values `[-30, 30]`. Objects and underlying motor training data are
shared; this split measures learner generalization within the chosen simulator.

Feedback contains the requested seven-dimensional command **before** hidden
rotation, previous terminal success reward, attempt boundary and normalized
public attempt/chunk clocks. Post-rotation commands, theta, scene/episode IDs and
graded simulator outcomes never enter the bridge. Images and proprioception
remain visible through native K/V. Success is latched over an attempt and emitted
only at its final chunk; only the next chunk receives that reward in feedback.
Reset/settling runs inside the environment without learner recurrence. Memory
starts fresh per trial and persists across every attempt within that trial.

## Stochastic policy and PPO

The policy samples independent Gaussian residual velocities with declared fixed
sigma (default .05) on seven active native model channels at every Euler pass.
The residual mean is bounded by `residual_scale` (default 1.0). Other padded
native channels remain unchanged. Initial native noise is drawn from a separate
checkpointed generator. The action variable for PPO is the joint collection of
sampled auxiliary residual velocities. Its log density sums over all four
passes, eight model horizon slots and seven active channels: 224 scalar Gaussian
samples per chunk. It is not a density for the marginal decoded physical action,
and it is never divided by dimensionality. Fixed-sigma entropy is constant.

Behavior records retain detached CPU native K/V, masks, public feedback,
per-Euler noisy actions, native velocities, suffix features, times and sampled
residuals. Replaying an optimizer epoch begins with initial memory and processes
every chunk chronologically, recomputing current bridge and critic features.
Saved bridge features and behavior means never substitute for current features.
The native motor is not rerun or placed in the optimizer graph during replay.

PPO uses complete-trial returns with gamma and GAE lambda one, terminal bootstrap
zero, and no return break at an attempt boundary. Each epoch fits the entire
saved global batch and takes one optimizer step; no truncated memory or execution
microbatch changes that weighting. Defaults are four epochs, AdamW at .0001,
clip .2, value coefficient .01, entropy coefficient .01 and gradient norm cap 1.
An optional explicit `--target-kl` stops remaining epochs after the joint KL
exceeds its cap; actual optimizer/replay counts and the early stop are recorded.
The default reward offset is zero. If explicitly changed, centering is applied
only to attempt-ending reward chunks, and public feedback/scoring stay raw.
CTM decay parameters are projected to [0,15] immediately after optimizer steps.

The first two updates must reproduce saved joint likelihoods and critic values
before optimization. The driver records numerical replay error, per-epoch joint
KL/clip fraction/max log ratio, finite gradient norm, nonzero gradient count,
phase times and peak memory. Native parameters must have no gradients and retain
their original fingerprint after both smoke updates and at completion.

## Allocation, evaluation and recovery

Run CPU tests on allocated Linux resources. Live work requires a one-visible-GPU
Slurm step or an exclusive Baige allocation. A two-GPU Baige parent first runs
the existing allocation probe with expected count two, then passes its immutable
report through `--allocation-probe` to both children. Each child declares its
physical `--expected-gpu-uuid`; the driver verifies selected CUDA PCI identity
against that scheduler map and records actual SAPIEN renderer binding. A single
GPU allocation can use the ordinary allocation-local probe path.

The target is 100 updates under a 9,000-second driver cap and a longer scheduler
allocation. The initial two updates qualify execution, then the same source,
optimizer and trial contract continue. Frozen evaluations at updates 0/10/50/100
use bounded common-seed held-out trials under normal memory, reset between
attempts and reward blinding. Native initial noise is shared across these
conditions; evaluations use the Gaussian mean residual and restore caller RNG.
The default two lanes provide one trial per held-out theta per condition; these
small diagnostics cannot establish a reliable learning or architecture claim.

Source, checkpoint/norm-stat/source-file hashes, selected device, global batch,
optimizer and stochastic-policy settings are saved before training. Each complete
update saves an atomic checkpoint containing bridge/critic, optimizer, global
Python/NumPy/Torch/CUDA RNG, private scene sampler and explicit motor generator.
Private trial plans and command hashes are separate audit artifacts. Signals and
time caps stop after a complete batch/update. Failures leave the previous complete
checkpoint intact. Resume it into a fresh output directory with the exact same
source and contract; a partial trial or partial optimizer epoch is never resumed.
