# Stage 0: compare frozen-pi0.5 steering interfaces

Owner: chat `01a19b58-9846-79b1-8a8c-f340f2c2f84d`, 2026-11-02 Shanghai.
Human requested the original Stage 0 screen, then asked to test all interfaces
at this stage. This extends the
[improvement-gate proposal](../robocasa-ppo-improvement-gate-20261102-r1/PLAN.md).
The existing AD/DPT evaluation remains its owner's immutable experiment.

## What this stage can establish

Screen intervention headroom before expensive PPO training. Recovery under
random search is evidence that an interface can change useful decisions;
it is not evidence that PPO learns to exploit it. A negative finite random
screen does not exclude an interface. Only a subsequent simulator-reward PPO
learning curve and paired held-out evaluation can establish improvement.
History collection for AD/DPT follows that positive control, then the matched
Flora-reward positive control. Keep the full chronological data from those
learners, including failed runs and optimizer boundaries.

## Candidate families and qualification

| Family | Stage 0 treatment | Prerequisite before rollout |
| --- | --- | --- |
| Action residual | Original 6-D eef residual, bounds 0.25 and 0.5 of action norm std; initial actor std 0.5 | Zero control reproduces native actions; unchanged gripper/base dimensions |
| Initial noise | Native independent Gaussian control plus bounded 32-D tied-noise steering | Record exact noise map, native/tied distributions and seeds separately; unchanged decoder |
| Conditioning/KV residual | Bounded low-dimensional code modifying frozen conditioning | Native-path zero equivalence, finite nonzero intervention, explicit injection site/projection/scale; no outcome-selected scaling |
| Denoiser velocity residual | Bounded code modifying frozen flow velocity | Native-path zero equivalence, exact denoising schedule and coordinates, explicit projection/scale; no outcome-selected scaling |
| Direct action head | A head from frozen prefix/public-state features that bypasses the action expert | First qualify a supervised motor-distillation warm start using retained source records, with base-action features removed. Charge this training separately. Random untrained actions are not a meaningful comparison |
| Sample-and-select | Report best-of-8 native episode recovery as search headroom; separately specify and qualify a realizable candidate scorer | Simulator-oracle best-of-8 is diagnostic only. It is not a deployed selector or an upper bound on all chunkwise selection policies. Report candidate-inference cost separately from physical interactions |

The conditioning/velocity mappings, direct-head warm start and realizable
selector are not currently RoboCasa-qualified. They must receive concrete
implementation/scale/data contracts and numerical qualification before their
own pilot. Do not silently substitute extra action-residual arms for them or
claim that all six have been tested from the original three-arm screen.

## Shared cases and readout

- Four tasks: OpenDrawer, PickPlaceDrawerToCounter, CoffeeSetupMug and
  TurnOffStove. CounterToCabinet is excluded as in the original proposal.
- Horizon 600 physical policy controls, 12 native H50 chunks, no success
  termination, simulator terminal success for this stage. No Flora scoring.
- Prospectively identify the first 20 native failures per task from a fresh
  ordered TRAIN screening bank, disjoint from DEV/ID/OOD. Native failure is
  determined by an independent anchor motor draw, not one of the eight
  subsequent search draws. Retain all inspected cases and their costs.
- Use eight new draws per case for the native resampling control and each
  qualified continuous interface. Pair motor seeds across applicable arms,
  pair actor draws across the two action bounds, and record exact native
  scene/state/camera/public-state fingerprints before physical actions.
- Conditioning, velocity and noise have different units. Calibrate a
  declared scale on public observations without success labels; report the
  resulting decoded-action perturbation, not an assertion that equal latent
  bounds are equal interventions. Random screens do not rank PPO learnability.
- Report per-task per-draw success, per-case best-of-8 recovery, paired
  wins/losses and exact exploratory paired tests. Keep the failure-conditioned
  denominator explicit. Do not pool this with unrestricted native success.
- A realized selector and a competent direct head require their own honest
  baseline/readout. Their training and candidate-inference costs cannot be
  hidden in the no-SGD random screen.

## Resource proposal and authority

The original three-arm estimate is 1,152,000 policy controls and about
3.7 GPU-hours at the proposal's unverified 87 controls/s/GPU. Seven
eight-draw rollout arms would be 2,688,000 controls and about 8.6 GPU-hours
at that rate, before native-failure identification, startup, qualification
and any direct-head/scorer fitting. Real throughput must be measured on the
actual allocation; an estimate is not runtime qualification.

Propose an aggregate **12 GPU-hour ceiling**, at most four allocated GPUs
at once, at most three hours per scheduler allocation, with all pilot/setup
and failed attempts charged. Use project NVMe and verify capacity/quota/write
access. Require actual allocated GPU/CUDA probes and an advancing real
episode. A finite cap may leave a family incomplete; report that explicitly
and diagnose before proposing a successor. No automatic fresh allocation.

**Expanded resource ceiling and direct-head/scorer training await the human
budget decision.** The user's original Stage 0 request authorizes preparation
and the original residual/native screen. No expanded GPU job has launched.
No Stage 1 PPO or multi-task history collection is authorized by this plan.

## Preservation

Pin reviewed latest main on each fresh execution worktree; never alter active
AD/DPT, calibration or PPO jobs. Save per-chunk public features, actual control,
motor noise, base/executed actions, terminal truth, case identity and timings.
For any supervised or PPO fitting preserve optimizer, sampler and RNG recovery
states. Monitor owned jobs every 20 minutes and update only this chat's row in
the shared coordination file. Verified findings get a durable summary after
the evidence audit.
