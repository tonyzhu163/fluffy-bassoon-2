# RoboTwin native-horizon source experiment

The human accepted roughly 1M source actions per learner, parallel collection,
and about 100 PPO iterations, then required whole episodes and K4 trials with
benchmark-specific cutoffs. This replaces exact 10k-action iteration cuts.
The earlier 400-step experiment remains a separate, protected history bank.

## Collection and updates

Each method completes **100 rounds of six full K4 trials**, one trial per
source task in each balanced block: **600 trials / 2,400 episodes per learner**.
Native episode limits come from RoboTwin commit `dca9ec6`:

| Source task | Maximum executed actions per episode |
| --- | ---: |
| handover_mic | 600 |
| scan_object | 500 |
| pick_diverse_bottles, place_a2b_left, place_object_stand, place_phone_stand | 400 |

Success ends an episode early. A benchmark timeout remains a truncation with
bootstrapping. No episode or trial is cut to hit an action target. The maximum
source cost is `100 × 4 × (600 + 500 + 4 × 400) = 1,080,000` actions per learner;
1.1M is a physical safety reservation. Trial counts and task/seed schedules are
matched; realized action counts can differ and are reported separately.
The pinned horizon-file SHA256 is
`41c62bd26184977bae45aee555edd0cc21de5cddf560c2595921fc23a1e3b551`.
PPO clocks use the current task horizon, which is saved with the behavior data.

Persistent PPO and the separately trained episode-reset PPO update once per
round: five epochs, LR 1e-6, minibatch limit 16 trials, and rollback/halt if
reported KL exceeds .1 or metrics become nonfinite. Thus each completes 100
PPO updates. SAC retains 500 decision warmup, batch 256 and UTD 20; its 100
rounds describe collection groups, not its much more frequent SGD updates.

The common frozen motor is Motus pi0.5, H32, full 1,024-dimensional noise bounded
at 1.5, with the same 2,062 public features. New source seed 20261030 was fixed
before outcomes. The old numerical calibration transfers only LR 1e-6; all new
learners initialize fresh. No diagnostic-trained weights are transferred.

## Evaluation and preservation

Evaluate frozen source snapshots after rounds 10, 25, 50 and 100. Source
validation uses a K4 trial on each of six tasks. Two fixed target tasks,
adjust_bottle and move_stapler_pad, use K8 and report episodes 1/2/4/8. Both have
400-action limits. Persistent PPO has normal, memory-reset and reward-blind
conditions; trained-reset PPO has its normal condition. SAC has frozen and
online conditions, independently restored with fresh target replay. Target
updates cannot change source training. The PPO worker owns the native Gaussian
references before and after each evaluation, avoiding duplicate references.

Retain raw observations, actions, native feedback, actual durations, policy
versions and complete replay/optimizer/sampler/RNG/coordinator recovery,
including pending PPO rollouts and evaluation cursors. Offline AD/DPT training
would use this versioned corpus in a separate supervised phase.

## Parallel workers and accounting

Three exclusive one-GPU workers run the three methods in parallel. This reduces
elapsed time; it does not imply a threefold GPU-hour saving. At the earlier
25.87 actions/s, source collection alone extrapolated to about 11.6 hours per
worker. Current measured rates and limitations are in LAUNCH-AUDIT.json.
Operational 12-hour driver / 13-hour scheduler slices checkpoint and resume;
the stopping criterion is the completed round/trial schedule.

The immutable parent manifest composes fixed child action caps: PPO 1,226,400,
SAC 1,194,400 and reset PPO 1,168,800. Their maximum new cost is 3,589,600,
including source safety reservations and all planned validation/target/native
interactions. An additional 1M reservation protects the prior ledger, counted
once: its audited actual cost is 190,394, already including 13,026 earlier
actions. No child reimports those prior actions. The conservative parent bound
is 4,589,600; unused reservations do not authorize extra scientific scope.

Active source pin: `db0184089fa88a5f96c153f9b80029bb91b77def`.
Parent: `robotwin-native-wholetrial-20261030-r2`, manifest SHA256
`1f80c08310dfa4543ec7a748ca6e2bf9aad589cd55978a402210abb0d83bdbc0`.
Each actual allocation passed its GPU/CUDA probe, 112 focused Linux tests,
and real native-noise parity with maximum error 0.0 before collection.

## Scale and execution optimization

[VL-RLA Appendix A.2.1](https://arxiv.org/html/2610.19517v1#A1.SS2.SSS1)
reports 25 teacher PPO iterations × 256 environments × four rounds × 400 steps:
10.24M nominal action slots, or 204,800 decisions at H50. Termination masking
reduces valid data; actual final per-task corpus counts are not supplied.
Student supervised training is a separate stage. This experiment's roughly 1M
per learner is an initial scale, not a reproduction of that nominal capacity.

The human asked to check the ongoing optimization before deciding on 10M.
The shared-motor/process-isolated batched collector is being developed in the
robotwin-collector-speed worktree (candidate b6fb953). CPU qualification is
recorded; a completed GPU throughput/parity receipt was not available at this
check. Compare identical cases, horizons, rewards and update schedules before
using a measured optimization gain for a larger allocation. Existing workers
remain pinned while the candidate is qualified.

LIBERO is independently owned. Its future recipe also needs complete trials
and appropriate suite limits. OpenPI's reference cutoffs are Spatial 220,
Object 280, Goal 300 and LIBERO-10 520; LIBERO-90 uses 400. See the
[OpenPI evaluation code](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py).
