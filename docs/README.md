# Documentation index

Method contracts and results by benchmark. Live status is in
[coordination](../outputs/coordination.md); dated plans and evidence are in
`outputs/<run-id>/`.

## Question, design and paper

- [In-context RL: literature review and a CTM RL loop](in-context-rl-review.md): Literature review (RL², AD, PEARL, MELD, AMAGO, TrMRL, Vintix, ECET, VLRLA, survey), pi0.5 language steering, and the proposed CTM in-context RL loop
- [Audit: in-context RL with CTMs in this codebase](ctm-icrl-audit.md): Audit written in fluffy-bassoon: gaps, synchronization/tick risks and the recommended CTM adapter. File paths in it refer to fluffy-bassoon at the time
- [Three design axes for in-context steering of frozen VLAs](frozen-vla-icrl-design.md): Training method, memory architecture and steering interface; staged DSRL/noise and RL² comparison
- [Reward-driven adaptation inside a VLA's context](native-vla-context-icrl.md): Proposal: direct reward-history conditioning inside pi0.5, training/replay requirements, latent-search comparison and paper decision gates
- [Frozen-VLA steering: experiment summary](frozen-vla-steering-summary.md)
- [Learning to Steer: In-Context Reinforcement Learning for Frozen VLAs](paper/DRAFT.md)
- [Paper decisions and claim boundaries](paper/DECISIONS.md)

## Shared trial environments and architectures

- [Cross-attempt RL environment contract](../src/fluffy_bassoon/meta_rl/DESIGN.md): Trial lifetime and feedback, tasks, recurrent PPO, CTM control, looped Transformer
- [Meta-RL evaluation record](../src/fluffy_bassoon/meta_rl/EVALUATION.md)
- [Continuous Memory Machine recurrent core](../src/lerobot_policy_ctm/architectures/continuous_memory_machine/DESIGN.md): The paper-aligned CMM core and its few-shot regression calibration
- [CMM evaluation status](../src/lerobot_policy_ctm/architectures/continuous_memory_machine/EVALUATION.md)

## Meta-World

- [State-based CTM meta-RL qualification](metaworld-ctm.md): ML1 Push and goal-visible ML10, recurrent baselines, success metrics, complete recovery and GPU-hour accounting
- [Single-seed ML10 method comparison](metaworld-comparison.md)
- [RL²-PPO on the common ML10 protocol](metaworld-rl2.md)
- [ECET on goal-visible Meta-World ML10](metaworld-ecet.md)
- [Official AMAGO1 ML10 baseline](metaworld-amago.md)
- [METTA paper-based network implementation](metaworld-metta.md)
- [Optional Meta-World environment workers](metaworld-workers.md)

## ManiSkill and the native pi0.5 bridge

- [ManiSkill pi0.5 in-context RL proof of concept](maniskill-pi05-icrl.md): Frozen pi0.5 motor, hidden per-trial rotation, outcome cache, training stages, controls, environment, assets and launch commands
- [Fast A1 cached rotation pilot](maniskill-rotation-cache-poc.md)
- [Frozen pi0.5 feature bridge](pi05-feature-bridge.md)
- [Stochastic flow-residual policy and exact PPO replay](pi05-bridge-policy.md)
- [Live native pi0.5 bridge PPO](pi05-live-bridge-ppo.md)
- [Balanced native support and fresh-probe qualification](ctm-support-probe.md)
- [Benchmark-specific CTM warm starts](ctm-warmstart.md)

## RoboMimic

- [DSRL on unchanged RoboMimic Can](robomimic-can-dsrl.md)
- [Can DSRL / RL² evaluation](robomimic-can-dsrl/EVALUATION.md)

## LIBERO with a frozen pi0.5

- [Frozen VLA steering on standard LIBERO](frozen-vla-libero.md)
- [Frozen VLA steering evaluation](frozen-vla-libero/EVALUATION.md)
- [Frozen VLA history-controller contract](frozen-vla-history.md)
- [Continuous noise distillation for frozen-VLA steering](frozen-vla-cnd.md)
- [Paired CND/DPT corpus for frozen π₀.₅ steering](frozen-vla-cnd-corpus.md)
- [LIBERO task generalization with frozen VLA steering](frozen-vla-multitask.md)
- [Direct RL² steering of a frozen VLA](frozen-vla-rl2.md)
- [Frozen pi0.5 constant-noise headroom pilot](frozen-vla-noise-headroom.md)
- [Frozen RLinf pi0.5 LIBERO-130 motor](frozen-vla-rlinf-libero.md)
- [Matched source-data training arms](frozen-vla-rlinf-training.md)
- [LIBERO as the starting benchmark for transferable steering](libero-transfer.md): Independent LIBERO-40 to strict LIBERO-90 split, Plus/PRO claim boundaries and adaptation controls
- [LIBERO-40 transfer evaluation status](libero-transfer/EVALUATION.md)
- [LIBERO collection execution evaluation](libero-collection-speed/EVALUATION.md)
- [Cached-prefix scalar motor adapter](libero-cached-prefix-b1.md)
- [Task57 ordinary DSRL with frozen pi0.5](vla-task57-dsrl.md)

## LIBERO source40 pipeline

- [All40 corpus to offline controller milestones](frozen-vla-libero/ALL40-OFFLINE-MILESTONES.md)
- [Approved PPO and frozen-validation execution](frozen-vla-libero/APPROVED-PPO-VALIDATION-EXECUTION.md)
- [Common-runtime source-data robot comparison](frozen-vla-libero/COMMON-RUNTIME-SOURCE-COMPARISON.md)
- [Fixed9 frozen mean/dispersion diagnostic](frozen-vla-libero/DISTRIBUTION-DIAGNOSTIC.md)
- [What the steering controller learns across trials](frozen-vla-libero/META-TASK.md)
- [Zero-interaction motor numerical replay](frozen-vla-libero/MOTOR-NUMERICAL-REPLAY.md)
- [Paired LIBERO90 noise-interface controls](frozen-vla-libero/NOISE-CONTROLS.md)
- [Offline context probe](frozen-vla-libero/OFFLINE-CONTEXT-PROBE.md)
- [Paired memory replication](frozen-vla-libero/PAIRED-MEMORY-REPLICATION.md)
- [Paired AD/DPT-style data from source40 PPO](frozen-vla-libero/PPO-CORPUS.md)
- [Frozen source-PPO target support/query execution](frozen-vla-libero/PPO-TARGET-SUPPORT-QUERY.md)
- [Frozen final-source PPO target wave](frozen-vla-libero/PPO-TARGET-WAVE.md)
- [Pretraining scale: primary-source audit](frozen-vla-libero/PRETRAINING-SCALE.md)
- [Frozen per-task source SAC competence](frozen-vla-libero/SOURCE-SAC-COMPETENCE.md)
- [Complete-case source40 collection preparation](frozen-vla-libero/SOURCE40-COLLECTION.md)
- [Whole-case source data amount diagnostic](frozen-vla-libero/SOURCE40-DATA-SCALES.md)
- [Fresh PPO source rollouts](frozen-vla-libero/SOURCE40-PPO.md)
- [Shared source40 PPO stream](frozen-vla-libero/SOURCE40-SHARED-PPO.md)
- [Source40 history assembly and matched controller training](frozen-vla-libero/SOURCE40-TRAINING.md)
- [Source-subset optimizer-scale qualification](frozen-vla-libero/SUBSET-SCALE-QUALIFICATION.md)
- [Bounded source-subset target comparison](frozen-vla-libero/SUBSET-TARGET-EVALUATION.md)
- [Fixed500 supervised-controller target first wave](frozen-vla-libero/TARGET-FIRSTWAVE.md)
- [Six-case ordinary DSRL learning-onset baseline](frozen-vla-libero/TARGET-ONLINE-DSRL-SIX.md)
- [Fresh target-online DSRL learning-onset diagnostic](frozen-vla-libero/TARGET-ONLINE-DSRL.md)
- [Fixed500 two-goal robot comparison](frozen-vla-libero/TWO-GOAL-ROBOT-COMPARISON.md)
- [Two-source-goal subset qualification](frozen-vla-libero/TWO-GOAL-SUBSET.md)
- [Complete-trial source PPO batches](vla-source40-batch14-ppo.md)
- [Proposed cumulative PPO source continuation](vla-source40-batch14-production.md)
- [Source40 CPU qualification launcher](vla-source40-cpu-qualification.md)
- [DPT validation labels from the matching TRAIN teacher](vla-source40-dpt-validation-labels.md)
- [Interim100 LIBERO90 readout profile](vla-source40-interim100-readout.md)
- [Cumulative source40 SAC top-up](vla-source40-sac-topup.md)
- [Fit from immutable complete top-up prefixes](vla-source40-topup-fitting.md)
- [Source40 interim fitting at 25/100 updates](vla-source40-topup-interim-fitting.md)

## RoboTwin

- [RoboTwin setup for VLRLA-style adaptation](robotwin-vlrla.md): Pinned runtime setup, six seen/six unseen tasks and bridge/history reproduction requirements
- [RoboTwin VLRLA preparation evaluation](robotwin-vlrla/EVALUATION.md)
- [VLRLA: published experimental protocol](vlrla-protocol.md): Exact RoboTwin six/six task split, collection and optimization settings, bridge/simp training, simulation and real-robot evaluation, and reproduction gaps
- [Native RoboTwin AD/DPT source-history exporter](robotwin-history-corpus.md)
- [Native RoboTwin Transformer AD/DPT full/static policy](robotwin-history-policy.md)
- [Fresh-process native K8 wrapper](robotwin-history-live.md)
- [Native frozen-history live evaluation adapter](robotwin-history-evaluation.md)
- [Production-native RoboTwin vector candidate](robotwin-production-vector.md)
