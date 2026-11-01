# First frozen RoboTwin history baseline

The first paired development panel completed all **18 K8 units / 144 episodes**. At this early100-update endpoint, the full-history AD and DPT students show **no advantage over static or same-weight episode-reset controls**. The two DPT-reset stapler successes occurred without persistence across episodes; they do not establish ICRL.

| Condition | adjust_bottle | move_stapler_pad |
| --- | --- | --- |
| Native Gaussian noise | 6/8 | 0/8 |
| Native zero noise | 6/8 | 0/8 |
| Frozen source SAC5 mean | 5/8 | 0/8 |
| AD full history | 5/8 | 0/8 |
| AD static | 5/8 | 0/8 |
| DPT full history | 5/8 | 0/8 |
| DPT static | 6/8 | 0/8 |
| AD full, episode reset | 6/8 | 0/8 |
| DPT full, episode reset | 5/8 | 2/8 |

This is one paired K8 trial per development task and condition, with one fitting seed. It is an early functional comparison, not a statistical generalization result or an evaluation of the eventual100-round source-trained PPO/SAC endpoints. Full/static students have independently fitted weights; full/reset pairs share exact weights.

All conditions used the same requested and accepted task cases, initial raw-observation hashes and feature hashes. Bottle E5 consistently accepted fallback seed22265019 instead of requested22265018. Every E1 decision in each same-weight full/reset pair has identical stored feature, noise, issued motor-target bytes, rewards, duration and termination traces. DPT-reset stapler wins were E4/E6; E1 failed identically in both variants. There is no initial-state mismatch explaining the reset comparison.

All four students were fit for100 updates on the same early SAC source bank:1,539 decisions,120 episodes,30 complete K4 trials and47,602 source actions, five trials per six source goals. Copying or fitting that bank added no robot interactions. The fixed DPT labels use the source SAC5 deterministic mean teacher; that teacher is feature-conditioned and its competence is not independently qualified. These results alone do not distinguish limited source coverage, teacher quality, fitting or history utilization as the cause of weak performance. Source collection continues under the original100-round contract.

## Verification and cost

Baige **job-ekg1ivu6j8pg** ended **Success** at2026-10-31 00:21:02 Shanghai. Scheduler running time was49m11s (**2,951 GPU-seconds**); the container timeline was49m13s. Runtime source was2fd2096906e52aabeaeb984f333fe1aaa6c4e196; all18 children and the wrapper exited0.

The panel consumed **47,137 physical primitive control actions** and zero SGD updates. Independent JSON/hash/NPY-mask audit passed22,135 checks, reconciled all3,100 ledger events and1,540 decisions, and found pending=uncertain=0. Model admission identities, matched cases, real native-noise parity0.0 and renderer receipts passed. Pinned runtime guards recomputed steering and frozen-motor identity after each K8 before publishing results; a separate literal post-motor hash field is not stored.

The57,600-action/5,400-GPU-second panel reservation is closed: unused10,463 actions and2,449 scheduler seconds are sealed, not assigned to another experiment. Original source ledgers/caps and trained streams are unchanged.

Detailed outcomes at E1/2/4/8, paired differences, exact hashes and audit limits are in [the independent audit](independent-audit/SUMMARY.md). Scheduler evidence is [the terminal receipt](receipts/scheduler-terminal.json) and [screenshot](receipts/scheduler-terminal.png). No RGB payload-wide rehash or model-forward work was performed by the independent audit.
