# Eight-hour RoboCasa babysit: results and handoff

The requested watch ran from 2026-11-04 23:05 through 2026-11-05 07:05 Shanghai. The already-running continuation was then followed through its successful scheduler release at 07:08:47. All owned GPU allocations are released. [Window receipt](WATCH-WINDOW-COMPLETE.json), [final observation](CHECK-FINAL.json), [checkpoint integrity](CONTINUATION-CLOSE-01.json).

## PPO and GRPO baseline

On 512 fresh paired cases at the fixed 40-update endpoints, SFT succeeded 311/512 (60.74%), PPO 303/512 (59.18%) and GRPO 322/512 (62.89%). The prespecified GRPO−PPO contrast is +3.71 percentage points, paired 95% bootstrap interval [−0.78,+8.40], exact McNemar p=.1374: no winner is established. The earlier PPO advantage on DEV64 did not repeat. This is one trained checkpoint/seed per method.

A separate controlled panel supports **four PPO epochs over one**: PPO four epochs 355/512 versus PPO one epoch 315/512, +7.81 points, Holm-adjusted p=.000775, with the same 3.072M TRAIN controls but 960 versus 240 optimizer steps. It uses a different case bank from the PPO/GRPO panel and cannot directly rank those endpoints. This is the supported recipe change for the proposed new sources. [Baseline result and independent audit](BASELINE-RESULT.md).

## Current joint meta-learners

AD converged at 14,800 and selected update 13,800 by the original validation NLL. DPT converged and selected 16,200. Their selected checkpoints remained identical across ID and OOD evaluation.

DIT and IC-IQL continued from full recovery on the exact original D71 source, corpus and optimizer/sampler/RNG/validation contract. They stopped at the operational invocation boundary, with actor updates **11,384** and **11,328**; their value phases remain 2,700 and 2,600. Both are **unconverged**. CPU integrity checks passed complete optimizer/sampler/Python/NumPy/Torch/CUDA-RNG recovery, all learner/best-value/optimizer tensors finite, and stable checkpoint SHA before/after reading. Their full recovery files remain on project storage. They have no final robot-performance readout; the earlier 2-case interim diagnostic is not promoted to a final result. [Recovery evidence](CONTINUATION-CLOSE-01.json), [validation traces](figures/training-handoff-final-01/training-progress.png).

The current models are frozen-VLM-feature **direct-action heads**. The later experiment learns KV steering through a frozen native action expert. Zero deployment SGD was used in all reported readouts.

## Audited ID and OOD curves

Each entry below is success out of 50 cases. `q0` is the first query; `q5` is the actual query after five complete preceding episodes. Reset clears cross-episode memory while retaining the same fitted head. Cumulative ever-success is a separate outcome.

| Task | AD q0 / q5 / reset q5 | DPT q0 / q5 / reset q5 | SFT q5 | Fixed130 native q5 |
| --- | --- | --- | --- | --- |
| OpenDrawer (ID) | 46 / 44 / 46 | 45 / 44 / 45 | 34 | 32 |
| CloseToasterOvenDoor (ID) | 28 / 27 / 29 | 29 / 27 / 29 | 21 | 26 |
| CoffeeSetupMug (ID) | 29 / 28 / 30 | 27 / 27 / 28 | 15 | 19 |
| CloseFridge (OOD) | 8 / 13 / 10 | 10 / 9 / 9 | 36 | 41 |
| TurnOnMicrowave (OOD) | 0 / 1 / 0 | 4 / 3 / 4 | 8 | 9 |

AD and DPT show strong ID behavior, weak transfer, and no consistent observed advantage from retaining previous episodes. ID PPO controls use each source task’s fixed130 teacher; the OOD PPO control uses the prospectively designated OpenDrawer130 teacher. Native controls have no learned memory; their query-to-query variation is not adaptation. These are descriptive single-fit comparisons with different training data/objectives, without a multiplicity-adjusted adaptation or method-ranking claim.

[Complete 0–5-context curves](figures/ad-dpt-audited-02/query-curves.png) · [AD ID](AD-ID50-RESULT.md) · [AD OOD](AD-OOD50-RESULT.md) · [DPT ID](DPT-ID50-RESULT.md) · [DPT OOD](DPT-OOD50-RESULT.md).

Across the eight final native-control/AD/DPT groups, independent raw audits passed **9,000 episodes, 5,670,000 physical controls, 27,000 videos and 17,037,000 decoded frames**, including commands, draws, feedback, native initial-state matching and memory continuity. [Aggregate hash-bound audit](FINAL-READOUT-AUDIT.json). The baseline TEST512 has its separate independent audit. Earlier SFT repeat outcomes were stable, but exact trajectory determinism is not claimed.

## Five-source steering stage

Only OpenDrawer and Toaster have independently confirmed effective source policies. Coffee 130 remains exploratory; Counter-to-Cabinet and Faucet have no trained source policy. The five-effective-task condition has not been met.

The concrete next proposal trains only Counter-to-Cabinet 40 and Faucet 40 with four-epoch PPO, then tests their fixed final checkpoints and Coffee 130 on fresh paired 512-case panels. It requests **70 GPU-hours** and explicit adoption of its new fixed-endpoint source-readiness gate. No approval was received and none of those sources was launched. The stopped old selected-checkpoint sweep and held sources stay stopped. [Reviewable proposal](SOURCE5-PROPOSAL.md), [CPU recipe qualification](SOURCE5-CPU-PREFLIGHT.json).

After five sources pass the adopted gate, a new five-task corpus and fitted native-KV learner must be qualified before ID then the locked 15 tasks: five atomic seen, five composite seen, five composite unseen, with queries after 1–5 actual episodes. The native-KV primitive passed numerical/device qualification; a fitted five-task steerer has not been trained. Context 384 is planned for the longer histories. [Transfer protocol](TRANSFER-PLAN.md).

The 70 GPU-hour request covers the source campaign only. The later 50-case four-method full/reset transfer matrix plus one frozen anchor is estimated at about 238 GPU-hours of inference, before fitting and timing margin; it needs its own measured reservation. This forecast is not authorization.

## Compute and storage

Current meta accounting is **73.3764/96 GPU-hours**, leaving **22.6236**; the separate PPO/GRPO TEST512 used **3.4533/7.5 GPU-hours**. Charges run from scheduler creation through verified release and include qualifications and continuations. The separate PPO ablation budget was not borrowed. [Allocation ledger](META-ALLOCATION-LEDGER.json).

Closed owned raw outputs were copied to private cold storage with every leaf SHA verified, including reads through stable original paths, before duplicate hot removal. All four native panels and both completed student ID panels are safely cold archived; final DPT ID duplicate cleanup completed at07:19:40 Shanghai. AD/DPT OOD raw outputs remain on project storage. No archive helper remains active. [Retention close](RETENTION-CLOSE.json). Active source datasets, runtimes, model caches and resumable checkpoints remain available on project storage.
