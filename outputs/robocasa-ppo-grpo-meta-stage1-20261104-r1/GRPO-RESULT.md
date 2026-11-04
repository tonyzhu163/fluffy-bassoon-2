# OpenDrawer GRPO at 3,072,000 TRAIN controls

GRPO ended with 41 successes on the 64 original DEV cases, the same count as its initial SFT policy. Seven cases changed in each direction. The middle checkpoints reached 43 and 44 successes. The last ten TRAIN collections had 850 successes in 1,280 attempts (66.41%), compared with 830/1,280 (64.84%) in the first ten, a gain of 1.56 percentage points. The run missed the prespecified ten-point learning gate.

| Completed updates | DEV successes / 64 | Gain vs. initial (pp) | Paired wins / losses | Exact two-sided McNemar p |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 41 | — | — | — |
| 10 | 38 | −4.69 | 5 / 8 | .5811 |
| 20 | 43 | +3.13 | 7 / 5 | .7744 |
| 30 | 44 | +4.69 | 8 / 5 | .5811 |
| 40 | 41 | 0 | 7 / 7 | 1 |

On both actor ranks the reported approximate KL reached 0.030235 at update 5, narrowly exceeding the Stage 1 reporting limit of 0.03. Peak clip fraction was 0.07836. The policy-ratio clip was ±0.2. The crossing predates the restart, and training continued to the intended endpoint.

Each update used 16 new scene seeds with eight attempts per seed and sparse first-success rewards. PPO used 128 different starts per update. Both ran the same number of episodes and controls. GRPO's periodic DEV curve and PPO's 34→34→36→34→26/64 curve used different sampler streams, including at initialization. The [four-policy matched evaluation](../robocasa-ppo40-matched-dev64-20261104-r1/PLAN.md) compares the saved endpoints using common draws.

The first attempt stopped at DEV10 after saving the complete update-10 checkpoint. An environment process compiling a second scene produced different mesh XML despite identical compiled state. All 22 checkpoint files verified. The second attempt replayed the 64-case DEV10 in fresh evaluation processes before further training. See the [failure reproduction](GRPO-DEV10-FAILURE-DIAGNOSIS.json), [recovery qualification](GRPO-RESUME10-QUALIFICATION.json) and [final DEV receipt](GRPO-DEV040.json).

All 80 actor-rank records have finite metrics and unchanged frozen VLM weights. Each rank took 240 optimizer steps. All 80 pre-update likelihood checks had zero maximum chunk log-ratio difference. The audit rechecked 320 raw training reset snapshots spanning 640 scene groups. Of these, 588 had both successful and failed attempts, 43 succeeded on every attempt and nine failed on every attempt. All 40 DEV reset reports matched across five panels and eight ranks. The shared action ledger charged 3,267,072 controls including 3,072 conservatively reserved for three interrupted reset calls. All 22 files in the update-40 checkpoint matched their saved SHA-256 hashes. The [final audit](GRPO-FINAL-AUDIT.json) contains all 40 TRAIN collections, five DEV panels, checkpoint bindings and per-phase counts.

The successful continuation used 9.629 scheduler GPU-hours. Across both GRPO attempts the total was 13.120 GPU-hours. Together with PPO, Stage 1 used 30.247 of 64 GPU-hours. [Scheduler closeout](GRPO-SCHEDULER-CLOSEOUT.json) · [all-attempt ledger](BUDGET.json).
