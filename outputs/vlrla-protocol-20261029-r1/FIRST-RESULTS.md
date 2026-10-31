# Starting evidence for the DSRL / in-context VLA comparison

Snapshot: 2026-10-29, Asia/Shanghai. These are existing completed experiments,
not new training results from the VLRLA protocol audit. Preserve their
original motor, task, source and resource contracts.

## Can: controller gain without established memory gain

The source-pinned two-seed comparison used the same frozen low-dimensional
Can diffusion motor, 28,800 primitive training steps per arm, eight paired
held-out initial scenes and K4 retries per scene.

| Controller | Any-step successes / 64 |
| --- | ---: |
| Frozen DSRL-NA | 31 |
| Retry-sampled frozen DSRL-NA | 29 |
| RL² GTrXL, persistent | 41 |
| Same RL² weights, memory reset between attempts | 40 |
| Separately trained episode-reset GTrXL | 39 |
| RL², reward-blind evaluation | 39 |

RL²'s by-attempt successes across seeds are **11, 9, 11, 10 out of 16**.
There is no increasing retry-success curve. The persistent-memory lesion
changes one outcome and the trained-reset gap is two outcomes. This supports
an executable matched-noise-interface pipeline and a descriptive controller
score difference, but does not establish across-episode adaptation, a CTM
advantage, convergence or a VLA result. DSRL spent 24,016 of 28,800 steps in
warmup; only six PPO updates were performed. Seeds/scenes/attempts are nested,
not 64 independent tasks.

Source: [Can evaluation](../../docs/robomimic-can-dsrl/EVALUATION.md), Baige
`job-lrdwz9x2s3kk`, source `b9829df25eeee7cd59d0345fbc4e114b68e0b49f`.
The compact underlying [result copy](starting-evidence/can-results.json)
retains the original source JSON without reinterpreting it as a new run.

## LIBERO VLA: small panels already at ceiling

Completed clean-source AD/DPT controller candidates were tested on two
development variants each of Plus and Pro, one nominal scene per variant,
K4 attempts. This is one goal family, not a blind or broad transfer test.

| Controller | Plus | Pro |
| --- | ---: | ---: |
| Native frozen motor | 8/8 | 8/8 |
| AD full / episode-reset / trained static | 8/8 each | 8/8 each |
| DPT full / episode-reset / trained static | 8/8 each | 8/8 each |
| Frozen source-replay SAC | 8/8 | 8/8 |
| Source-replay SAC with target updates | 8/8 | 7/8 |

The source-replay SAC control is not the original online DSRL source-training
recipe. Frozen/online SAC also differ in deterministic/stochastic sampling,
so their difference does not isolate weight updates. The clean source corpus
was 24/24 successful; useful failure/recovery learning was not established.

Evidence: [compact source results](starting-evidence/libero-results.json),
Baige Plus `job-immogbrrev71`, Pro `job-4cemq3xgqeiq`, source
`041678bd896669be87f34b8716d32d705a85dbb6`. Each job completed 72 episodes;
these 8-episode method cells are correlated development trials.

## RoboTwin readiness and next result

The existing preparation imported all twelve VLRLA tasks and found three
candidate multitask motors. It did not qualify a common motor, rendering,
real episodes or DSRL learning. The cached `adjust_bottle`-specific motor can
qualify plumbing but cannot establish the proposed six-source-task setup.

The first new result should establish native motor competence and ordinary
noise-steering improvement on source-task scenes, then compare persistent
GTrXL/PPO against the matched episode-reset learner and DSRL. Default training
trials are K4; evaluation extends to K8 with indices 1/2/4/8. The full 25-update
VLRLA-sized budget is a reference, pending measured runtime/capacity, rather
than a job already launched by this record.
