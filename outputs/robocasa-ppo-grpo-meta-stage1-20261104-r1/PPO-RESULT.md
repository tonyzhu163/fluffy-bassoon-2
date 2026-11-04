# Stage 1 PPO completed: numerical checks pass, learning gate fails

Baige Pro6K job `job-bylh99ylx9ie` completed successfully at Nov 4, 2026,
18:04:29 Shanghai. The eight-GPU allocation is terminal. Source d9a1c616,
RLinf 8f9bfa76, OpenDrawer, 40 iterations, one epoch, 128 episodes per update.

TRAIN success increased from 809/1,280 (63.20%) in updates 0–9 to
825/1,280 (64.45%) in updates 30–39: **+1.25 pp**, below the +10 pp gate.

| Policy update | DEV successes / 64 | Change from baseline | Paired wins / losses | Exact two-sided p |
| --- | ---: | ---: | ---: | ---: |
| 0 | 34 | — | — | — |
| 10 | 34 | 0 pp | 5 / 5 | 1 |
| 20 | 36 | +3.125 pp | 9 / 7 | .8036 |
| 30 | 34 | 0 pp | 7 / 7 | 1 |
| 40 | 26 | −12.5 pp | 5 / 13 | .09625 |

All 80 update/rank records have finite losses and unchanged frozen VLM.
All 80 pre-SGD likelihood/value parity checks pass. Every update performs
six optimizer steps, reaching 240 per actor rank. Maximum approximate KL
is .020879, maximum clip fraction .082534. All five DEV panels pass native
physical/public reset matching across all eight environment ranks.
Explained variance improves from −.9106 to .4158; value loss .46125→.05124.

The predeclared learning gate fails. The observed final DEV decline is
statistically inconclusive and is not a causal comparison against old PPO.
See the [root cause analysis](ROOT-CAUSE-ANALYSIS.md) before interpreting it.
No independent TEST, source admission, Stage 2 or student fitting is implied.

Worker-clock consumption is 61,488 GPU-seconds (17.08 GPU-hours). Conservative
creation-to-scheduler-completion consumption is 61,656 GPU-seconds
(17.1267 GPU-hours), including allocation/startup overhead. The all-attempt
Stage 1 cap remains 64 GPU-hours; GRPO is not yet submitted.

The [complete artifact audit](PPO-AUDIT-LIVE.json) and
[scheduler closeout](PPO-SCHEDULER-CLOSEOUT.json) bind the evidence. The full
final checkpoint contains model weights, distributed optimizer state and
all actor/rollout/environment RNG and sampler sidecars; bulk storage
archival must be content/hash verified before reclaiming this run's files.
