# Scope and provenance

Review date 2026-11-20. Repository base `d2f102e238bc3a24d3452cf8a0fbbec7c5853868`. Simulation is the application. Literature review only, with no model training, target interaction, or GPU allocation.

| Workstream | Independent question | Record |
| --- | --- | --- |
| Scale worker, local Codex CLI | Which papers vary pretraining breadth or volume, in what units, and with what held-out results? | [Scale evidence](SCALE.md) |
| Latents worker, local Codex CLI | What supervision, interventions and test-time information distinguish parameter identification from a useful hidden context? | [Latent evidence](LATENT.md) |
| Robotics worker, local Codex CLI | Which manipulation studies test new bodies, frozen motors, probe histories or a bypass, and what do those comparisons establish? | [Robotics evidence](ROBOT.md) |

We began with the repository's [prior frozen-policy review](../../docs/literature/frozen-vla-offline-icrl-review.md) and [embodiment factory](../embodiment-meta-factory-20261120-r1/DECISION.md). We followed original papers on task-diversity sweeps, action permutations, hidden-parameter MDPs, online system identification, probing, rapid motor adaptation, unseen-arm imitation, and history-conditioned manipulation. We checked reported units, held-out variable, deployment inputs and motor training. This was a targeted narrative review, including dated preprints, without a systematic database screening or pooled effect estimate.

Workers had a short CPU-only evidence pass each and stopped after saving source-linked notes. The synthesis uses independent settings and reset trajectories as the proposed experimental units. The prior two-source design fixes the application and budget for the proposed first comparison. No experimental configurations, random seeds, or machine results were generated in this review.
