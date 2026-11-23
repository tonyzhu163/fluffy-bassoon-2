# Paired native-success results

SymMeta improved native success for each of the three reader seeds on all four panels, both overall and on held-out shifts. On the primary 96 queries, the seed-average advantage covered all eight tasks for each π₀.₅ panel. UR5e–π₀ gained on six tasks and lost on both LIBERO-Object tasks. Its +9.7-point overall effect has weaker support under either sign-flip analysis.

| Panel | Queries | Seed 41 / 42 / 43 gain (pp) | Mean gain (pp) | 95% task bootstrap (pp) | Query p | Task p |
|---|---|---:|---:|---:|---:|---:|
| UR5e π₀.₅ | All 96 | +20.8 / +16.7 / +21.9 | +19.8 | [+11.8, +31.2] | 7.15e-05 | 0.0078 |
| UR5e π₀.₅ | Held-out 64 | +25.0 / +18.8 / +25.0 | +22.9 | [+14.6, +33.3] | 0.00046 | 0.0078 |
| IIWA π₀.₅ | All 96 | +25.0 / +25.0 / +22.9 | +24.3 | [+16.7, +31.9] | 3.75e-06 | 0.0078 |
| IIWA π₀.₅ | Held-out 64 | +25.0 / +21.9 / +17.2 | +21.4 | [+14.6, +28.6] | 0.000762 | 0.0078 |
| Kinova3 π₀.₅ | All 96 | +18.8 / +25.0 / +24.0 | +22.6 | [+14.9, +29.9] | 1.67e-06 | 0.0078 |
| Kinova3 π₀.₅ | Held-out 64 | +14.1 / +20.3 / +21.9 | +18.8 | [+8.9, +28.6] | 0.00126 | 0.0156 |
| UR5e π₀ | All 96 | +10.4 / +5.2 / +13.5 | +9.7 | [+1.7, +19.4] | 0.0901 | 0.0781 |
| UR5e π₀ | Held-out 64 | +14.1 / +3.1 / +12.5 | +9.9 | [-3.1, +22.9] | 0.17 | 0.2500 |

**Pairing and verification.** Baige Linux SQLite read the 64 final shards over SSH. Each match required agreement on plan index, suite, task, interface, regime, and ground-truth yaw, scale and grip. Reset ID 13, BASE variant and reset seed are reconstructed from the shared official evaluator plan. All eight UR5e–π₀.₅ seed-41 shard hashes match the published merge manifest, including the Goal-6 replacement. All 118 available published method-by-regime success counts and the four published seed-41 win/loss/tie triples match. The task table follows official shard order.

**Inference.** The primary outcome is the percentage-point difference in binary native task success. For a single seed, the exact two-sided query test gives each discordant pair either sign with equal probability. For the seed average, one sign multiplies the entire within-query contrast, preserving its magnitude and counting the shared flow outcome once. The task sensitivity flips whole task totals across all 256 sign assignments. Both tails use absolute statistics and include equality. The 95% empirical percentile interval uses all 8⁸ ordered resamples of eight whole tasks with replacement and holds the reader seeds fixed.

**Limits.** Sign-flip probabilities are conditional on independent sign symmetry at the indicated unit. Shared task/reset structure can violate query independence. Eight fixed tasks give coarse task probabilities and limited information about generalization. The percentile interval and sign-flip test use different reference distributions and can disagree near zero, as on UR5e–π₀. The held-out subset overlaps the primary set. All p-values are unadjusted. These analyses do not estimate uncertainty over new training seeds or independently replicated rollouts.

**Evidence.** [Keyed outcomes](outcomes.csv), [per-seed and seed-average statistics](statistics.csv), [ordered task effects](tasks.csv), [aggregate checks](aggregates.csv), [paired checks](paired_checks.csv), [shard paths and hashes](sources.csv), [Linux runtime and source revisions](runtime.txt), [reanalysis driver](reanalyze.py), and [LaTeX appendix](appendix.tex).
