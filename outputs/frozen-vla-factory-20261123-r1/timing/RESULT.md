# Phase-conditioned waits added no uniquely recoverable development query

On 32 held-out queries, the gripper-conditioned schedule selected a recorded successful branch on 19, versus 18 for seven constant waits. It won three pairs and lost two (+3.1 percentage points, 95% whole-task bootstrap interval 0.0 to +9.4). The only net task gain was one nominal bowl-into-drawer query (LIBERO-10 task 3). The two delayed Goal 1 placements traded one win for one loss, as did the two nominal Spatial 1 placements. The difference against default was also one query (+3.1 points, interval −6.3 to +12.5).

| Schedule or selector | BASE / 16 | S20 / 16 | Total / 32 |
| --- | ---: | ---: | ---: |
| Five moves, replan (fitting-selected fixed) | 16 | 2 | 18 |
| Ten moves, fourteen waits | 8 | 5 | 13 |
| Ten moves, seven waits | 14 | 4 | 18 |
| Ten moves, gripper-conditioned seven or fourteen waits | 15 | 4 | 19 |
| Five-control motion readout, with or without phase action | 16 | 3 | 19 |
| Same readout with support lag, with or without phase action | 16 | 3 | 19 |
| Fitting-selected public-lag table | 16 | 5 | 21 |
| Per-query hindsight ceiling, with or without phase action | 16 | 6 | 22 |

The principal failure was limited action coverage under delay. All four branches failed on ten of sixteen S20 queries. On every query where the phase schedule worked, at least one phase-independent schedule also worked. Relative to fourteen constant waits, the phase rule gained eight queries and lost two (+18.8 points, interval 0.0 to +37.5). The smaller effect against seven constant waits weakens the case for extra phase logic. The archived rule also lengthens the first burst's wait, so these comparisons cannot isolate the effect of responding to a later sign change.

**Selection on new tasks.** A paired-return ridge readout fit on the eight source tasks used five shared query controls and zero support. A second fit added a lag bit from up to 80 public controls of one earlier default attempt. The phase action received the highest score on zero of 32 held-out queries in both fits. Both fits scored 19, trading one win and one loss against each other. The source-fitted two-entry lag table chose default for ordinary motion and fourteen waits for delayed motion. It missed one reachable query. The outcome of that query and the per-case ceilings are available only to the scorer. [Inputs, fit and falsifier](PLAN.md) · [Counts, choices and task-cluster intervals](metrics.json)

**What transfers.** In the separate finger-friction replay, all 16 ordinary/reduced task-material pairs had a complete two-tape menu. Within each pair, default and paced tapes had identical first five commands and observed wrist motion. The sixth command differed on every pair. No gripper-command sign change occurred before the fork. Applying the zero-support source readout at that fork chose default on all 16 (14 recorded hits, versus 12 for pacing and a ceiling of 15). The archive contains no phase-conditioned tape here. The two reduced-friction default losses exhausted their recorded tapes with control budget remaining. These eight tasks overlap the timing panel. [Original replay](../../libero-xemb-friction-dev-20261119-r1/RESULT.md)

The compressed FK exports for Kinova3/π0.5, IIWA/π0.5 and UR5e/π0 each contain 96 historical endpoints and command-map diagnostics. They lack the required timing-branch table. The timing result applies to the LIBERO-Pro2.0 BASE/S20 perturbation of one frozen π0.5 policy. It establishes neither cross-arm replication nor an official test score. Only two paired starts per development task were measured, and earlier worker comparisons changed three paced outcomes on matched resets. The intervals resample the eight observed tasks with replacement, conditional on this source fit and these archived branches. [Cross-arm evidence](../../symmeta-fk-crossarm-20261123-r1/evidence/) · [Original branch comparison](../../libero-decision-pretraining-20261119-r1/RESULT.md)

**Recommendation.** Retain the public-lag table as the timing baseline. Deprioritize scaling this gripper-sign scheduler. To test a different phase architecture, collect matched continuations from the same pre-contact states on delayed cases that fail this entire menu. Include an opening-only long wait to separate its effect from later transition waits.

## Provenance and reproduction

Offline lookup of previously executed simulator outcomes. The phase rule runs inside the archived branch using the current frozen proposal. Both readouts choose one complete continuation at the common five-control boundary. The support bit uses commands and observed wrist positions, including the collector's fallback for short supports. No per-case oracle action, timing label, task ID or outcome enters the readouts. The source and target each contain eight disjoint tasks, two support/query reset pairs and two timing regimes. The retained branch summaries and common-prefix checks are available at this pin. The menu02 per-control NPZ tapes were retained on Baige.

- Research base `757a44cd4374833cad790852df8b87ddf412e638`. Source [menu02 collector](../../libero-decision-pretraining-20261119-r1/collect.py) at `b90ee5a0c049d301529d0bf661d0a682b858ed78` used frozen OpenPI `pi05_libero` and the checkpoint `pi05-libero-openpi-20261027-r1`. [Matched endpoints](../../libero-decision-pretraining-20261119-r1/reader01-summary.json) · [Public motion features](../../libero-first-controls-20261119-r1/prefix-readout.json)
- Local Linux CPU, Python 3.12.16, NumPy 2.5.4, exit 0, zero new simulator episodes, zero motor updates, 0 GPU-hours. No stochastic fitting. The percentile interval uses the exact discrete distribution of drawing eight complete tasks with replacement. Two runs produced identical JSON, SHA-256 `927a059645c6bf9b5795af110f17fb2e6b8e13dc230cf52a47e837e0fd8690db`. The [separate summary of the same episodes](../../libero-first-controls-20261119-r1/paired-evidence.json) reproduced the primary win/loss counts, interval and oracle comparison.

Run from the repository root with the project environment installed.

```sh
lane=outputs/frozen-vla-factory-20261123-r1/timing
OPENBLAS_NUM_THREADS=1 uv run --no-sync python "$lane/screen.py" | cmp "$lane/metrics.json" -
```
