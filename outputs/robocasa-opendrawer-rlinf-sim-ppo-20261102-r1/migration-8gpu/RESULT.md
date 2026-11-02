The requested eight-GPU migration is running on Baige `job-h3d8va4slqg5`, source `9ed1d19d42a346c6330c4d7f55303a48d21158ad`, with eight rollout/environment ranks and two colocated native actor ranks. All eight distinct allocated GPU probes passed at 19:23 Shanghai. The first migrated native update completed at 19:30, advancing global iteration 6→7 and all 198 actual Adam parameter-state counters 96→112. The VLM stayed frozen and losses/gradients were finite.

The first migrated update used the unchanged global 128 episodes / 51,200 controls / 1,024 H50 chunk slots, global minibatch 256 and four optimization epochs. Its 83/128 first-success and 83/128 final-success outcomes are a TRAIN observation. The final learning and paired held-out improvement gates remain pending until iteration 18 and the original after panel.

The complete step-7 checkpoint was independently hash-verified at 19:31:59: native model/value/Adam/LR state plus two actor, eight rollout and eight environment sidecars. A selective allocated-CPU read of the actual DCP tensors confirmed all 198 Adam step counters, both learning rates (5e-6 / 1e-4), and scheduler continuation from epoch 6 to 7. The original step-6 files and chronological batches remain immutable dependencies. See [first migrated update](../attempt-r4/ITERATION7-VERIFIED.json) and [actual Adam/LR continuity](../attempt-r4/ADAM-LR-CONTINUITY.json).

| Native timed operation | Prior two-GPU mean, four updates | First eight-GPU update | Speed ratio |
| --- | ---: | ---: | ---: |
| Full PPO iteration | 349.1 s | 274.2 s | 1.27× |
| Rollout collection | 266.4 s | 191.1 s | 1.39× |
| Actor training | 80.8 s | 80.5 s | 1.00× |
| Native reset/bootstrap | 131.6 s | 84.8 s | 1.55× |
| Native simulator interaction | 108.4 s | 90.9 s | 1.19× |
| Policy prediction | 35.2 s | 20.9 s | 1.68× |

The measured gain comes from collection. Two actor ranks retain the qualified rank-local advantage/mask reduction and Adam schedule; their training time remains about 80 seconds. Native simulator/reset work remains substantial. These are early native timing measurements across different fresh case banks; the table uses the first eight-GPU update. The eight-GPU allocation uses more GPU-seconds per update while reducing elapsed time.

The ordinary next update (global iteration 8) completed in 267.7 seconds, with 185.4 seconds for collection and 80.2 seconds for actor training. This gives about 1.30× elapsed-time improvement against the same prior mean. At 19:37:27 the run had completed 8/18 updates and 96/128 episodes of iteration 9; no extra diagnostic episodes or competing test allocation were used. See [latest ordinary-run timing/accounting snapshot](../attempt-r4/LATEST.json). A full-update eight-GPU allocation consumes about three times the GPU-seconds of the measured two-GPU execution; eight GPUs were explicitly requested and these measurements show the resulting tradeoff.

Fifteen-second GPU telemetry observed maxima of approximately 33–35 GiB on the two learner GPUs and 13–14 GiB on the other six GPUs, against about 96 GiB capacity each. The old two-GPU memory evidence is one snapshot at about 40 GiB/card, so it does not establish a matched peak-memory comparison. Per-process learner peaks are also retained in the update receipts.

The old owned allocation `job-qdtkpou7rv50` stopped at 19:16:38 after checkpoint 6 was fully verified. Six complete updates / 768 TRAIN episodes were retained. During stop, the next partial bank consumed 1,600 resolved controls and left an 800-control unresolved reservation; all 2,400 remain charged. The partial bank is preserved on disk and excluded from SGD. Iteration 7 resumes from step 6 and explicitly incurs that repeated case-prefix cost. Projected complete-run controls are 975,200 of the original 1,000,000 ceiling, leaving 24,800 headroom. See [stop/accounting disposition](STOPPED-AND-ACCOUNTED.json).

The original 64 before cases and their metrics (35 first-success / 34 final-success) were reused with zero new evaluation controls. Environment/rollout sidecars and canonical cases were transparently repartitioned by global case seed; old RNG evidence was retained and deterministic new-world rollout streams were recorded. This preserves the global case progression and scientific schedule while allowing numerical and stochastic world-size differences. It does not assert bitwise trajectory continuation.

The new entry point accepts a `CHECKPOINT_AND_EXIT` marker in its own output directory, polled after a complete update. It saves native DCP plus all worker sidecars and exits before another rollout. Allocated-Linux control-flow qualification passed; the actual native full-boundary path was independently verified on the first migrated update. No operator-exit marker was used during this continuing run.

Continue the existing parent 20-minute monitor against this exact job/output, with checkpoints 12/18 and the original paired 64-case after evaluation. Reactivate the existing `flow_sde_ppo` subagent for required continuation; do not duplicate the run or create a second owner. Flora, residual Stage 1 and unrelated allocations were not changed by this migration.
