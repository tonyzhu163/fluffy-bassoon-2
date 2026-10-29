# Final single-seed ML10 comparison, 2026-10-29

All four original jobs completed 80 rounds and 10M training transitions. Final saved-artifact audit: 2026-10-29T08:38:51.974111+08:00 on Baige. Source `cd0128f03a060aff2d538a91f35058aa822df845` remained clean and immutable. No training or model execution occurred during this audit.

## Final normal evaluation

| Method | Train success | Held-out success | Active GPU h | Training GPU h | Launcher GPU h | Parameters | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CTM T1 / cached100 | 16.0% | 0.0% | 2.0236 | 1.2897 | 2.0250 | 237,347 | 1.687 |
| RL²-PPO / GRU64 | 24.8% | 0.0% | 0.8367 | 0.4005 | 0.8383 | 21,960 | 0.668 |
| ECET | 24.8% | 21.6% | 9.0709 | 8.5437 | 9.0722 | 69,194 | 0.147 |
| AMAGO1 / 10 replay passes | 12.8% | 0.0% | 2.6583 | 1.2001 | 2.6625 | 5,403,181 | 27.946 |

Active GPU-hours are measured learner wall time on one reserved RPBZZZ6, including CPU simulation, optimization, periodic evaluation and final paired evaluations. Launcher time also includes launch/probe overhead; neither is a scheduler billing estimate. Training-only time excludes evaluation. Plot endpoints charge all final evaluation time; the repeated final normal outcomes replace the identical round80 normal point. AMAGO has 4,179,817 trainable and 5,403,181 total parameters. Peak memory is CUDA allocated memory, not reserved memory or total device usage.

## Paired final conditions

| Method | Train normal / reset / blind | Held-out normal / reset / blind | First K5 normal train / held-out |
| --- | --- | --- | --- |
| CTM T1 / cached100 | 16.0% / 18.4% / 14.0% | 0.0% / 0.0% / 0.0% | 16.0% / 0.0% |
| RL²-PPO / GRU64 | 24.8% / 22.8% / 16.0% | 0.0% / 0.0% / 0.0% | 28.0% / 0.0% |
| ECET | 24.8% / 30.0% / 23.2% | 21.6% / 21.2% / 6.4% | 24.0% / 22.0% |
| AMAGO1 / 10 replay passes | 12.8% / 12.0% / 11.2% | 0.0% / 0.0% / 0.0% | 10.0% / 0.0% |

ECET ties RL² at the final training-family endpoint while costing about 10.84 times as many active GPU-hours. Only ECET has nonzero final held-out success in this four-method screen. Its normal/reset held-out scores are close (21.6/21.2%); training-family reset is higher (30.0/24.8%). Reward-blind held-out success drops to6.4%. These controls do not demonstrate a persistent-memory advantage. CTM is below RL² on final training success, uses more time, and has zero final held-out success. The requested advantage over the strongest baseline on training performance, generalization or training efficiency is not established for this CTM setting.

These are single-seed observations, with ten evaluation lanes and correlated episodes, rather than confidence intervals or architecture rankings. ECET evaluation fluctuates substantially; its best intermediate checkpoint is not selected as the final endpoint.

## Work and recovery audit

| Method | Baige job | Optimizer steps | Forward / replay sample transitions | Final checkpoint SHA256 |
| --- | --- | ---: | ---: | --- |
| CTM T1 / cached100 | `job-rczwk9dd1v1e` | 4,000 | 100,000,000 | `a34dff8decfef013bde1f74310c67709b5ae0689a2b1f89ed49848db51a58b18` |
| RL²-PPO / GRU64 | `job-j400lp4d3m6m` | 1,600 | 100,000,000 | `a9634a8124ffb129211c00ea43d80028b3abdb2c6a97ddd8d0901d737bb03de3` |
| ECET | `job-wrlr0utzsayo` | 6,251,200 | 200,000,000 | `824de988a0ac8c3c78aeb3d28d2d9e5ea77e919ce2a99697487a09bce331c80a` |
| AMAGO1 / 10 replay passes | `job-dm64v27p0a30` | 2,083 | 99,984,000 | `e0ea9c8d542adfded6fdc656bd3fea5f3bfd21e1d75d5399ee8627026fd31532` |

Each method used exactly 10M new training transitions and5M evaluation transitions. All160saved evaluation conditions (40per method) were independently recomputed from binary10-by25raw outcome arrays, checking equal family weighting, per-episode curves, family labels, finite native-return arrays and repeated final normal arrays. Training rounds are contiguous1–80; PPO/sample exposure, final counter summaries, complete checkpoint hashes, GPU probes and successful qualification identities match. Restricted standard-library metadata inspection verified policy/normalizer, optimizer, sampler and Python/NumPy/Torch/CUDA RNG recovery without importing Torch or reading tensor storage. AMAGO additionally retains replay, scheduler, scaler and exploration recovery. Its unchanged terminal checkpoint hash also preserves the previously audited immutable replay manifest.

ECET has independent nonempty actor/critic AdamW states (49/7state entries), 6,251,200total optimizer steps and200Mcombined actor/critic forward samples. Its terminal checkpoint counters include all5M evaluation transitions and exactly match summary.json. The temporary periodic checkpoint evaluation lag seen during monitoring is absent at completion. Launcher-active time32660seconds is below its12-hour hard cap; all other original methods remain below their6-hour caps.

The initial terminal monitoring parser expected a training field on the final-only evaluation row. The terminal-aware saved-artifact audit supersedes it; no training failed and no allocation or retry was involved. Source timer nesting explains small subsecond-to-one-second differences between summed inner collection/update timers and training totals; exact final training totals match stored GPU-hour counters.

## Protocol limits

Seed1000 only; common Meta-World3.1.1 goal-visible task pools,50balanced lanes, five500-step episodes per training trial,80collections, eight simulator workers,16CPUs/64GiB and one RPBZZZ6per method. Evaluation covers fresh training-family instances and five held-out families, every5collections plus initial/final, ten balanced lanes and25episodes. Every saved family and episode has equal weight.

Ten PPO epochs do not equalize optimizer steps or gradient horizons. CTM uses cached100 TBPTT; RL² is the qualified garage-inspired PPO adaptation of an originally TRPO method; ECET retains native minibatch32, separate actor/critic learners, gamma.9 and sampled short context; AMAGO1 uses the accepted ten sampled replay-pass schedule adaptation (2,083steps/99,984,000samples), rather than PPO. Models and other method-specific hyperparameters differ. This is not a METTA reproduction or a controlled architecture-only comparison.

ECET receives no public-clock features by its original encoder contract. CTM/RL²/AMAGO retain their training clock encoding during K25evaluation, beyond the K5training range; AMAGO also uses its documented positional lookup extension. First-K5companion scores expose this limitation without replacing the K25headline. The separate frozen-clock interventions and1.25Mwarm continuations are excluded from these curves, costs and endpoints.

## Artifacts

- `checks/20261029-0835/completion-audit.json`: full raw-condition/counter/recovery audit.
- `checks/20261029-0835/audit_saved_artifacts.py` and `plot_final.py`: reproducible saved-artifact inspection and plotting.
- `figures/ml10-train_eval-gpu-hours-20261029-0835.png` and `.pdf`: requested training success versus active GPU-hours.
- Matching held-out, training-only-time and environment-step PNG/PDF companions, and `ml10-first5-gpu-hours-20261029-0835.*`.
- `figures/ml10-final-curves-20261029-0835.csv`: observed curve coordinates, including first-K5scores.

All original full checkpoints, failed admissions and immutable source worktrees are preserved. The narrow four-method monitor was deleted after the final artifact mirror was verified. No original comparison work remains; other campaigns retain their own monitors.
