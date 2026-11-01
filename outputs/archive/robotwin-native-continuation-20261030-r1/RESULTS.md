# Trained source continuation — first real rollovers verified

All three source learners resumed on Baige at executor e825821 with original DB scientific lineage and unchanged100-round action/episode/trial contracts. All actual GPU/CUDA probes and native-noise parity checks passed on three distinct allocated devices. No trained weights, replay, pending PPO trials, optimizer or RNG state were discarded. The original selected cases and closed episode1 prefixes were preserved without repeated actions or gradients.

| Learner | Baige job | First verified boundary: complete rounds/K4 | Source actions | Updates / cumulative SGD |
| --- | --- | --- | --- | --- |
| Persistent PPO | job-zgdkrn0qqueh | 40 /240 |384,343 |40 /200 |
| Feature-conditioned noise SAC | job-gv77bzhyeroo |41 /246 |411,497 |255,800 per optimizer /767,400 total |
| Independently trained reset PPO |job-0sz3b9z9h7t5 |41 /246 |392,770 |41 /205 |

At these boundaries all1,188,610 physical source actions are retained in complete trials. The first slices added51,373 new physical source actions; the already-charged SAC239/reset223 prefixes became committed exactly once. Each child exited0 at a whole round with no pending PPO data. Complete coordinator and learner ZIP CRC/hash checks passed, the motor was unchanged, and the supervisor verified the next hash-bound manifest and ledger head before launching child0002. First child process times were730.49/781.60/803.81seconds, including startup/recovery/teardown rather than only collection. These are immutable boundary receipts, not the currently advancing cursors.

[Independent read-only first-rollover audit](checks/20261030-2325-first-rollovers.json) binds checkpoint identities, optimizer/source counters, full recovery files and handoffs. Each fresh native process keeps the existing trained state. This verifies the operational continuation; it does not establish the underlying native GIL root cause is fixed or demonstrate adaptation performance. Full100-round collection/evaluation remains in progress.
