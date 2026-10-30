# Earlier 400-step run: terminal audit

Baige job `job-340yo6ui9wmf`, source `c3b6281`, ended on 2026-10-30 at
12:53:51 Shanghai. Scheduler duration: 112m11s / 6,731 GPU-seconds. Exit 0,
controlled wall stop. Prior allocated use was 2,672 seconds, giving 9,403
seconds in total. This does not complete its original 128k-per-arm target.
The human subsequently requested the new native-horizon whole-trial recipe;
retain this bank and do not automatically restart its older contract.

| Source arm | Complete K4 trials | Stage-cut trials | Episodes | Actions | Source updates |
| --- | ---: | ---: | ---: | ---: | --- |
| Persistent PPO | 27 | 1 | 109 | 39,488 | 3 PPO updates / 15 optimizer steps |
| SAC | 27 | 1 | 109 | 39,138 | 15,560 actor, critic and temperature steps each |
| Trained-reset PPO | 27 | 1 | 110 | 40,179 | 3 PPO updates / 15 optimizer steps |

Total: 81 complete K4 trials, 328 source episodes and 118,805 source actions.
Collection successes were 16/109, 17/109 and 16/110 respectively. These are not
controlled estimates of policy improvement. All 107 common initial cases
matched their public-feature hashes across the three arms in this one process.

The 32k-per-arm frozen source/target/native evaluation stage completed, with
its snapshots and cursors retained. It uses one K8 trial per target/condition
and 400-step limits, so it is not the expanded native-horizon experiment or a
robust generalization result.

All-phase corpus: 501 resets including preflight, 5,793 complete chunks,
177,368 new executed actions, and no unresolved decisions. Storage: 2,155,947,840
frame bytes, 120,828,132 decision-array bytes and 2,205,859,196 acting-weight
bytes. The prior ledger charges 190,394 actions including 13,026 earlier ones:
118,805 source, 25,293 source validation, 31,670 target and 1,600 native actions.
No pending or uncertain physical charge remains.

The read-only CPU audit verified SHA256 and ZIP CRC for individual and
coordinator recoveries, matching source counters, all three SAC optimizers at
step 15,560, replay with 1,278 rows, private RNG streams and target critic.
Both PPO optimizers are at step 15 with four pending on-policy trials each.
Global RNG, samplers and evaluation cursors are present; the motor fingerprint
is unchanged. See [recovery audit](RECOVERY-AUDIT.json). Raw data and complete
checkpoints remain protected on Baige; local checkpoint copies are retained
outside Git.
