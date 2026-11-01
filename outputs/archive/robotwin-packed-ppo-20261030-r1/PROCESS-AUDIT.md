# Packed simulator process audit

Read-only terminal audit of Baige `job-230io8i5cmpd`, run
`robotwin-packed-ppo-20261030-r1`, source
`0a116e745cdbd004b3ab4b7cca3f8fed40bd9604`. The remote worktree matches
that commit. No jobs, retries, source edits or ledger initialization were
performed by this audit. Raw artifacts remain under
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-packed-ppo-20261030-r1/`.

| Measured result | Persistent PPO | Reset PPO |
| --- | ---: | ---: |
| Child simulator processes | 6 | 6 |
| Completed K4 trials / episodes | 6 / 24 | 6 / 24 |
| Physical source actions | 10,510 | 10,800 |
| Collection time, including resets, corpus and motor | 235.551 s | 236.176 s |
| Measured source actions/s | 44.619 | 45.729 |
| Replay log-probability/value max errors | All six 0 / 0 | All six 0 / 0 |
| PPO updates / optimizer steps | 1 / 5 | 1 / 5 |
| Parent exit code | 0 | 0 |

All twelve distinct worker PIDs returned four native reset receipts each.
Every actual renderer PCI was the allocated `0000:a2:00.0`; every reset
receipt reported zero policy actions. Both idle parent scalar scenes were
released before spawning the pools. Each group reached `checkpointed`,
and the ordinary-resume fence was absent at terminal inspection. Ledger
charged and actual actions matched group physical counts, with zero pending
or uncertain actions and no unresolved decisions.

The packed envelope took 412.824 s for 21,310 source actions, or 51.620
actions/s including model startup and both PPO updates. The roughly
one-second GPU sampler observed a peak of 70,385,664,000 bytes
(65.552 GiB) out of 102,641,958,912 bytes. This is below the proposed
70–80 GiB occupancy target. Sampled GPU utilization peaked at 100%, with
a 24.621% mean over the complete envelope; this does not show sustained
GPU saturation. Final sampled allocation was zero bytes with no parent PIDs.

The native consoles were not error-free: persistent/reset logged
1,431/1,467 OIDN invalid-handle errors, plus 25 unsupported-CUDA device
errors each. Neither console contained GIL, fatal-Python, segmentation-fault,
RuntimeError or ProductionVectorError text. Optional Curobo imports failed
seven times per console. Requested seed 1,499,316,795 retried to
1,499,316,799, and 1,719,334,128 to 1,719,334,130, identically in both
learners; actual seeds are preserved in native receipts.

This verifies one fresh-controller packed collection/update round with
independent simulators and B1 motor/controller execution. It does not
establish a speedup against a matched scalar baseline, optimal process
capacity, long-run native stability, sparse/full RGB equivalence, a fix for
the GIL failures in separate production jobs, or migration of a trained
production recovery. Raw terminal metrics, artifact SHA-256 values, phase
timings, renderer receipts and limits are in
[independent-process-audit.json](independent-process-audit.json).
