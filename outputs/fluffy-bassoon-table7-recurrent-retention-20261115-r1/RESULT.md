# Baige Table 7 and recurrent-flow archive

On November 15, 2026, the archive released **45.73 GiB** of allocated PFS space. Shared PFS free space rose from **122.47 GiB to 168.07 GiB**, measured at 16:46 Shanghai time after both batches.

| Scope | Paths released | BOS GiB | PFS GiB reclaimed |
| --- | ---: | ---: | ---: |
| Table 7 r1, r2 and r3 initializers, handoff and qualification payloads, and training checkpoints through r3 step 10,000 | 49 | 44.42 | 44.42 |
| Recurrent-flow seed 1000, 1001 and 1002 initializers | 3 | 32.71 | 0.00 |
| Checkpoint payloads from six recurrent-flow carry/reset training runs | 58 | 1.31 | 1.31 |

The three recurrent-flow initializer files still have hard links in the separate T1 flow run. Those links keep **32.71 GiB** allocated and allow the T1 slim checkpoints to load. The archive stores each original initializer with the SHA-256 in its source receipt.

All BOS objects passed SHA-256 readback and a restore to separate scratch. The Table 7 r3 step 10,000 model and full training-state hashes also match the original recovery audit. The run directories on PFS retain their configurations, logs, audits and paired-cue traces. The scheduler check found no active consumers of the selected paths.

[Verification manifest](VERIFICATION.json) · [Restore instructions](RESTORE.md)
