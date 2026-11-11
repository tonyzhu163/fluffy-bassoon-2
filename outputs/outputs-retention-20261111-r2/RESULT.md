# Baige output retention, November 11, 2026

**Reclaimed 470.49 GiB from PFS.** Free space rose from
**213.02 GiB to 683.50 GiB**.
No jobs were stopped.

| Action | Payload | GiB released |
| --- | --- | ---: |
| Deleted | Four unused step-3 tensor files from the failed MIKASA r3 comparison qualification | 63.79 |
| Removed duplicate copies | MIKASA r2 and r3 failed-qualification initializers, both recoverable from the identical verified BOS object | 21.80 |
| Archived | Completed MIKASA reset, stream, and looped/recurrent carry checkpoints and selected initializers | 246.74 |
| Archived | Older Sidney RoboTwin `train03`, `train06`, and `train07` payloads | 51.13 |
| Archived | RoboCasa stage-1 GRPO resume predecessor | 15.96 |
| Archived | Step-1 and step-2 checkpoints from both critic-parity arms | 63.87 |
| Archived | Cited evidence from three RoboCasa qualification groups | 7.19 |

The archive covers **4,788 payloads in 54 verified BOS
objects**, totaling **384.89 GiB** of original PFS
allocation. All four failed BOS transfers passed full readback and restore on
retry. The initial unverified objects were removed. The original compact
configs, logs, numerical results and cold pointers remain available.
[Verification](VERIFICATION.json) · [Restore instructions](RESTORE.md).

## Experiment outputs after this pass

| Project | Audited before, GiB | After verified releases, GiB |
| --- | ---: | ---: |
| `fluffy-bassoon-2/outputs` | 757.45 | 619.30 |
| `fluffy-bassoon/outputs` | 489.79 | 157.46 |
| `worldrank/runs` | 17.25 | 17.25 |
| **Total** | **1,264.49** | **794.01** |

These output totals subtract this pass's measured releases from the
[full census](../storage-audit-baige-20261111-r1/RESULT.md). Concurrent changes
outside the selected paths are reflected in the live PFS free-space reading.

The priority hot runs are RoboTwin Sidney PPO `ppo02` and GRPO `grpo01`, and
the RoboCasa stage-1 PPO and GRPO final endpoints. The held critic comparison
retains both complete step-3 states and its initializer for evaluation.
The large remaining RoboCasa output roots also include the canonical Flow10M
teacher lineages and source5 training inputs. Paused steering recoveries and
other current or queued inputs remain on PFS. The original project's
remaining output allocation includes MIKASA campaign inputs and recoveries
outside the completed checkpoint selections above.
[Hot checkpoint check](HOT-CHECK.json) · [Selection](PLAN.json) · [Additional selection](ADDITIONAL.json).

The qualification deletion kept the original failure logs and small run
records. The earlier MIKASA step-2 tarball is still cold. RoboCasa qualification
artifacts cited by reward replay, numeric checks or the snapshot audit were
preserved in BOS. [Deletion receipt](MIKASA-QUALIFICATION-DELETION.json) ·
[r2 duplicate](MIKASA-R2-DUPLICATE-INITIALIZER.json) · [r3 duplicate](MIKASA-DUPLICATE-INITIALIZER.json).
