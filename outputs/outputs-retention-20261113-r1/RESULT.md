# Baige output archive and PFS release

Completed November 13, 2026 Shanghai time. All 14 archive parts passed a full BOS readback and an independent restore to staging before 6,166 source files were released. PFS gained 47.03 GiB of allocated space, from 221.71 GiB free before release to 268.74 GiB at verification. The BOS payload objects occupy 40.54 GiB.

| Released from PFS | Allocated GiB | BOS payload GiB |
| --- | ---: | ---: |
| Two completed RoboCasa OpenDrawer TEST512 panels | 11.22 | 9.02 |
| Sidney RoboTwin PPO and GRPO step-one checkpoints | 20.81 | 20.38 |
| Sidney RoboTwin PPO and GRPO reset records | 10.55 | 6.69 |
| Sidney RoboTwin PPO and GRPO sampling records | 4.45 | 4.45 |

Both TEST512 panels still have 3,102 JSON records and two logs apiece on PFS. The two Sidney initial checkpoints and the two step-two checkpoints each have all 30 expected files, including the four DCP shards. LIBERO was outside the release plan. The owned Baige scheduler showed one active KV ICIQL evaluator with its own BOS banks and shared policy inputs. The release ran without intervening in that job.

[Plan](PLAN.json) · [Verification and per-part BOS receipts](VERIFICATION.json) · [Restore instructions](RESTORE.md) · [BOS audit](BOS-AUDIT.md)
