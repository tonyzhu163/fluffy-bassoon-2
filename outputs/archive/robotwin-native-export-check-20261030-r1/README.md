# Native RoboTwin exporter qualification

Recorded 2026-10-30. The source owner executed Baige CPU-only allocation
`job-2j83ht020h15` at exact clean exporter commit
`9c27e027ffed1f66a5d56c79e142e7a55faba717`.
The allocation requested zero GPUs, four CPUs and 16 GiB RAM.

The attached logs record 34 focused exporter/PPO/SAC tests passing, followed
by a real SAC five-round source export: 1,539 paired AD/DPT decisions,
120 episodes, 30 complete K4 trials and 47,602 physical actions. Selected
native durations reconcile with the frozen source and per-task counters.
The converter used zero new robot actions, VLA calls or optimizer updates.

Remote output:
`/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-export-20261030-r2/`.

| Artifact | SHA256 |
| --- | --- |
| `export/ad.npz` | `b1ae47a6f31323ef65d252d1099ffbf8140bd5d018af3a7eafb428f71b17be9b` |
| `export/dpt.npz` | `b59015234c00f4ed3ec6aa2e25302c80b7b357332fc57d9668c3604cb510f6d7` |

This qualifies source-prefix conversion and causal deterministic saved-teacher
labeling. It does not establish deterministic teacher competence, student
learning, frozen robot performance or ICRL.

The first allocated attempt at `ce530bb16af99bd610ea6e12eaac0e75909597f2`
failed a test that incorrectly expected equal output at different public
episode clocks. The qualified pin corrects that expectation to compare each
episode against fresh-state replay of its own actual inputs. Production clock
semantics were unchanged.

Later explicit partition/case provenance and the native trainer/controller
are a separate candidate lineage starting at `54a8494`. Qualification of this
exporter does not qualify those later changes. Re-export the same frozen bank
with the enhanced exporter before native training; recollection is unnecessary.
