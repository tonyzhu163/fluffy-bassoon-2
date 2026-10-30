# First native RoboTwin AD/DPT export

The saved-data conversion completed on Baige CPU allocation
`job-2j83ht020h15`, source `9c27e027ffed1f66a5d56c79e142e7a55faba717`.
All 34 focused exporter/PPO/SAC tests passed in 3.71 seconds; the real SAC
source-prefix export exited 0. No robot actions, VLA queries or SGD were added.

| Retained source data | Count |
| --- | ---: |
| Source tasks | 6 |
| Complete K4 trials | 30, five per task |
| Complete episodes | 120 |
| Noise decisions | 1,539 |
| Executed primitive actions | 47,602 |

The paired AD and DPT datasets retain the same actual histories, outcomes and
native clocks. AD targets are executed behavior noise; DPT targets are the
frozen source-trained SAC actor's deterministic physical mean. Inputs retain
the native 2,062 features plus 1,024 previous physical noise coordinates and
four feedback/clock fields. Targets have 1,024 coordinates bounded by 1.5.
The source action sum matches the frozen global and per-task counters.

Artifacts remain on the project filesystem:

- Root: `/mowenjie/fluffy-bassoon-2/outputs/robotwin-native-export-20261030-r2/export/`.
- AD: `ad.npz`, 52,833,353 bytes, SHA256
  `b1ae47a6f31323ef65d252d1099ffbf8140bd5d018af3a7eafb428f71b17be9b`.
- DPT: `dpt.npz`, 52,912,660 bytes, SHA256
  `b59015234c00f4ed3ec6aa2e25302c80b7b357332fc57d9668c3604cb510f6d7`.
- Index: `index.json`, SHA256
  `29f5f72588530645a789f91735816268f412aebcffdb4d872c0fdc9d7d27e9b2`,
  pinning 7,667 input files. Both NPZ file hashes were independently rechecked.

Only the immutable five-round SAC source prefix was selected. Later source
records, validation, targets and native references are excluded. Copied source
data are charged once by the original collection ledger.

This qualifies conversion and saved-label correctness. Deterministic-mean
teacher competence, a fitted native student, native deployment and robot
adaptation results remain unqualified. The existing LIBERO trainer is
incompatible. The exporter owner is implementing the native full/static
trainer and frozen wrapper; no duplicate student implementation is started.
The first attempt's invalid reset-clock test is preserved under the r1 output;
the retry changes the test reference, preserving production clocks and code.

See [independent audit](receipts/export-audit.json), [tests](receipts/tests.log)
and [allocation receipt](receipts/runtime.json).
