# RoboTwin stage50 and guard qualification — 2026-10-31

The separate Linux CPU gate passed, and the three direct source-policy
stage50 evaluations are complete. Neither result changes the earlier SAC5-bank
AD/DPT first-fit panel or completes the100-round source-training objective.

## Parent-death guard software qualification

Baige `job-k1upj2ycy60q` completed Success, exit0, on0GPUs/4CPUs/16GiB.
Scheduler time was71s, ending00:44:14 Shanghai: zero GPU-seconds. All123 tests
passed without skips:

| Package | Implementation commit | Passed tests |
| --- | --- | ---: |
| Source owner control | `c62be99615e797fbec108e1f1b5e232e4a591494` | 45 |
| Frozen-panel parent-death guard | `c166171b847ada9a20a9629a07d5a0100b517097` | 31 |
| Trained paired launcher | `c353970a316837b13277ec65587e58e5cb25ce56` | 47 |

The31-test guard phase covers the guard/wrapper/adapter checks, including
parent SIGKILL and race behavior, and took1.97s. Tests used owned temporary
processes and tiny synthetic CPU optimizer fixtures. Actual production source
signals, robot actions, VLA queries and production SGD were zero. Owner
evidence is committed/pushed at
`80943996d033155e17e242b045f82904c255b89f`.

That owner commit also retains the first frozen panel's results, independent
audits and terminal scheduler receipt. Their bytes match the earlier captured
hashes; the local qualification evidence also matches the committed files.

This supersedes the guard's earlier pending-gate status with **allocated Linux
CPU-qualified**. It does not establish later live admission or deployment,
whole-container teardown, or any model-performance claim. The completed
first frozen panel remains pinned to2fd2096 and is not repeated or modified.

## Direct source-policy stage50 results

| Source learner | Cumulative source actions at snapshot | Source learner updates / optimizer steps | Source validation |
| --- | ---: | ---: | ---: |
| PPO | 479,674 | 50 /250 | 4/24 |
| Separately trained reset PPO | 479,895 | 50 /250 | 2/24 |
| SAC | 502,356 | 314,500 /943,500 | 3/24 |

Each snapshot contains300 complete K4 trials. These action counts describe
the cumulative source snapshots; they are not additional charges introduced
by this amendment. PPO and SAC update counters have different meanings and
are not comparable training iterations. All three checkpoint file hashes
match their saved identities, source counters were unchanged during evaluation,
pending source trials were zero, and saved reports confirm source RNG
restoration.

| Stage50 target condition | adjust_bottle | move_stapler_pad |
| --- | ---: | ---: |
| Persistent PPO, normal | 7/8 | 0/8 |
| Same-weight PPO, memory reset | 7/8 | 0/8 |
| Same-weight PPO, reward blind | 7/8 | 0/8 |
| Separately trained reset PPO | 7/8 | 0/8 |
| Frozen SAC | 3/8 | 0/8 |
| Target-updating SAC | 5/8 | 1/8 |

All24 source-validation cases match across three arms. All16 target cases
match across six conditions in actual seed and initial feature/raw-observation
hashes. Consistent reset retries preserve pairing. Recorded episode actions
and chunk counts match raw manifests according to the owner audit. Native
before/after references used different seeds and only one episode per task;
they do not support a paired native comparison with this table.

SAC online uses a separate fresh-target replay clone per task. Bottle adds
1,340 target learner updates /4,020 optimizer steps over2,054 actions;
stapler adds1,920 /5,760 over2,941 actions. Source replay is excluded. These
updates are additional target adaptation cost, not progress toward the source
quota. Frozen SAC, PPO and separately trained reset PPO target trials add no
SGD.

The PPO controls show **no stage50 memory advantage**. Two development tasks
and one K8 trial per condition do not establish general adaptation. The
source50 SAC model is distinct from the early SAC5 teacher used to label the
AD/DPT bank; this receipt does not qualify that teacher as expert or relabel
the student data. Final100-round source endpoint results remain open.

## Provenance and scope

Scientific lineage is `db0184089fa88a5f96c153f9b80029bb91b77def`; stage50
execution is `e825821a7c2231952bcf1042162f3470bd8ecdea`. The owner's saved
stage audit was checked at00:56:17 Shanghai on2026-10-31 and performed saved
JSON/artifact SHA inspection without model computation, tests, process
control or ledger access. It streamed checkpoint bytes for hashes; initial
frame identities are saved recorder metadata, not a fresh raw-RGB rehash.

[Captured provenance](ROBOTWIN-STAGE50-GUARD-PROVENANCE-20261031.json) retains
source hashes, three source checkpoint identities/counters,12 target result
rows, pairing summary, qualification receipt and scheduler accounting. This
amendment independently checked JSON consistency for40 case groups, outcome
vectors, stage counters and qualification arithmetic. It did not repeat
checkpoint streaming or launch tests/models. The stage50 audit files are
captured local owner artifacts and are not asserted to be committed at8094399.

The owner reports controlled pauses of the PPO/reset source parents for a
finite-child/checkpoint audit before a planned trained-N6 migration, while
SAC continues. That is a dated owner report, not a fresh process-status claim
or permission to signal jobs here. Original100-round scope/caps remain binding.
This amendment changes no allocation, source checkpoint, budget or active run.
