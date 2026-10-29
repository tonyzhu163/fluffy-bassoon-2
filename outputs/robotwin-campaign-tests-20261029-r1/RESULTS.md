# Shared-budget and raw-corpus integration checks passed

Baige `job-ytyeswlw54wg`, immutable source
`065777ceccf1c52a69d7def92399e35774b4bddd`, exited0. The actual one-GPU
allocation/CUDA probe passed. **84 tests passed** in12.67seconds across all9
focused files. The only warning is an unrelated installed Dash pytest hook
deprecation.

Checks include total/per-method action ceilings and prior-receipt deduplication,
conservative unresolved-intent recovery, exact final action cuts, lossless raw
RGB/state/noise/action/outcome round trips, native-feedback copy/identity
preservation, real PPO behavior likelihood/value/history metadata, immutable
acting-policy snapshots, loaded motor/runtime admission, and an interrupted
driver that resumes validation/native/target evaluation without repeating
completed units or spending those actions again.

These are allocated runtime/unit/integration checks using explicit fake
simulator fixtures where needed; this job performed **zero robot-control
actions**. The separate real-simulator camera parity gate passed before these
changes. End-to-end raw recording plus actual learner updates remain startup
requirements for the next source-training slice. No learned-performance or
meta-learning claim follows from this test result.

The prepared next command is
`scripts/diagnostics/robotwin_campaign_train.sh` at this exact source commit,
using the shared1Mledger,272k source actions per arm,8K4trials/PPOupdate and
the qualified final-camera mode. The proposed two-hour,one-GPU slice requires
the resource approval described in the [campaign protocol](../robotwin-campaign-1m-20261029-r1/PROTOCOL.md).
