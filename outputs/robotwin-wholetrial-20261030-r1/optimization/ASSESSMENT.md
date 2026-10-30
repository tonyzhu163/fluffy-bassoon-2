# RoboTwin batching assessment

Checked 2026-10-30 17:00 Shanghai. These are allocated Baige GPU measurements;
they do not establish a production training speedup.

| Spawned simulator lanes / VLA batch | Actions/s including resets |
| --- | ---: |
| 1 | 18.947 |
| 4 | 44.924 |
| 8 | 76.843 |

The first benchmark therefore shows 4.06 times throughput at batch eight
against its own batch-one reference. Our production collector already uses
final-camera capture, so the older full-to-final-camera 2.49 times gain must
not be counted again. The benchmark excludes production corpus I/O and does
not use the heterogeneous native task limits. Batch eight also does not divide
the fixed global batch of six complete K4 trials.

The initial PPO replay failed. The corrected narrow two-lane check at source
`1382e8dd9cfc40c66f93a49677e8e16cba16415e`, Baige
`job-ynkqj1uxbvg5`, executed 3,020 actions and passed the original likelihood
and value replay tolerances with all measured errors exactly zero. The small
controller executes one lane at a time; the motor and simulators can batch.
The frozen motor was unchanged. No SGD update was performed.

Persistent PPO collected 1,600 actions in 39.903 seconds; reset PPO collected
1,420 in 43.438 seconds. Scene resets used 21.088 and 25.262 seconds,
respectively, over half the measured time. This is a remaining bottleneck.
The console also contains OIDN invalid-handle/unsupported-CUDA warnings;
replay agreement does not qualify image quality or production rendering.

The optimization owner retains sole production integration ownership against
the active `db0184089fa88a5f96c153f9b80029bb91b77def` collector. Required
integration includes native 600/500/400 horizons, six complete K4 trials per
round, physical reservations, lossless corpus/native-action hooks and full
atomic recovery of pending PPO, optimizer, sampler and RNG state. Existing
three training jobs continue on their immutable source. A future migration
requires a reviewed exact pin and corpus-inclusive end-to-end measurement.

Receipts: [initial benchmark](r3-report.json), [corrected replay](r4-report.json).
The original 12,000-action qualification envelope covers all attempts once;
the owner must reconcile its final scheduler time and reservation before any
new qualification allocation. The human has not authorized a 10M source
expansion.
