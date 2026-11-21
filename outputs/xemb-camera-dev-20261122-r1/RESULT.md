# BASE and S15 on simulated UR5e

With frozen π₀.₅, active EquivFlow and active DIS each succeeded on all four BASE queries and failed on all four matched S15 queries. An oracle given the exact command transform showed the same four paired losses. Even perfect command calibration was insufficient for these camera-shifted rollouts.

| Command calibration | BASE native success | S15 native success | BASE mean absolute yaw error | S15 mean absolute yaw error |
| --- | ---: | ---: | ---: | ---: |
| Active EquivFlow-Procrustes | 4/4 | 0/4 | 22.41° | 18.03° |
| Active DIS + Procrustes | 4/4 | 0/4 | 8.21° | 12.87° |
| Nominal identity | 1/4 | 0/4 | 45.00° | 45.00° |
| Exact command transform | 4/4 | 0/4 | 0.00° | 0.00° |

DIS had lower mean yaw error under both cameras. The EquivFlow-minus-DIS gap narrowed from 14.20° on BASE to 5.16° on S15, a paired change of −9.04°. It narrowed on three of four pairs. This goes against the planned hypothesis that S15 would widen the gap. The two readers tied on native success on every query. Both views presented to the frozen motor also changed under S15. The oracle outcomes therefore limit what native task success can distinguish about the readers here. [Paired results](evidence/summary.json) · [Predictions, outcomes and controls per case](evidence/cases.json)

## Scope and execution

This development comparison used one LIBERO Goal task, two support/query layout pairs (11/13 and 12/14), and two oblique yaw/gain settings (+45°/1.0 and −45°/0.75). Both readers used the same 15-control support per cell. The EquivFlow checkpoint used training seed 41. DIS projection and nominal gain were fitted on source Panda/Sawyer tapes. Each query used a separate start from support. Initial physics, probe commands and query noise matched across cameras. All four methods shared the reset and opening policy proposal within each cell. Four matched queries per camera support a description of this panel only. [Plan](PLAN.md) · [Reader provenance](evidence/readers.json) · [Admitted cases](evidence/admitted.json)

Baige Distributed Training job `job-i9lfqfmnkmr6` ran on one Pro6K at `034cbf44`, completed 32/32 episodes and exited 0. The launcher recorded 442 seconds (0.123 GPU-hours). Full native episodes, issued commands and compressed support tapes remain on shared Baige storage. [Artifact paths and checksums](evidence/manifest.json)
