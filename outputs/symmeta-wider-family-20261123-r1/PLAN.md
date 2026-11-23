# SymMeta inner loops on interfaces outside yaw/gain/sign

Offline, exploratory. No simulator or VLA rollouts; under 1 GPU-hour.

## Question

SymMeta's solver inverts the three-parameter function that generates the benchmark shifts. Does a more general inner loop on the same learned reader match it inside that family and beat it outside? And does a Perceiver trained on continuously sampled interfaces close the gap to the fixed solvers?

## Method

Recorded 15-control support traces are relabeled to sampled planar interfaces `R(yaw) @ shape` by remapping the issued commands and the frame of proprioception ([wider_family.py](../../src/fluffy_bassoon/xemb/wider_family.py), tested in `tests/test_xemb_wider_family.py`). Images are unchanged, so the relabeled trace is physically exact for the remapped commands. Families: the paper's (`family`), unequal axis gains (`aniso`), shear, axis swap, a general 2x2 map including reflections, and a one-chunk (250 ms) command delay on the recorded traces.

- Source: the 1,466 Panda and Sawyer stage-3 traces. Test: the 96 locked UR5e supports, 4 interface draws per trace and family.
- Fixed solvers on frozen readers (the paper's seed 41–43 checkpoints and a reader retrained with continuous relabeling): the paper's Procrustes and scale head; `factored`, a least-squares 2x2 map from commands to proprioceptive travel plus a Procrustes frame rotation from travel to visual vectors; `visual`, one 2x2 least-squares map from commands to visual vectors. Each has an unregularized form and a ridge form that shrinks to the paper's solution, with the ridge chosen on the source arms.
- Learned inner loops, 3 seeds: the paper's Perceiver trained with continuous in-family relabeling at the sweep's schedule, with warmup and a lower rate, and with 4 times the updates; and a Perceiver that outputs the 2x2 map, trained in-family and on all families.
- Score: `|A inv(A_hat) - I|_F / sqrt(2)`, the planar command map the frozen policy is left with after correction, next to the same quantity with no adaptation. Yaw error is reported where a yaw is defined, and the recorded condition repeats the paper's OOD yaw number as an anchor.
- Fixed solvers are also run at 6, 12 and 24 chunks where the test supports have them.

## What would count against the current design

A general solver or the map-output Perceiver matching Procrustes on `recorded` and `family` and beating it on the other families at 15 controls. What would support it: the general solvers losing accuracy in-family at 15 controls because three policy commands rarely span the plane (the summary reports the command spread), and recovering only with longer support.

## Limits

- Relabeled commands follow the remapped distribution, not what the policy would issue under the new interface, and can leave the unit command box; the summary reports that fraction.
- Recorded commands that were clipped by the simulator are treated as unclipped.
- Operator error is not task success. Run closed-loop success only where this ranking is not decisive.
- Stored traces hold one token per 5-control chunk, so per-control Perceiver tokens are not tested here.

## Run

```bash
bash outputs/symmeta-wider-family-20261123-r1/run_wider_family_1gpu.sh <pin> <attempt> [stage3-attempt]
```

Writes `<attempt>/wider-family-summary.json` and `<attempt>/run.log`. A synthetic code-path check ran on CPU on 2026-11-23 (`--synthetic --step-scale 0.01`); its numbers are not evidence. The launcher has not been run on Baige.
