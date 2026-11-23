# Where a closed-form inner loop differs from a tuned Perceiver

Offline, exploratory. No simulator or VLA rollouts; about 0.5 GPU-hours on one RTX 4090 (112,500 updates, the size of the wider-family run).

## Question

The [wider-family run](../symmeta-wider-family-20261123-r1/RESULT.md) showed that a Perceiver trained on continuously relabeled interfaces with a lower rate and warmup matches SymMeta's yaw error on the paper's held-out shifts. Does a closed-form inner loop still do anything that Perceiver does not? Four comparisons on the 96 locked UR5e supports, same relabeling and operator error as that run:

1. **Number of training interfaces.** SymMeta and the tuned Perceiver trained on the paper's 7 interfaces, on 16 and 64 fixed in-family draws, and on a fresh draw per trace. Every set is produced by relabeling the same 1,466 Panda and Sawyer traces, so only interface diversity changes.
2. **Unshifted interface.** Every reader's operator error when the true interface is the nominal one (`nominal`). A reader-agnostic gate, which leaves the interface alone when the estimated correction is below a threshold chosen on the source arms, shows what abstaining costs on real shifts.
3. **Support and range changes.** Readers trained at 3 chunks, scored at 2 and 1; in-family yaw with gains of 0.3–0.5 and 2–3, outside the trained 0.65–1.5.
4. **Delay and wider families.** A new reader, `soft`, against Perceivers given the same augmentation (half the training traces delayed by one chunk; all families for the map-output Perceiver).

## The `soft` reader

[apply_soft](eval_soft_inner.py) keeps SymMeta's command-blind visual vectors and confidences. The frame rotation is a weighted Procrustes solve from proprioceptive travel to those vectors, which command timing does not enter. The in-frame 2x2 map is [soft_shape](../../src/fluffy_bassoon/xemb/wider_family.py): least squares from commands to travel, shrunk towards the best rotation-plus-gain fit of the same pairs, so the data decide only the directions the commands excite. It is solved at command lags of 0 and 1 chunk and mixed by how well each lag is explained. Beyond the SymMeta reader it learns five scalars: the ridge, three for the lag evidence and a yaw concentration scale, which is the only parameter the yaw likelihood trains. The summary reports whether that concentration and the share of the map supplied by the prior track the actual error.

## What would count against a closed-form inner loop

The tuned Perceiver matching SymMeta at 7 and 16 training interfaces, at 1 and 2 chunks and at out-of-range gains, and its delay-augmented variants matching `soft` on the delay and wider-family conditions. Then the solver has no measured advantage and the paper's reader should be the Perceiver with relabeling. What would support it: the Perceiver needing many interfaces or degrading off its training support while the closed-form readers do not.

## Limits

- Stored traces hold one token per 5-control chunk and end at 3 chunks. Per-control tokens and learned pairing across more than one lag are not tested. A delayed trace has two usable command–motion pairs.
- The unshifted test uses clean UR5e scenes. The 19/32 to 3/32 regression of the streaming reader on the native Panda interface was under LIBERO-PRO perturbations, which these traces do not contain.
- The 7-interface sets are relabeled from the stage-3 traces, not the paper's original 1,344-trace archive; the schedule-only control on that archive is the separate [seven-schedule run](../symmeta-seven-schedule-20261123-r1/PLAN.md).
- SymMeta and `soft` use a peak rate of 0.0015 and the Perceivers 0.0003 with 200 warmup updates, each reader's best known schedule. `soft` adds 200 warmup updates.
- Relabeled commands can leave the unit box, strongly so for gains of 0.3–0.5. Operator error is not task success.

## Run

```bash
bash outputs/symmeta-soft-inner-20261123-r1/run_soft_inner_1gpu.sh <pin> <attempt> [stage3-attempt]
```

Writes `<attempt>/soft-inner-summary.json` and `<attempt>/run.log`. A synthetic code-path check of four models ran on CPU on 2026-11-23 (`--synthetic --step-scale 0.01 --seeds 41`); its numbers are not evidence. The launcher copies the wider-family launcher with the script name changed and has not been run on Baige.
