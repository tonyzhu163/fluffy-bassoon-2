# How much does each reader change an interface that needs no correction?

Offline, exploratory. No simulator or VLA rollouts; three readers and three seeds at 2,500 updates each, about 0.1 GPU-hours on one RTX 4090 by scaling the wider-family run.

## Question

On the unmodified Panda interface the streaming reader cut native success from 19/32 to 3/32, with a median peak yaw correction of 64.6° ([findings](../../docs/FINDINGS.md)). Is a correction on an unshifted interface a property of the readers on clean scenes, and does a rule that leaves the interface alone when the estimated correction is small remove it without costing accuracy on real shifts?

The 96 locked UR5e supports are relabeled to the nominal interface (`nominal`), to four in-family draws each (`family`), and scored as recorded. Readers: SymMeta trained on the paper's seven interfaces and on continuously relabeled ones, the tuned Perceiver, and the paper's SymMeta and Perceiver checkpoints. For each reader the gate threshold is the one with the lowest operator error on the Panda and Sawyer traces, half relabeled to nominal and half in-family. The summary gives operator error with and without the gate, the share of traces left alone, and on `nominal` the mean and 90th-percentile absolute yaw estimate and the share above 15°.

## What would count against a gate

A nominal error that is already small without it, or a gate that raises error on `family` and `recorded` by about as much as it removes on `nominal`.

## Limits

- Clean UR5e scenes and three-chunk supports. The 19/32 to 3/32 regression was a streaming reader under LIBERO-PRO perturbations, which these traces do not contain. A small nominal error here does not show that regression is fixed.
- The gate is a threshold on the estimate, not a learned confidence.
- Operator error is not task success.

## Run

```bash
bash outputs/symmeta-false-correction-20261123-r1/run_false_correction_1gpu.sh <pin> <attempt> [stage3-attempt]
```

Writes `<attempt>/false-correction-summary.json` and `<attempt>/run.log`. A synthetic code-path check ran on CPU on 2026-11-23 (`--synthetic --step-scale 0.01 --seeds 41`); its numbers are not evidence. The launcher is the wider-family launcher with the script name changed and has not been run on Baige.

This run replaces a wider draft that also compared a closed-form reader with a soft prior on delay, short supports and gains outside the trained range. That comparison was dropped when the paper stopped claiming an architectural advantage.
