# Independent RoboTwin R4 qualification audit

2026-10-30 Asia/Shanghai. Read-only audit of Baige **job-ynkqj1uxbvg5**,
source `1382e8dd9cfc40c66f93a49677e8e16cba16415e`, remote run
`robotwin-collector-speed-20261030-r4`. No new tests, simulator episodes, model
computations or GPU jobs were performed for this audit. Existing rollout
artifacts were read on Linux CPU with CUDA hidden.

## Supported findings

The [report](report.json) and [exit receipt](exit-code.txt) agree: exit 0,
status passed, **3,020 executed actions** inside a 3,251-action ceiling.
The two raw saved rollout groups independently total the same 3,020 actions,
across eight complete episodes with no external budget cuts. Both source
tasks (`handover_mic`, `place_phone_stand`) use two episodes per trial.

| Controller | Physical actions | Decisions per lane | Group wall time | Replay max logp/value error |
| --- | ---: | --- | ---: | --- |
| Persistent memory | 1,600 | 26 / 26 | 39.903 s | 0 / 0, both lanes |
| Episode-reset memory | 1,420 | 20 / 26 | 43.438 s | 0 / 0, both lanes |

Controller inference microbatch is 1; simulator process count is 2. Saved
policy inputs have dimension 3,090 (2,062 motor features plus 1,024 previous
noise coordinates plus four feedback fields). Saved actions have 1,024
coordinates. All four reported old-policy replay errors are exactly zero;
neither group performed an optimizer step. The report records an unchanged
frozen motor. The original checkpoint's before/after fingerprint values were
not exported, so this audit verifies the reported boolean and pinned check
implementation rather than independently recomputing the model fingerprint.

The [GPU probe](gpu-probe.json) passed in the named scheduler container. All
12 native renderer receipts bind to the same allocated PCI device,
`0000:63:00.0`. Two distinct worker PIDs are recorded per group. These receipts
and raw rollout summaries are included in [audit-snapshot.json](audit-snapshot.json).
The report, probe and exit file mirrors match their remote SHA256 hashes;
the snapshot records hashes for the original log and binary rollout files.

## Limits and unresolved runtime diagnostics

This was `--ppo-only`: scalar-render comparison, 1/4/8-lane timing sweep and
compiled-kernel inference arrays are empty. It therefore establishes no
paired rendering equivalence, compilation gain, optimal capacity or speedup.
The probe explicitly records `timing_comparison_eligible=false`. Descriptive
group rates are 40.10 and 32.69 actions/s; they are not controlled speedup
measurements. Reset time alone accounts for 21.09 and 25.26 seconds of the
respective groups, making reset overhead the largest recorded phase.

Final-observation capture was enabled. **The console contains 416 OIDN
`invalid handle` errors and eight `unsupported device type: CUDA` errors.**
They continued during successful collection. Exit 0 and reproducible policy
replay do not establish that the denoiser, image quality or renderer temporal
state is correct. Rendering is not a clean-runtime qualification; these
errors require diagnosis before a production migration claim. Four Python
tracebacks are optional `curobo` import failures, consistent with the selected
MPLIB planner; no terminal Python exception or segmentation-fault text is
present in this R4 log. Vulkan fallback and missing optional PyTorch3D warnings
also remain.

One reset-controller episode succeeded at action 220; all other episodes
reached the 400-action wrapper limit without success. This tiny untrained
probe does not compare controller performance: exploration draws were not
paired across arms. It also does not qualify the production six-task corpus,
native task-specific horizons, K4/K8 trials, SGD updates, learning improvement,
or ordinary resume/admission of production data. The legacy multi-lane driver
remains explicitly qualification-only.
