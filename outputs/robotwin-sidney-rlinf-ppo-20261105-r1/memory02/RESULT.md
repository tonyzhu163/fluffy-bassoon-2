# Batch-end trim leaves a large reset peak

The64-scene job `job-p7hd216qprbd` completed the baseline and native batch-end
trim modes, then reached its worker timeout during the per-scene-close mode's
last camera read. No VLA/learner was loaded and no scientific update ran.

| Mode | Reset peak RSS (GiB) | RSS after reset (GiB) | Live glibc (GiB) |
| --- | ---: | ---: | ---: |
| Baseline, first real reset | 55.06 | 55.34 | 53.07 |
| Native trim at batch end, second reset | 106.18 | 55.48 | 53.25 |

The200ms peak sampler shows why post-reset RSS alone is insufficient. With
batch-end trimming,52.12GiB of freed glibc allocations remain in arenas, but their
resident pages are returned by the end. The transient peak is almost twice the
first-reset footprint. Four times this106.18GiB peak plus conservative184GiB
model/runtime/rollout/checkpoint allowances is608.73GiB, above the unchanged
486.4GiB Ray threshold for a512GiB allocation. This rejects batch-end trim alone
for the proposed retry. It is an admission projection, not a full-phase memory
measurement.

Both completed modes passed nonblank raw-camera and finite-state checks, plus
retained RGB/state, Python/NumPy/Torch RNG, scene metadata and motor-input
identity around manual GC/trim. Per-scene reset peak and paired reset/motor
checks did not complete and are not admitted. Cleanup returned process RSS to
3.31GiB with0.92GiB live glibc allocations. A fresh paired-only diagnostic will
supply the missing result.

All6,400physical holds and2,560setup/reset allowances (including the unresolved
last reset) remain charged to originalrank0, totaling8,960for this attempt and
80,640cumulative rank0. Scheduler duration1,750GPU-seconds; creation to release
1,762; worker1,717. Source`c7f68c1e0c0280a3efdf91866cb1b31e7b8f5c91`.

Raw evidence: [RESULT.json](RESULT.json), [EXIT.json](EXIT.json),
[GPU probe](gpu-probe.json), [profile log](profile.log).
