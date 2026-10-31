# Command diagnostic CPU qualification — 2026-11-01

On Baige `job-6n2x5ay22eok`, the exact `06b14256` source passed **42 diagnostic + 42 curve tests**, with zero failures/errors/skips. The separate optional `bbb200ef` physical bridge passed 5/8 and failed 3/8. Each failure reached the synthetic fake motor, which lacks the `.training` attribute required by the existing frozen-model check. The optional bridge remains unqualified; no retry occurred.

The combined wrapper and scheduler correctly report failure because the optional suite failed. The independently complete 84-case result remains valid for its source. Actual pre-mask GPU count was zero, CUDA remained uninitialized, and cgroups enforced 2 CPU cores / 16GiB. The inclusive CPU charge was 93 seconds: prior 1,682 → 1,775 / 1,800, leaving 25 phase seconds (17 in this 110-second envelope). No actual model, cache, simulator or robot actions ran. This receipt does not grant GPU, fit, source or readout admission.
