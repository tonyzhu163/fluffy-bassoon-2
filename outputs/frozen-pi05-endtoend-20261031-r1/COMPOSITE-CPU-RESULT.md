# Corrected composite — allocated CPU qualification, 2026-10-31

Baige `job-mjw4gq2e94d3` failed on exact clean source `e5f685d14a10b8878036e62798d294200bfbfee4`. All16 component file hashes match the bound integration manifest, and the requested seven test files ran once: **87 passed, 4 failed**, no skips/errors,60.843 seconds. Actual pre-mask GPU count0, uninitialized CUDA and cgroups2CPU/16GiB are verified.

Three parametrized source-orchestration fixtures fail at `tests/test_robotwin_steering_source.py:266`: they compare the snapshot's empty `unresolved_decisions` mapping `{}` with integer0. The fourth is a real readout integration error at `steering_history_readout.py:320`: `CampaignBudget.remaining(phase=PHASE)` omits the required keyword-only `method`. That test stops before physical episode execution. These are exact failed-call evidence, without treating the composite as qualified.

The failed attempt charged134 conservative CPU-worker-seconds. Phase CPU use is551/1,800, leaving1,249 in total:466 remain in this same600-second qualification hold and783 are unassigned. No allowance was renewed, no retry submitted, and failed output/source bytes remain immutable. GPU/actions remain zero; no actual VLA, simulator, source collection, data collection or fit occurred. No capacity GPU is admitted. The exact JUnit/log/clock/ticket/attestation copies match their saved hashes. The previously passing components need no unrelated suite rerun; a successor must address the concrete failed paths.

![Composite CPU qualification failed](screenshots/composite-cpu-terminal.jpg)
