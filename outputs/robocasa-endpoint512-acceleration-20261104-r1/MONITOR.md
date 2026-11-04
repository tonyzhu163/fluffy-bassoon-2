# Matched512 completion monitor

Final evaluation and independent audit completed on 2026-11-04. All GPUs were
released by 15:31:24 Shanghai; no work remains for this monitor. Deletion is
pending publication and mirroring of the final evidence.

- Automation: `finish-accelerated-matched-512-evaluation`
- Owner chat: `01a1a512-959d-7f42-82b2-9323662ff4dd`
- Final job: `job-2cpg9wglwz5d`, three one-GPU containers
- Execution source: `d9cfdeb4a10e1bf1b2805cc97fa692dd418077d5`
- Audit source: `47ee095ea0499bb991a6f97b0b14507b38a0ca51`
- Conservative all-attempt cost: 25,205/27,600 GPU-seconds
- Scope: original 512 paired cases per task, policies0/130, B8, 600 controls,
  original ABBA/common RNG, exact McNemar/Holm-2 and zero SGD

[Verified result](RESULT.md), [raw audit](FINAL-AUDIT.json),
[cost and release](COST-CLOSEOUT.json), [execution record](EXECUTION.json).

Keep the frozen model/runtime, source checkpoints, shared caches and original
staging copies. No additional compute or cleanup is authorized by this monitor.
