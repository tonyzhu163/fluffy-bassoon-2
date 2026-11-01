# Startup failure audit

Audited 2026-10-29 22:29 Asia/Shanghai. Baige job `job-jrjmcfwezo2d`,
immutable source `ad49861bd4624d6564b53127edef508b70189050`, exited **139**.
Scheduler reports 14 minutes 49 seconds running, ending 20:35:02 Shanghai.

The actual one-GPU CUDA allocation probe passed. All **31 focused tests
passed** in 34.11 seconds. The frozen Motus pi0.5 checkpoint loaded as a
3.35B-parameter model with native H32, and the runtime emitted `runtime_ready`
after 125.65 seconds. This does **not** establish simulator or learner startup.

The first `handover_mic` reset repeatedly printed
`018_microphone is not exist model file!`. The installed object inventory
contained only `001_bottle` from the earlier smoke. Upstream `SubEnv.setup_task`
catches setup exceptions and increments the seed without a retry bound; the
missing asset therefore caused hundreds of new renderer scenes before a
pybind11/GIL error and segmentation fault. Renderer receipts did bind to the
allocated GPU, but no successful reset, native-noise parity, real episode,
source transition, SAC optimizer update or PPO optimizer update was recorded.
There is **no resumable learner checkpoint**.

Next: install the pinned public object assets while preserving existing files;
add a structural preflight and bounded setup retries; validate inside a fresh
immutable GPU job. The prior 45 GPU-minute envelope allows a **28-minute**
retry after reserving 2 minutes for r1 and the measured 14m49s for r2. No
larger training allocation is authorized by this audit.

Raw evidence: `console.log`, `exit-code.txt`, `gpu-probe.json`, and
`experiment/{progress.jsonl,contract.json,runtime-allocation.json}`.
