# Allocated Linux checks for staged RoboTwin learning

Prepared 2026-10-30 Asia/Shanghai. One Baige GPU, 16 CPU, 128 GiB RAM,
8 GiB shared memory; hard scheduler ceiling two minutes, automatic retries off.
This fits the original remaining170seconds (42m10s already conservatively
accounted of45minutes). No real simulator policy actions or motor computation
are requested. Run all nine focused test files, including stage interruption,
exact action caps, native noise reproducibility and source RNG/replay isolation.

Mac work is limited to AST/shell checks. All test execution occurs in the
actual scheduler allocation after its GPU-count/CUDA probe. Preserve this source
commit, job exit code, tests log, GPU probe and terminal scheduler screenshot.
The pending two-hour source-learning allocation is separate and unlaunched.
