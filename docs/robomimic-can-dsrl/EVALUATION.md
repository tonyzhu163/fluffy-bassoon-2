# Can DSRL / RL² evaluation

No learner comparison or real-simulator performance result is qualified yet.

The first allocated runtime check used source
`c37ec48143bf2eadb81539f2b8748f64e45f2eab` on Baige
`job-9pi2eysq5lcx`. All 15 focused CPU tests passed in160.85seconds and the
exclusive RTX4090/CUDA attestation passed. Can initialization then failed on
RoboMimic0.3's missing legacy `mujoco_py` import. No Can episode, learner update,
trajectory parity or recovery continuation ran, and native success is unknown.

Raw evidence is preserved in
[`qualification-r2`](../../outputs/robomimic-can-dsrl-20261029-r1/qualification-r2/).
Runtime repair provenance and new submission receipts are in the parent run
directory. Current job state belongs in `outputs/coordination.md`.

The GTrXL qualification on Baige `job-pnesyoazimd0`, source
`c1032117b953acd94592336629a1d28c4f563d62`, passed all69 focused CPU tests
in132.51seconds inside its allocated RTX4090 container. The complete CUDA
allocation probe and real tensor/own-PID worker attestation also passed. These
check architecture equations, replay, resets and recovery contracts; they do
not establish Can simulator parity, learner performance or a memory benefit.
Those real-simulator and learning checks remain pending at the14:37 Shanghai
artifact audit. Evidence: `qualification-r4/` in the parent run directory.
