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
