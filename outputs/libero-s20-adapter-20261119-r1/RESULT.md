# S20 motor qualification and scheduled retry

Baige Distributed Training job `job-wop2qheu0jfh`, pinned to `b57f600c`,
passed the one-GPU probe and completed 12 optimizer updates on the cached
bowl demonstrations. The subsequent live smoke failed while importing
PyOpenGL EGL before its first rollout. Full training and development
evaluation never started. The two distributed steps consumed 0.128 GPUh.
Including development smoke attempts, the campaign consumed 0.214 GPUh
before scheduling the retry.

The replacement build includes the `a2c43644` observation-tail sampling
change and both from-reset arms. Across the 1,411 indexed demonstrations,
eligible partial chunks increased from 1 to 12,695. All 192 sampled windows
matched the stored observations and following-action targets exactly. It
selects a separate fixed timing flag
for the from-reset comparison using fitting resets. The scheduler launcher
loads the shared GLVND dispatch libraries and checks NVIDIA EGL before its
training smoke. The same dispatch libraries restored the missing EGL symbol
in the simulator interpreter on Baige. A clean scheduled fit will start with
fresh optimizer and sampler state. [Experiment design](PLAN.md)
