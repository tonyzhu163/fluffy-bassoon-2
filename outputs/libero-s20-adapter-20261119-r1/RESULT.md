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
in the simulator interpreter on Baige. [Experiment design](PLAN.md)

`job-ryv45l3e307r` at `41068e3c` passed the Pro6K device probe. It exited
after 17 seconds in the EGL smoke because PyOpenGL still could not load its
OpenGL entry point. The fit and rollouts remain unexecuted. Cumulative
worker use was 0.219 GPUh. The next launcher uses the desktop libraries
and NVIDIA vendor manifest from the successful distributed LIBERO-90
qualification. A CPU-only check in the S20 simulator interpreter loaded
the EGL, OpenGL, and X11 symbols successfully. The S20 12-update batch-two
fit also passed on the development RTX 4090. Full batch-four throughput and rendering in a
scheduled S20 container still need qualification. [Qualified distributed runtime](../libero90-full-action-20261119-r1/EXECUTION.md)
