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
fit also passed on the development RTX 4090. [Qualified distributed runtime](../libero90-full-action-20261119-r1/EXECUTION.md)

Baige Distributed Training job `job-str8aynnl1t7` started on one Pro6K
GPU at `758c3522`. The renderer reported NVIDIA RPBZZZ6. The 12-update
adapter smoke passed and the zero-correction commands matched the native
solver exactly. The subsequent native smoke completed all 24 episodes with
paired reset and shared-opening checks passing. The GPU stages used 521 s
before the full fit. In the two-query smoke all arms passed BASE, and the
paced arm was the only success on the S20 reset. This panel qualifies the
runtime and evaluator. [Design](PLAN.md)

## Full adapter fit

`dist03fit` completed all 2,048 updates at batch four, using 8,192 windows,
balanced equally across BASE and S20. The windows came from 1,299 distinct
demonstrations and 602 task/reset combinations across all 24 fitting tasks.
They contained 79,008 valid action targets, including those from 505 partial
windows. The audit matched every sampled window to the fitting manifest and
verified the final checkpoint. The worker used 521 s and handed the frozen
checkpoint to the development evaluator. [Fit evidence](dist03-fit.json)

On held-out fitting-task resets, flow-matching loss for BASE/S20 was
0.482/1.765 for frozen π0.5, 0.255/0.795 for the true timing flag, and
0.257/0.852 for the swapped flag. This measures prediction of observed
demonstrator actions. Native success is measured on the separate live
development panel.

## Native development benchmark

The frozen checkpoint completed 32 fitting episodes for control selection
and 256 development episodes across 32 matched cases. All paired reset and
opening checks passed. The separate fixed choices for after-opening and
from-reset evaluation both selected the BASE flag on fitting cases.
[Paired development evidence](dist03-dev.json)

| Controller | BASE | S20 | Total |
| --- | ---: | ---: | ---: |
| Frozen π0.5 | 16/16 | 1/16 | 17/32 |
| Same solver with zero correction | 16/16 | 1/16 | 17/32 |
| Frozen π0.5 with pacing | 9/16 | 5/16 | 14/32 |
| True timing after shared opening | 3/16 | 0/16 | 3/32 |
| Fixed flag after shared opening | 3/16 | 0/16 | 3/32 |
| Public lag after shared opening | 3/16 | 0/16 | 3/32 |
| True timing from reset | 4/16 | 1/16 | 5/32 |
| Fixed flag from reset | 4/16 | 2/16 | 6/32 |

The public lag rule identified the timing setting on all 32 cases. True
timing had zero paired wins and losses against fixed after the shared
opening, and zero wins and one loss against fixed from reset. Either true
timing arm added zero successes to the 21-case union of frozen π0.5,
pacing and zero correction. The single additional success in the 22-case
union across all arms was the BASE flag from reset on the S20 LIBERO-Object
task 0, reset 19. The learned correction improved held-out demonstration
loss and degraded live execution. This adapter provides little action
coverage for a timing-history learner on this panel. The official test
tasks remain reserved.

The evaluator exited 0 at 17:17:49 Shanghai on Nov 19. It used 2,793
seconds on the allocated Pro6K GPU. Cumulative worker use for qualification,
fitting and evaluation was 1.284 GPUh, including prior attempts. Raw
episodes and the frozen checkpoint are on the Baige project volume under
this run ID.
