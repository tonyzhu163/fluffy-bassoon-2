# OpenPI on the released LIBERO-Pro2.0 evaluator

Qualify the published scheduler and evaluation protocol on a predeclared
development subset. Use LIBERO-Pro2.0 at e5500c69 and the project development
task IDs from the campaign split. These eight IDs all belong to the release's
training partition. Keep the official test partition reserved.

Select BASE, S15 camera calibration, S20 fixed timing, D17 grasp failure,
and D18 object drop. Use every valid case among the eight tasks in the
release catalog. The drawer-opening task publishes no D17 or D18 case.
Two paired seed/reset combinations (142/0 and 143/3) yield 76 single-episode
rollouts across 38 published instances. The release determines original
instructions, action schema, perturbation parameters, native success and
suite horizons. There are no instruction or environment overrides.

Run the released benchmark planner, scheduler, policy runner and result
export. The project provides an adapter for original JAX OpenPI π0.5 with
bundled LIBERO normalization, ten flow steps and five actions executed per
replan. The adapter clears its queue on every reset and samples explicit
policy noise using the episode seed. There is no cross-episode context.
The server binds to the development container's loopback interface.
Simulator and policy inference share the single allocated GPU.

Report native success by case and suite, and account separately for
infrastructure failures. Interpret D17/D18 by released case assignment,
which can include episodes that never reach the event trigger. The panel
qualifies the release evaluation path and profiles a few mechanisms for
later label collection. It is a development-subset baseline. The full
released Pro2.0 robustness evaluation, the paired retry study on the official
test IDs, and the separate LIBERO-Plus evaluation follow their frozen
protocols. Reserve up to two GPU-hours for this development run.
