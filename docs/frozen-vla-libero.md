# Frozen VLA steering on standard LIBERO

`meta_rl.vla_libero` uses an unchanged standard LIBERO scene, sparse task reward,
robot proprioception and two RGB cameras. A simulator worker and an OpenPI
worker communicate over allocation-local loopback, allowing their separately
qualified Python environments to remain intact. Run both only on allocated
Linux. The GPU probe, actual renderer process context and actual motor compute
context must all identify the same allocation GPU.

The frozen motor is released `pi0_libero` (native horizon 50) or `pi05_libero`
(native horizon 10), using OpenPI source
`215abfb217dbac7d5f1273282331b9b1866c0479`. Each checkpoint carries its own verified
receipt and normalization assets; the requested config must match that receipt.
The driver supplies the native initial-noise argument. Steering predicts a
32-dimensional coordinate repeated over the native horizon. Native evaluation
instead samples every native noise coordinate independently. A separate tied
Gaussian baseline measures the effect of tying. Executed action chunks stop at
success or the original episode horizon; logs retain only executed actions,
their primitive rewards, and their exact interaction count.

The source controller is the authors' unchanged `PixelSACLearner` at
`nakamotoo/dsrl_pi0@7f48937d4553e95244cd81c79236a3256df80597`: a small RGB64 encoder,
public eight-dimensional robot state, three 128-wide hidden layers, ten critics,
mean reduction, automatic target entropy -16, learning rates 1e-4/3e-4/3e-4 and
gamma .999. The original launcher uses 500 warmup steering chunks, batch256,
20 updates per collected chunk and query frequency20. Deviations are explicit
CLI parameters in the immutable run contract. Initial source exploration and
the tied Gaussian baseline are unbounded Gaussian; trained SAC and student
coordinates lie in [-1,1]. A chunk success pays zero, otherwise -1. Time limits
bootstrap the source critic; terminal feedback still reaches the student.

Every source episode has a raw trace and the behavior checkpoint's update
identity. Complete boundary recovery contains actor, critic, target critic,
temperature, all optimizer states, replay, source RNGs and minibatch sampler.
Qualification constructs a separate learner from recovery and compares its
next sampled action and complete optimizer update. Native noise replay and
the frozen visual feature extractor also receive numerical parity checks.

The student interface uses the final shared-prior SAC actor's frozen visual
bottleneck50 plus robot state8. Historical raw RGB64 observations are encoded
again with this one fixed encoder; no representation drifts across source
learning checkpoints. AD predicts actual chronological executed coordinates;
DPT changes only query labels to a saved teacher's decisions. Context retains
the actions actually executed and their observed outcomes. Full trial groups
are split before training windows are sampled.

Initial qualification can distill chronological source episodes to check the
complete software path. A single stationary task and arbitrary consecutive
episodes do not by themselves establish a useful adaptation strategy. For the
robot performance experiment, pass explicit `--retry-train-states`,
`--retry-validation-states` and `--retry-max-steps`. The retry collector restores
the complete same shared SAC prior independently for each scene, repeats that
scene for K attempts, and performs ordinary SAC updates between attempts.
The fixed shared-prior encoder serves every trial. Training and validation
scene indices are disjoint from each other and the evaluation bank. Retain
teacher generation cost separately from deployment interaction count.

Deployment freezes both motor and student weights. Evaluation pairs methods
on initial states and scene seeds and reports per-attempt success plus success
at least once. Comparators are the native motor, tied Gaussian motor, frozen
ordinary SAC, the history Transformer, a separately trained static controller
with matched initialization/data/update budget, and the history Transformer
with state cleared between attempts. Final performance and interaction costs
come from actual own-history robot rollouts; imitation loss is a diagnostic.
