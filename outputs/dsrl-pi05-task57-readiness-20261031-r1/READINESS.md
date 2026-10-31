# Frozen pi0.5 / authors' LIBERO90:57 DSRL readiness

Static read-only review, 2026-10-31. Reviewed parent source
`8fefa89d054984c3dde85017b4dc2cc8db617b70`; no tests, model computation,
simulator, allocation or new dataset was run. Ordinary target-online SAC first;
no meta-learning or source40-corpus admission is needed for this experiment.

## Decision

Reuse the qualified authors' PixelSAC learner and OpenPI motor. A narrow task57
driver is needed: the existing protected entrypoints cannot honestly express
this experiment. Do not change the backbone, SAC learner, or old run contracts.
No SAC-math defect was found in the current qualified target collection path.

This is a **pi0.5-adapted DSRL baseline**, not a literal reproduction of the
paper's pi0 result. The authors selected task57 (cream cheese to tray); it is
not chosen from our prior success/failure scores. The paper reports roughly
20% to100% at10k online samples over3 seeds. A new single seed0 run is a pilot,
and the actual pi0.5 native headroom is still unknown.
[Paper §5.5/C.5](https://arxiv.org/html/2506.15799v2#S5.SS5),
[authors' task selection](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_sim.py#L111-L120).

## Fidelity that is already present

| Setting | Released LIBERO recipe | Current qualified pi0.5 path |
| --- | --- | --- |
| Motor | pi0_libero, H50 /execute20 | Entire official OpenPI40 pi05_libero frozen, H10 /execute10,10 flow steps |
| Actor observation | 64×64 rotated agentview RGB +8D robot state | Same; optional4104D prefix is **not** its actor input |
| Actor /critic /temperature LR | 1e-4 /3e-4 /3e-4 | Same |
| Network | PixelSAC3×128,10 critics,mean critic reduction | Same pinned authors' implementation |
| Noise | 32D coordinate tiled over motor horizon; learned bound1 | Same32D coordinate, tiled over10; Gaussian warmup unbounded |
| Warmup /batch /UTD | replay>500 queried chunks /256 /20 | Same; Gaussian until first completed optimizer burst |
| Collection/update order | Whole episode at fixed weights, then chunks×20 updates | Same |
| Reward/mask | −1 per query; final0 on success; bootstrap timeout | Same |
| Discount | .999^20 stored per query | .999^(actual executed duration); deliberate H10 adaptation |
| Stop | 500000 optimizer updates, possible final episode-burst overshoot | Existing onset/long drivers stop at episode milestones; needs task57 profile |
| Evaluation | Initial independent full-horizon Gaussian; learned stochastic SAC;10 episodes /10k optimizer updates | Existing primitives implement both, but old drivers use one query per episode milestone |

Sources: `vla_libero.py:314-418` (`OfficialSAC`),
`vla_source40_corpus.py:46-101` (`OnlineSourceSimulator`),
`vla_libero.py:265-311` (`FrozenOpenPI`),
`vla_target_online_dsrl.py:164-220` (`frozen_query`).
[Published launcher](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/scripts/run_libero.sh#L17-L34),
[SAC parameters](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/launch_train_sim.py#L29-L52),
[update clock](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L97-L155),
[reward and masks](https://github.com/nakamotoo/dsrl_pi0/blob/7f48937d4553e95244cd81c79236a3256df80597/examples/train_utils_sim.py#L270-L287).

The old RoboTwin frozen-noise PPO /1024D feature-SAC results used different
controllers and interfaces; they do not establish that this32D PixelSAC path
cannot learn. Native PPO updates the motor expert and is a separate later arm.

## Concrete driver gaps, not hypothetical learner bugs

1. Onset MAIN binds tasks0/1 and16k actions; six MAIN binds0,10,...50 and64k;
   long MAIN binds old0/40 ancestors and240 episodes. None admits57 or a fresh
   500k-update schedule. Do not bypass or mutate their guards.
2. `run_case:298-301,317-320` insists every training and query episode has one
   nominal state hash. Passing task57 alone would repeat state0. A small new
   profile should reuse collection/recovery primitives but explicitly schedule
   a population; preserve the old fixed-case guard for the old experiment.
3. The legacy `vla_libero --mode teacher` defaults to task57 but is unsuitable
   as the new launcher: `max_steps` means primitive steps (`:480-505`), its
   exact replay==500 branch switches to the random actor before an optimizer
   update (`:488` vs `:503`), and it lacks the target all-in journal/clock.
   The current `OnlineSourceSimulator` already fixes the warmup boundary using
   `learner.updates==0`; do not introduce this legacy defect into the new route.
4. The task57 profile must report optimizer updates AND query records AND actual
   primitives/settling/evaluation costs. 10k replay rows is not10k primitives,
   and500k optimizer updates is not500k simulator actions.

## Minimal single-seed schedule for the narrow new driver

- Fresh PixelSAC seed0. No actor/replay/optimizer from source40 or old targets.
- Outcome-independent training cycle through task57 states0..39. Query panel
  states40..49, same predeclared physical states and stochastic seed bank at
  each checkpoint. Record actual initial hashes, not only seeds. This explicit
  bank is a declared adaptation of the authors' fresh `env.reset()` population.
- Initial10-query native panel uses independent `(10,32)` Gaussian noise.
  Reuse identical held-out physical cases and seeds for each10-query learned
  stochastic-SAC panel. A final native rerun is an additional stability control.
- Stop after the first complete episode/update burst with updates>=500000.
  At H400/ex10, overshoot is at most780 updates. For an unmodified whole-burst
  collector, evaluate after the first burst crossing each10k threshold and
  report the actual update count. Do not label these exact10k-update snapshots.
  Avoid replaying every crossed threshold with identical weights if one burst
  crosses several thresholds (not possible here with800 maximum burst).
- Optional first allocation can stop at a full episode after10k updates and
  resume the SAME full contract to500k; this is an execution slice, not another
  cold start or a new method. No headroom conclusion before native measurement.
- Full actor/critics/targets/temperature/Adam, replay, sampler/Python/NumPy/JAX
  RNG and typed motor RNG saved after every episode+burst. Preserve raw current
  and next image/wrist/state, instruction, actual32D coordinate and tiled motor
  noise, executed actions, primitive rewards, duration, success/timeout and
  complete physical intent journal. Query never enters replay or changes RNG.

## Bounded cost proposal, not allocation authority

One fresh seed, <=10 cumulative GPU-hours including setup/failure/retries and
<=1M actual simulator actions including settling/qualification/evaluation.
Stop with full recovery when either ceiling prevents the next whole episode.

At most500 pre-update query records plus25039 updated records are needed for
the completed-burst500k stop: <=25539 training queries. At10 actions/query and
at most10 settling actions per episode, a deliberately loose all-outcome bound
is510780 training+settling actions. Initial native plus50 learned panels cost
<=510×410=209100 actions; final native10 costs<=4100. Total<=723980 before
qualification/retries, leaving276020 inside1M. With full400-step failures,
training is instead about261k all-in actions; successful shorter episodes
change the primitive/settling balance and must be measured.

Actual old task0/40 totals were5065s/153480 updates and5406s/182400 updates.
Straight-line500k extrapolations are4.58h/4.12h; these include old collection
and few queries, not the new dense evaluation schedule. Measured complete
postwarmup400-action/800-update episodes were20.27-22.32s. Thus10h is a ceiling,
not a guarantee of completing500k and all panels. The first slice measures the
real query/IO/update rates; do not silently extend it or claim convergence.

Prior240-episode results were task0:72/240,8154 query records/153480 updates;
task40:0/240,9600/182400. Fixed state0 and single query/checkpoint, different
targets, motor and optimizer budget cannot refute the published task57 result.
Evidence: `outputs/libero40-target-online-dsrl-long-audit-20261031-r1/README.md`
and `outputs/libero40-target-online-dsrl-long-plan-20261030-r1/budget.json`.

## Exact existing runtime/CLI reuse

Motor Python `/mowenjie/fluffy-bassoon/envs/openpi-215abfb/bin/python`;
OpenPI `/mowenjie/fluffy-bassoon/third_party/openpi-215abfb`;
DSRL `/mowenjie/fluffy-bassoon-2/deps/dsrl-pi0-7f48937-20261029-r1/repo`,
pin `7f48937d4553e95244cd81c79236a3256df80597`;
overlay `/mowenjie/fluffy-bassoon-2/envs/dsrl-pi0-overlay-20261029-r1`;
checkpoint `/mowenjie/fluffy-bassoon/inputs/pi05-libero-openpi-20261027-r1`,
READY receipt `ec064cce32f5003e2e3979c5712d7d6385018f6bfcc59cb2161105881e70d170`.

The existing server command is executable inside the actual allocation with
its qualified simulator PYTHONPATH/rendering environment (the new driver can
reuse `TargetServer` to launch it):

```sh
/mowenjie/envs/wam-libero-plus-py310/bin/python \
  -m fluffy_bassoon.meta_rl.vla_libero --mode env-server \
  --suite libero_90 --task-id 57 --horizon 400 \
  --fresh-environment-on-reset --port 8769 --output "$TASK_ATTEMPT/server-57"
```

`TargetServer`, `OfficialSAC`, `OnlineSourceSimulator`, `run_episode`,
`frozen_query`, `JournaledSimulator` and the generic begin/finish cumulative
worker-clock functions are the reusable implementation, not a new RL stack.
Use the qualified `vla_target_online_dsrl_six.sh` environment/probe setup, but
**its existing7200s/fixed-six contract must not be reused for a10h task57 job**.
There is no existing truthful full-run task57/500k CLI at this reviewed pin;
root owns the narrow new profile and exact launch command after implementation.
