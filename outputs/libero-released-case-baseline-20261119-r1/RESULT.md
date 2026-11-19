# Frozen π0.5 on released LIBERO-Pro2.0 cases

The released evaluator completed all 76 planned single-episode rollouts.
The panel used 38 released instances across the eight project development
tasks, all within the released training partition. The scheduler supplied
original instructions, case parameters, native success and horizons. The
same frozen JAX OpenPI π0.5 ran from each independent episode reset with an
empty action queue and no prior-attempt history. [Protocol](PLAN.md) ·
[Scheduler evidence](release02-evidence.json)

| Released case | Native success |
| --- | ---: |
| BASE, nominal | 16/16 |
| S15, fixed camera calibration | 0/16 |
| S20, fixed timing | 2/16 |
| D17, grasp failure | 13/14 |
| D18, object drop | 9/14 |
| Development subset total | 40/76 |

The nominal motor solved both chosen resets on every task. The only S20
successes were on Goal 0, the middle-drawer task. D18 had five misses
across five task IDs, while the motor succeeded on every corresponding
BASE start. This makes D18 a candidate for a further recovery panel.
The evaluator results identify case assignment. To attribute a miss to
an actual drop, first log whether and when the released event triggered.
The D17 and D18 denominators are smaller because the drawer-opening task
publishes neither case. [Released perturbations](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/PERTURBATIONS.md)

We used the pinned release planner, scheduler, runner and export functions
with the qualified Baige simulator environment. The upstream planner
verified all 339 official test instances at the catalogue level. The scored
run above uses a development subset and paired seed/reset choices. Full
Pro2.0 robustness, the repeated-attempt evaluation on the official test
tasks and LIBERO-Plus remain separate measurements. [Evaluation guide](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/docs/remote-evaluation-guide.md)

## Execution and verification

The first scheduler run stopped on a decoded gripper value of about
−1.0007, beyond the announced lower bound of −1. The rerun clips commands
to the advertised bounds, following the release's π0 adapter. Of 19,034
issued commands, 9,547 changed in the gripper coordinate and three also
changed in vertical translation. The largest corrections were 0.0187
and 0.0192, respectively. This is adapter preprocessing for the published
relative controller. [Upstream π0 adapter](https://github.com/LIBERO-Pro-Team/LIBERO-Pro2.0/blob/e5500c690c1498eca1b7050350654964190cdf70/libero_remote/runner/pi0.py) ·
[OpenPI adapter](evaluate.py)

Baige development RTX 4090. The completed `release02` used `62a0b749`
and released scheduler ID `run_496a4ffe50764fedba039815a6b4870e`.
It exited 0 at 12:42:35 Shanghai on Nov 19. All 76 exports matched their
planned suite, task, case, seed, reset and horizon. The counts in the
export matched the accepted controller steps and action-bound log.
The accepted worker used 1,269 GPU-seconds, with 57 more in the first
attempt, for 0.368 GPU-hours total. Raw planner, scheduler and export
artifacts are on Baige under
`/mowenjie/fluffy-bassoon-2/outputs/libero-released-case-baseline-20261119-r1/`.
