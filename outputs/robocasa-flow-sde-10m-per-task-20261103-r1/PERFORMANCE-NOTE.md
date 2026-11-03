# Timing units and comparison with earlier runs

At18:09Shanghai on2026-11-03, recent complete production intervals are about
257seconds/update, including rollout collection and actor optimization.
OpenDrawer is8/130 and Toaster6/130. At that measured rate the130-update run
takes9.3elapsed hours after startup, approximately74GPU-hours per task on
eight GPUs. The five independent source tasks total about370GPU-hours plus
qualification/startup/evaluation/recovery overhead. Parallel execution reduces
calendar time; queued tasks still need their own full allocation/runtime.
This is an early throughput forecast, not a guaranteed completion time.

TRAIN is128episodes ×600executed controls ×130updates =9,984,000controls per
task,49,920,000for all five. Before/after64cases each add76,800per task.
Every task's actual Q1 episode records verify128complete600-control episodes.
Reset/settling controls, failed attempts and unresolved reservations are
accounted separately. Flow integration steps, action dimensions and repeated
SGD epochs are not new environment controls. [Measured intervals and control
counts](ETA-20261103-1810.json).

The old1MOpenDrawer run's ordinary eight-GPU update took267.7seconds for
51,200controls; the current update takes about257.3seconds for76,800controls.
That is roughly191versus298controls/s across the allocation, a1.56×gain.
The old scientific collection was32envs ×4rounds; this one uses128envs ×1round.
Both collect128episodes before an update. Horizon400→600also increases
decisions and optimization steps per update,16→24. The old complete run's
9.49GPU-hour all-attempt bound is not9.49elapsed hours: its prefix used two
GPUs and its continuation eight. See the [measured migration timings](../robocasa-opendrawer-rlinf-sim-ppo-20261102-r1/migration-8gpu/RESULT.md).

The joint steering plan's earlier40figure is40GPU-hours per arm, equivalent
to about5elapsed hours on eight GPUs. Its initial forecast was38.6–38.8GPU-h;
later measured forecasts rose to49–52GPU-h and the separate steering jobs
were held for review. That plan uses48envs/GPU,384episodes/update and50updates
of small velocity/KV steering policies around a frozen expert. The current
campaign trains the full action expert/value head over130policy versions and
uses16envs/GPU. Its measured actor optimization alone is about115–116seconds
per update. The two workloads do not share an elapsed-time forecast. Sources:
[joint plan](../robocasa-joint-steering-scale50-20261103-r1/PLAN.json),
[later joint projection](../research-coordination-20261103-r1/CHECK-20261103-1350.json).
