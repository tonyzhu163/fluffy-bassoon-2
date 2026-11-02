# Native pi0.5 first-success timing

Human request 2026-11-02: obtain first-success p99 per task to anchor a
conservative cutoff strictly below750 controls. This is measurement only;
no PPO, Flora scoring, task substitution, or change to active runs.

The saved250-episode native qualification and its raw monitors contain final
success and physical accounting, but no stepwise success. Timing cannot be
recovered from those records. The new run instruments every native control,
preserving the simulator's original final-success metric separately.

Repeat the same five prospectively fixed50-case headroom banks: OpenDrawer,
PickPlaceCounterToCabinet, PickPlaceDrawerToCounter, TurnOffStove, CoffeeSetupMug.
Use the qualified RoboCasa365 pi0.5 initializer and native H50/10-flow-step
inference. Execute all750 controls (Coffee600), without success termination.
One shared frozen motor per GPU, up to8 concurrent simulator lanes, from the
qualified batch8 collector. A fixed task/case/chunk seed schedule is saved
before any episode. Batch-size numerical drift is already qualified and
remains a limitation; this is a new sample, not exact replay of the old250.

Record one-based `first_success_step` after reset (reset controls excluded),
every successful step, final success, ever success, native case metadata,
physical counters and all three native-camera videos. Timing and simulator
truth stay exclusively in monitoring files, outside motor inputs/PPO records.
Failures have null first-success time; never impute their horizon as success.

Report each task's successful count, nearest-rank empirical p99, maximum and
rounded-up multiple of50 separately for final-success episodes and ever-success
episodes. Use the larger rounded result as the conservative candidate. With
fewer than100 successes, this empirical p99 equals the sample maximum; it is
not a statistically established population99% guarantee. Preserve all failures
as right-censored observations. If the rounded result is750, report that a
sub750 cutoff is unsupported rather than force it to700. Changing training
horizons or Flora calibration is a separate, explicitly recorded migration.

Resource ceiling: one Baige4-node ×2GPU job,64CPU/256GiB per node,20minutes
(9,600GPU-s),250 complete episodes/180,000policy controls,212,000all-in controls
(128 reset reserve/episode),64GiB retained output. No automatic retry or
borrowing. Check the40-experiment-GPU limit before submission; dev is excluded.
Run core CPU tests in allocated Linux, then source-pin/deploy via Git. Check
every actual allocation GPU and one completed real episode before treating
the remaining cohort as qualified. Preserve partial evidence on any failure.
