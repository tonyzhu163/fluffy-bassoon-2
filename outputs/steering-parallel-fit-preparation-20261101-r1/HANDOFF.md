# Concurrent scientific fits: code-only preparation

Based on reviewed main `940aef2e22b6e4c8829a2ef199ba5738e6d6148e`, in a new
detached `steering-parallel-fits` worktree. No model, numerical test, resource
allocation or job was executed by this chat. Actual CPU qualification,
deployment, capacity selection, GPU probes, allocations and observation remain
with the existing execution parent / `finish_assembly`.

## Behavior

The small `steering_history_parallel` module reuses `steering_history.fit_commands`
and the unchanged `vla_history_train`. Both source assemblies must pass the
existing actual source/A2/teacher gates. Their full profile and source prefix must
match, and dataset manifests must retain the same whole-lineage TRAIN/VAL
partitions and provenance. Event-replay trajectories cannot substitute for
qualified RL histories or DPT targets. The old fixed4D screen admitted no arm;
the current event replay bank is still unqualified. There is no actual fit launch.

The exact science profile remains AD full/static and DPT full/static, seed0,
2,500 updates, width128/layers2/heads4, global batch8, memory256, LR3e-4,
validation every250 over16 fixed batches, and final endpoint selection. The
existing360s per-fit /1,440 summed fitting seconds are still profile fields;
the parent chooses execution capacity that can reach the full endpoint. This
code neither shortens the endpoint nor promotes a wall-clock partial.

`prepare` emits four existing-trainer argv templates into one immutable manifest.
`run` binds only allocated CUDA devices, CPU thread counts and optional execution
microbatches. Four selected fits are four separate `Popen`/exec interpreters, all
started before the first wait. They retain the matched seed while owning
independent process-global RNGs, private sampler/RNGs, optimizers, caches and
existing full trainer recoveries. No parent Torch model or learner is shared.
Only scheduler-provided GPU visibility is used. The parent may put all four on
one qualified GPU if measured capacity permits, place them on different allocated
devices, or run disjoint fit-ID subsets in separate qualified jobs.

Each fit atomically claims `processes/<id>/` and has its own `<id>/` trainer output.
The launcher never overwrites an existing process/output identity or retries a
failure. Command, host/device/execution choices, PID, log hashes, exit status and
existing artifact references are saved per fit. A regular failed fit does not
cancel the other independent processes. SIGINT/TERM is forwarded only to this
launcher's own live children; existing complete recoveries remain on disk.

`collect` writes an aggregate only after all four child processes exited normally
and the existing `load_student` admits each actual candidate at update2,500 with
the exact profile/seed/data/teacher contracts. Each `latest.pt` must contain the
matching final weights, optimizer, sampler and Python/NumPy/Torch/CUDA RNG fields.
Missing, partial, inconsistent or interrupted fits leave the aggregate absent.
No auto-resume or checkpoint pooling is added. Existing trainer recoveries remain
the resumable artifacts; any recovery is controlled by the execution owner.

## Qualification handoff

Run only the new focused test file on an actual allocated Linux CPU worker:

```
python -m pytest -q tests/test_steering_history_parallel.py --junitxml=<fresh-receipt>/junit.xml
```

Expected12 synthetic cases. No scientific fit runs in this test. The source/model
admission functions are explicitly isolated in unit fixtures, while production
always calls the existing actual gates and final loader. The process test uses
four REAL standard-library child interpreters and a four-way start barrier; a
serial launch/wait implementation fails. It checks distinct PIDs, independent
same-seed draws, retained scheduler visibility, output claims, independent
failure, fixed-profile rejection of100 updates, split/command mutation rejection,
and all-four-only aggregation with complete/tampered/partial recovery fixtures.
Existing trainer and model-admission tests are unchanged and need no blind rerun.

No allocation is requested here. The parent can choose a bounded CPU envelope
within its actual authorization and settle it using the original receipt process.
Record exact qualified pin, clean source/import/cgroup/zero-GPU evidence, JUnit,
terminal status and conservative CPU-worker cost. Refresh main before deployment;
do not modify active or ended run trees.

## Result analysis

The aggregate keeps per-fit metrics references and full/static endpoint deltas
within AD and within DPT. Train NLL is the final sampled global batch; VAL NLL
averages16 fixed held-out batches. Read the complete curves before diagnosing
underfit or overfit. AD and DPT have different target distributions, so their raw
NLLs are not an objective ranking. Failed/partial fits first indicate capacity or
execution/optimization problems; inspect their logs, update counts, timing and
recovery before drawing architecture conclusions. Poor completed fits require
checking source improvement, teacher competence, labels and normalization.

Frozen simulator readout must still compare full/static/reset and genuine reward
and wrong-condition history controls, plus independent held-out cases/seeds.
Completion of imitation fits supplies no robot-generalization, reward-causal or
ICRL result. Current artifacts contain code preparation only, no finished four-fit
results to analyze.
